"""End-to-end benchmark orchestrator.

Pipeline per model:
1. Generate + render the puzzle set once; every model gets the same directory.
2. Invoke the external agentic harness ONCE, handing it the puzzle directory
   and the task spec written by task.py — the prompts and the agent's verifier
   live here, so the harness stays a general-purpose runner and any other one
   can be substituted. It drives a single persistent agent through the puzzles
   sequentially (the first is a warmup) and streams one JSONL record per round.
3. Grade each produced image by sending it to a cheap vision "grader LLM"
   that transcribes it back to a 9x9 grid.
4. Verify each grid against the puzzle clues and Sudoku rules, and report
   warmup-vs-steady-state stats to expose within-session learning.
"""
import asyncio
import json
import os
import random
import shlex
import time
import tomllib
from pathlib import Path

import httpx

from .generator import generate_puzzle
from .grader import parse_grader_response, verify
from .openrouter import FatalAPIError, extract_grid
from .get_leaderboard import summarize as _summarize
from .task import TRANSCRIPTION_UNAVAILABLE, write_spec as write_task_spec
from .renderer import render_puzzle
from .trials import (
    DEFAULT_TRIALS_DIR,
    _entry_key,
    _rank_key,
    load_trial,
    merge as merge_trial,
    puzzle_fingerprint,
    save_trial,
    trial_id,
)
from .vision import Reader, grade_image, self_check

DIFFICULTY_CLUES = {
    "easy": 42,
    "medium": 36,
    "hard": 30,
    "expert": 26,
}

DEFAULT_HARNESS_CMD = "sudoku-agent-harness"
DEFAULT_HARNESS_ID = "harness"
DEFAULT_GRADER_MODEL = "google/gemma-4-26b-a4b-it"
DEFAULT_VISION_MIDDLEWARE_CMD = ["python", "vision-middleware/transcribe.py"]

# The middleware's exit code for a failure re-running cannot fix. Kept in
# sync with EXIT_CONFIG_ERROR in vision-middleware/transcribe.py; any other
# command used as the middleware should follow the same convention.
MIDDLEWARE_EXIT_CONFIG_ERROR = 3

# Graded per-model records go in their own subdirectory of the results dir.
# The top level holds the run's shared artifacts — puzzles.json, task.json,
# leaderboard.json, puzzle_images/, solutions-<harness>/ — and mixing one
# JSON per model in among them made "which files are model records?" a
# question of exclusion lists rather than location.
FINAL_SUBDIR = "final"


class FatalRunError(RuntimeError):
    """A misconfiguration that makes the rest of the run worthless.

    A dead grader model or a dead transcriber model fails identically for
    every model still queued, so carrying on spends the budget producing
    results nothing verified — and, with the middleware, results recorded as
    if they had alt text they never got. Raised so the run stops at the first
    occurrence instead of at the end of the model list.
    """


class Benchmark:
    def __init__(
        self,
        models,
        n_puzzles=100,
        seed=None,
        output_dir="results",
        concurrency=4,
        difficulty="mixed",
        harness_cmd=DEFAULT_HARNESS_CMD,
        harness_id=DEFAULT_HARNESS_ID,
        grader_model=DEFAULT_GRADER_MODEL,
        harness_timeout=300,
        fresh=False,
        verbose=False,
        trials_dir=DEFAULT_TRIALS_DIR,
        expensive_models=None,
        model_categories=None,
        vision_middleware=False,
        vision_middleware_cmd=None,
        vision_middleware_config=None,
    ):
        self.models = models
        self.n_puzzles = n_puzzles
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.concurrency = concurrency
        self.difficulty = difficulty
        self.harness_cmd = shlex.split(harness_cmd) if isinstance(harness_cmd, str) else list(harness_cmd)
        self.harness_id = harness_id
        self.grader_model = grader_model
        self.harness_timeout = harness_timeout
        self.fresh = fresh
        self.verbose = verbose
        self.trials_dir = Path(trials_dir) if trials_dir else None
        self.expensive_models = set(expensive_models or [])
        # model_categories maps model_id → TOML section name (e.g. "open-weight", "proprietary")
        self.model_categories = model_categories or {}
        self.vision_middleware = vision_middleware
        # Whether transcriptions actually reached the prompts. `middleware`
        # in a result or a trial entry must mean "this run used alt text",
        # not "alt text was requested" — otherwise a run that fell back is
        # recorded as evidence about a middleware it never used.
        self.vision_middleware_applied = False
        self.vision_middleware_cmd = (
            shlex.split(vision_middleware_cmd) if isinstance(vision_middleware_cmd, str)
            else list(vision_middleware_cmd) if vision_middleware_cmd is not None
            else DEFAULT_VISION_MIDDLEWARE_CMD
        )
        self.vision_middleware_config = vision_middleware_config
        # puzzle id → seconds the middleware spent transcribing that
        # image. Charged to every model's round for that puzzle, so a
        # middleware run's times stay comparable with a run without it.
        self.transcription_seconds = {}

    def generate_puzzles(self):
        if self.seed is not None:
            random.seed(self.seed)

        if self.difficulty == "mixed":
            tiers = list(DIFFICULTY_CLUES.keys())
        else:
            tiers = [self.difficulty]

        puzzles = []
        for i in range(self.n_puzzles):
            tier = tiers[i % len(tiers)]
            n_clues = DIFFICULTY_CLUES[tier]
            puzzle, solution = generate_puzzle(n_clues=n_clues)
            puzzles.append(
                {
                    "id": i,
                    "difficulty": tier,
                    "puzzle": puzzle,
                    "solution": solution,
                    "image": render_puzzle(puzzle),
                }
            )
            print(f"  puzzle {i + 1}/{self.n_puzzles} ({tier})")
        return puzzles

    def _transcription_charge(self, puzzle_id):
        """Seconds of middleware time to add to this puzzle's round.

        A transcription is produced once and reused by every model, but a
        model running without the middleware pays for reading the image
        inside its own round. Charging each model the time its puzzle's
        transcription took keeps `elapsed` meaning the same thing on both
        sides: what it cost to get that answer, whichever path produced it.
        Without this, enabling the middleware looks like a free speedup —
        the work moved out of the measured round rather than disappearing.

        A failed transcription is charged too: the time was spent before the
        round could start, and a middleware that fails slowly is not free.

        None when the run had no middleware, so the record keeps the
        harness's own figure untouched.
        """
        if not self.vision_middleware_applied:
            return None
        seconds = self.transcription_seconds.get(puzzle_id)
        return round(seconds, 3) if seconds else None

    async def _run_harness_session(self, model, puzzles_dir, solutions_dir, task_spec):
        """Invoke the harness ONCE for the whole puzzle set.

        The harness drives a single persistent agent across every puzzle
        sequentially and streams one JSONL record per round to stdout. We
        consume those lines live so progress is visible during long runs.

        `task_spec` is the file that makes this a Sudoku benchmark: the prompts
        and the agent's tools live in task.py and are handed over here, so the
        harness is a general-purpose runner and any other one can be dropped in
        without re-implementing the task.

        Returns (rounds_by_puzzle_id, summary, error_or_None).
        """
        argv = self.harness_cmd + [
            "--model", model,
            "--task", str(task_spec),
            "--inputs-dir", str(puzzles_dir),
            "--output-dir", str(solutions_dir),
            "--timeout", str(self.harness_timeout),
        ]
        # Without this the harness resumes: rounds it already completed for
        # this model against this exact puzzle set are replayed from its state
        # file rather than re-run.
        if self.fresh:
            argv.append("--fresh")
        # Strip VIRTUAL_ENV so `uv run --project ...` doesn't warn about the
        # benchmarker's active venv clashing with the harness's project venv.
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        # In verbose mode let the harness's stderr stream to our terminal so
        # smolagents' step-by-step logs are visible live.
        stderr_dest = None if self.verbose else asyncio.subprocess.PIPE
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr_dest,
                env=env,
            )
        except FileNotFoundError as e:
            return {}, {}, f"Harness not found: {e}"

        rounds = {}
        summary = {}
        fatal_error = None
        async for line in proc.stdout:
            line = line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("summary"):
                summary = rec
                continue
            pid = rec.get("item_id", rec.get("puzzle_id"))
            if pid is None:
                # The harness reports fatal setup errors as a bare
                # {"success": false, "error": ...} line with no puzzle_id.
                if rec.get("error"):
                    fatal_error = rec["error"]
                continue
            rounds[pid] = rec
            status = "OK" if rec.get("success") else f"FAILED ({rec.get('error', '?')[:80]})"
            print(
                f"  [{model}] round {rec.get('round')} "
                f"(puzzle {pid}): solved={status} in {rec.get('elapsed', 0):.1f}s",
                flush=True,
            )

        await proc.wait()
        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = (await proc.stderr.read()).decode(errors="replace")

        error = None
        if not rounds:
            error = (
                fatal_error
                or stderr_text.strip()
                or f"harness produced no round records (exit code {proc.returncode})"
            )[:500]
        return rounds, summary, error

    async def _grade_one(
        self, client, semaphore, model, puzzle, round_rec, solutions_dir,
        session_error=None, reader=None,
    ):
        """Grade a single solution image produced by the harness.

        Tries the OpenRouter grader first. If that fails, falls back to
        local pixel-level transcription (deterministic, free, and available
        during every run), so a transient API error never becomes a lost
        data point.
        """
        # Checked for every puzzle, whatever the harness claimed, so the
        # count reflects what is actually on disk rather than what was
        # reported. A model can fail its round and still have written an
        # image, or claim success and have written nothing.
        solution_path = solutions_dir / f"puzzle_{puzzle['id']:03d}.png"
        record = {
            "puzzle_id": puzzle["id"],
            "difficulty": puzzle["difficulty"],
            "output_image": solution_path.exists(),
        }
        if round_rec is not None:
            # The round is charged its puzzle's transcription time (see
            # `_transcription_charge`). The harness's own figure is kept
            # beside it so the adjustment is visible rather than baked in.
            harness_elapsed = round_rec.get("elapsed")
            charge = self._transcription_charge(puzzle["id"])
            record["harness_elapsed"] = harness_elapsed
            record["transcription_elapsed"] = charge
            record["elapsed"] = (
                harness_elapsed + charge
                if harness_elapsed is not None and charge
                else harness_elapsed
            )
            record["round"] = round_rec.get("round")
            # Per round, not per run: a resumed session can replay rounds that
            # ran the other way. Copied so a graded record says for itself
            # whether it had alt text, without consulting the state file.
            record["middleware"] = round_rec.get("middleware")
            record["input_tokens"] = round_rec.get("input_tokens")
            record["output_tokens"] = round_rec.get("output_tokens")
            record["total_tokens"] = round_rec.get("total_tokens")
            record["harness_final_answer"] = round_rec.get("final_answer")

        if round_rec is None:
            # Attribute the failure to the session-level error when there is
            # one; otherwise the generic message hides the real cause.
            record["error"] = (
                f"harness session failed: {session_error}"
                if session_error
                else "harness produced no record for this puzzle"
            )
            return record
        if not round_rec.get("success"):
            record["error"] = f"harness failure: {round_rec.get('error', 'unknown')}"
            return record

        if not record["output_image"]:
            record["error"] = f"harness reported success but {solution_path.name} is missing"
            return record

        output_bytes = solution_path.read_bytes()

        # --- primary path: OpenRouter grader ---
        grader_error = None
        grader_text = None
        async with semaphore:
            try:
                grader_text = await extract_grid(client, output_bytes, self.grader_model)
            except FatalAPIError as e:
                # Not this image's problem: the grader model is unusable, so
                # every remaining puzzle and model would fail the same way,
                # and each one still costs a harness run to produce.
                raise FatalRunError(
                    f"grader model {self.grader_model!r} is unusable "
                    f"(HTTP {e.status}). Nothing further can be graded — fix "
                    f"--grader-model and re-run. ({e})"
                ) from e
            except Exception as e:
                grader_error = f"grader call failed: {type(e).__name__}: {e}"

        # --- fallback: local pixel-level transcription ---
        local = None
        if grader_text is None and reader is not None:
            local = grade_image(reader, puzzle, solution_path)
            if not local.get("read_error"):
                # The local reader could parse the image — use its verdict.
                record["grader_text"] = "[local fallback]"
                record["parsed_grid"] = local.get("grid")
                record["verdict"] = {
                    k: v for k, v in local.items()
                    if k not in ("source", "grid", "read_error")
                }
                record["verdict"]["error_type"] = (
                    local.get("error_type", "LOCAL_FALLBACK")
                    + (" (local fallback)" if grader_error else "")
                )
                if grader_error:
                    record["grader_error"] = grader_error
                return record

        # Neither path produced a usable result.
        if grader_text is None:
            record["error"] = grader_error or "grader produced no response"
            if local is not None and local.get("read_error"):
                record["error"] += f"; local fallback: {local['read_error']}"
            return record

        record["grader_text"] = grader_text
        grid = parse_grader_response(grader_text)
        record["parsed_grid"] = grid
        if grid is None:
            record["verdict"] = {
                "correct": False,
                "error_type": "GRADER_PARSE_ERROR",
            }
        else:
            record["verdict"] = verify(puzzle["puzzle"], grid)
        return record

    async def run_model(self, model, puzzles, api_key, out_dir, puzzles_dir,
                        task_spec, reader=None):
        """Run one model end-to-end: a single sequential harness session,
        then concurrent grading of everything it produced."""
        safe = model.replace("/", "_").replace(":", "_")
        solutions_dir = out_dir / f"solutions-{self.harness_id}" / safe
        solutions_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{model}] starting sequential session over {len(puzzles)} puzzles...")
        rounds, summary, harness_error = await self._run_harness_session(
            model, puzzles_dir, solutions_dir, task_spec
        )
        if harness_error:
            print(f"  [{model}] harness error: {harness_error}")
        if summary:
            n_rounds = summary.get("n_rounds", summary.get("n_puzzles"))
            print(
                f"  [{model}] session done: {summary.get('n_solved')}/"
                f"{n_rounds} produced an image in "
                f"{summary.get('total_elapsed', 0):.1f}s total"
            )

        print(f"  [{model}] grading {len(puzzles)} solution images...")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/sudoku-vision-benchmark",
            "X-Title": "Sudoku Vision Benchmark",
        }
        semaphore = asyncio.Semaphore(self.concurrency)
        results = []
        async with httpx.AsyncClient(headers=headers) as client:
            tasks = [
                asyncio.create_task(
                    self._grade_one(
                        client, semaphore, model, p, rounds.get(p["id"]),
                        solutions_dir, harness_error, reader=reader,
                    )
                )
                for p in puzzles
            ]
            for coro in asyncio.as_completed(tasks):
                r = await coro
                results.append(r)
                pid = r.get("puzzle_id")
                if "error" in r:
                    print(f"  [{model}] puzzle {pid}: ERROR - {r['error']}")
                else:
                    ok = r["verdict"].get("correct", False)
                    et = r["verdict"].get("error_type", "?")
                    mark = "PASS" if ok else f"FAIL ({et})"
                    print(f"  [{model}] puzzle {pid}: {mark}")

        results.sort(key=lambda r: r["puzzle_id"])

        # The session's wall clock is charged what its rounds were, so the
        # total keeps agreeing with the sum of the rounds it covers.
        charged = sum(r.get("transcription_elapsed") or 0 for r in results)
        if summary and charged:
            summary["harness_total_elapsed"] = summary.get("total_elapsed")
            summary["transcription_elapsed"] = round(charged, 3)
            if summary.get("total_elapsed") is not None:
                summary["total_elapsed"] = round(
                    summary["total_elapsed"] + charged, 3
                )

        return results, summary

    async def _transcribe_puzzle(self, puzzle, img_dir, semaphore):
        """Run the vision middleware for one puzzle image.

        Returns (puzzle_id, transcription_text, fatal_message, seconds).
        The text is None when this puzzle failed; `fatal_message` is set only
        for a failure the next puzzle would repeat — a model ID the endpoint
        does not serve, a rejected key — which stops the run rather than
        quietly transcribing nothing. `seconds` is how long the middleware
        took, timed from inside the semaphore so a queued puzzle is not
        charged for waiting its turn: what a model would pay for this
        transcription is the call, not the benchmark's own batching.
        """
        pid = puzzle["id"]
        image_path = img_dir / f"puzzle_{pid:03d}.png"

        async with semaphore:
            task_desc = (
                "Solve a 9x9 Sudoku puzzle. The image shows a partially filled "
                "9x9 grid with some digits (clues) already placed and empty "
                "cells to fill. Transcribe the grid in a clear text format "
                "(e.g. a 9-line grid with digits and dots/zeros for blanks) "
                "so that someone who cannot see the image can solve the puzzle."
            )
            argv = list(self.vision_middleware_cmd) + [
                "--image", str(image_path),
                "--task", task_desc,
            ]
            if self.vision_middleware_config:
                argv += ["--config", str(self.vision_middleware_config)]

            started = time.perf_counter()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                elapsed = time.perf_counter() - started
                if proc.returncode != 0:
                    detail = stderr.decode(errors="replace").strip()[:300]
                    print(
                        f"  [vision-middleware] puzzle {pid}: FAILED "
                        f"(exit {proc.returncode}) after {elapsed:.1f}s: {detail}",
                        flush=True,
                    )
                    # The middleware reports a configuration failure — a model
                    # the endpoint does not serve, a rejected key — with its
                    # own exit code, because the next puzzle would fail the
                    # same way and so would every model still queued.
                    if proc.returncode == MIDDLEWARE_EXIT_CONFIG_ERROR:
                        # Just the failure, not the progress chatter that
                        # precedes it on the same stream.
                        reason = next(
                            (ln for ln in reversed(detail.splitlines())
                             if ln.startswith("error:")),
                            detail.splitlines()[-1] if detail else "",
                        )
                        return pid, None, reason or "configuration error", elapsed
                    return pid, None, None, elapsed
                text = stdout.decode(errors="replace").strip()
                if not text:
                    # Exit 0 with nothing on stdout is a failure, not a
                    # transcription; counting it would put an empty block in
                    # the prompt under text telling the model to rely on it.
                    print(
                        f"  [vision-middleware] puzzle {pid}: FAILED "
                        f"(exit 0 but no output after {elapsed:.1f}s)",
                        flush=True,
                    )
                    return pid, None, None, elapsed
                print(
                    f"  [vision-middleware] puzzle {pid}: OK "
                    f"({len(text)} chars in {elapsed:.1f}s)",
                    flush=True,
                )
                return pid, text, None, elapsed
            except Exception as e:
                elapsed = time.perf_counter() - started
                print(
                    f"  [vision-middleware] puzzle {pid}: ERROR "
                    f"{type(e).__name__} after {elapsed:.1f}s: {e}",
                    flush=True,
                )
                return pid, None, None, elapsed

    async def _run_vision_middleware(self, puzzles, img_dir):
        """Transcribe every puzzle image through the vision middleware.

        Returns (transcriptions, n_ok, seconds): a dict mapping puzzle id →
        alt text for EVERY puzzle, how many of those are real transcriptions,
        and a dict mapping puzzle id → the seconds that puzzle's transcription
        took, which every model's round for it is later charged. A puzzle
        the middleware failed on still gets an entry, carrying
        ``TRANSCRIPTION_UNAVAILABLE`` — the prompts are shared across rounds,
        so the alt attribute is always rendered and a missing entry would show
        the model an empty one.

        Raises FatalRunError when the middleware is misconfigured, or when it
        transcribed nothing at all. Both mean every model in the run would be
        prompted exactly as if `--vision-middleware` had not been passed,
        which is not the experiment that was asked for and costs a full run to
        discover afterwards.
        """
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [
            asyncio.create_task(self._transcribe_puzzle(p, img_dir, semaphore))
            for p in puzzles
        ]
        transcriptions, n_ok, fatal = {}, 0, None
        seconds = {}
        for coro in asyncio.as_completed(tasks):
            pid, text, fatal_msg, elapsed = await coro
            seconds[pid] = elapsed
            if fatal_msg and fatal is None:
                fatal = fatal_msg
            if text is None:
                transcriptions[pid] = TRANSCRIPTION_UNAVAILABLE
            else:
                transcriptions[pid] = text
                n_ok += 1

        if fatal is not None:
            raise FatalRunError(
                f"vision middleware is misconfigured: {fatal.rstrip('.')}. "
                f"Fix {self.vision_middleware_config or 'vision-middleware/models.toml'} "
                f"and re-run; nothing was graded."
            )
        if not n_ok:
            raise FatalRunError(
                f"vision middleware transcribed 0/{len(puzzles)} puzzles. "
                f"Every model would run the standard prompts, so the run "
                f"would measure the opposite of what --vision-middleware asks."
            )
        return transcriptions, n_ok, seconds

    def _middleware_model(self):
        """The transcriber the middleware is configured to use, for the record.

        Read from the same config the middleware itself reads: the explicit
        --vision-middleware-config, else models.toml beside the middleware
        script. Best-effort — a custom middleware need not use a config at
        all, and the audit index is still worth writing without this.
        """
        candidates = []
        if self.vision_middleware_config:
            candidates.append(Path(self.vision_middleware_config))
        script = next(
            (a for a in reversed(self.vision_middleware_cmd) if a.endswith(".py")),
            None,
        )
        if script:
            candidates.append(Path(script).resolve().parent / "models.toml")
        for path in candidates:
            try:
                return tomllib.loads(path.read_text())["vision"]["model"]
            except (OSError, KeyError, tomllib.TOMLDecodeError):
                continue
        return None

    def _save_transcriptions(self, transcriptions, img_dir):
        """Write the middleware's output to the results dir, for auditing.

        The transcriptions otherwise live only inside the prompts in
        task.json, so nothing afterwards can check what the middleware
        actually said about each puzzle — and a middleware run's timings and
        accuracy only mean something next to that text. Each puzzle's
        transcription gets its own file, beside an index naming the
        transcriber and recording, per puzzle, whether it produced anything
        and the seconds every model's round for it was charged.

        Best-effort: an audit copy that cannot be written must not stop a run
        whose prompts already carry the text.
        """
        out_dir = self.output_dir / "transcriptions"
        entries = []
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            for pid in sorted(transcriptions):
                text = transcriptions[pid]
                ok = text != TRANSCRIPTION_UNAVAILABLE
                name = None
                if ok:
                    name = f"puzzle_{pid:03d}.txt"
                    (out_dir / name).write_text(text)
                entries.append({
                    "puzzle_id": pid,
                    "image": f"puzzle_{pid:03d}.png",
                    "file": name,
                    "ok": ok,
                    "chars": len(text) if ok else 0,
                    # The figure added to this puzzle's round for every model
                    # (see `_transcription_charge`), whether it succeeded or
                    # not: the run waited for it either way.
                    "elapsed_s": round(self.transcription_seconds.get(pid) or 0.0, 3),
                })
            charged = [e["elapsed_s"] for e in entries if e["elapsed_s"]]
            (out_dir / "index.json").write_text(json.dumps({
                "model": self._middleware_model(),
                "command": self.vision_middleware_cmd,
                "config": (
                    str(self.vision_middleware_config)
                    if self.vision_middleware_config else None
                ),
                "images_dir": str(img_dir),
                "n_puzzles": len(entries),
                "n_ok": sum(1 for e in entries if e["ok"]),
                "total_elapsed_s": round(sum(charged), 3),
                "avg_elapsed_s": round(sum(charged) / len(charged), 3) if charged else None,
                "puzzles": entries,
            }, indent=2) + "\n")
        except OSError as e:
            print(f"  [vision-middleware] could not save transcriptions "
                  f"for auditing: {e}")
            return None
        return out_dir

    @staticmethod
    def _blanket_failure(results):
        """The one error every round of this model died of, or None.

        A model that fails every single round with the *identical* message is
        rarely a model problem: a broken harness dependency, an exhausted key,
        a bad endpoint. The signature is the error text with the round-specific
        tail cut off, so two rounds that failed the same way match.
        """
        if not results:
            return None
        errors = set()
        for r in results:
            if r.get("verdict", {}).get("correct") or r.get("output_image"):
                return None
            error = r.get("error")
            if not error:
                return None
            errors.add(error[:120])
        return errors.pop() if len(errors) == 1 else None

    def summarize(self, model, results, session_summary=None):
        """Per-model metrics. The implementation lives in get_leaderboard.py so that
        summarize_results.py can rebuild the identical numbers from disk."""
        return _summarize(model, results, session_summary)

    def _record_trial(self, summaries, puzzles_meta, *, standings=True):
        """Merge summaries into the permanent trial for this puzzle set.

        Called once per model, as soon as that model has finished all its
        rounds, rather than once at the end of the run. A benchmarking session
        is long and routinely gets stopped partway; recording as we go means
        every model that actually completed is in the record, and a harness
        change can be compared against it while the rest of the run is still
        going. Each entry is stamped with the commit it came from, so the
        record says which code produced it.

        Only complete, uninterrupted runs are eligible: a session that stopped
        after two of five puzzles could otherwise post a 100% accuracy that
        outranks an honest full run.

        With `standings=False` the outcome is reported as one line per model
        instead of the full best-so-far table, which would otherwise be
        reprinted after every model.
        """
        if self.trials_dir is None:
            return
        if self.seed is None:
            print(
                "\nNot recording a trial: runs without --seed generate different "
                "puzzles each time, so their scores are not comparable."
            )
            return

        eligible, excluded = [], []
        for s in summaries:
            session = s.get("session") or {}
            if s.get("n_puzzles") != self.n_puzzles:
                excluded.append((s["model"], "incomplete run"))
            elif session.get("interrupted"):
                excluded.append((s["model"], "session interrupted"))
            else:
                eligible.append(s)

        tid = trial_id(self.seed, self.n_puzzles, self.difficulty)
        path = self.trials_dir / f"{tid}.json"
        fingerprint = puzzle_fingerprint(puzzles_meta)

        try:
            trial, changes = merge_trial(
                load_trial(path),
                eligible,
                tid=tid,
                seed=self.seed,
                n_puzzles=self.n_puzzles,
                difficulty=self.difficulty,
                fingerprint=fingerprint,
                middleware=self.vision_middleware_applied,
            )
        except ValueError as e:
            print(f"\nNot recording a trial: {e}")
            return

        save_trial(path, trial)

        by_key = {_entry_key(e): e for e in trial["entries"]}
        if standings:
            print(f"\n=== Trial {tid} ({path}) ===")
        for key, what in changes.items():
            model, mw = key if isinstance(key, tuple) else (key, False)
            mw_tag = " [mw]" if mw else ""
            if standings:
                print(f"  {what:<9} {model}{mw_tag}")
            else:
                rev = (by_key.get(key) or {}).get("rev")
                print(
                    f"  Trial {tid}: {what} — {model}{mw_tag}"
                    + (f"  [rev {rev}]" if rev else "")
                )
        for model, why in excluded:
            if standings:
                print(f"  {'skipped':<9} {model} ({why})")
            else:
                print(f"  Trial {tid}: not recorded — {model} ({why})")
        if standings:
            self._print_trial_standings(trial)
        return trial

    def _print_trial_standings(self, trial=None):
        """Print the trial's best-so-far table.

        Read-only by default: the entries were merged model by model as the
        run went, so re-merging here would count every model as a second
        attempt and inflate its `runs`.
        """
        if self.trials_dir is None or self.seed is None:
            return
        tid = trial_id(self.seed, self.n_puzzles, self.difficulty)
        path = self.trials_dir / f"{tid}.json"
        if trial is None:
            trial = load_trial(path)
            if not trial:
                return
            print(f"\n=== Trial {tid} ({path}) ===")
        print("  best-so-far standings:")
        for i, e in enumerate(trial.get("entries") or [], 1):
            imgs = e.get("n_output_images")
            img_col = (
                f"  imgs={imgs}/{e.get('n_puzzles', '?')}" if imgs is not None else ""
            )
            mw = "yes" if e.get("middleware") else " --"
            rev = e.get("rev") or "?"
            print(
                f"    {i}. {e['model']}: acc={e['accuracy'] * 100:5.1f}%  "
                f"avg={e['avg_elapsed_s']:6.2f}s  tokens={e.get('total_tokens', 0):>9,}"
                f"  mw={mw}{img_col}  (runs={e.get('runs', 1)}, rev={rev})"
            )

    def _model_run_order(self, models):
        """Flat sort: least missing → never-run → open-weight → fewest failures → least expensive.

        1st: Models with an existing solutions dir but *some* PNGs missing
             (interrupted mid-run).  Sorted by least-missing first.
        2nd: Models never run (no solutions subdir at all).
        3rd: Everything else (all PNGs present).  Sorted by fewest failures
             then least expensive.

        Within each of the three groups the same sub-sort applies:
          open-weight before proprietary, fewest failures first, and
          non-expensive before expensive.
        """
        _cat_prio = {"open-weight": 0, "proprietary": 1}
        n = self.n_puzzles

        def _key(model):
            safe = model.replace("/", "_").replace(":", "_")
            sol_dir = self.output_dir / f"solutions-{self.harness_id}" / safe

            expensive = 1 if model in self.expensive_models else 0
            cat_order = _cat_prio.get(self.model_categories.get(model, ""), 2)

            if not sol_dir.is_dir():
                return (1, 0, cat_order, 0, expensive, model)

            png_count = len(list(sol_dir.glob("*.png")))
            missing = n - png_count

            if missing > 0:
                return (0, missing, cat_order, 0, expensive, model)

            failed = 0
            results_path = self.output_dir / FINAL_SUBDIR / f"{safe}.json"
            if not results_path.exists():
                # Runs made before the records moved into final/.
                results_path = self.output_dir / f"{safe}.json"
            if results_path.exists():
                try:
                    records = json.loads(results_path.read_text())
                    if isinstance(records, list):
                        failed = sum(
                            1 for r in records
                            if r.get("verdict", {}).get("correct") is False
                        )
                except (json.JSONDecodeError, OSError):
                    pass
            return (2, 0, cat_order, failed, expensive, model)

        return sorted(models, key=_key)

    def run(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill it in."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {self.n_puzzles} puzzles (difficulty={self.difficulty})...")
        puzzles = self.generate_puzzles()

        puzzles_meta = [
            {
                "id": p["id"],
                # The seed that generated this set. Carried per puzzle so the
                # file stays a plain list — every reader here indexes it as
                # one — while still saying which run it belongs to. None when
                # the run was unseeded, and therefore not reproducible.
                "seed": self.seed,
                "difficulty": p["difficulty"],
                "puzzle": p["puzzle"],
                "solution": p["solution"],
            }
            for p in puzzles
        ]
        (self.output_dir / "puzzles.json").write_text(json.dumps(puzzles_meta, indent=2))

        # The harness reads puzzles from this directory. Every model gets the
        # exact same directory, so the puzzle set is identical across models.
        img_dir = self.output_dir / "puzzle_images"
        img_dir.mkdir(exist_ok=True)
        for p in puzzles:
            (img_dir / f"puzzle_{p['id']:03d}.png").write_bytes(p["image"])

        # The task itself — prompts, filenames, and the verifier the agent is
        # required to use — written where the harness can read it. Rewritten
        # every run so an edit to task.py takes effect immediately.

        transcriptions = None
        if self.vision_middleware:
            print(
                f"\nRunning vision middleware "
                f"({self.vision_middleware_cmd[0]}) on {len(puzzles)} puzzles..."
            )
            transcriptions, n_ok, self.transcription_seconds = asyncio.run(
                self._run_vision_middleware(puzzles, img_dir)
            )
            charged = [s for s in self.transcription_seconds.values() if s]
            print(
                f"Vision middleware: {n_ok}/{len(puzzles)} puzzles transcribed"
                + (
                    f", {sum(charged):.1f}s of transcription "
                    f"({sum(charged) / len(charged):.1f}s per puzzle, added to "
                    f"every model's round for that puzzle so the times stay "
                    f"comparable with a run without the middleware)"
                    if charged else ""
                )
            )
            # A total failure raises rather than falling back: see
            # `_run_vision_middleware`.
            self.vision_middleware_applied = True
            saved = self._save_transcriptions(transcriptions, img_dir)
            if saved is not None:
                print(f"Transcriptions saved for auditing: {saved}")

        task_spec = write_task_spec(
            self.output_dir / "task.json",
            transcriptions=transcriptions,
        )
        print(f"Task spec for the harness: {task_spec}")

        # Calibrate the local vision reader once from the puzzle images we just
        # rendered. If it passes its self-check it becomes the fallback grader
        # for every model, so a transient API error never loses a data point.
        reader = None
        try:
            candidate = Reader.calibrate(img_dir, puzzles_meta)
            check = self_check(candidate, img_dir, puzzles_meta)
            if check["all_ok"]:
                reader = candidate
                print(
                    f"Local grader: calibrated and self-checked OK "
                    f"({check['n_ok']}/{check['n_checked']} puzzles)"
                )
            else:
                print(
                    f"Local grader: FAILED self-check "
                    f"({check['n_ok']}/{check['n_checked']} puzzles reproduced) "
                    f"— API grader errors will NOT be rescued."
                )
                for c in check["checks"]:
                    if not c["ok"]:
                        print(f"  puzzle {c['puzzle_id']}: {c.get('error', 'mismatch')}")
        except Exception as e:
            print(f"Local grader: could not calibrate ({e}) — API fallback disabled.")

        summaries = []
        # (error signature, models it has hit in a row). Two different models
        # failing every round the same way is a broken run, not two broken
        # models; carrying on spends a harness session per model to rediscover
        # it. See `_blanket_failure`.
        blanket = (None, [])

        # Sort models so the ones that need the most work run first: interrupted
        # models with missing outputs first, then never-run models, then complete
        # models with fewest failures first.  Every tier sorts non-expensive
        # before expensive, then open-weight before proprietary.
        sorted_models = self._model_run_order(self.models)
        if sorted_models != self.models or self.expensive_models or self.model_categories:
            missing_count = sum(
                1 for m in sorted_models
                if (self.output_dir / f"solutions-{self.harness_id}"
                    / m.replace("/", "_").replace(":", "_")).is_dir()
                and len(list((self.output_dir / f"solutions-{self.harness_id}"
                    / m.replace("/", "_").replace(":", "_")).glob("*.png")))
                < self.n_puzzles
            )
            new_count = sum(
                1 for m in sorted_models
                if not (self.output_dir / f"solutions-{self.harness_id}"
                        / m.replace("/", "_").replace(":", "_")).is_dir()
            )
            expensive_in_run = self.expensive_models & set(sorted_models) if self.expensive_models else set()

            tiers = []
            if missing_count:
                tiers.append(f"{missing_count} interrupted (missing runs)")
            if new_count:
                tiers.append(f"{new_count} never run")
            tiers.append("complete")
            if expensive_in_run:
                tiers.append(f"({len(expensive_in_run)} expensive deferred)")
            print(f"Priority order: {' > '.join(tiers)}")

        for model in sorted_models:
            print(f"\n=== Running {model} via {self.harness_cmd[0]} ===")
            results, session_summary = asyncio.run(
                self.run_model(
                    model, puzzles, api_key, self.output_dir, img_dir, task_spec,
                    reader=reader,
                )
            )

            safe = model.replace("/", "_").replace(":", "_")
            final_dir = self.output_dir / FINAL_SUBDIR
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / f"{safe}.json").write_text(
                json.dumps(results, indent=2)
            )

            signature = self._blanket_failure(results)
            if signature is not None and signature == blanket[0]:
                blanket[1].append(model)
                if len(blanket[1]) >= 2:
                    raise FatalRunError(
                        f"{len(blanket[1])} models in a row failed every round "
                        f"with the same error, so the cause is the run, not the "
                        f"models: {', '.join(blanket[1])} — {signature.strip()}. "
                        f"Nothing further would be measured; the models already "
                        f"finished are recorded."
                    )
            else:
                blanket = (signature, [model] if signature else [])

            summary = self.summarize(model, results, session_summary)
            summary["middleware"] = self.vision_middleware_applied
            summaries.append(summary)
            # Into the permanent record now, not at the end of the run: this
            # model is finished, and a session stopped later must not lose it.
            self._record_trial([summary], puzzles_meta, standings=False)
            print(f"\nSummary for {model}:")
            print(
                f"  Correct: {summary['n_correct']}/{summary['n_puzzles']} "
                f"({summary['accuracy'] * 100:.1f}%)"
            )
            print(
                f"  Output images: {summary['n_output_images']}/{summary['n_puzzles']} "
                f"({summary['output_rate'] * 100:.1f}%)"
            )
            print(f"  Avg time: {summary['avg_elapsed_s']:.2f}s")
            print(
                f"  Tokens: {summary['total_tokens']:,} "
                f"(in {summary['input_tokens']:,} / out {summary['output_tokens']:,})"
            )
            learning = summary.get("learning") or {}
            if learning.get("warmup_elapsed_s") is not None:
                steady = learning.get("steady_state_avg_elapsed_s")
                speedup = learning.get("speedup_after_warmup")
                print(
                    f"  Warmup: {learning['warmup_elapsed_s']:.1f}s -> steady state: "
                    + (f"{steady:.1f}s" if steady is not None else "n/a")
                    + (f" ({speedup:.1f}x faster)" if speedup else "")
                )

        summaries.sort(key=_rank_key)
        (self.output_dir / "leaderboard.json").write_text(json.dumps(summaries, indent=2))

        print("\n=== Leaderboard (correctness, then avg time) ===")
        for s in summaries:
            learning = s.get("learning") or {}
            speedup = learning.get("speedup_after_warmup")
            suffix = f"  warmup-speedup={speedup:.1f}x" if speedup else ""
            print(
                f"  {s['model']}: acc={s['accuracy'] * 100:5.1f}%  "
                f"avg={s['avg_elapsed_s']:6.2f}s  "
                f"tokens={s['total_tokens']:>9,}{suffix}"
            )

        # Every model was recorded as it finished; this only reads the record
        # back, so nothing is counted as a second attempt.
        self._print_trial_standings()
