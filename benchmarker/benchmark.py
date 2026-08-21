"""End-to-end benchmark orchestrator.

Pipeline per model:
1. Generate + render the puzzle set once; every model gets the same directory.
2. Invoke the external agentic harness ONCE with --puzzles-dir. The harness
   drives a single persistent agent through the puzzles sequentially (the
   first is a warmup) and streams one JSONL record per round.
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
from pathlib import Path

import httpx

from .generator import generate_puzzle
from .grader import parse_grader_response, verify
from .openrouter import extract_grid
from .renderer import render_puzzle
from .trials import (
    DEFAULT_TRIALS_DIR,
    load_trial,
    merge as merge_trial,
    puzzle_fingerprint,
    save_trial,
    trial_id,
)

DIFFICULTY_CLUES = {
    "easy": 42,
    "medium": 36,
    "hard": 30,
    "expert": 26,
}

DEFAULT_HARNESS_CMD = "sudoku-agent-harness"
DEFAULT_GRADER_MODEL = "google/gemini-flash-1.5"


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
        grader_model=DEFAULT_GRADER_MODEL,
        harness_timeout=300,
        verbose=False,
        trials_dir=DEFAULT_TRIALS_DIR,
    ):
        self.models = models
        self.n_puzzles = n_puzzles
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.concurrency = concurrency
        self.difficulty = difficulty
        self.harness_cmd = shlex.split(harness_cmd) if isinstance(harness_cmd, str) else list(harness_cmd)
        self.grader_model = grader_model
        self.harness_timeout = harness_timeout
        self.verbose = verbose
        self.trials_dir = Path(trials_dir) if trials_dir else None

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

    async def _run_harness_session(self, model, puzzles_dir, solutions_dir):
        """Invoke the harness ONCE for the whole puzzle set.

        The harness drives a single persistent agent across every puzzle
        sequentially and streams one JSONL record per round to stdout. We
        consume those lines live so progress is visible during long runs.

        Returns (rounds_by_puzzle_id, summary, error_or_None).
        """
        argv = self.harness_cmd + [
            "--model", model,
            "--puzzles-dir", str(puzzles_dir),
            "--output-dir", str(solutions_dir),
            "--timeout", str(self.harness_timeout),
        ]
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
            pid = rec.get("puzzle_id")
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
        session_error=None,
    ):
        """Grade a single solution image produced by the harness."""
        record = {
            "puzzle_id": puzzle["id"],
            "difficulty": puzzle["difficulty"],
        }
        if round_rec is not None:
            record["elapsed"] = round_rec.get("elapsed")
            record["round"] = round_rec.get("round")
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

        solution_path = solutions_dir / f"puzzle_{puzzle['id']:03d}.png"
        if not solution_path.exists():
            record["error"] = f"harness reported success but {solution_path.name} is missing"
            return record

        output_bytes = solution_path.read_bytes()
        async with semaphore:
            try:
                grader_text = await extract_grid(client, output_bytes, self.grader_model)
            except Exception as e:
                record["error"] = f"grader call failed: {type(e).__name__}: {e}"
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

    async def run_model(self, model, puzzles, api_key, out_dir, puzzles_dir):
        """Run one model end-to-end: a single sequential harness session,
        then concurrent grading of everything it produced."""
        safe = model.replace("/", "_").replace(":", "_")
        solutions_dir = out_dir / "solutions" / safe
        solutions_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{model}] starting sequential session over {len(puzzles)} puzzles...")
        rounds, summary, harness_error = await self._run_harness_session(
            model, puzzles_dir, solutions_dir
        )
        if harness_error:
            print(f"  [{model}] harness error: {harness_error}")
        if summary:
            print(
                f"  [{model}] session done: {summary.get('n_solved')}/"
                f"{summary.get('n_puzzles')} produced an image in "
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
                        solutions_dir, harness_error,
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
        return results, summary

    def summarize(self, model, results, session_summary=None):
        n = len(results)
        n_correct = sum(1 for r in results if r.get("verdict", {}).get("correct"))
        n_errors = sum(1 for r in results if "error" in r)
        elapsed = [r["elapsed"] for r in results if r.get("elapsed") is not None]
        avg_time = sum(elapsed) / len(elapsed) if elapsed else 0.0

        per_diff = {}
        for r in results:
            d = r.get("difficulty")
            per_diff.setdefault(d, {"n": 0, "correct": 0})
            per_diff[d]["n"] += 1
            if r.get("verdict", {}).get("correct"):
                per_diff[d]["correct"] += 1

        tok_in = sum(r.get("input_tokens") or 0 for r in results)
        tok_out = sum(r.get("output_tokens") or 0 for r in results)

        return {
            "model": model,
            "n_puzzles": n,
            "n_correct": n_correct,
            "n_errors": n_errors,
            "accuracy": n_correct / n if n else 0.0,
            "avg_elapsed_s": avg_time,
            "input_tokens": tok_in,
            "output_tokens": tok_out,
            "total_tokens": tok_in + tok_out,
            "tokens_per_correct": (tok_in + tok_out) / n_correct if n_correct else None,
            "per_difficulty": per_diff,
            "learning": self._learning_stats(results),
            "session": session_summary or {},
        }

    @staticmethod
    def _learning_stats(results):
        """Split warmup (round 1) from steady state (rounds 2+).

        The whole point of running one persistent agent sequentially is to see
        whether it gets faster/better after building its own infrastructure.
        """
        by_round = sorted(
            (r for r in results if r.get("round") is not None),
            key=lambda r: r["round"],
        )
        if not by_round:
            return {}

        warmup = by_round[0]
        rest = by_round[1:]

        def _avg(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        steady_time = _avg([r.get("elapsed") for r in rest])
        warmup_time = warmup.get("elapsed")
        speedup = (
            warmup_time / steady_time
            if warmup_time and steady_time and steady_time > 0
            else None
        )

        return {
            "warmup_elapsed_s": warmup_time,
            "warmup_correct": bool(warmup.get("verdict", {}).get("correct")),
            "warmup_total_tokens": warmup.get("total_tokens"),
            "steady_state_avg_elapsed_s": steady_time,
            "steady_state_avg_total_tokens": _avg(
                [r.get("total_tokens") for r in rest]
            ),
            "steady_state_accuracy": (
                sum(1 for r in rest if r.get("verdict", {}).get("correct")) / len(rest)
                if rest else None
            ),
            "speedup_after_warmup": speedup,
        }


    def _record_trial(self, summaries, puzzles_meta):
        """Merge this run's summaries into the permanent trial for this puzzle set.

        Only complete, uninterrupted runs are eligible: a session that stopped
        after two of five puzzles could otherwise post a 100% accuracy that
        outranks an honest full run.
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
            )
        except ValueError as e:
            print(f"\nNot recording a trial: {e}")
            return

        save_trial(path, trial)

        print(f"\n=== Trial {tid} ({path}) ===")
        for model, what in changes.items():
            print(f"  {what:<9} {model}")
        for model, why in excluded:
            print(f"  {'skipped':<9} {model} ({why})")
        print("  best-so-far standings:")
        for i, e in enumerate(trial["entries"], 1):
            print(
                f"    {i}. {e['model']}: acc={e['accuracy'] * 100:5.1f}%  "
                f"avg={e['avg_elapsed_s']:6.2f}s  tokens={e.get('total_tokens', 0):>9,}"
                f"  (runs={e.get('runs', 1)})"
            )

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

        summaries = []
        for model in self.models:
            print(f"\n=== Running {model} via {self.harness_cmd[0]} ===")
            results, session_summary = asyncio.run(
                self.run_model(model, puzzles, api_key, self.output_dir, img_dir)
            )

            safe = model.replace("/", "_").replace(":", "_")
            (self.output_dir / f"results_{safe}.json").write_text(
                json.dumps(results, indent=2)
            )

            summary = self.summarize(model, results, session_summary)
            summaries.append(summary)
            print(f"\nSummary for {model}:")
            print(
                f"  Correct: {summary['n_correct']}/{summary['n_puzzles']} "
                f"({summary['accuracy'] * 100:.1f}%)"
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

        summaries.sort(key=lambda s: (-s["accuracy"], s["avg_elapsed_s"]))
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

        self._record_trial(summaries, puzzles_meta)
