"""smolagents-based agentic Sudoku harness.

The harness is invoked ONCE per model with a directory of puzzles and drives
the agent through them sequentially, using the SAME CodeAgent and the SAME
working directory across all rounds. `reset=False` on subsequent runs so
that:

- the model sees prior turns in its conversation (episodic memory),
- the Python interpreter state persists (variables, helper functions),
- files the agent writes to the working directory persist round-to-round.

The first puzzle is framed as a warmup — the agent is told there will be N
puzzles total and is encouraged to set up reusable infrastructure. Rounds
2+ get a terse "next puzzle" prompt so we don't repeatedly re-inflate the
system context.

Per-round stats are emitted as JSONL to stdout so the benchmarker can
stream progress in real time. All smolagents console output — including the
live token stream from `stream_outputs=True` — is routed to stderr so stdout
stays a clean, machine-readable JSONL channel.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image


WARMUP_PROMPT = """You are about to solve a SERIES of {n_puzzles} 9x9 Sudoku puzzles in a single session. This is puzzle 1 of {n_puzzles} — a warmup / practice round.

This is a test of YOUR reasoning. A program that solves Sudoku is trivial to write and proves nothing; the entire point of this benchmark is whether you can do the deductions yourself.

## The one rule that matters

**No code may look at, test, or reason about Sudoku digits or constraints. Ever.**

Code is for PIXELS ONLY: reading the image, and drawing the output image. Every decision about which digit goes in which cell must happen in your own written reasoning.

Specifically forbidden — all of these count as cheating, no matter how they are framed:

- A solver, backtracker, brute-forcer, or recursive search.
- Constraint propagation, candidate elimination, or pencil-mark computation in code.
- Code that checks whether a row/column/box is valid, has duplicates, or is complete. Do not write your own checker, assertion script, or "sanity check" over the digits — a checker you control becomes a search loop. A verifier is provided; see below.
- Importing, installing, or calling any Sudoku library, and looking up the answer.
- Asking another model or service to solve it.

Do not try to find a reading of this rule that permits your script. If a piece of code takes digits as input, it is forbidden. The only arrays your code should ever touch are pixel arrays.

Your working directory is archived and inspected after the run. Writing a solver and reporting that you "solved it by hand" is the worst possible outcome — far worse than failing honestly.

If you cannot finish a puzzle by reasoning, say so and return "IMPOSSIBLE". An honest failure is a legitimate result.

## Reading the puzzle

**The puzzle image is attached to this message — look at it directly.** The same image is on disk as `input.png` in your working directory.

Read the clue digits and their positions from the image you were shown. Prefer this over writing OCR or pixel-inspection code — you can already see the digits, and writing extraction code costs most of a round. Only fall back to reading them with code if you genuinely cannot make out the image; such code may classify glyphs, but must never use Sudoku rules to decide or correct a reading.

Use `input.png` on disk for what vision can't give you precisely: grid geometry for rendering (image dimensions, grid bounds, cell size), and as the base image to draw onto.

Start by transcribing the grid into text in your reasoning, using `.` for empty cells, and re-check that transcription against the image before solving. A misread clue wastes the whole round.

## How to solve it

Use `rNcN` notation (r1c1 is top-left, r9c9 bottom-right). Work these techniques in order — always exhaust the cheap ones before reaching for an expensive one:

1. **Hidden singles (start here — usually the most productive).** For each digit 1-9, take one box, row, or column at a time: if the digit can legally go in only one empty cell of that unit, it goes there. Scanning digit-by-digit across all 27 units finds most placements in easy and medium puzzles.
2. **Cross-hatching.** For a given digit, mentally strike out every row and column that already contains it. In any box, the digit must live in the cells that survive; if only one survives, place it.
3. **Naked singles.** For a single empty cell, list which digits its row, column, and box do NOT already contain. If exactly one digit remains, place it.
4. **Keep candidate lists (pencil marks) once scanning dries up.** Write out the possible digits per empty cell in your reasoning and maintain them as you place digits. This is what makes the harder techniques visible.
5. **Locked candidates.** If within a box a digit's only possible cells all sit in one row (or column), that digit can be eliminated from the rest of that row (or column) — and vice versa.
6. **Naked pairs/triples.** If two cells in a unit have exactly the same two candidates, those digits belong to those two cells; remove them from every other cell in the unit. Same idea for three cells sharing three candidates.
7. **Hidden pairs/triples.** If two digits can only go in the same two cells of a unit, those cells hold exactly those digits; strip their other candidates.
8. **X-Wing and similar** only for genuinely hard grids, after the above stop yielding.

Practical advice:

- Start with the digit that already appears most often on the board — it is the most constrained and places fastest.
- After every placement, immediately update the affected row, column, and box; a stale candidate list causes contradictions later.
- Re-scan for hidden singles each time you place a few digits. Puzzles usually re-open.
- Never guess. If you are stuck, you have missed something — re-run step 1 digit by digit across all 27 units rather than branching.
- Before rendering, sanity-check by eye that each row, column, and 3x3 box holds 1-9 once, then confirm with the provided verifier (next section).

Show your deductions as you go. A brief trace ("r3c7=4, hidden single in box 3") is expected, and it is the evidence that the reasoning was yours.

## Checking your work

A verifier is provided as `verify_sudoku.py` in your working directory. Use it instead of writing your own — that is the whole reason it exists.

```
python verify_sudoku.py --text "483957261 915362748 ..."     # 81 digits, any spacing
python verify_sudoku.py --image output.png --reference input.png
```

It prints `VALID`, or `INVALID` with the units that failed. Two things to know:

- `--text` checks the Sudoku constraints on a COMPLETE grid. It rejects partial grids, and it never tells you which cell is wrong or what the answer is. It confirms your reasoning held together; it cannot do the reasoning for you.
- `--image` checks the rendered file structurally: dimensions match, original clues untouched, no cell left blank. It does not read your digits, so run `--text` too.

Use it to confirm a solution you have already reasoned out. If it says INVALID, go back to your deductions and find the mistake — do not start permuting digits and re-running it. Every call is logged and reviewed; a long run of calls is indistinguishable from a search and will be treated as one.

## Output image

- `output.png` MUST have the same dimensions (width, height) as `input.png`.
- `output.png` MUST use the same 9x9 grid layout and cell coordinates as `input.png`.
- The original clue digits must remain in their original positions.
- Every empty cell in the input must be filled with the digit you deduced.
- Each digit centered in its cell in a clear, readable font.

## Reusable infrastructure

Because this is a series, set up rendering infrastructure now:

- Write helper functions and save them to files in the working directory — they persist across rounds.
- Work out the render geometry once (image size, grid bounds, cell centers, font size) and reuse it; it is identical for every puzzle in this session.
- Write your renderer to take the 81 digits as an explicit argument. It draws what you tell it and decides nothing.
- On later rounds you will just be told "next puzzle" — your prior conversation, code, and files carry over. Only the reasoning has to be redone.

## Notes

- If a tool or module is unavailable, log that fact and continue without it. This helps the user extend the harness for future runs.
- When writing text onto an image, don't try to identify or match the source font. Pick a clear, readable font and move on. Readability > fidelity.

End this round by ensuring `output.png` exists in the working directory.
"""

NEXT_PUZZLE_PROMPT = """Next puzzle (round {round} of {n_puzzles}).

The new puzzle image is attached to this message — read its clues by looking at it. The same image has replaced `input.png` in your working directory, and the grid geometry is unchanged, so reuse your renderer rather than rediscovering the layout.

Same rule as before: code draws pixels, you do all the Sudoku reasoning. No solver, no candidate elimination in code, and don't write your own checker — use `verify_sudoku.py` once you have reasoned out the answer. Show your deductions, and return "IMPOSSIBLE" if you genuinely cannot solve it.

Save the solution as `output.png`.
"""


def default_archive_dir() -> Path:
    """`archive/` at the harness repo root, regardless of the caller's cwd."""
    return Path(__file__).resolve().parent.parent / "archive"


STATE_FILENAME = ".harness_state.json"
VERIFIER_FILENAME = "verify_sudoku.py"


def _install_verifier(td: Path) -> bool:
    """Drop the shipped verifier into the agent's working directory.

    The agent is forbidden from writing constraint-checking code, so it is
    given one. Best-effort: if the copy fails the round still runs, just
    without the safety net.
    """
    src = Path(__file__).resolve().parent / VERIFIER_FILENAME
    try:
        shutil.copy(src, td / VERIFIER_FILENAME)
        return True
    except OSError as e:
        print(f"[harness] could not install {VERIFIER_FILENAME}: {e}", file=sys.stderr)
        return False


def _puzzle_set_fingerprint(puzzle_paths: list[Path]) -> str:
    """Content hash of the whole puzzle set.

    Resuming is only safe against the identical set of puzzles. Hashing the
    file contents (not just names) means regenerating puzzles with a new seed
    invalidates the state even though the filenames are unchanged.
    """
    h = hashlib.sha256()
    for p in sorted(puzzle_paths, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _load_state(output_dir: Path, model_id: str, fingerprint: str) -> dict:
    """Return {puzzle_name: round_record} for rounds already completed.

    Empty if there is no state, it belongs to another model, or the puzzle set
    has changed since it was written.
    """
    state_path = output_dir / STATE_FILENAME
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[harness] ignoring unreadable state file: {e}", file=sys.stderr)
        return {}

    if state.get("puzzle_set") != fingerprint:
        print(
            "[harness] puzzle set changed since last run - ignoring saved state "
            "and starting fresh.",
            file=sys.stderr,
        )
        return {}
    if state.get("model") != model_id:
        print(
            f"[harness] saved state is for {state.get('model')!r}, not "
            f"{model_id!r} - ignoring.",
            file=sys.stderr,
        )
        return {}

    done = {}
    for rec in state.get("rounds", []):
        name = rec.get("puzzle_name")
        # Only trust a record whose solution image is actually still there.
        if name and rec.get("success") and (output_dir / name).exists():
            done[name] = rec
    return done


def _save_state(output_dir: Path, model_id: str, fingerprint: str, rounds: list) -> None:
    """Persist completed rounds so an interrupted run can resume.

    Written atomically so a Ctrl+C mid-write can't leave a corrupt file.
    Best-effort: never raises.
    """
    try:
        state_path = output_dir / STATE_FILENAME
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"model": model_id, "puzzle_set": fingerprint, "rounds": rounds},
                indent=2,
            )
        )
        os.replace(tmp, state_path)
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        print(f"[harness] failed to save state: {e!r}", file=sys.stderr)


def _token_counts(agent) -> tuple[int, int]:
    """Session-cumulative (input, output) token counts from the agent's monitor.

    Returns (0, 0) if the monitor is unavailable or hasn't recorded anything,
    so token accounting can never break a run.
    """
    try:
        usage = agent.monitor.get_total_token_counts()
        return int(usage.input_tokens or 0), int(usage.output_tokens or 0)
    except Exception:  # noqa: BLE001 - metrics are best-effort
        return 0, 0


def _safe_model_name(model_id: str) -> str:
    """Turn `moonshotai/kimi-k3` into a single safe path segment."""
    return model_id.replace("/", "_").replace(":", "_")


def _archive_working_dir(td: Path, archive_dir: Path, model_id: str, rounds: list) -> Path | None:
    """Snapshot the agent's working directory for post-hoc cheat inspection.

    Copies every file the agent left behind — its own scripts especially —
    plus a session.json of the round records. Any previous archive for this
    model is replaced so stale runs can't be mistaken for the current one.

    Never raises: losing the archive must not fail an otherwise good session.
    """
    try:
        dest = Path(archive_dir).resolve() / _safe_model_name(model_id)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # input.png/output.png are just the current round's scratch copies;
        # the puzzles and solutions are already saved by the benchmarker.
        shutil.copytree(
            td, dest,
            ignore=shutil.ignore_patterns("input.png", "output.png", "__pycache__"),
        )
        (dest / "session.json").write_text(
            json.dumps({"model": model_id, "rounds": rounds}, indent=2)
        )
        return dest
    except Exception as e:  # noqa: BLE001 - archiving is best-effort
        print(f"[harness] failed to archive working dir: {e!r}", file=sys.stderr)
        return None


class _RoundDeadline:
    """Per-round soft timeout, enforced between agent steps.

    smolagents' CodeAgent takes no `timeout` argument, so we register this as
    a step callback. It can only fire between steps — a single step that hangs
    inside a model call or subprocess will overrun it. The benchmarker's
    subprocess-level timeout is the hard backstop.
    """

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.start: float | None = None

    def reset(self) -> None:
        self.start = time.perf_counter()

    def __call__(self, memory_step, agent=None) -> None:
        if self.start is None or self.seconds <= 0:
            return
        elapsed = time.perf_counter() - self.start
        if elapsed > self.seconds:
            raise TimeoutError(
                f"Round exceeded soft timeout of {self.seconds}s "
                f"(elapsed {elapsed:.1f}s)"
            )


def run(
    model_id: str,
    puzzles_dir: Path,
    output_dir: Path,
    timeout: int,
    archive_dir: Path | None = None,
    fresh: bool = False,
) -> dict:
    """Run the agentic harness across a whole directory of puzzles.

    Args:
        model_id: OpenRouter model ID (e.g. "openai/gpt-4o").
        puzzles_dir: Directory containing puzzle_NNN.png files.
        output_dir: Directory where output_NNN.png files will be written.
        timeout: Per-round soft timeout in seconds.
        archive_dir: Where to copy the agent's working directory when the
            session ends, under `<archive_dir>/<model>/`. Defaults to
            `archive/` at the harness repo root. Pass an explicit path to
            relocate it. This is the audit trail: every script the agent
            wrote is preserved so you can check whether it cheated (e.g.
            by writing a Sudoku solver, which the task prompt forbids).
        fresh: Ignore any saved state and re-solve every puzzle. By default a
            run resumes: puzzles already solved for this model against this
            exact puzzle set are skipped and their stored records replayed.

    Returns:
        dict with `success` (bool), `rounds` (list of per-round records),
        `n_solved` (int), `total_elapsed` (float), `error` (str, optional).
        Each round is also emitted as one JSONL line on stdout as it completes.
    """
    from rich.console import Console
    from smolagents import AgentLogger, CodeAgent, LiteLLMModel, LogLevel

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (check .env file).")

    puzzles_dir = Path(puzzles_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not puzzles_dir.exists():
        raise FileNotFoundError(f"Puzzles directory not found: {puzzles_dir}")

    puzzle_paths = sorted(puzzles_dir.glob("puzzle_*.png"))
    if not puzzle_paths:
        raise FileNotFoundError(
            f"No puzzle_*.png files found under {puzzles_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = _puzzle_set_fingerprint(puzzle_paths)
    completed = {} if fresh else _load_state(output_dir, model_id, fingerprint)
    if completed:
        print(
            f"[harness] resuming: {len(completed)}/{len(puzzle_paths)} puzzles "
            f"already solved for {model_id}. Note the agent starts fresh, so it "
            f"cannot reuse infrastructure the interrupted run built.",
            file=sys.stderr,
        )

    model = LiteLLMModel(
        model_id=f"openrouter/{model_id}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    deadline = _RoundDeadline(timeout)
    agent = CodeAgent(
        tools=[],
        model=model,
        additional_authorized_imports=[
            # Meta
            "importlib", # importlib.reload
            # Image manipulation
            "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
            "PIL.ImageOps", "PIL.ImageFilter",
            "numpy",
            # Filesystem / IO
            "os", "os.path", "io", "pathlib", "shutil", "glob", "tempfile",
            # Data handling
            "json", "base64", "re", "collections", "itertools", "math",
            # Shell / external tools
            "subprocess",
        ],
        max_steps=100,
        stream_outputs=True,
        # CodeAgent takes no `timeout` kwarg; the per-round budget is enforced
        # between steps by this callback instead.
        step_callbacks=[deadline],
        # With stream_outputs=True, smolagents renders the live token stream
        # (and all its other logging) through its logger's rich Console, which
        # defaults to stdout. stdout is our JSONL stats channel, so send the
        # agent's terminal output to stderr instead.
        logger=AgentLogger(
            level=LogLevel.INFO,
            console=Console(file=sys.stderr, highlight=False),
        ),
    )

    # Hold onto the real stdout for JSONL records, then point sys.stdout at
    # stderr for the whole session. smolagents (and anything the agent's own
    # code prints) would otherwise corrupt the machine-readable stdout channel.
    jsonl_out = sys.stdout

    # ONE persistent working dir across all rounds.
    total_start = time.perf_counter()
    rounds = []
    interrupted = False
    # Rounds executed in THIS process, as opposed to replayed from state. The
    # first one gets the warmup prompt and resets the agent, because on resume
    # the agent has no memory of the interrupted run.
    executed = 0
    with tempfile.TemporaryDirectory(prefix="sudoku-harness-") as td:
        td = Path(td)
        cwd = os.getcwd()
        _install_verifier(td)
        os.chdir(td)
        sys.stdout = sys.stderr
        try:
            n_puzzles = len(puzzle_paths)
            for i, puzzle_path in enumerate(puzzle_paths):
                # Already solved against this exact puzzle set: replay the
                # stored record so the caller still sees every puzzle.
                prior = completed.get(puzzle_path.name)
                if prior is not None:
                    record = {**prior, "resumed": True}
                    rounds.append(record)
                    jsonl_out.write(json.dumps(record) + "\n")
                    jsonl_out.flush()
                    print(
                        f"[harness] round {i + 1}/{n_puzzles}: "
                        f"{puzzle_path.name} already solved, skipping",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                # Refresh input.png with the current round's puzzle.
                input_dst = td / "input.png"
                output_src = td / "output.png"
                if output_src.exists():
                    output_src.unlink()
                shutil.copy(puzzle_path, input_dst)

                puzzle_id = _puzzle_id_from_name(puzzle_path)
                first_executed = executed == 0
                if first_executed:
                    prompt = WARMUP_PROMPT.format(n_puzzles=n_puzzles)
                else:
                    prompt = NEXT_PUZZLE_PROMPT.format(
                        round=i + 1, n_puzzles=n_puzzles
                    )

                img = Image.open(input_dst)
                deadline.reset()
                # The monitor only resets on round 1 (reset=True), so its
                # counts accumulate across the session; diff them per round.
                tokens_before = _token_counts(agent)
                start = time.perf_counter()
                error = None
                final_answer = None
                print(
                    f"\n===== Round {i + 1}/{n_puzzles}: "
                    f"{puzzle_path.name} =====",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    final_answer = agent.run(
                        prompt,
                        images=[img],
                        # Reset only on the first round executed in this
                        # process; later rounds keep the session going.
                        reset=first_executed,
                    )
                except KeyboardInterrupt:
                    # Let everything already finished persist, then stop.
                    interrupted = True
                    print(
                        f"\n[harness] interrupted during {puzzle_path.name}; "
                        f"{len(rounds)} completed round(s) saved. Re-run the "
                        f"same command to resume.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                except Exception as e:  # noqa: BLE001 - one bad round must not kill the whole session
                    error = f"{type(e).__name__}: {e}"
                executed += 1
                elapsed = time.perf_counter() - start

                dest = output_dir / puzzle_path.name
                success = output_src.exists() and error is None
                if success:
                    shutil.move(str(output_src), str(dest))

                in_after, out_after = _token_counts(agent)
                round_in = max(0, in_after - tokens_before[0])
                round_out = max(0, out_after - tokens_before[1])

                record = {
                    "puzzle_id": puzzle_id,
                    "puzzle_name": puzzle_path.name,
                    "round": i + 1,
                    "success": success,
                    "elapsed": elapsed,
                    "input_tokens": round_in,
                    "output_tokens": round_out,
                    "total_tokens": round_in + round_out,
                    "final_answer": _stringify(final_answer),
                }
                if error is not None:
                    record["error"] = error
                elif not success:
                    record["error"] = "Agent finished without producing output.png"

                rounds.append(record)
                # Streaming stats: one JSONL line per round as it completes.
                # Written to the REAL stdout, bypassing the stderr redirect.
                jsonl_out.write(json.dumps(record) + "\n")
                jsonl_out.flush()
                # Persist after every round so Ctrl+C at any point keeps all
                # the work done so far.
                _save_state(output_dir, model_id, fingerprint, rounds)
        except KeyboardInterrupt:
            # Ctrl+C outside agent.run() (e.g. while copying files).
            interrupted = True
            print(
                f"\n[harness] interrupted; {len(rounds)} completed round(s) "
                f"saved. Re-run the same command to resume.",
                file=sys.stderr,
            )
        finally:
            _save_state(output_dir, model_id, fingerprint, rounds)
            sys.stdout = jsonl_out
            os.chdir(cwd)
            # Must happen before the TemporaryDirectory context exits and
            # deletes everything, and in `finally` so a crashed session is
            # still archived (those are the runs worth inspecting).
            archived_to = _archive_working_dir(
                td,
                archive_dir if archive_dir is not None else default_archive_dir(),
                model_id,
                rounds,
            )
            if archived_to is not None:
                print(f"[harness] archived working dir -> {archived_to}", file=sys.stderr)

    return {
        # An interrupted session did not cover the puzzle set, so it is not a
        # successful run even though the rounds it did finish are saved.
        "success": not interrupted,
        "interrupted": interrupted,
        "rounds": rounds,
        "n_solved": sum(1 for r in rounds if r["success"]),
        "n_resumed": sum(1 for r in rounds if r.get("resumed")),
        "total_elapsed": time.perf_counter() - total_start,
        # Only counts this process's work; replayed rounds contribute nothing.
        "total_input_tokens": sum(r.get("input_tokens", 0) for r in rounds if not r.get("resumed")),
        "total_output_tokens": sum(r.get("output_tokens", 0) for r in rounds if not r.get("resumed")),
    }


def _puzzle_id_from_name(path: Path) -> int | None:
    """Extract NNN from puzzle_NNN.png; None if the filename doesn't match."""
    stem = path.stem  # e.g. "puzzle_042"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _stringify(x):
    if x is None:
        return None
    s = str(x)
    return s if len(s) <= 2000 else s[:2000] + "...[truncated]"
