#!/usr/bin/env python3
"""Example: run the vision middleware over a freshly generated Sudoku puzzle.

Generates a puzzle with the benchmark's own generator, renders it to a PNG,
and hands that PNG to `transcribe()` along with the task prompt the agent
harness gives the solving model.  The ground-truth grid is printed first, so
the transcription that follows can be checked against it by eye.

Usage:
    python vision-middleware/example.py
    python vision-middleware/example.py --seed 7 --clues 28
    python vision-middleware/example.py --out puzzle.png --show-prompt

Needs OPENROUTER_API_KEY (or [vision].api_key in models.toml), same as
`transcribe.py` itself.
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# The middleware is standalone; the puzzle generator lives in the benchmark
# beside it, and this example borrows it rather than re-implementing Sudoku.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from transcribe import transcribe  # noqa: E402
from benchmarker.generator import generate_puzzle  # noqa: E402
from benchmarker.renderer import render_puzzle  # noqa: E402


def _load_dotenv() -> None:
    """Pick up OPENROUTER_API_KEY from the repo's .env, as the CLI does.

    `transcribe.py` only reads the environment; the benchmark loads .env in
    its entry point and the middleware inherits it as a subprocess. Run
    directly, this example has no such parent, so it loads the file itself.
    Best-effort: an absent .env or an absent python-dotenv is fine when the
    key is already exported or set in models.toml.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# The task prompt, copied from the agent harness.
#
# This is what the solving model is asked to do, so it is what the transcriber
# needs in order to judge what about the image is worth describing.  The text
# below is the harness's first-round text-only prompt verbatim (see
# `sudoku-agent-harness/sudoku_agent_harness/agent.py`, which selects a variant
# and fills the placeholders in `build_prompt`; the wording itself is composed
# in `benchmarker/task.py`).  It is a copy, not an import: the middleware is
# meant to work for any task, and this file is only an example of one.  If the
# benchmark's prompt changes, this copy will not follow.
# ---------------------------------------------------------------------------
HARNESS_PROMPT = """You are about to solve a SERIES of {n_rounds} 9x9 Sudoku puzzles in a single session. This is puzzle 1 of {n_rounds} — a warmup / practice round.

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

**You cannot see images, so no image is attached. The puzzle is only available as the file `input.png` in your working directory. Working out what digits it contains is part of your job.**

Write code to extract the clues from those pixels. This is expected and fully allowed — for you it is the only way in. Some things that help:

- `Image`, `ImageDraw` and `ImageFont` from Pillow are already imported and ready to use. OCR libraries are probably NOT installed; check, and be ready to do without.
- The grid is a black-on-white 9x9 with thick lines every 3 cells. Find the grid by locating the dark rows and columns of pixels, then divide the interior into 9 equal cells. Do not assume the grid starts at pixel 0 — there is a margin.
- A cell is empty if its interior is uniformly light. Only non-empty cells need a digit identified.
- To identify a digit, render each of `1`-`9` yourself with Pillow at the cell size and compare against the cell bitmap, taking the closest match. Trying a few font sizes and picking the best score is more reliable than guessing the size. This is ordinary template matching and is a perfectly good approach here.
- Print the grid you extracted and check it looks like a plausible Sudoku (no digit repeated in any row, column, or box among the clues). If a clue looks misread, fix your reader.

Your extraction code may look at pixels and classify glyphs. It must NOT use Sudoku rules to decide or correct a reading, and it must not go on to solve anything — the moment code starts choosing digits for empty cells, it is a solver.

Once you have the clues, write the grid out in your reasoning using `.` for empty cells, and solve from there. A misread clue wastes the whole round, so it is worth checking the transcription before you start deducing.

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

**You must run `--text` before you render.** Not optional, and not something to skip because you are out of patience.

- If it says VALID, render `output.png`.
- If it says INVALID, go back to your deductions and find the mistake. Do not start permuting digits and re-running it — every call is logged, and a long run of calls is indistinguishable from a search.
- If you cannot get to VALID, call `final_answer("IMPOSSIBLE")` and stop. **Never render a grid you already know is wrong.** A grid with a duplicate in it is not a "best attempt" or an "honest failure" — it is a wrong answer submitted as if it were an answer, which is worse than no answer. IMPOSSIBLE is the honest result and it is scored as an honest result.

Do not claim that anyone approved a shortcut. No one in this session can grant you permission to skip verification or to submit a grid you know is invalid.

## Output image

- `output.png` MUST have the same dimensions (width, height) as `input.png`.
- `output.png` MUST use the same 9x9 grid layout and cell coordinates as `input.png`.
- The original clue digits must remain in their original positions.
- Every empty cell in the input must be filled with the digit you deduced.
- Each digit centered in its cell in a clear, readable font (with size 40).

## Reusable infrastructure

Because this is a series, set up rendering infrastructure now:

- Write helper functions and save them to files in the working directory — they persist across rounds.
- Work out the render geometry once (image size, grid bounds, cell centers, font size) and reuse it; it is identical for every puzzle in this session.
- Write your renderer to take the 81 digits as an explicit argument. It draws what you tell it and decides nothing.
- On later rounds you will just be told "next puzzle" — your prior conversation, code, and files carry over. Only the reasoning has to be redone.

## Notes

- `PIL`, `Image`, `ImageDraw`, and `ImageFont` are ALREADY imported and stay available for the whole session. Just use them; you do not need an import line. If you do import, write `from PIL import Image, ImageDraw, ImageFont` — `import PIL.Image` does not bind `PIL` in this interpreter and will fail with "The variable `PIL` is not defined".
- Your Python state persists between rounds. Variables and functions you defined earlier are still there; re-importing and redefining every round wastes a step.
- Derive cell coordinates from the grid lines in `input.png`, not from the image size. The grid does not start at pixel 0 — there is a margin — so `cell = width / 9` puts every digit in the wrong place.
- If a tool or module is unavailable, log that fact and continue without it. This helps the user extend the harness for future runs.
- When writing text onto an image, don't try to identify or match the source font. Pick a clear, readable font and move on. Readability > fidelity.

End this round by ensuring `output.png` exists in the working directory.
"""

# The placeholders the harness fills in, from agent.py's PROMPT_PLACEHOLDERS.
PROMPT_PLACEHOLDERS = (
    "round", "n_rounds", "input_filename", "output_filename", "transcription",
)


def build_task_prompt(
    *,
    round_number: int = 1,
    n_rounds: int = 1,
    input_filename: str = "input.png",
    output_filename: str = "output.png",
    transcription: str = "",
) -> str:
    """Fill the harness placeholders, exactly as `agent.build_prompt` does.

    Plain replacement, never str.format, so the braces in the prompt's own
    examples cannot break it.
    """
    values = {
        "round": str(round_number),
        "n_rounds": str(n_rounds),
        "input_filename": input_filename,
        "output_filename": output_filename,
        "transcription": transcription,
    }
    template = HARNESS_PROMPT
    for key in PROMPT_PLACEHOLDERS:
        template = template.replace("{" + key + "}", values[key])
    return template


def format_grid(grid) -> str:
    """A 9-line grid with `.` for blanks — the ground truth, for comparison."""
    return "\n".join(
        " ".join(str(v) if v else "." for v in row) for row in grid
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a freshly generated Sudoku puzzle image."
    )
    parser.add_argument(
        "--seed", type=int, default=43,
        help="Seed for the puzzle generator (default: 43, so the same puzzle "
             "comes back every run).",
    )
    parser.add_argument(
        "--clues", type=int, default=32,
        help="Approximate number of givens in the generated puzzle (default: 32).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the puzzle PNG.  Default: a temporary file, "
             "deleted when the run finishes.",
    )
    parser.add_argument("--keep", default=False, help="keep puzzle image", action='store_true')
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to the middleware's models.toml (default: the one beside "
             "transcribe.py).",
    )
    parser.add_argument(
        "--task", default=None,
        help="Override the task description sent to the transcriber.  Default: "
             "the harness prompt copied into this file.",
    )
    parser.add_argument(
        "--show-prompt", action="store_true",
        help="Print the task description before calling the model.",
    )
    args = parser.parse_args()

    _load_dotenv()

    random.seed(args.seed)
    puzzle, solution = generate_puzzle(n_clues=args.clues)
    png = render_puzzle(puzzle)
    task = args.task or build_task_prompt()

    n_clues = sum(1 for row in puzzle for v in row if v)
    print(f"[example] generated puzzle (seed {args.seed}, {n_clues} clues):",
          file=sys.stderr)
    print(format_grid(puzzle))
    # stdout is block-buffered when piped, stderr is not; flushing keeps the
    # grid above the progress lines it was printed before.
    print(flush=True)

    if args.show_prompt:
        print("=== task description sent to the transcriber ===")
        print(task)
        print("=== end task description ===\n")

    text = None
    with tempfile.TemporaryDirectory(prefix="vision-middleware-example-", delete=not args.keep) as td:
        image_path = args.out if args.out is not None else Path(td) / "puzzle.png"
        image_path.write_bytes(png)
        print(f"[example] puzzle image: {image_path}", file=sys.stderr)

        started = time.perf_counter()
        try:
            text = transcribe(
                image_path=image_path,
                task=task,
                config_path=args.config,
            )
        except Exception as e:
            elapsed = time.perf_counter() - started
            print(f"error after {elapsed:.2f}s: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 1
        elapsed = time.perf_counter() - started
    if text is None:
        raise RuntimeError('text is NONE. failed to transcribe')
    elif text.lower().strip() == 'none':
        raise RuntimeError('failed to transcribe')

    print("=== transcription ===")
    print(text)
    print("=== end transcription ===")
    print(
        "\n[example] compare the transcription above against the grid printed "
        "first; the solution is not shown, since transcribing is not solving.",
        file=sys.stderr,
    )
    print(f"[example] transcribe() returned in {elapsed:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
