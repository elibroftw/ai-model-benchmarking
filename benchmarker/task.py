"""The Sudoku task: everything an agent harness must be TOLD, not know.

The harness is a general-purpose agent runner. It knows how to keep one agent
alive across a series of rounds, hand it an input file, collect an output file,
time it, and stream stats. It knows nothing about Sudoku — and it must not, or
every alternative harness would have to re-implement the same task.

So the task lives here, on the benchmark side, and is passed to whatever
harness is configured as a JSON spec file (see `write_spec`). That spec is the
whole task: the prompts, the filenames the agent reads and writes, and any
files to drop into the agent's working directory (here, the verifier the agent
is required to use instead of writing its own).

Prompts are chosen along two axes the harness resolves at run time:

* first round vs. later rounds — round 1 is a warmup that sets up reusable
  infrastructure and carries the full rules; later rounds are terse, because
  the session is persistent and re-sending the rules wastes context.
* vision vs. text-only — whether the harness could attach the image. A
  text-only model has to read `input.png` with code, which changes the
  instructions substantially. The harness detects this (a provider rejecting
  images) and switches, so both variants are supplied up front.

The only placeholders a harness has to fill are `{round}`, `{n_rounds}`,
`{input_filename}` and `{output_filename}`. Everything else is plain text, and
the harness substitutes by literal replacement — a task prompt containing
braces (JSON examples, dict literals) must not break a run.
"""
import json
from pathlib import Path

# Files handed to the agent, and expected back from it. Names the harness uses
# verbatim; the prompts below refer to them by the same names.
INPUT_FILENAME = "input.png"
OUTPUT_FILENAME = "output.png"

# Which files in the inputs dir are rounds, in sorted order.
INPUT_GLOB = "puzzle_*.png"

# Copied into the agent's working directory at the start of a session.
ASSETS_DIR = Path(__file__).resolve().parent / "agent_assets"
VERIFIER_FILENAME = "verify_sudoku.py"


VISION_READING_SECTION = """## Reading the puzzle

**The puzzle image is attached to this message — look at it directly.** The same image is on disk as `input.png` in your working directory.

Read the clue digits and their positions from the image you were shown. Prefer this over writing OCR or pixel-inspection code — you can already see the digits, and writing extraction code costs most of a round. Only fall back to reading them with code if you genuinely cannot make out the image; such code may classify glyphs, but must never use Sudoku rules to decide or correct a reading.

Use `input.png` on disk for what vision can't give you precisely: grid geometry for rendering (image dimensions, grid bounds, cell size), and as the base image to draw onto.

Start by transcribing the grid into text in your reasoning, using `.` for empty cells, and re-check that transcription against the image before solving. A misread clue wastes the whole round.
"""

TEXT_ONLY_READING_SECTION = """## Reading the puzzle

**You cannot see images, so no image is attached. The puzzle is only available as the file `input.png` in your working directory. Working out what digits it contains is part of your job.**

Write code to extract the clues from those pixels. This is expected and fully allowed — for you it is the only way in. Some things that help:

- `Image`, `ImageDraw` and `ImageFont` from Pillow are already imported and ready to use. OCR libraries are probably NOT installed; check, and be ready to do without.
- The grid is a black-on-white 9x9 with thick lines every 3 cells. Find the grid by locating the dark rows and columns of pixels, then divide the interior into 9 equal cells. Do not assume the grid starts at pixel 0 — there is a margin.
- A cell is empty if its interior is uniformly light. Only non-empty cells need a digit identified.
- To identify a digit, render each of `1`-`9` yourself with Pillow at the cell size and compare against the cell bitmap, taking the closest match. Trying a few font sizes and picking the best score is more reliable than guessing the size. This is ordinary template matching and is a perfectly good approach here.
- Print the grid you extracted and check it looks like a plausible Sudoku (no digit repeated in any row, column, or box among the clues). If a clue looks misread, fix your reader.

Your extraction code may look at pixels and classify glyphs. It must NOT use Sudoku rules to decide or correct a reading, and it must not go on to solve anything — the moment code starts choosing digits for empty cells, it is a solver.

Once you have the clues, write the grid out in your reasoning using `.` for empty cells, and solve from there. A misread clue wastes the whole round, so it is worth checking the transcription before you start deducing.
"""

WARMUP_PROMPT = """You are about to solve a SERIES of {n_rounds} 9x9 Sudoku puzzles in a single session. This is puzzle 1 of {n_rounds} — a warmup / practice round.

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

{reading_section}
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

VISION_NEXT_LINE = "The new puzzle image is attached to this message — read its clues by looking at it. The same image has replaced `input.png` in your working directory, and the grid geometry is unchanged, so reuse your renderer rather than rediscovering the layout."

TEXT_ONLY_NEXT_LINE = "A new puzzle has replaced `input.png` in your working directory. No image is attached — run the extraction code you already wrote to read its clues. The grid geometry is unchanged, so reuse both your reader and your renderer rather than rediscovering the layout."

NEXT_PUZZLE_PROMPT = """Next puzzle (round {round} of {n_rounds}).

{next_reading_line}

Same rule as before: code draws pixels, you do all the Sudoku reasoning. No solver, no candidate elimination in code, and don't write your own checker.

Run `python verify_sudoku.py --text "..."` before rendering. If it won't come back VALID, answer "IMPOSSIBLE" rather than rendering a grid you know is wrong. Your imports, variables, and helper functions from earlier rounds are all still loaded.

Save the solution as `output.png`.
"""


def build_spec():
    """The task as a plain dict: prompts already composed, assets resolved.

    The two reading sections are folded into the round prompts here rather
    than left for the harness to assemble — how this task's instructions come
    together is the task's business, not the runner's.
    """
    return {
        "task": "sudoku-vision",
        "description": (
            "Solve a series of 9x9 Sudoku puzzles given as images, by reasoning "
            "rather than by writing a solver, and render each answer as an image."
        ),
        "input_filename": INPUT_FILENAME,
        "output_filename": OUTPUT_FILENAME,
        "input_glob": INPUT_GLOB,
        # Absolute so a harness running from any cwd can find them.
        "assets": [str(ASSETS_DIR / VERIFIER_FILENAME)],
        "prompts": {
            "first_round": {
                "vision": WARMUP_PROMPT.replace(
                    "{reading_section}", VISION_READING_SECTION
                ),
                "text_only": WARMUP_PROMPT.replace(
                    "{reading_section}", TEXT_ONLY_READING_SECTION
                ),
            },
            "next_round": {
                "vision": NEXT_PUZZLE_PROMPT.replace(
                    "{next_reading_line}", VISION_NEXT_LINE
                ),
                "text_only": NEXT_PUZZLE_PROMPT.replace(
                    "{next_reading_line}", TEXT_ONLY_NEXT_LINE
                ),
            },
        },
    }


# Sections injected when the vision middleware is active.  The harness
# itself is unchanged: the task spec carries the transcription, so every
# harness — including the default one — reads it without knowing where it
# came from.
#
# For vision-capable models the transcription is supplementary — the image is
# still attached, and the transcription helps catch misreads.
# For text-only models the transcription replaces the self-extraction step,
# so the model gets the clues directly and only touches the image for
# geometry.

VISION_WITH_MIDDLEWARE_SECTION = """## Reading the puzzle

**The puzzle image is attached to this message — look at it directly.** The same image is on disk as `input.png` in your working directory.

The image also carries a transcription of its contents as alt text.  Use the alt text as your primary source for the clue digits and their positions — it should save you from reading every digit off the image:

<img src="{input_filename}" alt="\n{transcription}" />

Cross-check a few cells against the attached image to confirm the transcription is accurate.  If you find a discrepancy, trust the image and correct that cell in your reasoning.  A misread clue wastes the whole round.

Use `input.png` on disk for what vision can't give you precisely: grid geometry for rendering (image dimensions, grid bounds, cell size), and as the base image to draw onto.
"""

TEXT_ONLY_WITH_MIDDLEWARE_SECTION = """## Reading the puzzle

**The puzzle image is on disk as `{input_filename}` for rendering purposes only.** You cannot see it, so its contents are described to you as alt text:

<img src="{input_filename}" alt="\n{transcription}" />

You do **not** need to extract the digits from the image — the alt text has them.  Use `input.png` **only** for its geometry: image dimensions, grid bounds, cell size, and as the base image to draw the solution onto.

If the transcription looks inconsistent with Sudoku rules (e.g. a duplicated clue digit in a row), trust the image over the transcription for that cell — the image is the ground truth.
"""


# Used as the alt text when the middleware fails for a puzzle. The prompts are
# shared by every round, so the alt attribute is always present; leaving it
# empty would sit under text telling the model to rely on it, and would tell a
# text-only model in the same breath not to read the image.
TRANSCRIPTION_UNAVAILABLE = (
    "TRANSCRIPTION UNAVAILABLE - the pre-transcription step failed for this "
    "puzzle. Disregard any instruction above about not needing to read the "
    "image: for this round you must read the clue digits from the image file "
    "yourself."
)


def _as_alt_text(text: str) -> str:
    """Make a transcription safe to sit inside an HTML alt attribute.

    The transcription comes from an arbitrary vision model, so it can contain
    anything. Only the quote characters actually matter — they would end the
    attribute early and the grid would spill into the markup.
    """
    return text.replace('"', "&quot;").replace("\r\n", "\n")


def build_spec_with_transcriptions(transcriptions: dict[int, str] | None = None):
    """Like `build_spec`, but with {transcription} placeholders in the prompts.

    When ``transcriptions`` is non-empty the prompts are rewritten to reference
    ``{transcription}``, which the harness fills from ``puzzle_NNN.txt`` files
    saved alongside the puzzle images.  The prompts for vision-capable models
    treat the transcription as supplementary; text-only prompts use it as the
    primary clue source.
    """
    if not transcriptions:
        return build_spec()

    spec = build_spec()
    # The spec is the only channel between the benchmarker and the harness, so
    # the alt text travels in it, keyed by the input file it describes. Nothing
    # is written next to the images: the harness looks up the current round's
    # file name and fills {transcription} from this map.
    spec["transcriptions"] = {
        f"puzzle_{pid:03d}.png": _as_alt_text(text)
        for pid, text in transcriptions.items()
    }

    # Rewrite prompts to include {transcription} placeholder.
    vision_first = WARMUP_PROMPT.replace(
        "{reading_section}", VISION_WITH_MIDDLEWARE_SECTION
    )
    text_first = WARMUP_PROMPT.replace(
        "{reading_section}", TEXT_ONLY_WITH_MIDDLEWARE_SECTION
    )

    # The transcription block must be repeated in the later-round prompts, not
    # merely referred to. Each round is a fresh message with its own puzzle, and
    # the harness fills {transcription} per round from that puzzle's .txt file;
    # a prompt that says "use the transcription" without carrying one leaves the
    # model with nothing — and tells a text-only model not to read the image.
    vision_next = NEXT_PUZZLE_PROMPT.replace(
        "{next_reading_line}",
        VISION_NEXT_LINE
        + "\n\nThis puzzle's image carries its own alt text.  Use it as your "
          "primary source for the clue digits, cross-checking a few cells "
          "against the attached image:\n\n"
          "<img src=\"{input_filename}\" alt=\"\n{transcription}\" />",
    )
    text_next = NEXT_PUZZLE_PROMPT.replace(
        "{next_reading_line}",
        TEXT_ONLY_NEXT_LINE
        + "\n\nThis puzzle's contents are described to you as alt text, so "
          "you do not need to extract the digits.  Use `{input_filename}` "
          "only for its geometry.\n\n"
          "<img src=\"{input_filename}\" alt=\"\n{transcription}\" />",
    )

    spec["prompts"] = {
        "first_round": {
            "vision": vision_first,
            "text_only": text_first,
        },
        "next_round": {
            "vision": vision_next,
            "text_only": text_next,
        },
    }

    return spec


def write_spec(path, transcriptions=None):
    """Write the task spec where a harness can read it, and return the path.

    Written fresh on every run: the prompts are code, and a stale spec left in
    a results directory would silently keep running the previous wording.

    When ``transcriptions`` (a dict mapping puzzle id → text) is provided the
    spec's prompts are rewritten to embed the transcriptions via the
    ``{transcription}`` placeholder the harness already knows how to fill.
    """
    spec = build_spec_with_transcriptions(transcriptions)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2))
    return path
