# Sudoku Vision + Image Editing Benchmark

The purpose of this benchmark is an intelligent test based on SIMPLE Rules where experience does help, but unlike chess there is no need to memorize anything but the rules and some common patterns.

## Running

```sh
uv run run_benchmark.py \
    --harness-cmd "uv run --project ./sudoku-agent-harness sudoku-agent-harness" \
    --n-puzzles 3 \
    --seed 42 \
    --skip-expensive \
    -v
```

Models are read from `models.toml` by default. Drop `--skip-expensive` to
include the models marked `expensive = true`; marking one is inline, so it
stays in whichever category it belongs to:

```toml
{ id = "moonshotai/kimi-k3", expensive = true },
```

Each model gets **one sequential session**: the harness is invoked once with
the whole puzzle directory and drives a single persistent agent through the
puzzles in order. The first puzzle is a warmup where the agent can build
reusable infrastructure; later rounds reuse it. That's deliberate — it tests
whether a model learns within a session, not just whether it can one-shot a
puzzle. The benchmarker reports warmup-vs-steady-state timing so you can see
the learning curve.

```txt
results/
├── puzzles.json                       # generated puzzles + solutions
├── puzzle_images/puzzle_NNN.png       # what each model saw (harness input dir)
├── solutions/<model>/puzzle_NNN.png   # what each model produced
├── <model>.json                       # per-puzzle records incl. round number
└── leaderboard.json                   # sorted by accuracy, then avg time
```

Useful flags: `-v` streams the harness's live agent logs (pair with
`--concurrency 1`), `--grader-model` swaps the vision model used to read
solution images, `--difficulty` pins a single tier.

`results/` is scratch and is overwritten by every run. The durable record is
`trials/`, which is committed:

```txt
trials/
└── seed-42-n5-mixed.json    # one file per puzzle set
```

Each trial keeps only each model's **best** result, ranked by correctness,
then time, then tokens. Re-running the same seed merges into the existing
file, so adding one new model doesn't discard everything already measured —
the run prints whether each model was `new`, `improved`, or `kept`.

Only complete, uninterrupted runs are recorded, so a session stopped after
two of five puzzles can't post a 100% that outranks an honest full run. The
puzzle content is fingerprinted, so if the generator changes and a seed
starts producing different puzzles the merge is refused rather than mixing
incomparable numbers. Unseeded runs aren't recorded at all. `--no-trials`
skips the whole mechanism.

## Grading

1. Correctness
2. Time to solve
3. Cost effectiveness

All agents are given the same generated puzzles. The winner is determined by the best time-to-solve assuming that correctness is 100% for all agents.

## Testing

### Phase 1 Setup

Agents are given two phases. The first phase is setup. We will tell the agent what is expected of them and to prepare themselves to do any preplanning steps presently so that when they are given a new puzzle, they do not have to spend time doing anything repetitive.
The AI is not allowed to write a sudoku solver.

The AI should be smart enough to realize that they should create their own script/harness for answering questions. There will be a sample sudoku given to all AI agents which showcases the input format.

Then we will ask AI to solve the sample sudoku (if it hasn't already) so that it can ensure its ready for grading.

Time: 10 minutes?

### Phase 2 Mass Grading

- Within a session, an agent will be given a sudoku, it answers it, it receives another sudoku. 100 sudokus is probably enough to make a judgement call. During each test-run, all agents will receive the same 100 sudokus.
- The sudokus are generated OUTSIDE the working directory, as to avoid cheating. A future version of the benchmark would run the AI in a docker container to guarantee no cheating.

## Leaderboard

Test harness is only applicable to Best model + new models. If an older model has lost in both correctness and cost effectiveness, it should not be tested ever again. A leaderboard will memorialize to avoid ambiguity if a model is outdated very quickly.

## Implementation Draft

Here’s a concrete, end-to-end plan for testing AI agents on Sudoku using vision input → visual output → automated grading. It’s structured so you can implement it incrementally and keep the focus on reasoning/intelligence rather than memorization or perception hacks.

---

## 1. High-Level Pipeline

```
[Puzzle Generator]
       ↓ (renders as image/HTML)
[Agent receives screenshot of puzzle]
       ↓ (agent solves internally, draws/writes answers into grid area)
[Agent outputs final solution image]
       ↓ [Grader: Image → 9x9 numeric matrix]
[Sudoku Verifier + Clue-Match Check]
       ↓ [Metrics & Intelligence Analysis]
```

---

## 2. Phase-by-Phase Plan

### Phase 1: Puzzle Generation & Input Preparation
Goal: Provide clean, varied, dynamically generated puzzles so agents can’t rely on memorized solutions.

- **Generator**: Use a Sudoku generator that guarantees:
  - Valid puzzle with exactly one solution.
  - Controlled difficulty via number of givens and required solving depth (e.g., basic elimination vs X-Wing/Swordfish chains).
- **Difficulty tiers** (example):
  - Easy: ≥40 clues (mostly single-candidate logic)
  - Medium: 32–39 clues (pairs/triples, hidden singles)
  - Hard: 25–31 clues (advanced patterns, some lookahead)
  - Expert: ≤24 clues (chains, guessing/backtracking needed for humans; tests deeper search/planning)
- **Rendering**:
  - Render puzzles as screenshots of a simple UI:
    - 9×9 grid with clear borders.
    - Original clues in one style (e.g., black serif), empty cells blank or lightly shaded.
    - Leave an editable canvas area where the agent can “write” digits.
  - Optionally inject realistic variations for robustness testing:
    - Different fonts, slight rotation, subtle noise/compression, different background colors.

### Phase 2: Agent Interaction Setup
Goal: Clearly define how the agent perceives and responds so grading is reliable.

Two practical modes (choose one based on your AI’s capabilities):

- **Mode A (Recommended for clean evaluation)**: Interactive UI task
  - Agent sees a screenshot of a web page with:
    - The puzzle grid at fixed coordinates.
    - An input field or clickable cells where it can place digits.
    - Instruction text: “Fill the empty cells with the correct numbers.”
  - Agent interacts (via tool calls, mouse/keyboard simulation, or direct DOM edits) and then captures a final screenshot of the completed grid as its output image.

- **Mode B (Pure vision-to-image)**: Static input → drawn solution
  - Input: Single static screenshot of puzzle only.
  - Output requirement: Agent returns an image where it has visually written/drawn digits into all empty cells of the same grid layout.
  - This better tests generative/visual reasoning but complicates grading (requires robust handwritten digit recognition).

Recommendation: Start with Mode A; switch to B once your grader is stable.

### Phase 3: Output Capture & Preprocessing
Goal: Ensure consistent, crop-friendly solution images for the grader.

- Agent must output a single image containing only the final filled Sudoku grid (or you programmatically crop it).
- Standardize:
  - Resolution (e.g., 1080×1080 or fixed pixel-per-cell size).
  - Grid area bounding box (either enforced by your UI template or detected via edge/grid detection in grading step).

---

### Phase 4: Grading Pipeline (Image → Text → Verification)

This is the critical part. Build it modularly so you can isolate perception errors from reasoning errors.

#### Step 1: Grid Localization & Cell Segmentation
- Input: Final solution image.
- Tasks:
  - Detect main Sudoku grid region using contour detection or template matching.
  - Detect horizontal and vertical lines to define the 9×9 cell layout (or use fixed geometry if UI is standardized).
  - Extract each of the 81 cells as ROIs with consistent size, deskewed and contrast-normalized.

#### Step 2: Digit Recognition per Cell
- For each cell:
  - Classify as “empty” or “filled”.
  - If filled, classify digit 1–9.
- Options:
  - **Lightweight CNN classifier** trained on extracted Sudoku digits (typed and/or handwritten depending on Mode A/B). Far more reliable than generic OCR for this structured task.
  - Or use Tesseract/PaddleOCR with careful preprocessing; tune for single-digit recognition only.
- Output: 9×9 integer matrix `M` where:
  - `0` = originally empty and still empty (shouldn’t happen in solved puzzle)
  - `1–9` = recognized digit

#### Step 3: Clue Preservation Check
- Compare `M` with the original puzzle grid `C`:
  - For every position `(r,c)` where `C[r][c] != 0`, verify `M[r][c] == C[r][c]`.
  - If mismatch → flag as “clue violation” (agent overwrote or misread an initial clue).

#### Step 4: Sudoku Verification Logic
A simple deterministic checker:
- Every row contains digits 1–9 exactly once.
- Every column contains digits 1–9 exactly once.
- Every 3×3 box contains digits 1–9 exactly once.
Return:
- `VALID` / `INVALID`, plus optionally which constraints are violated (for error analysis).

#### Step 5: Grading Output Structure
For each puzzle instance, store:
- Puzzle ID & difficulty tier
- Original grid and final recognized grid
- Clue-match result
- Sudoku validity result
- Error type classification:
  - `PERCEPTION_ERROR`: OCR/cell segmentation failure (detect via local sanity checks or known-given mismatches)
  - `REASONING_ERROR`: Valid perception but invalid Sudoku logic
  - `COMPLETE`: Correct solution

---

### Phase 5: Designing for Intelligence Testing (Not Just “Can It Solve One?”)

To emphasize reasoning/intelligence over rote knowledge or lucky pattern matches:

1. **Dynamic, unseen puzzles**
   - Generate all test puzzles at evaluation time. Never reuse puzzles across runs. Prevents memorization or lookup from training data.

2. **Difficulty scaling**
   - Plot solve rate vs difficulty tier. Agents relying on shallow heuristics will plateau at medium/hard; stronger reasoning agents maintain higher accuracy into hard/expert tiers.

3. **Robustness variants (adaptability tests)**
   - Change visual style: different fonts, colors, grid thicknesses.
   - Add mild noise/compression or slight rotation.
   - Test whether performance degrades only due to perception or also due to failure to generalize reasoning across layouts.

4. **Separate perception from reasoning**
   - Run a subset of puzzles where you feed the grader’s recognized grid directly back into verification without going through the agent again. This gives you:
     - Baseline grader accuracy.
     - Ability to attribute failures correctly (e.g., “agent solved it but OCR misread 4 as 9”).

5. **Optional advanced test: Stepwise validation**
   - If your setup allows, record each number the agent places in order.
   - Verify after each step that no rule is violated. Early violations can indicate guesswork or weak lookahead vs systematic reasoning.

---

### Recommended Tools & Stack (Practical)

- **Puzzle generation**: `sudoku-python`, `pysudokus`, or a simple backtracking generator in Python.
- **UI/Rendering**: Simple HTML/CSS grid served via Flask/FastAPI; screenshot with Playwright/Selenium if simulating real browser usage.
- **Image processing**: OpenCV, Pillow (grid detection, cell cropping).
- **Digit recognition**:
  - Train a small CNN or use an existing single-digit classifier on your rendered digit style.
  - Alternatively: Tesseract/PaddleOCR with heavy preprocessing + confidence thresholds.
- **Verification & metrics**: Pure Python scripts; store results in JSON/CSV for analysis.

---

### Key Pitfalls & How to Avoid Them

| Risk | Mitigation |
|------|------------|
| OCR errors look like reasoning failures | Use a high-quality digit classifier on cropped cells; log confidence scores; separate “perception” vs “logic” error categories. |
| Agent cheats by solving via internal code but drawing randomly | Grade only the final image; for stricter tests, require consistency checks or stepwise validation if your framework supports it. |
| Overfitting to one visual style | Test on multiple renderings (fonts, colors, slight distortions). |
| Too easy → doesn’t differentiate intelligence levels | Use clear difficulty tiers and report performance per tier. |

---

If you tell me:
- What kind of agent you’re testing (multimodal LLM, vision-language model with tools, autonomous desktop agent, etc.),
- Whether Mode A or B is preferred,
I can turn this into a concrete implementation blueprint with sample code structures and prompts.

## Cost-Effective Runs

When testing the benchmark/harness, these models were expensive, so skip them for development.

1. Kimi K3
2. Claude Fable 5
3. MiniMax M3
