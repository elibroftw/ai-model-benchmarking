# sudoku-agent-harness

Agentic Sudoku-solving harness. Given a *directory* of puzzle images and an
OpenRouter model ID, this program runs a single persistent
[smolagents](https://github.com/huggingface/smolagents) `CodeAgent`
sequentially across every puzzle — sharing conversation history, Python
interpreter state, and working-directory files across rounds. The first
puzzle is a warmup; subsequent rounds reuse whatever infrastructure the
agent built.

That's the design choice that makes this a real intelligence test: it
measures whether an agent can *learn from its session*, not just one-shot
each puzzle independently.

It exists as a separate program (not a library) so the
[sudoku-vision-benchmark](../sudoku-vision-benchmark/) benchmarker can treat
it as an opaque box. Any harness matching the same CLI contract can be
substituted.

## Contract

```sh
sudoku-agent-harness \
    --model MODEL_ID \
    --puzzles-dir /path/to/puzzles/ \
    --output-dir /path/to/outputs/ \
    [--timeout 1200]
```

- `--puzzles-dir` — a directory containing `puzzle_NNN.png` files. The
  harness picks them up in sorted order and solves them **sequentially** in
  the same agent.
- `--output-dir` — where solution PNGs are written, one per puzzle. Each
  output file has the **same filename** as its corresponding input.
- Each solution PNG **must preserve the input's dimensions and 9x9 grid
  layout** so downstream graders can crop cells at known coordinates. The
  task prompt enforces this constraint on the agent.
- `--timeout` is per-round (soft), not total.

### Exit codes

- `0` — every round completed cleanly (an output PNG exists for each puzzle).
- `1` — at least one round finished without producing its output PNG.
- `2` — fatal error before or during setup (bad args, missing env, missing
  puzzles dir, etc.).

### Stdout

Streaming JSONL. One line per round as it finishes:

```json
{"puzzle_id": 0, "puzzle_name": "puzzle_000.png", "round": 1, "success": true, "elapsed": 42.1, "final_answer": "..."}
{"puzzle_id": 1, "puzzle_name": "puzzle_001.png", "round": 2, "success": true, "elapsed": 8.7, "final_answer": "..."}
```

A final summary line closes the run:

```json
{"summary": true, "n_puzzles": 100, "n_solved": 97, "total_elapsed": 812.3}
```

Callers should consume stdout line-by-line to stream progress; a line with
`"summary": true` marks the end.

Requires `OPENROUTER_API_KEY` in the environment or a `.env` file. The `.env`
is looked up from the directory you invoke the harness in first, then from the
harness repo, so running it from the benchmarker picks up that project's key.

## Resuming

Runs are resumable by default. After every round the harness writes
`.harness_state.json` into `--output-dir`, so you can Ctrl+C a model, run a
different one, and come back later — re-run the same command and it skips
what's already done:

```txt
[harness] resuming: 1/2 puzzles already solved for moonshotai/kimi-k3.
[harness] round 1/2: puzzle_000.png already solved, skipping
```

Skipped rounds are replayed onto stdout with `"resumed": true`, so callers
still see a record for every puzzle. Their tokens are excluded from the
session totals, which count only work this process actually did.

State is only reused when it is genuinely safe:

- The puzzle set must be **byte-identical** — the fingerprint hashes file
  contents, so regenerating puzzles with a new seed invalidates it even
  though the filenames are unchanged.
- The model must match.
- The solution image must still be on disk.

Otherwise the harness says why and starts fresh. Pass `--fresh` to force a
full re-run.

**Caveat:** a resumed session starts a brand-new agent with an empty working
directory, so it cannot reuse infrastructure the interrupted run built. The
first round it actually executes gets the warmup prompt again. Warmup-vs-
steady-state learning numbers are therefore only meaningful for a run that
completed in one go.

## Image accumulation across rounds

The session is deliberately persistent (`reset=False`) so the agent keeps its
memory between puzzles — that is what the warmup-vs-steady-state measurement
depends on. The side effect is that every round's puzzle image stays in the
conversation, so round N sends N images. Providers refuse well before a long
run finishes; DeepInfra caps at 8, which failed runs at round 9.

Before each round the harness strips images from all earlier steps, so a
request never carries more than the current puzzle. Only the pixels go: the
agent's own text — including the grids it transcribed and its deductions —
is left intact, so session memory still works and the learning signal is
preserved. It also keeps image tokens flat instead of growing every round.

## Text-only models

Not every model on the leaderboard accepts images. Sending one an attachment
gets a 404 from OpenRouter ("No endpoints found that support image input"),
which used to kill the whole session.

The harness now detects that on the first round, drops the attachment for the
rest of the session, and retries immediately with a prompt that tells the
agent the puzzle is only available as `input.png` and that extracting the
clues from those pixels is part of the job. That variant also hands it a
starting approach — locate the grid lines, split into cells, render each
digit with Pillow and template-match — since for a text-only model the image
reading is unavoidable overhead rather than the thing being measured.

Detection costs exactly one rejected request per model, and nothing after
that. `--no-image` skips even that for a model you already know is text-only.

The solving rules do not change: pixels in, reasoning yours, no solver.

## Solving rules and the verifier

The task prompt forbids the agent from writing any code that touches Sudoku
digits — no solver, no candidate elimination, and no self-written checker.
Code is for pixels only: reading the puzzle image and drawing the answer.
Models rationalise their way around vaguer wording ("a verification script,
not a solver"), so the prohibition names those framings explicitly.

To make that rule free rather than costly, each session gets
`verify_sudoku.py` in its working directory:

```sh
python verify_sudoku.py --text "483957261 915362748 ..."
python verify_sudoku.py --image output.png --reference input.png
```

It is deliberately a binary oracle. `--text` checks the constraints on a
**complete** grid only — a checker that scores partial grids is a search
heuristic — and reports which units failed, never which cell or digit. The
`--image` mode is structural: dimensions, clues untouched, no blank cells;
it never reads the agent's digits.

Every call is appended to `verifier_calls.log`, which the archive keeps. A
run that called it a handful of times was checking its work; a run that
called it hundreds of times was searching.

The prompt also carries actual Sudoku technique — hidden singles first, then
cross-hatching, naked singles, pencil marks, locked candidates, naked and
hidden pairs — so a model that can reason isn't held back by not knowing how
to start.

## Cheat auditing

The agent works in a throwaway directory, so by default everything it wrote
would vanish with the run. At the end of every session — including a crashed
one — that directory is snapshotted to:

```txt
archive/<model>/
├── <whatever scripts and notes the agent wrote>
└── session.json      # round records + final answers
```

The scratch `input.png` / `output.png` are skipped (the benchmarker already
keeps those); what's preserved is the agent's *own* code. That's the audit
trail: the task prompt forbids writing a Sudoku solver, and this is how you
check whether a model did it anyway. Re-running a model replaces its archive.

Override the location with `--archive-dir`. `archive/` is gitignored.

## Install

```sh
uv sync
```

## Run

```sh
uv run sudoku-agent-harness --model openai/gpt-4o \
    --puzzles-dir puzzles/ --output-dir solutions/
```

## Alternative harnesses

If you outgrow smolagents (e.g. you want a sandboxed shell, editor, browser,
or a more elaborate multi-tool loop), the same CLI contract can be
implemented by wrapping [OpenHands](https://github.com/All-Hands-AI/OpenHands).
OpenHands also uses [LiteLLM](https://github.com/BerriAI/litellm), so
OpenRouter models work out of the box. It's overkill for Sudoku, but a
natural fit for testing delegation-style agentic tasks (see the parent
benchmark's `TODO.md`).

Other options considered:

- **Claude Agent SDK** — best-in-class but Anthropic-only, so it'd defeat
  the multi-model comparison the benchmark exists for.
- **Roll-your-own tool-use loop on OpenRouter's function-calling** — cheaper
  than smolagents but reinvents the code-execution sandbox.
- **[general-agent](../general-agent/)** — sibling repo starting a
  skill-based agent framework. Once it's mature, we may wrap it as a
  drop-in `sudoku-agent-harness`-compatible CLI.
