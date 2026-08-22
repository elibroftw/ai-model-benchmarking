# sudoku-agent-harness

A general-purpose agentic harness for file-in / file-out tasks. Given a
*directory* of input files, an OpenRouter model ID, and a **task spec**, this
program runs a single persistent
[smolagents](https://github.com/huggingface/smolagents) `CodeAgent`
sequentially across every input — sharing conversation history, Python
interpreter state, and working-directory files across rounds. The first round
is a warmup; subsequent rounds reuse whatever infrastructure the agent built.

That's the design choice that makes this a real intelligence test: it
measures whether an agent can *learn from its session*, not just one-shot
each item independently.

**The harness holds no task of its own.** What the agent is asked to do — the
prompts, the filenames, the tools dropped into its working directory — arrives
in the `--task` spec. It is named for the benchmark it was built for, but it
knows nothing about Sudoku: that task is defined by
[sudoku-vision-benchmark](../sudoku-vision-benchmark/) in
`benchmarker/task.py` and passed in. Which is the point — a different harness
serves the same task by reading the same spec, with nothing to re-implement.

It exists as a separate program (not a library) so the benchmarker can treat
it as an opaque box. Any harness matching the same CLI contract can be
substituted.

## Contract

```sh
sudoku-agent-harness \
    --model MODEL_ID \
    --task /path/to/task.json \
    --inputs-dir /path/to/inputs/ \
    --output-dir /path/to/outputs/ \
    [--timeout 1200]
```

- `--task` — the task spec (below). Everything task-specific comes from here.
- `--inputs-dir` — a directory of input files, matched by the spec's
  `input_glob`. The harness takes them in sorted order and works through them
  **sequentially** in the same agent. `--puzzles-dir` is accepted as an alias
  for callers written against the older contract.
- `--output-dir` — where each round's output is collected. Each output file
  has the **same filename** as its corresponding input.
- `--timeout` is per-round (soft), not total.

Any constraint on the *content* of the output — for this benchmark, that a
solution PNG preserves the input's dimensions and 9x9 grid layout so graders
can crop cells at known coordinates — is the task's business, stated in the
task's prompt. The harness only moves files.

### The task spec

```json
{
  "task": "sudoku-vision",
  "input_filename": "input.png",
  "output_filename": "output.png",
  "input_glob": "puzzle_*.png",
  "assets": ["/abs/path/to/verify_sudoku.py"],
  "prompts": {
    "first_round": {"vision": "...", "text_only": "..."},
    "next_round":  {"vision": "...", "text_only": "..."}
  }
}
```

- `input_filename` / `output_filename` — what each round's input is copied to
  in the working directory, and what the agent is expected to leave behind.
- `input_glob` — which files in `--inputs-dir` are rounds.
- `assets` — files copied into the working directory at the start of a
  session. For this benchmark that's the verifier the agent is required to
  use instead of writing its own; another task might ship a reference
  document or a helper script. The harness neither knows nor cares what they
  are.
- `prompts` — `first_round` carries the full rules, `next_round` is the terse
  follow-up (the session is persistent, so re-sending the rules just burns
  context). Each has a `vision` and a `text_only` variant; the harness picks
  `text_only` when the model can't accept images, and a task that supplies
  only one variant gets it used for both.

Prompts may use `{round}`, `{n_rounds}`, `{input_filename}` and
`{output_filename}`. Substitution is literal replacement, not `str.format`,
so a prompt containing braces cannot break a run.

### Exit codes

- `0` — every round completed cleanly (an output file exists for each input).
- `1` — at least one round finished without producing its output file.
- `2` — fatal error before or during setup (bad args, missing env, missing or
  invalid task spec, missing inputs dir, etc.).

### Stdout

Streaming JSONL. One line per round as it finishes:

```json
{"item_id": 0, "item_name": "puzzle_000.png", "round": 1, "success": true, "elapsed": 42.1, "final_answer": "..."}
{"item_id": 1, "item_name": "puzzle_001.png", "round": 2, "success": true, "elapsed": 8.7, "final_answer": "..."}
```

`item_id` is the trailing number in the input's filename, so a caller can
correlate a round with the input it supplied without the harness knowing what
the inputs mean. (`puzzle_id` / `puzzle_name` were the pre-rename names;
readers on both sides still accept them, so old state files keep resuming.)

A final summary line closes the run:

```json
{"summary": true, "n_rounds": 100, "n_puzzles": 100, "n_solved": 97, "total_elapsed": 812.3}
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
[harness] resuming: 1/2 round(s) already done for moonshotai/kimi-k3.
[harness] round 1/2: puzzle_000.png already done, skipping
```

Skipped rounds are replayed onto stdout with `"resumed": true`, so callers
still see a record for every input. Their tokens are excluded from the
session totals, which count only work this process actually did.

State is only reused when it is genuinely safe:

- The input set must be **byte-identical** — the fingerprint hashes file
  contents, so regenerating the inputs invalidates it even though the
  filenames are unchanged.
- The model must match.
- The output file must still be on disk.

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

Everything in this section is the *task's* doing, not the harness's — it lives
in the benchmark's `benchmarker/task.py` and `benchmarker/agent_assets/`, and
is described here because it is what this harness is normally pointed at.

The task prompt forbids the agent from writing any code that touches Sudoku
digits — no solver, no candidate elimination, and no self-written checker.
Code is for pixels only: reading the puzzle image and drawing the answer.
Models rationalise their way around vaguer wording ("a verification script,
not a solver"), so the prohibition names those framings explicitly.

To make that rule free rather than costly, the task ships `verify_sudoku.py`
as an asset, and the harness installs it in the working directory:

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

The scratch input/output files named by the task spec are skipped (the caller
already has both); what's preserved is the agent's *own* code. That's the audit
trail: this task's prompt forbids writing a Sudoku solver, and this is how you
check whether a model did it anyway. Re-running a model replaces its archive.

Override the location with `--archive-dir`. `archive/` is gitignored.

## Install

```sh
uv sync
```

## Run

```sh
uv run sudoku-agent-harness --model openai/gpt-4o \
    --task ../sudoku-vision-benchmark/results/task.json \
    --inputs-dir puzzles/ --output-dir solutions/
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
  drop-in CLI-compatible replacement. Since the task arrives as a spec, that
  wrapper has no Sudoku code to write: read `--task`, run the prompts, write
  the output file.
