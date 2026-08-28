#!/usr/bin/env python3
"""Sudoku Vision Benchmark - grade LLMs via OpenRouter.

Usage:
    uv run cli/run_benchmark.py                             # reads models.toml
    uv run cli/run_benchmark.py --models-file custom.txt    # override file
    uv run cli/run_benchmark.py openai/gpt-4o ...           # override with positional args
    uv run cli/run_benchmark.py --model glm-5.3-flash       # one model, by id or unique substring

Every model in the run is graded on the SAME set of freshly-generated puzzles.
"""
import argparse
import sys
import tomllib
from pathlib import Path

# Running this as `uv run cli/<script>.py` puts cli/ on sys.path, not the repo
# root, so the benchmarker package has to be pointed at explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from benchmarker.benchmark import (  # noqa: E402
    Benchmark,
    DEFAULT_GRADER_MODEL,
    DEFAULT_HARNESS_CMD,
    DEFAULT_HARNESS_ID,
    FatalRunError,
)
from benchmarker.trials import DEFAULT_TRIALS_DIR  # noqa: E402

DEFAULT_MODELS_FILE = "models.toml"
# Category assumed for a plain-text models file, which carries no grouping.
DEFAULT_CATEGORY = "open-weight"
# Models under this category are recorded but never run.
DISABLED_CATEGORY = "disabled"


def _read_models_file(path, skip_expensive=False):
    """Load model IDs from a .toml manifest, or a plain one-per-line list.

    Returns (enabled, skipped, expensive_ids): ``enabled`` is a list of
    (id, category), ``skipped`` a list of (id, note), and ``expensive_ids``
    a set of model IDs that are marked expensive (enabled entries only).
    """
    path = Path(path)
    if path.suffix == ".toml":
        return _read_models_toml(path, skip_expensive=skip_expensive)
    models = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            models.append((line, DEFAULT_CATEGORY))
    return models, [], set()


def _read_models_toml(path, skip_expensive=False):
    """Parse a models.toml manifest.

    The [models] table maps a category name to a list of entries. Each entry
    is either a bare id string or an inline table `{ id, note, expensive }`.
    The `disabled` category is never run, and `expensive = true` entries are
    held back when skip_expensive is set.

    Returns (enabled, skipped, expensive_ids): ``enabled`` is a list of
    (id, category), ``skipped`` a list of (id, note), and ``expensive_ids``
    is a set of model IDs within the *enabled* list that are marked expensive.
    """
    data = tomllib.loads(path.read_text())
    section = data.get("models")
    if not isinstance(section, dict):
        raise TypeError(f"{path}: expected a [models] table of categories.")

    enabled, skipped, expensive_ids = [], [], set()
    for category, items in section.items():
        if not isinstance(items, list):
            raise TypeError(
                f"{path}: [models].{category} must be a list of models."
            )
        for i, item in enumerate(items):
            if isinstance(item, str):
                model_id, note, expensive = item, "", False
            elif isinstance(item, dict):
                model_id = item.get("id")
                note = item.get("note", "")
                expensive = bool(item.get("expensive"))
            else:
                raise TypeError(
                    f"{path}: [models].{category} entry #{i + 1} must be a "
                    f"string or an inline table."
                )
            if not model_id:
                raise ValueError(
                    f"{path}: [models].{category} entry #{i + 1} has no `id`."
                )
            if category == DISABLED_CATEGORY:
                # do not log disabled models
                continue
            elif expensive and skip_expensive:
                skipped.append((model_id, note or "marked expensive"))
            else:
                enabled.append((model_id, category))
                if expensive:
                    expensive_ids.add(model_id)
    return enabled, skipped, expensive_ids


def _manifest_entries(path):
    """Every model in the manifest as {id: (category, expensive, note)}.

    Unlike `_read_models_file` this keeps the disabled ones and ignores
    --skip-expensive: it answers "what does the manifest say about this id?",
    which is a different question from "what should this run cover?". Returns
    an empty dict for a manifest that cannot be read, since a named model runs
    with or without the manifest's blessing.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        if path.suffix != ".toml":
            return {
                line.strip(): (DEFAULT_CATEGORY, False, "")
                for line in path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            }
        section = tomllib.loads(path.read_text()).get("models") or {}
        entries = {}
        for category, items in section.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    entries[item] = (category, False, "")
                elif isinstance(item, dict) and item.get("id"):
                    entries[item["id"]] = (
                        category,
                        bool(item.get("expensive")),
                        item.get("note", ""),
                    )
        return entries
    except (tomllib.TOMLDecodeError, OSError, TypeError):
        return {}


def _resolve_model_id(wanted, entries, parser):
    """Turn what the user typed into one model ID.

    An exact ID always wins. Otherwise a case-insensitive substring match
    against the manifest resolves it, so `--model glm-5.3-flash` is enough for
    `z-ai/glm-5.3-flash` — these IDs are long and typing the vendor prefix
    adds nothing. An ambiguous abbreviation is an error listing the
    candidates, never a guess: running the wrong model costs a full session.

    Something the manifest has never heard of is returned as typed. The
    manifest is a convenience here, not an allowlist — a model can be tried
    before it is added.
    """
    if wanted in entries:
        return wanted
    matches = [mid for mid in entries if wanted.lower() in mid.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        parser.error(
            f"--model {wanted!r} matches {len(matches)} models: "
            f"{', '.join(sorted(matches))}. Use a full model ID."
        )
    return wanted


def main():
    parser = argparse.ArgumentParser(
        description="Grade LLMs on a Sudoku vision benchmark via OpenRouter. "
        "All models are tested against the same generated puzzles per run.",
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="OpenRouter model IDs. If omitted, models are read from --models-file "
        f"(default: {DEFAULT_MODELS_FILE}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Benchmark this ONE model instead of every enabled model in the "
        "manifest. Accepts a full OpenRouter ID or any unique substring of one "
        "(e.g. `glm-5.3-flash`). The model's category and `expensive` flag are "
        "still taken from the manifest when it lists it; a model the manifest "
        "does not list, or lists as disabled, runs anyway — naming it is the "
        "decision. Mutually exclusive with positional model IDs.",
    )
    parser.add_argument(
        "--models-file",
        default=None,
        help=f"Path to a models manifest. A .toml file is parsed as [[models]] "
        f"tables (`id`, optional `enabled`/`note`); any other extension is read "
        f"as one model ID per line. Defaults to {DEFAULT_MODELS_FILE} when no "
        f"positional models are given.",
    )
    parser.add_argument("--n-puzzles", type=int, default=100, help="Number of puzzles per model.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    parser.add_argument("--output-dir", default="results", help="Where to save results.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent grader-LLM requests. Solving is always sequential "
        "(one persistent agent per model), so this only affects grading speed.",
    )
    parser.add_argument(
        "--difficulty",
        default="mixed",
        choices=["mixed", "easy", "medium", "hard", "expert"],
    )
    parser.add_argument(
        "--harness-cmd",
        default=DEFAULT_HARNESS_CMD,
        help=f"Command to invoke the agentic harness subprocess "
        f"(default: {DEFAULT_HARNESS_CMD}). It is called ONCE per model and must "
        f"accept --model, --task, --inputs-dir, --output-dir, --timeout, work "
        f"through the inputs sequentially in one session, and stream one JSONL "
        f"record per round to stdout. The task itself — prompts and the agent's "
        f"verifier — is passed in via --task, so a harness needs no knowledge of "
        f"Sudoku.",
    )
    parser.add_argument(
        "--harness-id",
        default=DEFAULT_HARNESS_ID,
        help="Short label for the harness, used in the solutions directory name "
        f"(e.g. solutions-<id>/). Default: {DEFAULT_HARNESS_ID}.",
    )
    parser.add_argument(
        "--grader-model",
        default=DEFAULT_GRADER_MODEL,
        help=f"OpenRouter model ID used to transcribe output images "
        f"back into a 9x9 grid for verification (default: {DEFAULT_GRADER_MODEL}).",
    )
    parser.add_argument(
        "--harness-timeout",
        type=int,
        default=60 * 20,
        help="Per-puzzle timeout in seconds passed to the harness. Default is 20 minutes.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature, forwarded to the harness as --temperature. "
        "Omitted by default, which leaves the harness on its own default "
        "(0.1 for the one in this repo) and keeps the flags sent to a "
        "substituted harness to the documented set.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Forwarded to the harness: ignore its saved state and redo every "
        "round. By default the harness resumes, replaying rounds it already "
        "completed for a model against this exact puzzle set — which is what "
        "you want after an interrupted run, but not when you are re-testing a "
        "model whose earlier rounds you no longer trust.",
    )
    parser.add_argument(
        "--skip-expensive",
        action="store_true",
        help="Skip models marked `expensive = true` in the manifest. Useful for "
        "a cheap smoke run before committing to the costly models.",
    )
    parser.add_argument(
        "--trials-dir",
        default=DEFAULT_TRIALS_DIR,
        help=f"Permanent per-puzzle-set leaderboards, keeping each model's best "
        f"result and merging reruns (default: {DEFAULT_TRIALS_DIR}). Requires "
        f"--seed, since unseeded runs aren't comparable. Use --no-trials to skip.",
    )
    parser.add_argument(
        "--no-trials",
        action="store_true",
        help="Don't read or update the permanent trials directory.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Stream the harness's stderr live (smolagents step-by-step logs). "
        "Best paired with --concurrency 1; interleaved output is chaotic otherwise.",
    )
    parser.add_argument(
        "--vision-middleware",
        action="store_true",
        help="Before the model loop, transcribe every puzzle image once "
        "through a vision model (configured in vision-middleware/models.toml). "
        "Each transcription is carried in the task spec and given to the agent "
        "as that round's image alt text, so every model sees the same one and "
        "the transcriber runs once per puzzle, not once per model.  Applied to "
        "ALL models — text-only and vision-capable — for fairness.  Vision "
        "models still receive the image itself as well.",
    )
    parser.add_argument(
        "--vision-middleware-cmd",
        default=None,
        help="Command to invoke the vision middleware (default: python "
        "vision-middleware/transcribe.py).  Must accept --image and --task.",
    )
    parser.add_argument(
        "--vision-middleware-config",
        default=None,
        help="Path to the middleware's models.toml (default: "
        "vision-middleware/models.toml).",
    )
    args = parser.parse_args()

    if args.model and args.models:
        parser.error(
            "give either --model or positional model IDs, not both."
        )

    if args.model or args.models:
        # Explicitly named models still inherit what the manifest knows about
        # them — category (which orders the run) and the expensive flag — so a
        # single-model run is scheduled and recorded exactly like the same
        # model inside a full run.
        entries = _manifest_entries(args.models_file or DEFAULT_MODELS_FILE)
        if args.model:
            models = [_resolve_model_id(args.model, entries, parser)]
        else:
            models = [_resolve_model_id(m, entries, parser) for m in args.models]
        expensive_ids = {m for m in models if entries.get(m, (None, False))[1]}
        model_categories = {
            m: entries[m][0] for m in models if m in entries
        }
        for model_id in models:
            known = entries.get(model_id)
            if known is None:
                print(f"  {model_id}: not in the manifest — running it anyway")
            elif known[0] == DISABLED_CATEGORY:
                note = f" ({known[2]})" if known[2] else ""
                print(f"  {model_id}: disabled in the manifest{note} — "
                      f"running it because you named it")
    else:
        path = args.models_file or DEFAULT_MODELS_FILE
        if not Path(path).exists():
            parser.error(
                f"No models given on the command line and models file '{path}' does not exist."
            )
        try:
            entries, skipped, expensive_ids = _read_models_file(
                path, skip_expensive=args.skip_expensive
            )
        except (tomllib.TOMLDecodeError, TypeError, ValueError) as e:
            parser.error(f"Could not read models from '{path}': {e}")
        if not entries:
            parser.error(f"No enabled models in '{path}'.")
        models = [model_id for model_id, _ in entries]
        model_categories = dict(entries)

        by_category = {}
        for model_id, category in entries:
            by_category.setdefault(category, []).append(model_id)
        breakdown = ", ".join(
            f"{len(ids)} {cat}" for cat, ids in sorted(by_category.items())
        )
        print(f"Loaded {len(models)} enabled model(s) from {path} ({breakdown})")
        for model_id, note in skipped:
            reason = f" - {note}" if note else ""
            print(f"  skipping: {model_id}{reason}")

    print(f"Models to test on the same puzzle set: {', '.join(models)}")
    if expensive_ids:
        print(f"  Expensive: {', '.join(sorted(expensive_ids))}")

    bench = Benchmark(
        models=models,
        n_puzzles=args.n_puzzles,
        seed=args.seed,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        difficulty=args.difficulty,
        harness_cmd=args.harness_cmd,
        harness_id=args.harness_id,
        grader_model=args.grader_model,
        harness_timeout=args.harness_timeout,
        fresh=args.fresh,
        verbose=args.verbose,
        trials_dir=None if args.no_trials else args.trials_dir,
        expensive_models=expensive_ids,
        model_categories=model_categories,
        vision_middleware=args.vision_middleware,
        vision_middleware_cmd=args.vision_middleware_cmd,
        vision_middleware_config=args.vision_middleware_config,
        temperature=args.temperature,
    )
    try:
        bench.run()
    except FatalRunError as e:
        # A dead grader or transcriber model fails the same way for every
        # model still queued, so the run stops here instead of spending the
        # rest of the budget producing results nothing verified.
        print(f"\nRun aborted: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
