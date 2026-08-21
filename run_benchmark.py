#!/usr/bin/env python3
"""Sudoku Vision Benchmark - grade LLMs via OpenRouter.

Usage:
    uv run run_benchmark.py                             # reads models.toml
    uv run run_benchmark.py --models-file custom.txt    # override file
    uv run run_benchmark.py openai/gpt-4o ...           # override with positional args

Every model in the run is graded on the SAME set of freshly-generated puzzles.
"""
import argparse
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from benchmarker.benchmark import (  # noqa: E402
    Benchmark,
    DEFAULT_GRADER_MODEL,
    DEFAULT_HARNESS_CMD,
)
from benchmarker.trials import DEFAULT_TRIALS_DIR  # noqa: E402

DEFAULT_MODELS_FILE = "models.toml"
# Category assumed for a plain-text models file, which carries no grouping.
DEFAULT_CATEGORY = "open-weight"
# Models under this category are recorded but never run.
DISABLED_CATEGORY = "disabled"


def _read_models_file(path):
    """Load model IDs from a .toml manifest, or a plain one-per-line list."""
    path = Path(path)
    if path.suffix == ".toml":
        return _read_models_toml(path)
    models = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            models.append((line, DEFAULT_CATEGORY))
    return models, []


def _read_models_toml(path):
    """Parse a models.toml manifest.

    The [models] table maps a category name to a list of entries. Each entry
    is either a bare id string or an inline table `{ id, note }`. The
    `disabled` category is never run.

    Returns (enabled, skipped): `enabled` is a list of (id, category) and
    `skipped` a list of (id, note), so the run can report what it left out
    rather than silently dropping it.
    """
    data = tomllib.loads(path.read_text())
    section = data.get("models")
    if not isinstance(section, dict):
        raise TypeError(f"{path}: expected a [models] table of categories.")

    enabled, skipped = [], []
    for category, items in section.items():
        if not isinstance(items, list):
            raise TypeError(
                f"{path}: [models].{category} must be a list of models."
            )
        for i, item in enumerate(items):
            if isinstance(item, str):
                model_id, note = item, ""
            elif isinstance(item, dict):
                model_id, note = item.get("id"), item.get("note", "")
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
                skipped.append((model_id, note))
            else:
                enabled.append((model_id, category))
    return enabled, skipped


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
        f"accept --model, --puzzles-dir, --output-dir, --timeout, solve the puzzles "
        f"sequentially in one session, and stream one JSONL record per round to stdout.",
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
    args = parser.parse_args()

    if args.models:
        models = list(args.models)
    else:
        path = args.models_file or DEFAULT_MODELS_FILE
        if not Path(path).exists():
            parser.error(
                f"No models given on the command line and models file '{path}' does not exist."
            )
        try:
            entries, skipped = _read_models_file(path)
        except (tomllib.TOMLDecodeError, TypeError, ValueError) as e:
            parser.error(f"Could not read models from '{path}': {e}")
        if not entries:
            parser.error(f"No enabled models in '{path}'.")
        models = [model_id for model_id, _ in entries]

        by_category = {}
        for model_id, category in entries:
            by_category.setdefault(category, []).append(model_id)
        breakdown = ", ".join(
            f"{len(ids)} {cat}" for cat, ids in sorted(by_category.items())
        )
        print(f"Loaded {len(models)} enabled model(s) from {path} ({breakdown})")
        for model_id, note in skipped:
            reason = f" - {note}" if note else ""
            print(f"  skipping (disabled): {model_id}{reason}")

    print(f"Models to test on the same puzzle set: {', '.join(models)}")

    bench = Benchmark(
        models=models,
        n_puzzles=args.n_puzzles,
        seed=args.seed,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        difficulty=args.difficulty,
        harness_cmd=args.harness_cmd,
        grader_model=args.grader_model,
        harness_timeout=args.harness_timeout,
        verbose=args.verbose,
        trials_dir=None if args.no_trials else args.trials_dir,
    )
    bench.run()


if __name__ == "__main__":
    sys.exit(main())
