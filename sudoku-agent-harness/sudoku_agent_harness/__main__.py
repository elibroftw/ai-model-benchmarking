"""CLI entrypoint for sudoku-agent-harness.

The harness is invoked ONCE per model and drives a persistent CodeAgent
through every puzzle in `--puzzles-dir` sequentially. Per-round stats are
emitted as JSONL to stdout as each round finishes (so callers can stream
progress). A final summary JSON line is printed at the end.
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def _load_env():
    """Load .env from the invoking directory as well as the package's own tree.

    Bare `load_dotenv()` resolves relative to the *calling module's* directory,
    so running this harness from another project (as the benchmarker does)
    would miss that project's .env. Check the cwd first, then fall back.
    Existing environment variables always win, so an explicitly exported key
    is never overridden.
    """
    cwd_env = find_dotenv(usecwd=True)
    if cwd_env:
        load_dotenv(cwd_env)
    load_dotenv()


def main():
    _load_env()

    parser = argparse.ArgumentParser(
        description="Agentic Sudoku harness (smolagents + OpenRouter). "
        "Runs one persistent agent across every puzzle in --puzzles-dir.",
    )
    parser.add_argument("--model", required=True, help="OpenRouter model ID.")
    parser.add_argument(
        "--puzzles-dir",
        required=True,
        help="Directory containing puzzle_NNN.png files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where solution PNGs will be written (same filenames as inputs).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 20,
        help="Per-round soft timeout in seconds (default 1200).",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Where to snapshot the agent's working directory when the session "
        "ends, as <archive-dir>/<model>/. Defaults to archive/ at the harness "
        "repo root. Preserves every script the agent wrote so you can audit "
        "whether it cheated.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore saved state and re-solve every puzzle. By default a run "
        "resumes: puzzles already solved for this model against this exact "
        "puzzle set are skipped, so an interrupted run can be restarted.",
    )
    args = parser.parse_args()

    # Import late so `--help` is fast and doesn't require smolagents.
    from .agent import run

    try:
        result = run(
            args.model,
            Path(args.puzzles_dir),
            Path(args.output_dir),
            timeout=args.timeout,
            archive_dir=Path(args.archive_dir) if args.archive_dir else None,
            fresh=args.fresh,
        )
    except Exception as e:  # noqa: BLE001 - top-level: report and exit non-zero
        print(json.dumps({"success": False, "error": f"{type(e).__name__}: {e}"}))
        return 2

    # Per-round records were already streamed as JSONL from run().
    # Emit a final summary line the caller can distinguish by the `summary` key.
    summary = {
        "summary": True,
        "n_puzzles": len(result["rounds"]),
        "n_solved": result["n_solved"],
        "n_resumed": result.get("n_resumed", 0),
        "interrupted": result.get("interrupted", False),
        "total_elapsed": result["total_elapsed"],
        "total_input_tokens": result.get("total_input_tokens", 0),
        "total_output_tokens": result.get("total_output_tokens", 0),
    }
    print(json.dumps(summary))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
