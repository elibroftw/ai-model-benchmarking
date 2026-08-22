"""CLI entrypoint for the agent harness.

The harness is invoked ONCE per model and drives a persistent CodeAgent
through every input in `--inputs-dir` sequentially. What it asks the agent to
do comes entirely from `--task`: the harness carries no task of its own, so
the same spec can be handed to a different harness without re-writing the
task. Per-round stats are emitted as JSONL to stdout as each round finishes
(so callers can stream progress). A final summary JSON line is printed at the
end.
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
        description="Agentic file-in/file-out harness (smolagents + OpenRouter). "
        "Runs one persistent agent across every input in --inputs-dir, doing "
        "whatever --task says.",
    )
    parser.add_argument("--model", required=True, help="OpenRouter model ID.")
    parser.add_argument(
        "--task",
        required=True,
        help="Path to the task spec: a JSON file with the prompts, the input "
        "and output filenames, and any files to install in the agent's "
        "working directory. The task lives with whoever defines it, not here.",
    )
    parser.add_argument(
        "--inputs-dir",
        "--puzzles-dir",
        dest="inputs_dir",
        required=True,
        help="Directory of input files, one per round, in sorted order.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where each round's output is written (same filename as "
        "its input).",
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
        "--no-image",
        action="store_true",
        help="Don't attach the input image; use the task's text-only prompts, "
        "which tell the agent to read the input file itself. Only needed to "
        "skip the one failed request for a model already known to be "
        "text-only — otherwise this is detected automatically.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore saved state and redo every round. By default a run "
        "resumes: rounds already completed for this model against this exact "
        "input set are skipped, so an interrupted run can be restarted.",
    )
    args = parser.parse_args()

    # Import late so `--help` is fast and doesn't require smolagents.
    from .agent import run

    try:
        result = run(
            args.model,
            Path(args.inputs_dir),
            Path(args.output_dir),
            task=Path(args.task),
            timeout=args.timeout,
            archive_dir=Path(args.archive_dir) if args.archive_dir else None,
            fresh=args.fresh,
            send_images=not args.no_image,
        )
    except Exception as e:  # noqa: BLE001 - top-level: report and exit non-zero
        print(json.dumps({"success": False, "error": f"{type(e).__name__}: {e}"}))
        return 2

    # Per-round records were already streamed as JSONL from run().
    # Emit a final summary line the caller can distinguish by the `summary` key.
    summary = {
        "summary": True,
        # Named for the caller's benefit: one round per input file.
        "n_puzzles": len(result["rounds"]),
        "n_rounds": len(result["rounds"]),
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
