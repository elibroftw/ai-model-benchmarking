#!/usr/bin/env python3
"""CLI entry point for dsh-agent harness.

The harness is invoked ONCE per model with a task spec, an inputs directory,
and an output directory.  It drives a single persistent agent (through the
DSH API) across every input sequentially and streams one JSONL record per
round to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def _load_env():
    """Load .env from the invoking directory first, then from the package tree."""
    cwd_env = find_dotenv(usecwd=True)
    if cwd_env:
        load_dotenv(cwd_env)
    load_dotenv()


def main():
    _load_env()

    parser = argparse.ArgumentParser(
        description=(
            "Agentic file-in/file-out harness via DSH API. "
            "Runs one persistent agent across every input in --inputs-dir, "
            "doing whatever --task says."
        ),
    )
    parser.add_argument("--model", required=True, help="OpenRouter model ID.")
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Path to the task spec: a JSON file defining prompts (first_round "
            "and next_round, each with vision and/or text_only variants), "
            "input/output filenames, and optional asset files to install."
        ),
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
        help=(
            "Directory where each round's output is written (same filename as "
            "its input)."
        ),
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
        help=(
            "Where to snapshot the agent's working directory when the session "
            "ends, as <archive-dir>/<model>/.  Defaults to archive/ at the "
            "harness repo root."
        ),
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help=(
            "Don't attach the input image; use the task's text-only prompts."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore saved state and redo every round.  By default a run "
            "resumes: rounds already completed for this model against this "
            "exact input set are skipped."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress and diagnostics to stderr.",
    )
    args = parser.parse_args()

    # Import late so --help is fast.
    from .harness import run_harness

    archive_dir = (
        Path(args.archive_dir)
        if args.archive_dir
        else Path.cwd() / "archive"
    )

    exit_code = run_harness(
        model_id=args.model,
        task_path=Path(args.task),
        inputs_dir=Path(args.inputs_dir),
        output_dir=Path(args.output_dir),
        timeout=args.timeout,
        archive_dir=archive_dir,
        no_image=args.no_image,
        fresh=args.fresh,
        verbose=args.verbose,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()