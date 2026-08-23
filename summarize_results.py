#!/usr/bin/env python3
"""Summarize a results directory without re-running the benchmark.

`run_benchmark.py` prints a leaderboard and writes results/leaderboard.json
only when a run completes. This rebuilds the same view from whatever is on
disk, so an interrupted run is still readable.

It also verifies the solution images itself: each PNG is transcribed locally
and checked against its puzzle's clues and the Sudoku rules, which needs no
API and no trust in the agent's own account of its work. The transcriber is
first checked against puzzles.json, whose grids it must reproduce exactly.

Usage:
    uv run summarize_results.py                          # text report on results/
    uv run summarize_results.py --format markdown        # pasteable table
    uv run summarize_results.py --write-leaderboard      # (re)write leaderboard.json
    uv run summarize_results.py --results-dir other/     # another directory
"""
import argparse
import json
import sys
from pathlib import Path

from benchmarker.get_leaderboard import (
    collect,
    format_markdown,
    format_text,
    leaderboard,
)

DEFAULT_RESULTS_DIR = "results"
DEFAULT_MODELS_FILE = "models.toml"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory of per-model result JSON (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--models-file",
        default=DEFAULT_MODELS_FILE,
        help="Model manifest, used to turn result filenames back into model "
             f"IDs (default: {DEFAULT_MODELS_FILE})",
    )
    parser.add_argument(
        "--harness-id",
        default=None,
        help="If set, only the solutions-<id> directory for this harness is "
             "summarized. By default every solutions-* directory under "
             "--results-dir is reported in turn.",
    )
    parser.add_argument(
        "--no-verify-images",
        action="store_true",
        help="Skip transcribing the solution PNGs. Correctness then comes only "
             "from the run's grader verdicts, where it got any.",
    )
    parser.add_argument(
        "--hide-errors",
        action="store_true",
        help="Omit the `errors` column and the error-kind breakdown.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="Report format (default: text). json emits the full report.",
    )
    parser.add_argument(
        "--out",
        help="Write the report here instead of stdout.",
    )
    parser.add_argument(
        "--write-leaderboard",
        action="store_true",
        help="Also (re)write <results-dir>/leaderboard-<harness>.json from "
             "these records.",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f"no such directory: {results_dir}")

    # Discover every solutions-* directory under the results dir.
    solutions_dirs = sorted(results_dir.glob("solutions-*"))
    if not solutions_dirs:
        # Fall back to the old single solutions/ name for backwards compat.
        old = results_dir / "solutions"
        if old.is_dir():
            solutions_dirs = [old]

    if not solutions_dirs:
        sys.exit(f"no solutions-* or solutions/ directory found in {results_dir}")

    for sd in solutions_dirs:
        hid = sd.name.removeprefix("solutions-")
        try:
            report = collect(
                args.results_dir,
                models_file=args.models_file,
                verify_images=not args.no_verify_images,
                solutions_dir=sd,
            )
        except FileNotFoundError as e:
            print(f"skipping {sd.name}: {e}", file=sys.stderr)
            continue

        # Filter to a single harness when --harness-id is given.
        if args.harness_id is not None and hid != args.harness_id:
            continue

        if args.format == "json":
            text = json.dumps(report, indent=2)
        elif args.format == "markdown":
            text = format_markdown(report, hide_errors=args.hide_errors)
        else:
            text = format_text(report, hide_errors=args.hide_errors)

        # Prefix every report with a harness heading when there is more than
        # one solutions dir.
        if len(solutions_dirs) > 1 and args.format == "text":
            divider = f"{'=' * 20}  {sd.name} (harness: {hid})  {'=' * 20}"
            out_lines = [divider, text]
        else:
            out_lines = [text]

        combined = "\n".join(out_lines)

        if args.out:
            base = Path(args.out)
            # Append the harness id to avoid overwriting between harnesses.
            if len(solutions_dirs) > 1 and args.harness_id is None:
                out_path = base.with_stem(f"{base.stem}-{hid}")
            else:
                out_path = base
            out_path.write_text(combined + "\n")
            print(f"wrote {out_path}")
        else:
            print(combined)

        if args.write_leaderboard:
            lb_path = results_dir / f"leaderboard-{hid}.json"
            lb_path.write_text(json.dumps(leaderboard(report), indent=2))
            print(f"wrote {lb_path}")


if __name__ == "__main__":
    main()
