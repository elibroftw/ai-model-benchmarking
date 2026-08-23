#!/usr/bin/env python3
"""Report how many solution PNGs each model produced.

Usage:
    uv run cli/count_solution_images.py                  # reads results/solutions
    uv run cli/count_solution_images.py path/to/dir      # override root
"""
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path("results/solutions")


def count_pngs(root):
    """Map each subdirectory of root to its PNG count."""
    return {d.name: len(list(d.glob("*.png"))) for d in sorted(root.iterdir()) if d.is_dir()}


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    counts = count_pngs(root)
    if not counts:
        sys.exit(f"no subdirectories in {root}")

    by_count = defaultdict(list)
    for name, count in counts.items():
        by_count[count].append(name)

    for count in sorted(by_count, reverse=True):
        names = by_count[count]
        print(f"{count} png{'' if count == 1 else 's'} ({len(names)}):")
        for name in names:
            print(f"  {name}")

    total = sum(counts.values())
    print(f"\n{total} png{'' if total == 1 else 's'} across {len(counts)} directories")


if __name__ == "__main__":
    main()
