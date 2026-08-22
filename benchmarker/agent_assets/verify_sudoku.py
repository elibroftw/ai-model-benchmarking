#!/usr/bin/env python3
"""Sudoku solution verifier, provided to the agent so it need not write one.

The harness copies this into the agent's working directory each session. The
agent is forbidden from writing its own constraint-checking code; this exists
so that rule costs it nothing.

Deliberately a BINARY oracle over COMPLETE grids only:

- It refuses partial grids. A checker that scores incomplete grids is a search
  heuristic, and the agent could drive it as a solver.
- It never names a cell, a digit, or anything about the correct solution. At
  most it names which units failed, which tells the agent its reasoning went
  wrong somewhere without pointing at the answer.

Every call is appended to `verifier_calls.log` in the working directory, which
is archived with the session — a run that calls this hundreds of times was
searching, not reasoning.

Usage:
    python verify_sudoku.py --text "530070000 600195000 ..."
    python verify_sudoku.py --text-file grid.txt
    python verify_sudoku.py --image output.png --reference input.png

Exit code 0 = VALID, 1 = INVALID, 2 = could not check (bad input).
"""
import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_NAME = "verifier_calls.log"


def _log(argv, verdict):
    """Record the call so abuse is visible in the session archive."""
    try:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(LOG_NAME, "a") as f:
            f.write(f"{stamp}\t{verdict}\t{' '.join(argv)}\n")
    except OSError:
        pass  # Logging must never break the check.


def parse_grid(text):
    """Parse 81 digits into a 9x9 grid. Returns (grid, error)."""
    tokens = re.findall(r"[0-9.]", text)
    if len(tokens) != 81:
        return None, f"expected 81 digits, found {len(tokens)}"
    grid = []
    for r in range(9):
        row = []
        for c in range(9):
            t = tokens[r * 9 + c]
            row.append(0 if t in "0." else int(t))
        grid.append(row)
    return grid, None


def check_grid(grid):
    """Return a list of failing unit names; empty means valid.

    Reports the unit, never the offending cell or digit.
    """
    if any(v == 0 for row in grid for v in row):
        return ["incomplete (the grid still has empty cells)"]

    failures = []
    full = set(range(1, 10))
    for r in range(9):
        if set(grid[r]) != full:
            failures.append(f"row {r + 1}")
    for c in range(9):
        if {grid[r][c] for r in range(9)} != full:
            failures.append(f"column {c + 1}")
    for br in range(3):
        for bc in range(3):
            box = {grid[br * 3 + i][bc * 3 + j] for i in range(3) for j in range(3)}
            if box != full:
                failures.append(f"box r{br + 1}c{bc + 1}")
    return failures


def check_image(output_path, reference_path):
    """Structural checks on a rendered solution against the puzzle image.

    Verifies what can be established from pixels alone: same dimensions, the
    original clue cells untouched, and no cell left blank. It does NOT read
    the agent's digits, so it cannot tell the agent whether its answer is
    right — use --text for the constraint check.
    """
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return ["cannot check image: Pillow is not available"]

    try:
        out = Image.open(output_path).convert("RGB")
    except OSError as e:
        return [f"cannot open {output_path}: {e}"]

    failures = []
    if reference_path is None:
        return ["--image requires --reference to compare against"]
    try:
        ref = Image.open(reference_path).convert("RGB")
    except OSError as e:
        return [f"cannot open {reference_path}: {e}"]

    if out.size != ref.size:
        return [f"dimensions {out.size} do not match input {ref.size}"]

    w, h = ref.size
    # Infer the grid box from the reference's dark lines rather than assuming
    # a fixed geometry, so this keeps working if the renderer changes.
    gray = ref.convert("L")
    px = gray.load()
    dark_cols = [x for x in range(w) if sum(px[x, y] < 128 for y in range(h)) > h * 0.5]
    dark_rows = [y for y in range(h) if sum(px[x, y] < 128 for x in range(w)) > w * 0.5]
    if len(dark_cols) < 2 or len(dark_rows) < 2:
        return ["could not locate the grid in the reference image"]
    x0, x1 = dark_cols[0], dark_cols[-1]
    y0, y1 = dark_rows[0], dark_rows[-1]
    cw = (x1 - x0) / 9.0
    ch = (y1 - y0) / 9.0

    pad = 4
    blank_cells = 0
    changed_clues = 0
    for r in range(9):
        for c in range(9):
            box = (
                int(x0 + c * cw) + pad,
                int(y0 + r * ch) + pad,
                int(x0 + (c + 1) * cw) - pad,
                int(y0 + (r + 1) * ch) - pad,
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            ref_cell = ref.crop(box)
            out_cell = out.crop(box)
            ref_has_ink = min(ref_cell.convert("L").getextrema()) < 128
            out_has_ink = min(out_cell.convert("L").getextrema()) < 128
            if ref_has_ink:
                # A clue cell: must be untouched.
                if ImageChops.difference(ref_cell, out_cell).getbbox() is not None:
                    changed_clues += 1
            elif not out_has_ink:
                blank_cells += 1

    if changed_clues:
        failures.append(f"{changed_clues} original clue cell(s) were altered")
    if blank_cells:
        failures.append(f"{blank_cells} cell(s) left blank")
    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Verify a completed Sudoku. Reports VALID or INVALID.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="81 digits; 0 or . for empty (rejected as incomplete).")
    src.add_argument("--text-file", help="File containing the 81 digits.")
    src.add_argument("--image", help="Rendered solution PNG (structural checks only).")
    parser.add_argument("--reference", help="The puzzle image, required with --image.")
    args = parser.parse_args()

    # `is not None`, not truthiness: an empty --image must not silently
    # fall through to the text branch.
    if args.image is not None:
        failures = check_image(args.image, args.reference)
    else:
        try:
            raw = args.text if args.text is not None else Path(args.text_file).read_text()
        except OSError as e:
            print(f"INVALID: cannot read input ({e})")
            _log(sys.argv[1:], "ERROR")
            return 2
        grid, err = parse_grid(raw)
        if err:
            print(f"INVALID: {err}")
            _log(sys.argv[1:], "ERROR")
            return 2
        failures = check_grid(grid)

    if failures:
        print("INVALID: " + "; ".join(failures))
        _log(sys.argv[1:], "INVALID")
        return 1
    print("VALID")
    _log(sys.argv[1:], "VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
