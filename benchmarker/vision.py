"""Read a rendered Sudoku image back into a 9x9 grid, locally.

The benchmark's normal grading path sends each produced image to a vision LLM
(see grader.py). That costs money, is nondeterministic, and — as the current
`results/` shows — reports nothing at all when the API call fails, leaving
only the agent's own claim that it solved the puzzle.

This module is the deterministic alternative the TODO calls for. It exploits
what the benchmark guarantees: the output image has the same geometry as the
input, so the grid lines can be found by projection and each cell cropped
exactly. Digits are then matched against templates.

The templates are calibrated from the benchmark's OWN puzzle images, whose
contents are known exactly from puzzles.json — the renderer is deterministic,
so a clue digit in an input image is a pixel-perfect example of that digit in
the style every model started from. Installed fonts are added as fallbacks for
digits a model drew in some other style.

Because the ground truth is known for the input images, the reader can be
checked before it is trusted: `self_check` transcribes the inputs and compares
them against puzzles.json. If that does not come back clean, nothing this
module says about the solutions should be believed.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .grader import verify
from .renderer import _FONT_CANDIDATES

# Luminance at or below this counts as ink. The renderer draws pure black on
# white; the margin is for models that antialias or use a softer grey.
INK_LEVEL = 128

# Digits are normalized into a box this many pixels on a side before matching.
NORM = 24

# A cell with fewer ink pixels than this is empty. Rendered blanks measure
# exactly zero, and the smallest digits models actually draw measure ~10, so
# this only absorbs stray antialiasing.
MIN_INK = 4

# Fraction of the cell trimmed off each edge before looking for a digit, so a
# thick grid line never reads as ink.
CELL_INSET = 0.05

# A match this weak means the reader does not recognize the mark; below this
# the cell is reported unread instead of guessed.
MIN_SCORE = 0.45
# Best-vs-runner-up gap below this means the two candidates are too close to
# separate confidently.
MIN_MARGIN = 0.03


def _to_gray(path):
    with Image.open(path) as im:
        return im.convert("L")


def find_grid_lines(img):
    """Locate the 10 vertical and 10 horizontal grid lines.

    Returns (xs, ys) of line centers, or None if the image does not look like
    a 9x9 grid. Uses dark-pixel projections, so it does not care about the
    line widths, the margin, or the image scale.
    """
    w, h = img.size
    px = img.load()
    cols = [0] * w
    rows = [0] * h
    for y in range(h):
        for x in range(w):
            if px[x, y] <= INK_LEVEL:
                cols[x] += 1
                rows[y] += 1

    xs = _line_centers(cols, h)
    ys = _line_centers(rows, w)
    if len(xs) != 10 or len(ys) != 10:
        return None
    return xs, ys


def _line_centers(projection, span):
    """Centers of runs where at least half the pixels are ink.

    A grid line spans the whole image, so it dominates its projection bucket;
    digits never come close to half a line's worth of ink.
    """
    centers = []
    start = None
    threshold = 0.5 * span
    for i, v in enumerate(projection):
        if v >= threshold and start is None:
            start = i
        elif v < threshold and start is not None:
            centers.append((start + i - 1) // 2)
            start = None
    if start is not None:
        centers.append((start + len(projection) - 1) // 2)
    return centers


def _cell_box(xs, ys, r, c):
    inset = max(2, round(CELL_INSET * (xs[c + 1] - xs[c])))
    return (xs[c] + inset, ys[r] + inset, xs[c + 1] - inset, ys[r + 1] - inset)


def _bitmask(cell):
    """Ink of one cell as a bitmask, cropped to the mark and scale-normalized.

    Returns (mask, ink_pixels). Aspect ratio is preserved — squashing a `1`
    into a square box turns it into something a `2` template matches better.
    """
    w, h = cell.size
    px = cell.load()
    minx, miny, maxx, maxy = w, h, -1, -1
    ink = 0
    for y in range(h):
        for x in range(w):
            if px[x, y] <= INK_LEVEL:
                ink += 1
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)
    if ink < MIN_INK:
        return 0, ink

    crop = cell.crop((minx, miny, maxx + 1, maxy + 1))
    cw, ch = crop.size
    scale = NORM / max(cw, ch)
    tw, th = max(1, round(cw * scale)), max(1, round(ch * scale))
    canvas = Image.new("L", (NORM, NORM), 255)
    canvas.paste(crop.resize((tw, th), Image.LANCZOS),
                 ((NORM - tw) // 2, (NORM - th) // 2))

    cpx = canvas.load()
    mask = 0
    for y in range(NORM):
        for x in range(NORM):
            if cpx[x, y] <= INK_LEVEL:
                mask |= 1 << (y * NORM + x)
    return mask, ink


def _similarity(a, b):
    """Jaccard overlap of two ink bitmasks."""
    union = (a | b).bit_count()
    return (a & b).bit_count() / union if union else 0.0


def _font_paths():
    """Fonts to draw fallback templates with, widest variety first.

    Starts from the renderer's own candidates — a model that used PIL picked
    from the same list — then adds the other weights and families likely to
    be installed.
    """
    extra = []
    for path in _FONT_CANDIDATES:
        extra.append(path)
        for a, b in (("-Bold", "-Regular"), ("Bold", ""), ("bd.ttf", ".ttf")):
            if a in path:
                extra.append(path.replace(a, b))
    extra += [
        "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ]
    seen, out = set(), []
    for path in extra:
        if path not in seen and Path(path).exists():
            seen.add(path)
            out.append(path)
    return out


def font_templates(size=48):
    """Bitmask templates for 1-9 drawn in every font we can find."""
    templates = {}
    for path in _font_paths():
        try:
            font = ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
        for digit in range(1, 10):
            box = size * 2
            img = Image.new("L", (box, box), 255)
            ImageDraw.Draw(img).text(
                (box // 2, box // 2), str(digit), font=font, fill=0, anchor="mm"
            )
            mask, ink = _bitmask(img)
            if ink >= MIN_INK:
                templates.setdefault(digit, []).append(mask)
    return templates


class Reader:
    """Transcribes rendered Sudoku images into 9x9 grids."""

    def __init__(self, templates, calibrated_digits=frozenset()):
        self.templates = templates
        # Digits for which a pixel-exact example was taken from the benchmark's
        # own puzzle images, rather than only guessed at from installed fonts.
        self.calibrated_digits = frozenset(calibrated_digits)

    @classmethod
    def calibrate(cls, puzzle_images_dir, puzzles, with_fonts=True):
        """Build a reader from the input images plus installed fonts.

        `puzzles` is puzzles.json: each clue in an input image is a labelled
        example, so the digits every model started from are matched exactly.
        """
        templates = {}
        calibrated = set()
        images_dir = Path(puzzle_images_dir)
        for puzzle in puzzles or []:
            pid = puzzle.get("id")
            clues = puzzle.get("puzzle")
            if pid is None or not clues:
                continue
            path = images_dir / f"puzzle_{pid:03d}.png"
            if not path.exists():
                continue
            img = _to_gray(path)
            lines = find_grid_lines(img)
            if lines is None:
                continue
            xs, ys = lines
            for r in range(9):
                for c in range(9):
                    digit = clues[r][c]
                    if not digit:
                        continue
                    mask, ink = _bitmask(img.crop(_cell_box(xs, ys, r, c)))
                    if ink < MIN_INK:
                        continue
                    bucket = templates.setdefault(digit, [])
                    if mask not in bucket:
                        bucket.append(mask)
                    calibrated.add(digit)

        if with_fonts:
            for digit, masks in font_templates().items():
                bucket = templates.setdefault(digit, [])
                bucket.extend(m for m in masks if m not in bucket)

        return cls(templates, calibrated)

    def transcribe(self, path):
        """Read one image into a grid.

        Returns a dict with `grid` (9x9, 0 = empty) or None if the image is
        not a readable grid, plus the cells the reader was unsure about. A
        guess is never silently upgraded to a fact: uncertain cells are
        listed so a caller can refuse to score the image.
        """
        try:
            img = _to_gray(path)
        except (OSError, ValueError) as e:
            return {"grid": None, "error": f"unreadable image: {e}", "uncertain": []}

        lines = find_grid_lines(img)
        if lines is None:
            return {
                "grid": None,
                "error": "no 9x9 grid found in the image",
                "uncertain": [],
            }
        xs, ys = lines

        grid = [[0] * 9 for _ in range(9)]
        uncertain = []
        for r in range(9):
            for c in range(9):
                mask, ink = _bitmask(img.crop(_cell_box(xs, ys, r, c)))
                if ink < MIN_INK:
                    continue
                best_score, best_digit, runner_up = 0.0, 0, 0.0
                for digit, masks in self.templates.items():
                    score = max(_similarity(mask, m) for m in masks)
                    if score > best_score:
                        best_score, best_digit, runner_up = score, digit, best_score
                    elif score > runner_up:
                        runner_up = score
                grid[r][c] = best_digit
                if best_score < MIN_SCORE or best_score - runner_up < MIN_MARGIN:
                    uncertain.append(
                        {
                            "cell": [r, c],
                            "read_as": best_digit,
                            "score": round(best_score, 3),
                            "margin": round(best_score - runner_up, 3),
                        }
                    )
        return {"grid": grid, "error": None, "uncertain": uncertain}


def grade_image(reader, puzzle, path):
    """Verify one solution image against its puzzle's clues and Sudoku rules.

    Returns a verdict shaped like grader.verify's, so the same reporting works
    for both paths, tagged with where it came from.
    """
    read = reader.transcribe(path)
    if read["grid"] is None:
        return {
            "correct": False,
            "error_type": "IMAGE_UNREADABLE",
            "source": "local-transcription",
            "read_error": read["error"],
        }
    verdict = verify(puzzle["puzzle"], read["grid"])
    verdict["source"] = "local-transcription"
    verdict["grid"] = read["grid"]
    if read["uncertain"]:
        # Kept alongside the verdict rather than overriding it: a misread grid
        # almost never satisfies the clues and all 27 constraints, so a
        # "correct" verdict survives a shaky cell, while an "incorrect" one
        # with uncertain cells deserves a human look.
        verdict["uncertain_cells"] = read["uncertain"]
    return verdict


def self_check(reader, puzzle_images_dir, puzzles):
    """Transcribe the input images and compare against puzzles.json.

    This is the reader's own report card. The clues are known exactly, so any
    mismatch means the transcription is broken and its verdicts on the
    solution images cannot be trusted.
    """
    images_dir = Path(puzzle_images_dir)
    checks = []
    for puzzle in puzzles or []:
        pid = puzzle.get("id")
        clues = puzzle.get("puzzle")
        if pid is None or not clues:
            continue
        path = images_dir / f"puzzle_{pid:03d}.png"
        if not path.exists():
            checks.append({"puzzle_id": pid, "ok": False, "error": "image missing"})
            continue
        read = reader.transcribe(path)
        if read["grid"] is None:
            checks.append({"puzzle_id": pid, "ok": False, "error": read["error"]})
            continue
        wrong = [
            [r, c]
            for r in range(9)
            for c in range(9)
            if read["grid"][r][c] != clues[r][c]
        ]
        checks.append(
            {
                "puzzle_id": pid,
                "ok": not wrong,
                "cells_matched": 81 - len(wrong),
                "mismatched_cells": wrong[:10],
            }
        )
    return {
        "n_checked": len(checks),
        "n_ok": sum(1 for c in checks if c["ok"]),
        "all_ok": bool(checks) and all(c["ok"] for c in checks),
        "checks": checks,
    }
