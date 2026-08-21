"""Render a Sudoku puzzle to a PNG image."""
import io
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Bold.ttf",
    "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
]


def _load_font(size):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_puzzle(puzzle, cell_size=64, margin=16):
    """Render a 9x9 puzzle as PNG bytes.

    puzzle: 9x9 list of ints (0 = empty).
    """
    grid_px = cell_size * 9
    size = grid_px + 2 * margin
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    for i in range(10):
        thick = 4 if i % 3 == 0 else 1
        x = margin + i * cell_size
        draw.line([(x, margin), (x, margin + grid_px)], fill="black", width=thick)
        draw.line([(margin, x), (margin + grid_px, x)], fill="black", width=thick)

    font = _load_font(cell_size - 20)

    for r in range(9):
        for c in range(9):
            v = puzzle[r][c]
            if v == 0:
                continue
            text = str(v)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            cx = margin + c * cell_size + cell_size // 2
            cy = margin + r * cell_size + cell_size // 2
            draw.text((cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]),
                      text, fill="black", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
