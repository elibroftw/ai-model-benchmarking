---
name: scribbling-on-images
description: Writing text or digits ONTO an existing image. Prioritize readability; do NOT spend effort matching the original image's font unless the task explicitly requires font adherence.
---

# Scribbling on images

You've been asked to draw or write text onto an existing image (e.g. fill in
a form, annotate a screenshot, write digits into a grid). Follow this skill's
guidance unless the task **explicitly** demands otherwise.

## The rule

**Readability beats fidelity.** Pick a font that is clear at the rendered
size and get the text down. Do NOT:

- Attempt to identify the exact typeface used in the source image.
- Iterate to match stroke weight, kerning, or subtle glyph shapes.
- Loop over multiple candidate fonts comparing pixel similarity to the source.
- Ask questions about font choice — decide and move on.

The task at hand is almost never "faithfully reproduce this font." It's
"communicate information visually." Font-matching is the incidental detail;
the digit/letter itself is the payload.

## What "readable" means (concrete defaults)

- Use a **bold sans-serif** font. Common good picks that ship on Linux/macOS:
  - `/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf`
  - `/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf`
  - Any `arial-bold` / `helvetica-bold` on macOS.
- Size the text at roughly **60–75%** of the containing cell/box's smaller
  dimension. Bigger reads better; leave a small margin.
- Colour: **solid black** on light backgrounds, **solid white** on dark.
  Do not anti-alias if the rest of the image is pixel-crisp.
- Centre text in its target region. Use the font metrics (`textbbox`) to
  compute the offset — don't eyeball.

## PIL recipe (starting point, not gospel)

```python
from PIL import Image, ImageDraw, ImageFont

img = Image.open("input.png").convert("RGB")
draw = ImageDraw.Draw(img)

def readable_font(size):
    for path in [
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()

def write_centered(text, cx, cy, cell_px, colour="black"):
    font = readable_font(int(cell_px * 0.65))
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (r - l) / 2 - l, cy - (b - t) / 2 - t), text,
              fill=colour, font=font)
```

## When to break this rule

If the task explicitly says "match the original font" or "in the same style
as the existing text", then font-matching is the point of the exercise and
this skill does not apply. Otherwise: pick a clear font, place the text, ship.
