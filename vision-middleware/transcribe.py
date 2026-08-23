#!/usr/bin/env python3
"""Vision Middleware — transcribe an image for use by a text-only agent.

Given an image and a task description, sends both to a vision-capable
model (configured in models.toml) and prints a plain-text transcription
designed to be useful for completing that task without seeing the image.

Usage:
    python transcribe.py --image puzzle.png --task "Solve this Sudoku puzzle"
    python transcribe.py --image screenshot.png --task "Fill out this web form"

The output is the model's transcription on stdout.  stderr carries
progress messages.  Exit code 0 on success, non-zero on error.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import tomllib
from pathlib import Path

import httpx

SYSTEM_PROMPT = (
    "You are an image-transcriber. You transcribe images such that the "
    "output can be used to complete the task that is also provided. Your "
    "output will be placed as the alt of an img html tag. Do not complete "
    "the task. Use the task description to infer what about the image "
    "would be useful for someone who would not be able to see the image "
    "for themselves. Do not attempt to complete the task given."
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "models.toml"


def load_config(path: Path) -> dict:
    """Read the vision model config from a TOML file."""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = tomllib.loads(path.read_text())
    vision = data.get("vision")
    if not isinstance(vision, dict):
        raise ValueError(f"{path}: missing [vision] table")
    return vision


def transcribe(
    *,
    image_path: Path,
    task: str,
    config_path: Path | None = None,
) -> str:
    """Send an image + task description to the vision model; return its text."""
    config = load_config(config_path or DEFAULT_CONFIG)

    model = config.get("model")
    if not model:
        raise ValueError("models.toml: [vision].model is required")

    api_base = config.get("api_base", "https://openrouter.ai/api/v1")
    api_key = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "no API key: set OPENROUTER_API_KEY or [vision].api_key in models.toml"
        )

    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 2000)

    # Read and encode the image.
    image_bytes = image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    # Infer MIME type from extension.
    suffix = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Task: {task}\n\nTranscribe the image above so that someone who cannot see it can still complete this task.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/sudoku-vision-benchmark",
        "X-Title": "Sudoku Vision Middleware",
    }

    print(f"[vision-middleware] calling {model} @ {url} …", file=sys.stderr)
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=120.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"OpenRouter HTTP {e.response.status_code} for model '{model}': {e}"
        ) from e

    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"no choices in response: {data}")

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(
        f"[vision-middleware] done — "
        f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
        f"completion_tokens={usage.get('completion_tokens', '?')}",
        file=sys.stderr,
    )
    return content


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe an image for use by a text-only agent."
    )
    parser.add_argument(
        "--image", required=True, type=Path,
        help="Path to the image file to transcribe.",
    )
    parser.add_argument(
        "--task", required=True,
        help="Task description that the transcription should support.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=f"Path to models.toml (default: {DEFAULT_CONFIG}).",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    try:
        text = transcribe(
            image_path=args.image,
            task=args.task,
            config_path=args.config,
        )
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
