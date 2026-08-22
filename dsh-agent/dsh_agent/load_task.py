"""Task spec loader for dsh-agent harness.

A task spec is a JSON file that describes:
- The prompt text for first and subsequent rounds (vision + text-only variants)
- Input/output filename conventions
- Asset files to install in the agent's working directory
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Defaults that the spec doesn't need to repeat.
DEFAULT_INPUT_FILENAME = "input.png"
DEFAULT_OUTPUT_FILENAME = "output.png"
DEFAULT_INPUT_GLOB = "*"
# How long to wait (seconds) between polling the DSH API for a turn to finish.
POLL_INTERVAL = 2


def load_task(path: str | Path) -> dict:
    """Read a task spec JSON file, filling in defaults.

    Returns a dict with at least:
        task (str):          short name, defaults to the file stem
        prompts.first_round: {vision: str, text_only: str}
        prompts.next_round:  {vision: str, text_only: str}
        input_filename (str): basename the agent reads each round
        output_filename (str): basename the agent writes each round
        input_glob (str):     glob pattern to enumerate inputs
        assets (list):        files to install in the working directory

    Raises ValueError on any structural problem.
    """
    path = Path(path)
    try:
        spec = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"could not read task spec {path}: {e}") from e

    if not isinstance(spec, dict):
        raise ValueError(f"task spec {path} is not a JSON object")

    prompts = spec.get("prompts")
    if not isinstance(prompts, dict) or not prompts.get("first_round"):
        raise ValueError(
            f"task spec {path} needs prompts.first_round (and normally "
            f"prompts.next_round)"
        )

    for phase in ("first_round", "next_round"):
        variants = prompts.get(phase) or prompts["first_round"]
        if isinstance(variants, str):
            variants = {"vision": variants, "text_only": variants}
        if not variants.get("vision") and not variants.get("text_only"):
            raise ValueError(
                f"task spec {path}: prompts.{phase} has neither a vision "
                f"nor a text_only variant"
            )
        prompts[phase] = variants

    spec.setdefault("task", path.stem)
    spec.setdefault("input_filename", DEFAULT_INPUT_FILENAME)
    spec.setdefault("output_filename", DEFAULT_OUTPUT_FILENAME)
    spec.setdefault("input_glob", DEFAULT_INPUT_GLOB)
    spec.setdefault("assets", [])
    return spec


def validate_assets(assets: list, wd: Path) -> list[Path]:
    """Check that every asset listed in the spec exists on disk.

    Each asset entry is a dict with at least a ``path`` key (relative to
    the task spec's own directory).  Returns the list of resolved absolute
    paths.  Missing assets are logged to stderr but do not raise.
    """
    resolved = []
    for a in assets:
        src = a.get("path")
        if not src:
            continue
        dst = wd / src
        if dst.exists():
            resolved.append(dst)
        else:
            # The spec may point at a sibling file; try relative to itself.
            print(
                f"[dsh-agent] asset {src} not found at {dst}",
                file=sys.stderr,
            )
    return resolved