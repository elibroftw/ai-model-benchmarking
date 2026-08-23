#!/usr/bin/env python3
"""Print a markdown table from a trial JSON file (e.g. trials/seed-42-n3-mixed.json).

Usage:
    uv run cli/trial_table.py                                    # default trial
    uv run cli/trial_table.py trials/seed-42-n3-mixed.json       # explicit path
    uv run cli/trial_table.py --hide-errors                      # drop the errs column
"""
import argparse
import json
import sys
import tomllib
from pathlib import Path

DEFAULT_TRIAL = "trials/seed-42-n3-mixed.json"
DEFAULT_MODELS = "models.toml"


def load_costs(path):
    try:
        data = tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    raw = data.get("costs") or {}
    return {
        k: (float(v[0]), float(v[1]))
        for k, v in raw.items()
        if isinstance(v, (list, tuple)) and len(v) >= 2
    }


def load_types(path):
    """Heuristic: models under a ``# text-only`` guard are T, else V."""
    try:
        raw = Path(path).read_text()
    except OSError:
        return {}
    import re
    text_ids = set()
    in_text = False
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("#") and "text-only" in s.lower():
            in_text = True
            continue
        if s.startswith("#") and any(kw in s.lower() for kw in ("vision", "baseline", "disabled")):
            in_text = False
            continue
        if re.match(r'^[a-zA-Z_]\S*\s*=\s*\[', s):
            in_text = False
            continue
        if not in_text:
            continue
        m = re.search(r'"([^"]+)"', line)
        if m:
            text_ids.add(m.group(1))
    data = tomllib.loads(raw) if Path(path).exists() else {}
    mapping = {}
    for entries in (data.get("models") or {}).values():
        for entry in entries:
            mid = entry.get("id") if isinstance(entry, dict) else entry
            if isinstance(mid, str):
                mapping[mid] = "T" if mid in text_ids else "V"
    return mapping


def _cost_cell(entry, costs):
    acc = entry.get("accuracy", 0)
    if acc != 1.0:
        return "-"
    in_p, out_p = costs.get(entry["model"], (0, 0))
    if in_p == 0 and out_p == 0:
        return "-"
    cost = (entry["input_tokens"] * in_p + entry["output_tokens"] * out_p) / 1_000_000
    return f"${cost:.2f}"


def _cost_hour(entry, costs):
    in_p, out_p = costs.get(entry["model"], (0, 0))
    if in_p == 0 and out_p == 0:
        return "-"
    n = entry.get("n_puzzles", 0)
    avg = entry.get("avg_elapsed_s", 0) or 0
    total_sec = avg * n
    if total_sec <= 0:
        return "-"
    cost = (entry["input_tokens"] * in_p + entry["output_tokens"] * out_p) / 1_000_000
    return f"${cost * 3600 / total_sec:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Markdown table from a trial JSON.")
    parser.add_argument(
        "trial", nargs="?", default=DEFAULT_TRIAL,
        help=f"Path to the trial file (default: {DEFAULT_TRIAL})",
    )
    parser.add_argument(
        "--models-file", default=DEFAULT_MODELS,
        help=f"Model manifest for types & costs (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--hide-errors", action="store_true",
        help="Omit the errors column.",
    )
    args = parser.parse_args()

    trial_path = Path(args.trial)
    if not trial_path.exists():
        sys.exit(f"no such file: {trial_path}")
    trial = json.loads(trial_path.read_text())

    entries = trial.get("entries") or []
    if not entries:
        sys.exit(f"no entries in {trial_path}")

    costs = load_costs(args.models_file)
    types = load_types(args.models_file)

    headers = [
        "| # | model | type | mw | score | avg s | cost | $/h | images | tok_in | tok_out",
    ]
    if not args.hide_errors:
        headers[0] += " | errors"
    headers.append(
        "|--:|---|:--:|:--:|--:|--:|--:|--:|--:|--:|--:" +
        ("|--:" if not args.hide_errors else "|")
    )

    rows = []
    for i, e in enumerate(entries, 1):
        mw = "yes" if e.get("middleware") else "--"
        score = f"{e['n_correct']}/{e['n_puzzles']}"
        imgs = f"{e['n_output_images']}/{e['n_puzzles']}"
        row = (
            f"| {i} | `{e['model']}` | {types.get(e['model'], 'V')} | {mw} | "
            f"{score} | {e['avg_elapsed_s']:.1f} | {_cost_cell(e, costs)} | "
            f"{_cost_hour(e, costs)} | {imgs} | {e['input_tokens']:,} | "
            f"{e['output_tokens']:,}"
        )
        if not args.hide_errors:
            row += f" | {e['n_errors']}"
        row += " |"
        rows.append(row)

    print("\n".join(headers + rows))

    print()
    print(
        f"{trial['n_puzzles']} puzzles ({trial['difficulty']}), "
        f"seed={trial['seed']}, "
        f"updated {trial.get('updated_at', '?')}.  "
        f"`mw` — vision middleware active (yes / --).  "
        f"`cost` — total spend at OpenRouter prices, perfect models only.  "
        f"`$/h` — cost per wall-clock hour."
    )


if __name__ == "__main__":
    main()