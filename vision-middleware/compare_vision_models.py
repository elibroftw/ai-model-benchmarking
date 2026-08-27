#!/usr/bin/env python3
"""Compare vision models on transcription speed over the same set of puzzles.

Generates N Sudoku puzzles once, then asks every model under test to
transcribe every puzzle image, timing each `transcribe()` call.  Finishes
with a ranking of the models by response time.

Calls that fail — an exception, or a model that answers with nothing
usable (a null content field, or the literal word "None", which some of
these models emit when they cannot read the image) — are recorded as
failures and left out of the timing statistics.  A model with no
successful call at all is excluded from the ranking rather than ranked
last, since a model that never answers has no response time to compare.

The middleware picks its model from models.toml, and `transcribe()` takes
no model argument, so each model is driven through a generated config that
copies the base `[vision]` table with only `model` swapped out.

Usage:
    python vision-middleware/compare_vision_models.py
    python vision-middleware/compare_vision_models.py --puzzles 3
    python vision-middleware/compare_vision_models.py --models a/b,c/d --json out.json

Needs OPENROUTER_API_KEY (or [vision].api_key in models.toml), same as
`transcribe.py` itself.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from transcribe import ConfigError, load_config, transcribe  # noqa: E402
from example import _load_dotenv, build_task_prompt  # noqa: E402
from benchmarker.generator import generate_puzzle  # noqa: E402
from benchmarker.renderer import render_puzzle  # noqa: E402

# The models under test.  Vision-capable and open-weight, as models.toml
# requires; override with --models.
DEFAULT_MODELS = (
    "google/gemma-4-26b-a4b-it",
    # TODO
    # deepseek/deepseek-v4-flash-vision-exp
)


def _toml_value(value) -> str:
    """Serialize a scalar TOML value.  Only what a [vision] table holds."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # TOML basic strings escape the same way JSON strings do.
        return json.dumps(value)
    raise TypeError(f"cannot write {type(value).__name__} to config: {value!r}")


def write_model_config(base: dict, model: str, path: Path) -> Path:
    """Write a models.toml holding `base` with [vision].model set to `model`."""
    lines = ["[vision]", f"model = {_toml_value(model)}"]
    lines += [
        f"{key} = {_toml_value(value)}"
        for key, value in base.items()
        if key != "model"
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def is_usable(text) -> bool:
    """Whether a transcription counts as an answer.

    The middleware hands back the model's `content` verbatim, so a model
    that declines shows up either as JSON null or as the word "None" in
    prose.  Neither is a transcription.
    """
    if not isinstance(text, str):
        return False
    return text.strip().lower() not in ("", "none")


def make_puzzles(count: int, seed: int, n_clues: int) -> list[dict]:
    """Generate `count` puzzles, each reseeded so the set is reproducible."""
    puzzles = []
    for i in range(count):
        random.seed(seed + i)
        puzzle, _solution = generate_puzzle(n_clues=n_clues)
        puzzles.append({
            "index": i + 1,
            "seed": seed + i,
            "clues": sum(1 for row in puzzle for v in row if v),
            "png": render_puzzle(puzzle),
        })
    return puzzles


def run_trials(
    *,
    puzzles: list[dict],
    models: list[str],
    base_config: dict,
    task: str,
    workdir: Path,
) -> list[dict]:
    """Time every (puzzle, model) pair sequentially; return one row per call.

    Sequential on purpose: overlapping requests would measure contention
    as much as model speed.  The per-puzzle model order rotates so that no
    single model always pays for whatever the first call of a puzzle costs.
    """
    configs = {
        model: write_model_config(
            base_config, model, workdir / f"models-{i}.toml"
        )
        for i, model in enumerate(models)
    }
    # Models whose failure would repeat for every image (bad ID, rejected
    # key): dropped for the rest of the run instead of retried 10 times.
    dead: dict[str, str] = {}
    rows = []

    for puzzle in puzzles:
        image_path = workdir / f"puzzle-{puzzle['index']:02d}.png"
        image_path.write_bytes(puzzle["png"])
        order = models[(puzzle["index"] - 1) % len(models):] + \
            models[:(puzzle["index"] - 1) % len(models)]

        for model in order:
            label = f"puzzle {puzzle['index']}/{len(puzzles)} · {model}"
            if model in dead:
                print(f"[compare] skip {label}: {dead[model]}", file=sys.stderr)
                rows.append({
                    "puzzle": puzzle["index"], "model": model, "seconds": None,
                    "ok": False, "error": f"skipped: {dead[model]}",
                    "chars": None,
                })
                continue

            print(f"[compare] {label} …", file=sys.stderr)
            started = time.perf_counter()
            try:
                text = transcribe(
                    image_path=image_path,
                    task=task,
                    config_path=configs[model],
                )
            except ConfigError as e:
                elapsed = time.perf_counter() - started
                dead[model] = str(e)
                print(f"[compare] {label}: config error, dropping model — {e}",
                      file=sys.stderr)
                rows.append({
                    "puzzle": puzzle["index"], "model": model,
                    "seconds": elapsed, "ok": False, "error": str(e),
                    "chars": None,
                })
                continue
            except Exception as e:
                elapsed = time.perf_counter() - started
                print(f"[compare] {label}: failed after {elapsed:.2f}s — "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                rows.append({
                    "puzzle": puzzle["index"], "model": model,
                    "seconds": elapsed, "ok": False,
                    "error": f"{type(e).__name__}: {e}", "chars": None,
                })
                continue

            elapsed = time.perf_counter() - started
            ok = is_usable(text)
            if ok:
                print(f"[compare] {label}: {elapsed:.2f}s, {len(text)} chars",
                      file=sys.stderr)
            else:
                print(f"[compare] {label}: {elapsed:.2f}s but returned nothing "
                      f"usable ({text!r:.40})", file=sys.stderr)
            rows.append({
                "puzzle": puzzle["index"], "model": model,
                "seconds": elapsed, "ok": ok,
                "error": None if ok else "returned None",
                "chars": len(text) if isinstance(text, str) else None,
            })

    return rows


def summarize(rows: list[dict], models: list[str]) -> list[dict]:
    """Per-model stats over its successful calls, fastest median first."""
    stats = []
    for model in models:
        mine = [r for r in rows if r["model"] == model]
        times = [r["seconds"] for r in mine if r["ok"]]
        stats.append({
            "model": model,
            "attempts": len(mine),
            "successes": len(times),
            "median": statistics.median(times) if times else None,
            "mean": statistics.fmean(times) if times else None,
            "min": min(times) if times else None,
            "max": max(times) if times else None,
        })
    # Ranked models first (by median), then the ones with nothing to rank.
    return sorted(
        stats,
        key=lambda s: (s["median"] is None, s["median"] or 0.0),
    )


def print_report(rows: list[dict], stats: list[dict], puzzles: list[dict]) -> None:
    models = [s["model"] for s in stats]
    width = max(len(m) for m in models)

    print("\n=== per-call time in seconds (× = no usable transcription) ===")
    print("  puzzle  " + "  ".join(m.rjust(max(9, len(m))) for m in models))
    by_key = {(r["puzzle"], r["model"]): r for r in rows}
    for puzzle in puzzles:
        cells = []
        for model in models:
            row = by_key.get((puzzle["index"], model))
            if row is None or row["seconds"] is None:
                cell = "×"
            else:
                cell = f"{row['seconds']:.2f}" + ("" if row["ok"] else " ×")
            cells.append(cell.rjust(max(9, len(model))))
        print(f"  {puzzle['index']:>6}  " + "  ".join(cells))

    print("\n=== ranking by median response time "
          "(models with no usable transcription excluded) ===")
    rank = 0
    for s in stats:
        if s["median"] is None:
            continue
        rank += 1
        print(f"  {rank}. {s['model'].ljust(width)}  "
              f"median {s['median']:6.2f}s   mean {s['mean']:6.2f}s   "
              f"min {s['min']:6.2f}s   max {s['max']:6.2f}s   "
              f"({s['successes']}/{s['attempts']} usable)")
    if rank == 0:
        print("  (no model produced a usable transcription)")

    excluded = [s for s in stats if s["median"] is None]
    if excluded:
        print("\n  excluded:")
        for s in excluded:
            print(f"    {s['model'].ljust(width)}  "
                  f"0/{s['attempts']} usable transcriptions")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank vision models by transcription speed on one puzzle set."
    )
    parser.add_argument(
        "--puzzles", type=int, default=10,
        help="How many puzzles to generate and send to every model (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed; puzzle i uses seed+i, so the set is reproducible "
             "(default: 42).",
    )
    parser.add_argument(
        "--clues", type=int, default=32,
        help="Approximate number of givens per puzzle (default: 32).",
    )
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help="Comma-separated model IDs to compare (default: %(default)s).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Base models.toml, for api_base/temperature/max_tokens/api_key "
             "(default: the one beside transcribe.py).  Its [vision].model is "
             "ignored; --models decides.",
    )
    parser.add_argument(
        "--task", default=None,
        help="Override the task description sent to the transcriber.  Default: "
             "the harness prompt from example.py.",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Also write every timing and the summary to this file as JSON.",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("error: --models listed no models", file=sys.stderr)
        return 2
    if args.puzzles < 1:
        print("error: --puzzles must be at least 1", file=sys.stderr)
        return 2

    _load_dotenv()

    try:
        base_config = load_config(args.config or Path(__file__).resolve().parent
                                 / "models.toml")
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    task = args.task or build_task_prompt()
    print(f"[compare] generating {args.puzzles} puzzles "
          f"(seeds {args.seed}–{args.seed + args.puzzles - 1}, "
          f"~{args.clues} clues) …", file=sys.stderr)
    puzzles = make_puzzles(args.puzzles, args.seed, args.clues)
    print(f"[compare] {len(puzzles) * len(models)} calls: "
          f"{len(puzzles)} puzzles × {len(models)} models "
          f"({', '.join(models)})", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="vision-middleware-compare-") as td:
        rows = run_trials(
            puzzles=puzzles,
            models=models,
            base_config=base_config,
            task=task,
            workdir=Path(td),
        )

    stats = summarize(rows, models)
    print_report(rows, stats, puzzles)

    if args.json:
        args.json.write_text(json.dumps({
            "seed": args.seed,
            "clues": args.clues,
            "puzzles": [
                {k: v for k, v in p.items() if k != "png"} for p in puzzles
            ],
            "calls": rows,
            "summary": stats,
        }, indent=2) + "\n")
        print(f"\n[compare] wrote {args.json}", file=sys.stderr)

    # Non-zero when nothing could be ranked; a partial failure is still a result.
    return 0 if any(s["median"] is not None for s in stats) else 1


if __name__ == "__main__":
    sys.exit(main())
