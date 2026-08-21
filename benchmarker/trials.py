"""Permanent per-puzzle-set leaderboards.

`results/` is scratch — it is overwritten by every run and gitignored. A
trial is the opposite: a durable record, one file per puzzle set, holding
only each model's BEST result so far. Re-running a puzzle set merges into
the existing file rather than replacing it, so adding one new model doesn't
discard everything already measured.

A puzzle set is identified by (seed, n_puzzles, difficulty), which is what
determines the generated puzzles. The puzzles' content hash is stored too,
so a generator change that silently alters the puzzles for a given seed is
caught instead of merging incomparable numbers.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TRIALS_DIR = "trials"


def trial_id(seed, n_puzzles, difficulty) -> str:
    """Stable, human-readable name for a puzzle set."""
    return f"seed-{seed}-n{n_puzzles}-{difficulty}"


def puzzle_fingerprint(puzzles_meta) -> str:
    """Content hash of the generated puzzles.

    Two runs may share a seed yet produce different puzzles if the generator
    changes. Hashing the grids themselves detects that.
    """
    h = hashlib.sha256()
    for p in sorted(puzzles_meta, key=lambda x: x["id"]):
        h.update(str(p["id"]).encode())
        h.update(str(p["difficulty"]).encode())
        h.update(json.dumps(p["puzzle"], separators=(",", ":")).encode())
    return h.hexdigest()


def _rank_key(entry):
    """Sort key implementing the README's grading order.

    Correctness first, then time to solve, then cost. A non-positive time is
    treated as unknown so a run that errored out instantly cannot masquerade
    as the fastest.
    """
    elapsed = entry.get("avg_elapsed_s")
    if not elapsed or elapsed <= 0:
        elapsed = float("inf")
    tokens = entry.get("total_tokens") or float("inf")
    return (-(entry.get("accuracy") or 0.0), elapsed, tokens)


def is_better(new, old) -> bool:
    """True if `new` outranks `old` on correctness, then speed, then cost."""
    return _rank_key(new) < _rank_key(old)


def load_trial(path: Path) -> dict | None:
    """Read an existing trial file, or None if absent/unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def merge(existing, summaries, *, tid, seed, n_puzzles, difficulty, fingerprint):
    """Merge this run's summaries into an existing trial.

    Returns (trial, changes) where `changes` maps model id to one of
    "new", "improved", or "kept", so the caller can report what actually
    moved rather than claiming every model was recorded.

    Raises ValueError if the puzzle set's content hash disagrees with the
    stored one — merging those numbers would compare different puzzles.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if existing is None:
        trial = {
            "trial_id": tid,
            "seed": seed,
            "n_puzzles": n_puzzles,
            "difficulty": difficulty,
            "puzzle_fingerprint": fingerprint,
            "created_at": now,
            "updated_at": now,
            "entries": [],
        }
    else:
        stored = existing.get("puzzle_fingerprint")
        if stored and stored != fingerprint:
            raise ValueError(
                f"puzzle set for {tid} does not match the stored trial "
                f"(fingerprint {stored[:12]}... vs {fingerprint[:12]}...). "
                f"The generator likely changed. Delete the trial file or use "
                f"a different seed rather than merging incomparable results."
            )
        trial = dict(existing)
        trial.setdefault("entries", [])

    by_model = {e["model"]: e for e in trial["entries"]}
    changes = {}

    for summary in summaries:
        model = summary["model"]
        candidate = dict(summary)
        candidate["recorded_at"] = now
        prior = by_model.get(model)

        if prior is None:
            candidate["runs"] = 1
            by_model[model] = candidate
            changes[model] = "new"
        elif is_better(candidate, prior):
            candidate["runs"] = prior.get("runs", 1) + 1
            candidate["previous_best"] = {
                "accuracy": prior.get("accuracy"),
                "avg_elapsed_s": prior.get("avg_elapsed_s"),
                "total_tokens": prior.get("total_tokens"),
                "recorded_at": prior.get("recorded_at"),
            }
            by_model[model] = candidate
            changes[model] = "improved"
        else:
            # Keep the better prior result, but remember it was re-attempted.
            prior["runs"] = prior.get("runs", 1) + 1
            prior["last_attempt_at"] = now
            changes[model] = "kept"

    trial["entries"] = sorted(by_model.values(), key=_rank_key)
    trial["updated_at"] = now
    trial["puzzle_fingerprint"] = fingerprint
    return trial, changes


def save_trial(path: Path, trial) -> None:
    """Write a trial atomically so an interrupt can't truncate the record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(trial, indent=2) + "\n")
    tmp.replace(path)
