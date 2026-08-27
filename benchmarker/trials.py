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
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TRIALS_DIR = "trials"

# Suffixes whose changes cannot alter what a run measures: records and results
# (.json), configuration and the model manifest (.toml), prose (.md). A
# benchmarking session rewrites these constantly — the results dir alone is a
# stream of JSON — so counting them would mark every entry dirty and the flag
# would say nothing.
NON_SOURCE_SUFFIXES = {".json", ".toml", ".md"}

# The repo this package lives in, so a rev is read from the benchmark's own
# tree no matter which directory the run was invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent


def git_rev(root=None):
    """The commit a result was produced from, plus `-dirty` if source differs.

    The point of a permanent record is to see whether a change to the harness
    or the prompts moved the numbers, and that question is unanswerable if an
    entry does not say which code produced it.

    Dirtiness counts both tracked modifications and untracked files — both are
    code the commit does not describe — but only for files that are source. A
    run whose only local changes are to .json/.toml/.md was produced by
    exactly the committed code.

    Returns None when git is unavailable or this is not a repository: a
    missing rev is honest, an invented one is not.
    """
    def _git(*args):
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(root or REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        # Returned verbatim: a porcelain status line begins with two status
        # columns and a space, and stripping the output would eat the first
        # line's leading space, shifting that one path by a character.
        return proc.stdout if proc.returncode == 0 else None

    rev = (_git("rev-parse", "--short", "HEAD") or "").strip()
    if not rev:
        return None
    status = _git("status", "--porcelain", "-uall")
    if status is None:
        # HEAD read fine but the status did not, so whether the tree is clean
        # is unknown. Report the commit without claiming it is untouched.
        return rev
    for line in status.splitlines():
        if len(line) <= 3:
            continue
        path = line[3:]
        if " -> " in path:  # a rename is reported as "old -> new"
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if path and Path(path).suffix.lower() not in NON_SOURCE_SUFFIXES:
            return f"{rev}-dirty"
    return rev


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

    Correctness first, then time to solve, then cost. Accuracy dominates
    outright: a faster result never outranks a more accurate one, however much
    faster it was, and time only decides between results that scored the same.
    A non-positive time is treated as unknown so a run that errored out
    instantly cannot masquerade as the fastest.

    Every ranking in the project goes through this one key — the report rows
    and leaderboard.json (`get_leaderboard.collect`), which run of a model the
    permanent record keeps (`is_better`, used by `merge`), and the
    results-dir-beats-the-record check — so the guarantee holds everywhere
    rather than per call site.
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


def _entry_key(entry):
    """The composite identity of a trial entry: model + middleware flag."""
    return (entry["model"], bool(entry.get("middleware", False)))


def merge(existing, summaries, *, tid, seed, n_puzzles, difficulty, fingerprint,
          middleware=False, rev=None):
    """Merge this run's summaries into an existing trial.

    Each entry is keyed by ``(model, middleware)``, so a run with vision
    middleware enabled and a run without it are tracked as independent
    observations — they used different prompts and are not directly comparable.

    Existing entries that predate the middleware field are treated as
    ``middleware: false``.

    Returns (trial, changes) where ``changes`` maps (model, middleware) to one
    of "new", "improved", or "kept", so the caller can report what actually
    moved rather than claiming every model was recorded.

    Each recorded entry is stamped with ``rev``, the commit its run came from
    (see `git_rev`), so the record says which code produced each best result.
    An entry that is kept keeps the rev of the run that set it — the rev
    describes the result, not the last attempt at it.

    Raises ValueError if the puzzle set's content hash disagrees with the
    stored one — merging those numbers would compare different puzzles.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if rev is None:
        rev = git_rev()

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
        # Retroactive: entries written before the middleware field was added
        # were all from no-middleware runs.
        for e in trial.setdefault("entries", []):
            e.setdefault("middleware", False)
        trial.setdefault("entries", [])

    by_key = {_entry_key(e): e for e in trial["entries"]}
    changes = {}

    for summary in summaries:
        model = summary["model"]
        candidate = dict(summary)
        candidate["middleware"] = bool(middleware)
        candidate["recorded_at"] = now
        candidate["rev"] = rev
        key = _entry_key(candidate)
        prior = by_key.get(key)

        if prior is None:
            candidate["runs"] = 1
            by_key[key] = candidate
            changes[key] = "new"
        elif is_better(candidate, prior):
            candidate["runs"] = prior.get("runs", 1) + 1
            candidate["previous_best"] = {
                "accuracy": prior.get("accuracy"),
                "avg_elapsed_s": prior.get("avg_elapsed_s"),
                "total_tokens": prior.get("total_tokens"),
                "recorded_at": prior.get("recorded_at"),
            }
            by_key[key] = candidate
            changes[key] = "improved"
        else:
            # Keep the better prior result, but remember it was re-attempted.
            prior["runs"] = prior.get("runs", 1) + 1
            prior["last_attempt_at"] = now
            changes[key] = "kept"

    trial["entries"] = sorted(by_key.values(), key=_rank_key)
    trial["updated_at"] = now
    trial["puzzle_fingerprint"] = fingerprint
    return trial, changes


def save_trial(path: Path, trial) -> None:
    """Write a trial atomically so an interrupt can't truncate the record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(trial, indent=2) + "\n")
    tmp.replace(path)
