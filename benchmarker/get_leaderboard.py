"""Build a summary of a `results/` directory after the fact.

`Benchmark.run` prints a leaderboard as it goes, but that only happens when a
run finishes: an interrupted run leaves per-model JSON on disk and no summary
at all. Everything here reads that directory instead of re-running anything,
so the same numbers can be recovered from whatever a partial run left behind.

Three sources are read, and they mean different things:

* The solution images themselves. `vision.py` transcribes each PNG locally and
  `grader.verify` checks the grid against the puzzle's clues and the Sudoku
  rules. This is the strongest signal available here: deterministic, free, and
  independent of both the grading API and the agent's own account of its work.
  The transcriber is checked against puzzles.json before it is trusted.
* `final/<model>.json` — the benchmarker's graded records, from the grading
  API. `verdict.correct` is a real verification too, but only exists when that
  call succeeded. (Runs made before the records moved into `final/` left them
  at the top level; both are read.)
* `solutions/<model>/.harness_state.json` — the harness's own resume file. Its
  `success: true` means only "the agent finished without raising and wrote
  output.png"; the `final_answer` beside it is the agent's own claim about its
  work. Nothing there is verified, and a model that fabricates a grid looks
  identical to one that solved it.

So the state file is used to fill in models the run never graded (an
interrupt leaves the state file but no JSON) and to report what the harness
claimed — never to award correctness. Correctness comes from reading pixels.

The per-model metric functions (`summarize`, `learning_stats`) live here
rather than on `Benchmark` so the live run and this after-the-fact path can
never disagree.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from .trials import _rank_key
from .trials import (
    DEFAULT_TRIALS_DIR,
    is_better,
    load_trial,
    puzzle_fingerprint as trial_puzzle_fingerprint,
)
from .vision import Reader, grade_image, self_check

# Files in the results dir that are not per-model records. Only needed for
# the pre-`final/` layout, where model records sat beside the run's shared
# artifacts. Kept in sync with FINAL_SUBDIR in benchmark.py.
NON_MODEL_FILES = {"puzzles.json", "leaderboard.json", "task.json"}
FINAL_SUBDIR = "final"

# Written by the harness into each model's solutions dir. Kept in sync with
# STATE_FILENAME in sudoku_agent_harness/agent.py.
HARNESS_STATE_FILENAME = ".harness_state.json"


def summarize(model, results, session_summary=None):
    """Per-model metrics for one model's list of per-puzzle records."""
    n = len(results)
    n_correct = sum(1 for r in results if r.get("verdict", {}).get("correct"))
    n_errors = sum(1 for r in results if "error" in r)
    elapsed = [r["elapsed"] for r in results if r.get("elapsed") is not None]
    avg_time = sum(elapsed) / len(elapsed) if elapsed else 0.0

    per_diff = {}
    for r in results:
        d = r.get("difficulty")
        per_diff.setdefault(d, {"n": 0, "correct": 0})
        per_diff[d]["n"] += 1
        if r.get("verdict", {}).get("correct"):
            per_diff[d]["correct"] += 1

    n_images = sum(1 for r in results if r.get("output_image"))
    tok_in = sum(r.get("input_tokens") or 0 for r in results)
    tok_out = sum(r.get("output_tokens") or 0 for r in results)

    return {
        "model": model,
        "n_puzzles": n,
        "n_correct": n_correct,
        "n_errors": n_errors,
        # How many solution images the model actually produced. Separates
        # "answered and got it wrong" from "never produced an answer".
        "n_output_images": n_images,
        "output_rate": n_images / n if n else 0.0,
        "accuracy": n_correct / n if n else 0.0,
        "avg_elapsed_s": avg_time,
        "input_tokens": tok_in,
        "output_tokens": tok_out,
        "total_tokens": tok_in + tok_out,
        "tokens_per_correct": (tok_in + tok_out) / n_correct if n_correct else None,
        "per_difficulty": per_diff,
        "learning": learning_stats(results),
        "session": session_summary or {},
    }


def learning_stats(results):
    """Split warmup (round 1) from steady state (rounds 2+).

    The whole point of running one persistent agent sequentially is to see
    whether it gets faster/better after building its own infrastructure.
    """
    by_round = sorted(
        (r for r in results if r.get("round") is not None),
        key=lambda r: r["round"],
    )
    if not by_round:
        return {}

    warmup = by_round[0]
    rest = by_round[1:]

    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    steady_time = _avg([r.get("elapsed") for r in rest])
    warmup_time = warmup.get("elapsed")
    speedup = (
        warmup_time / steady_time
        if warmup_time and steady_time and steady_time > 0
        else None
    )

    return {
        "warmup_elapsed_s": warmup_time,
        "warmup_correct": bool(warmup.get("verdict", {}).get("correct")),
        "warmup_total_tokens": warmup.get("total_tokens"),
        "steady_state_avg_elapsed_s": steady_time,
        "steady_state_avg_total_tokens": _avg([r.get("total_tokens") for r in rest]),
        "steady_state_accuracy": (
            sum(1 for r in rest if r.get("verdict", {}).get("correct")) / len(rest)
            if rest else None
        ),
        "speedup_after_warmup": speedup,
    }


def _safe_name(model_id):
    """The filename the benchmark writes for a model ID."""
    return model_id.replace("/", "_").replace(":", "_")


def load_model_ids(models_file):
    """Map safe filename -> real model ID, so `a_b.json` reads as `a/b`.

    The mangling is lossy (both `/` and `:` become `_`), so the manifest is
    one way back; a harness state file, which stores the ID verbatim, is
    another and is preferred when present.
    """
    path = Path(models_file)
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError):
        return {}

    mapping = {}
    for entries in (data.get("models") or {}).values():
        for entry in entries:
            model_id = entry.get("id") if isinstance(entry, dict) else entry
            if isinstance(model_id, str):
                mapping[_safe_name(model_id)] = model_id
    return mapping


def load_model_types(models_file):
    """Map model ID -> 'T' (text-only) or 'V' (vision-capable).

    Determined by comment-guarded sections in the TOML: every model listed
    after a ``# text-only`` comment (and before the end of that list or a
    ``# vision-capable``/``# baseline``/``# disabled`` comment) is text-only;
    everything else is vision-capable.  Proprietary models always come back
    as vision-capable because that block has no ``# text-only`` guard.
    """
    path = Path(models_file)
    if not path.exists():
        return {}
    try:
        raw = path.read_text()
    except OSError:
        return {}
    data = tomllib.loads(raw)

    # Parse comment-guarded regions from the raw text so we know which models
    # sit below a ``# text-only`` line.
    text_only_ids = _parse_text_only_ids(raw)

    mapping = {}
    for entries in (data.get("models") or {}).values():
        for entry in entries:
            model_id = entry.get("id") if isinstance(entry, dict) else entry
            if not isinstance(model_id, str):
                continue
            mapping[model_id] = "T" if model_id in text_only_ids else "V"
    return mapping


def _parse_text_only_ids(raw_toml):
    """Return the set of model IDs appearing after a ``# text-only`` comment
    within the same TOML array-of-tables / array block."""
    import re
    text_only_ids = set()
    in_text_only = False
    for line in raw_toml.splitlines():
        stripped = line.strip()
        # A ``# text-only`` comment on its own line starts the region.
        if stripped.startswith("#") and "text-only" in stripped.lower():
            in_text_only = True
            continue
        # A ``# vision`` or ``# baseline`` or ``# disabled`` guard ends it.
        if stripped.startswith("#") and any(
            kw in stripped.lower()
            for kw in ("vision-capable", "vision", "baseline", "disabled")
        ):
            in_text_only = False
            continue
        # A new TOML key = [ block (non-comment, non-indented) resets the
        # region — we left the previous list.
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*\s*=\s*\[', stripped):
            in_text_only = False
            continue
        if not in_text_only:
            continue
        # Match ``"vendor/model"`` or ``{ id = "vendor/model"`` on a line.
        m = re.search(r'(?:")([^"]+)(?:")(?!.*"id")', line)
        if not m:
            m = re.search(r'id\s*=\s*"([^"]+)"', line)
        if m:
            text_only_ids.add(m.group(1))
    return text_only_ids


def load_costs(models_file):
    """Map model ID -> (input_price_per_1M, output_price_per_1M) from ``[costs]``.

    Returns (0, 0) for models with no entry so callers don't have to guard.
    """
    path = Path(models_file)
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError):
        return {}
    raw = data.get("costs") or {}
    return {
        model_id: (
            float(tuple_val[0]) if tuple_val[0] is not None else 0.0,
            float(tuple_val[1]) if tuple_val[1] is not None else 0.0,
        )
        for model_id, tuple_val in raw.items()
        if isinstance(tuple_val, (list, tuple)) and len(tuple_val) >= 2
    }


def load_disabled_ids(models_file):
    """Return the set of model IDs listed under ``[models] disabled``."""
    path = Path(models_file)
    if not path.exists():
        return set()
    try:
        data = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError):
        return set()
    disabled = set()
    for entry in (data.get("models") or {}).get("disabled") or []:
        model_id = entry.get("id") if isinstance(entry, dict) else entry
        if isinstance(model_id, str):
            disabled.add(model_id)
    return disabled


def puzzle_image_fingerprint(puzzle_images_dir):
    """Content hash of the rendered puzzle set, or None if it is missing.

    Must match `_puzzle_set_fingerprint` in sudoku_agent_harness/agent.py:
    that is the value stored in each state file, and comparing them is how a
    state file left over from an earlier puzzle set is detected.
    """
    d = Path(puzzle_images_dir)
    if not d.is_dir():
        return None
    paths = sorted(d.glob("*.png"), key=lambda p: p.name)
    if not paths:
        return None
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def load_harness_state(model_dir):
    """Read one model's `.harness_state.json`, or None if absent/unusable."""
    path = Path(model_dir) / HARNESS_STATE_FILENAME
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(state, dict) or not isinstance(state.get("rounds"), list):
        return None
    return state


def harness_claims(state, fingerprint=None):
    """What the harness reported for a model — its claims, not verified facts.

    `stale` is True when the state was written against a different puzzle set
    than the one currently in the results dir, which makes its rounds refer
    to puzzles nobody can compare against.
    """
    rounds = state.get("rounds") or []
    # `input_set` is the harness's task-agnostic name; `puzzle_set` is what
    # state files written before that rename carry.
    stored = state.get("input_set", state.get("puzzle_set"))
    return {
        "model_id": state.get("model"),
        "puzzle_set": stored,
        "stale": bool(fingerprint and stored and stored != fingerprint),
        "n_rounds": len(rounds),
        # Self-reported: the agent finished and wrote an image. NOT correctness.
        "n_success": sum(1 for r in rounds if r.get("success")),
        "n_resumed": sum(1 for r in rounds if r.get("resumed")),
        "n_errors": sum(1 for r in rounds if r.get("error")),
    }


def records_from_state(state, model_dir, difficulty_by_id=None):
    """Turn state-file rounds into records shaped like the graded ones.

    Deliberately produces no `verdict`: nothing in the state file has been
    checked, so these records carry timing, tokens and the agent's claim, and
    score zero correctness until a real grading pass runs.
    """
    model_dir = Path(model_dir)
    difficulty_by_id = difficulty_by_id or {}
    records = []
    for rec in state.get("rounds") or []:
        # item_* are the harness's generic keys; puzzle_* are the pre-rename
        # ones, still present in state files from earlier runs.
        pid = rec.get("item_id", rec.get("puzzle_id"))
        name = rec.get("item_name") or rec.get("puzzle_name") or (
            f"puzzle_{pid:03d}.png" if isinstance(pid, int) else None
        )
        record = {
            "puzzle_id": pid,
            "difficulty": difficulty_by_id.get(pid),
            # Checked on disk, like the grading path does, rather than trusting
            # the harness's claim that it wrote the file.
            "output_image": bool(name) and (model_dir / name).exists(),
            "elapsed": rec.get("elapsed"),
            "round": rec.get("round"),
            "input_tokens": rec.get("input_tokens"),
            "output_tokens": rec.get("output_tokens"),
            "total_tokens": rec.get("total_tokens"),
            "harness_final_answer": rec.get("final_answer"),
            "middleware": rec.get("middleware"),
        }
        if rec.get("error"):
            record["error"] = f"harness failure: {rec['error']}"
        elif not rec.get("success"):
            record["error"] = "harness reported failure"
        else:
            record["error"] = "not graded: no benchmarker record for this round"
        records.append(record)
    records.sort(key=lambda r: (r["puzzle_id"] is None, r["puzzle_id"]))
    return records


def _error_kind(message):
    """Coarse bucket for an error string: the text before its first colon."""
    head = str(message).splitlines()[0]
    return head.split(":")[0].strip()[:60] or "unknown"


def _error_breakdown(records):
    errors = {}
    for r in records:
        if "error" in r:
            kind = _error_kind(r["error"])
            errors.setdefault(kind, {"n": 0, "sample": str(r["error"])[:200]})
            errors[kind]["n"] += 1
    return errors


def collect(results_dir, models_file=None, verify_images=True, solutions_dir=None,
            trials_dir=DEFAULT_TRIALS_DIR):
    """Read a results dir into a report dict. Nothing here calls the network.

    With `verify_images`, every solution PNG on disk is transcribed locally and
    checked against its puzzle — the only correctness signal that does not
    depend on the grading API having worked or on the agent's own word.

    ``solutions_dir`` overrides the default ``<results_dir>/solutions``, and
    is how the caller iterates over multiple harnesses (``solutions-dsh``,
    ``solutions-smolagents``, …).

    ``trials_dir`` is read first and supplies each model's row: it is the
    permanent record of the best result for this puzzle set, while `results/`
    is scratch that any later run overwrites. A results dir that BEATS the
    record keeps its own numbers and is marked, that being a new best rather
    than a stale row. Pass None to report strictly what is on disk.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"no such results directory: {results_dir}")

    id_map = load_model_ids(models_file) if models_file else {}
    if solutions_dir is None:
        solutions_dir = results_dir / "solutions"
    else:
        solutions_dir = Path(solutions_dir)
    fingerprint = puzzle_image_fingerprint(results_dir / "puzzle_images")

    # Last-resort fallback for rows whose own records and state say nothing
    # (`_row_middleware` is what normally decides): the pre-`middleware`-field
    # layout, where a transcribed run left puzzle_NNN.txt beside the rendered
    # inputs. Deliberately NOT inferred from this run's transcriptions/ dir:
    # `results/` is scratch shared by every run, so a middleware run's
    # artifacts say nothing about a row whose records predate it.
    puzzle_images_dir = results_dir / "puzzle_images"
    was_middleware = (
        puzzle_images_dir.is_dir()
        and any(puzzle_images_dir.glob("puzzle_*.txt"))
    )

    puzzles = []
    puzzles_path = results_dir / "puzzles.json"
    if puzzles_path.exists():
        try:
            puzzles = json.loads(puzzles_path.read_text())
        except json.JSONDecodeError:
            puzzles = []
    difficulty_by_id = {p.get("id"): p.get("difficulty") for p in puzzles}
    puzzle_by_id = {p.get("id"): p for p in puzzles}

    # Local verification needs the clues to check against, so it is only
    # possible when puzzles.json is present alongside the rendered inputs.
    reader = None
    image_check = None
    if verify_images and puzzles:
        reader = Reader.calibrate(results_dir / "puzzle_images", puzzles)
        image_check = self_check(reader, results_dir / "puzzle_images", puzzles)
        if not image_check["all_ok"]:
            # The reader could not even reproduce the inputs it was calibrated
            # on, so its verdicts on the solutions are not evidence.
            reader = None

    # Harness state, keyed by the solutions subdir it was found in.
    states = {}
    if solutions_dir.is_dir():
        for d in sorted(p for p in solutions_dir.iterdir() if p.is_dir()):
            state = load_harness_state(d)
            if state is not None:
                states[d.name] = state

    summaries = []
    unreadable = []
    for path in model_record_paths(results_dir):
        try:
            records = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            unreadable.append((path.name, str(e)))
            continue
        if not isinstance(records, list):
            unreadable.append((path.name, "not a list of records"))
            continue
        summary = _model_summary(
            path.stem, records, states.get(path.stem), solutions_dir,
            fingerprint, graded=True, reader=reader, puzzle_by_id=puzzle_by_id,
        )
        # Which file this row was read from, so the report can say whether the
        # run predates the move into final/.
        summary["record_path"] = str(path)
        summaries.append(summary)

    # A run that dies mid-model leaves a state file and no JSON. Those rounds
    # are real work and belong in the report — clearly marked ungraded, since
    # only the harness's own word says anything about them.
    graded = {s["safe_name"] for s in summaries}
    for safe, state in states.items():
        if safe in graded:
            continue
        stale = harness_claims(state, fingerprint)["stale"]
        records = records_from_state(
            state, solutions_dir / safe, {} if stale else difficulty_by_id
        )
        summaries.append(
            _model_summary(
                safe, records, state, solutions_dir, fingerprint, graded=False,
                reader=reader, puzzle_by_id=puzzle_by_id,
            )
        )

    # Left over: a solutions dir with neither records nor a readable state file.
    accounted = {s["safe_name"] for s in summaries}
    orphans = []
    if solutions_dir.is_dir():
        for d in sorted(p for p in solutions_dir.iterdir() if p.is_dir()):
            if d.name not in accounted:
                orphans.append(
                    {
                        "safe_name": d.name,
                        "model": id_map.get(d.name, d.name),
                        "pngs_on_disk": len(list(d.glob("*.png"))),
                    }
                )

    # Real model IDs before anything keyed by them: a state file stores the ID
    # verbatim, so it beats un-mangling the filename via the manifest. The
    # trial lookup below needs both the ID and the middleware flag.
    model_types = load_model_types(models_file) if models_file else {}
    model_costs = load_costs(models_file) if models_file else {}
    for s in summaries:
        state_id = (s.get("harness") or {}).get("model_id")
        s["model"] = state_id or id_map.get(s["safe_name"], s["safe_name"])
        s["type"] = model_types.get(s["model"], "V")
        in_price, out_price = model_costs.get(s["model"], (0.0, 0.0))
        s["cost_per_1M_input"] = in_price
        s["cost_per_1M_output"] = out_price
        s.setdefault("middleware", was_middleware)

    # The trial is the source a row shows by default: `results/` is scratch
    # that every run overwrites, `trials/` is the permanent best-so-far for
    # this exact puzzle set. Keyed by (model, middleware), because a
    # middleware run and a plain run used different prompts and are separate
    # observations. A results dir that BEATS the record keeps its own numbers
    # and is marked — that is a new best, not a stale row.
    trial_entries, trial_meta = load_trial_entries(trials_dir, puzzles)
    for s in summaries:
        s["verified_accuracy"], s["verification_source"] = _verified_accuracy(s)
        found = trial_entries.get(s["model"])
        if found is None:
            continue
        if _disk_beats_trial(s, found):
            _note_beaten_trial(s, found)
        else:
            _apply_trial(s, found, same_run=_same_run_as_trial(s, found))

    # Ranked on whatever verification actually happened, best source first;
    # a row nothing verified has no score to rank on and sinks to the bottom.
    # `_rank_key` puts accuracy ahead of time unconditionally, so a faster row
    # never heads the table over a more accurate one.
    for s in summaries:
        s["verified_accuracy"], s["verification_source"] = _verified_accuracy(s)
    summaries.sort(key=_report_rank_key)

    # Drop models the manifest marks as disabled — they are noise.
    disabled_ids = load_disabled_ids(models_file) if models_file else set()
    if disabled_ids:
        summaries = [s for s in summaries if s["model"] not in disabled_ids]
        orphans = [o for o in orphans if o["model"] not in disabled_ids]

    # Counted here, not while merging: the rows just dropped were considered
    # too, and a tally that includes them describes no table anyone sees.
    trial_meta["n_from_trial"] = sum(
        1 for s in summaries if (s.get("trial") or {}).get("used")
    )
    trial_meta["n_beaten"] = sum(1 for s in summaries if _beat_trial(s))
    # Rows whose record IS this run: the numbers agree by construction, and
    # nothing "beat" anything.
    trial_meta["n_same_run"] = sum(
        1 for s in summaries if (s.get("trial") or {}).get("same_run")
    )
    # Models the record has never seen: reported from the results dir, and
    # unmarked, since there is nothing for them to have beaten.
    trial_meta["n_unrecorded"] = sum(1 for s in summaries if not s.get("trial"))

    report = {
        "results_dir": str(results_dir),
        "solutions_dir": str(solutions_dir),
        "models_file": str(models_file) if models_file else None,
        "harness_id": solutions_dir.name.removeprefix("solutions-"),
        "n_puzzles": len(puzzles),
        "difficulty_mix": _difficulty_mix(puzzles),
        "puzzle_fingerprint": fingerprint,
        # Which trial files were consulted, and how many rows they supplied.
        "trial": trial_meta,
        # The transcriber's report card on the input images, or None when the
        # local pass was skipped or impossible.
        "image_check": image_check,
        "summaries": summaries,
        "ungraded_models": orphans,
        "unreadable_files": unreadable,
        "totals": _totals(summaries),
    }
    # First key, so the JSON form opens with the same provenance line the text
    # and markdown forms print.
    return {"source": format_source(report, label=False), **report}


def _model_summary(
    safe, records, state, solutions_dir, fingerprint, *, graded,
    reader=None, puzzle_by_id=None,
):
    """One row of the report: what the images show, what was graded, what was
    claimed."""
    summary = summarize(safe, records)
    summary["safe_name"] = safe
    # False means nothing here was verified: the numbers come from the
    # harness's own state file and correctness is unmeasured, not zero.
    summary["graded"] = graded
    summary["source"] = "benchmarker" if graded else "harness-state"
    model_solutions = solutions_dir / safe
    # What is on disk now, versus what was on disk when grading ran. A
    # mismatch means the run was interrupted or the dir was touched since.
    summary["pngs_on_disk"] = (
        len(list(model_solutions.glob("*.png"))) if model_solutions.is_dir() else 0
    )
    summary["errors"] = _error_breakdown(records)
    # How many records the grader actually returned a verdict for. Without
    # this, "0% accuracy" and "never graded" look identical.
    summary["n_verdicts"] = sum(1 for r in records if r.get("verdict"))
    # Keyed by puzzle for comparison against the local read of the same image.
    summary["grader_verdicts"] = {
        r["puzzle_id"]: bool(r["verdict"].get("correct"))
        for r in records
        if r.get("verdict") and r.get("puzzle_id") is not None
    }
    summary["harness"] = harness_claims(state, fingerprint) if state else {}
    # The state file may be mid-rewrite by a running benchmark: only let it
    # speak for these records when it is for this puzzle set and covers at
    # least as many rounds as the row has.
    state_rounds = (state or {}).get("rounds") or []
    state_covers = bool(state) and not summary["harness"].get("stale") and (
        len(state_rounds) >= len(records)
    )
    mw, n_with, n_known = _row_middleware(
        records, state, state_covers_records=state_covers
    )
    if mw is not None:
        summary["middleware"] = mw
        # A session that mixed the two is flagged rather than averaged: the
        # rounds are not comparable, so the reader has to know.
        summary["middleware_rounds"] = [n_with, n_known]
    summary["local"] = (
        _local_verification(reader, model_solutions, puzzle_by_id or {})
        if reader is not None else {}
    )
    return summary


def _row_middleware(records, state, *, state_covers_records=False):
    """Whether this model's rounds ran with a transcription, per round.

    The flag is a property of a round, not of a results dir: a resumed
    session can replay rounds that ran the other way, and the two are not
    comparable. Read from the graded records first (`middleware`, or a
    `transcription_elapsed` charge, which only a middleware run has) — those
    are the rounds the row's numbers come from, so they are the only
    first-hand answer.

    The harness state file is second-hand and used only when the records say
    nothing, and only when it plausibly describes the same session
    (`state_covers_records`). It is rewritten by every run, so a run in
    progress would otherwise label a row whose numbers came from an earlier,
    differently-configured one.

    Returns (flag, n_with, n_known): `flag` is None when nothing trustworthy
    says either way, leaving the fallback in `collect` to decide.
    """
    sources = [records]
    if state_covers_records:
        sources.append((state or {}).get("rounds") or [])
    for rounds in sources:
        known = [
            r for r in rounds
            if r.get("middleware") is not None
            or r.get("transcription_elapsed") is not None
        ]
        if not known:
            continue
        n_with = sum(
            1 for r in known
            if r.get("middleware") or r.get("transcription_elapsed") is not None
        )
        return bool(n_with), n_with, len(known)
    return None, 0, 0


def _local_verification(reader, model_dir, puzzle_by_id):
    """Transcribe and verify every solution image this model left on disk.

    The denominator is the whole puzzle set: an image that was never produced
    is not a correct answer, it is a missing one.
    """
    per_puzzle = {}
    for pid, puzzle in sorted(puzzle_by_id.items()):
        if pid is None or not puzzle.get("puzzle"):
            continue
        path = Path(model_dir) / f"puzzle_{pid:03d}.png"
        if not path.exists():
            continue
        per_puzzle[pid] = grade_image(reader, puzzle, path)

    n_in_set = sum(1 for pid, p in puzzle_by_id.items()
                   if pid is not None and p.get("puzzle"))
    n_correct = sum(1 for v in per_puzzle.values() if v.get("correct"))

    # Per tier over the whole set, so a tier whose image is missing still
    # shows as attempted-and-not-solved rather than vanishing.
    per_diff = {}
    for pid, puzzle in puzzle_by_id.items():
        if pid is None or not puzzle.get("puzzle"):
            continue
        d = puzzle.get("difficulty")
        bucket = per_diff.setdefault(d, {"n": 0, "correct": 0})
        bucket["n"] += 1
        if per_puzzle.get(pid, {}).get("correct"):
            bucket["correct"] += 1

    error_types = {}
    for v in per_puzzle.values():
        if not v.get("correct"):
            et = v.get("error_type", "?")
            error_types[et] = error_types.get(et, 0) + 1
    return {
        "n_images_read": len(per_puzzle),
        "n_in_set": n_in_set,
        "n_correct": n_correct,
        "accuracy": n_correct / n_in_set if n_in_set else 0.0,
        "per_difficulty": per_diff,
        "error_types": error_types,
        # Images whose transcription had at least one shaky cell. A correct
        # verdict here is still trustworthy (a misread grid essentially never
        # satisfies every constraint); an incorrect one may be the reader's
        # fault and is worth a human look.
        "uncertain": {
            pid: v["uncertain_cells"]
            for pid, v in per_puzzle.items() if v.get("uncertain_cells")
        },
        "per_puzzle": per_puzzle,
    }


# What a trial entry supplies when it supersedes a row. Everything else on the
# row (model id, type, prices, what is on disk now) still describes this
# results dir.
TRIAL_METRIC_KEYS = (
    "n_puzzles", "n_correct", "n_errors", "n_output_images", "output_rate",
    "accuracy", "avg_elapsed_s", "input_tokens", "output_tokens",
    "total_tokens", "tokens_per_correct", "per_difficulty", "learning",
    "session",
)


def model_record_paths(results_dir):
    """Every graded per-model JSON in a results dir, newest layout first.

    Records live in `final/`. Runs made before that wrote them beside the
    run's shared artifacts at the top level, so those are read too — and a
    model present in both is taken from `final/`, which is where the current
    benchmarker writes. Returned sorted by model name, so report order does
    not depend on which layout a run used.
    """
    results_dir = Path(results_dir)
    by_stem = {}
    for path in sorted((results_dir / FINAL_SUBDIR).glob("*.json")):
        by_stem[path.stem] = path
    for path in sorted(results_dir.glob("*.json")):
        if path.name in NON_MODEL_FILES:
            continue
        by_stem.setdefault(path.stem, path)
    return [by_stem[k] for k in sorted(by_stem)]


def load_trial_entries(trials_dir, puzzles):
    """Each model's best recorded result for this exact puzzle set, by model.

    `results/` is scratch: a re-run that crashed, or one stopped after two
    puzzles, overwrites what a complete run left there. `trials/` is the
    durable side — one file per puzzle set, each model's best result — so a
    row that is worse on disk than in the trial is stale, not news.

    Matched on the puzzles' content hash, never on the seed alone: a
    generator change can produce different puzzles for the same seed, and
    those numbers must never be mixed in. A trial that does not describe
    exactly these puzzles is ignored.

    One row per model, not one per (model, middleware): the record keeps a
    middleware run and a plain run as separate observations, but the report
    ranks models, so each model is represented by whichever of its entries is
    best on the benchmark's own order — score, then time, then cost. The
    chosen entry's `middleware` flag rides along, so the `mw` column says
    which kind of run produced that model's best result. The entries that lost
    are kept under `others`, so nothing is hidden by the choice.

    Returns (entries, meta), where meta records which files were consulted so
    the report can say so.
    """
    meta = {
        "dir": str(trials_dir) if trials_dir else None,
        "files": [],
        # Rows shown from the record, and rows this results dir beat it on.
        "n_from_trial": 0,
        "n_beaten": 0,
    }
    if not trials_dir or not puzzles:
        return {}, meta
    d = Path(trials_dir)
    if not d.is_dir():
        return {}, meta
    try:
        want = trial_puzzle_fingerprint(puzzles)
    except (KeyError, TypeError):
        # puzzles.json without the grids cannot be fingerprinted; without it
        # there is no safe way to tell which trial describes these puzzles.
        return {}, meta

    entries = {}
    for path in sorted(d.glob("*.json")):
        trial = load_trial(path)
        if not isinstance(trial, dict):
            continue
        if trial.get("puzzle_fingerprint") != want:
            continue
        meta["files"].append(str(path))
        for entry in trial.get("entries") or []:
            if not isinstance(entry, dict) or not entry.get("model"):
                continue
            key = entry["model"]
            found = {
                "entry": entry,
                "trial_id": trial.get("trial_id", path.stem),
                "path": str(path),
                "others": [],
            }
            prior = entries.get(key)
            if prior is None:
                entries[key] = found
            elif is_better(entry, prior["entry"]):
                found["others"] = prior["others"] + [prior["entry"]]
                entries[key] = found
            else:
                prior["others"].append(entry)
    return entries, meta


def _same_run_as_trial(summary, found):
    """True when the record was written by the run sitting in the results dir.

    Time and tokens are the run's fingerprint: two sessions never agree on
    `avg_elapsed_s` to the last float digit, and the record stores exactly
    what these records produced. When they match, the record and the row are
    one result, so "which is better" is not a question — and the images on
    disk are the record's own images, which makes the local read the best
    verification available OF THE RECORD, not a foreign one.
    """
    entry = found["entry"]
    disk_avg = summary.get("avg_elapsed_s")
    entry_avg = entry.get("avg_elapsed_s")
    if disk_avg is None or entry_avg is None:
        return False
    return (
        abs(disk_avg - entry_avg) <= 1e-9 * max(abs(disk_avg), abs(entry_avg), 1.0)
        and summary.get("total_tokens") == entry.get("total_tokens")
        and summary.get("n_puzzles") == entry.get("n_puzzles")
    )


def _apply_trial(summary, found, *, same_run=False):
    """Report this model's recorded best instead of what is on disk.

    The trial is the default source: `results/` is scratch that every run
    overwrites, while the trial is the permanent record of each model's best
    result on this exact puzzle set, updated as each model finishes. What the
    swap displaced is kept under `trial.replaced`, so the row still says where
    its numbers came from.

    The local read and the grader verdicts are normally dropped: both describe
    the images sitting in the solutions dir right now, which belong to a
    different run, and pairing them with the record's score would invent a
    result neither produced. When the record came from THIS run
    (`same_run`) that reasoning inverts — the images are the record's own — so
    the local verification is kept and the row is scored on it. This is how a
    record whose accuracy was limited by a failed grader call gets read
    correctly without the row pretending to be a new result.
    """
    entry = found["entry"]
    summary["trial"] = {
        "trial_id": found["trial_id"],
        "path": found["path"],
        "recorded_at": entry.get("recorded_at"),
        "runs": entry.get("runs"),
        "used": True,
        "same_run": same_run,
        "middleware": entry.get("middleware"),
        # The model's other recorded entries — normally the same model's run
        # with the middleware the other way round — ranked below this one.
        "others": [
            {
                "middleware": o.get("middleware"),
                "accuracy": o.get("accuracy"),
                "avg_elapsed_s": o.get("avg_elapsed_s"),
                "total_tokens": o.get("total_tokens"),
                "recorded_at": o.get("recorded_at"),
            }
            for o in found.get("others") or []
        ],
        "replaced": {
            "accuracy": summary.get("verified_accuracy"),
            "verification_source": summary.get("verification_source"),
            "n_correct": summary.get("n_correct"),
            "n_puzzles": summary.get("n_puzzles"),
            "avg_elapsed_s": summary.get("avg_elapsed_s"),
        },
    }
    for key in TRIAL_METRIC_KEYS:
        if key in entry:
            summary[key] = entry[key]
    # The numbers now come from the entry, so `mw` must describe the entry's
    # run rather than whatever is sitting in the results dir.
    if entry.get("middleware") is not None:
        summary["middleware"] = bool(entry["middleware"])
        summary.pop("middleware_rounds", None)
    if not same_run:
        summary["local"] = {}
        summary["grader_verdicts"] = {}
        summary["n_verdicts"] = 0
    summary["source"] = "trial"


def _beat_trial(summary):
    """True when this row is the results dir's own, having beaten the record.

    The row carries the entry it beat but did not take its numbers, which is
    exactly what the `*` marks.
    """
    trial = summary.get("trial") or {}
    return bool(trial) and not trial.get("used")


def _disk_beats_trial(summary, found):
    """True when this results dir holds a better run than the record does.

    The record is what a row shows by default, so this is the exception worth
    marking: a result better than anything recorded for that model on this
    puzzle set — on the benchmark's own ranking, correctness then speed then
    cost. Normally the two agree, since each model is recorded as it finishes;
    they diverge while a run is still going, or when a run was made against a
    trials dir other than this one.

    Compared on the accuracy the RECORD is written in — the run's own grader
    verdicts (`accuracy`), not `verified_accuracy` from the report's fresh
    local pass. The two are different measurements, and comparing across them
    let a results dir displace a better record purely because the report had
    verified its images more thoroughly than the recording run managed to.
    A record written by this very run never loses to it either; see
    `_same_run_as_trial`.

    A row nothing verified never wins: unverified numbers are not a result.
    """
    if summary.get("verification_source") == "none":
        return False
    if _same_run_as_trial(summary, found):
        return False
    disk = {
        "accuracy": summary.get("accuracy") or 0.0,
        "avg_elapsed_s": summary.get("avg_elapsed_s"),
        "total_tokens": summary.get("total_tokens"),
    }
    return is_better(disk, found["entry"])


def _note_beaten_trial(summary, found):
    """Record that this run beat the entry, without touching the row's numbers.

    The row keeps what the results dir measured — that is the better result —
    and carries the record it beat, so the improvement can be read off the
    report rather than inferred.
    """
    entry = found["entry"]
    summary["trial"] = {
        "trial_id": found["trial_id"],
        "path": found["path"],
        "used": False,
        "beaten": {
            "accuracy": entry.get("accuracy"),
            "avg_elapsed_s": entry.get("avg_elapsed_s"),
            "total_tokens": entry.get("total_tokens"),
            "recorded_at": entry.get("recorded_at"),
            "rev": entry.get("rev"),
        },
    }


def _verified_accuracy(summary):
    """(accuracy, source) from the strongest verification available.

    Local transcription wins when it read anything: it checked the actual
    pixels, and unlike the grader it cannot be missing because an API call
    failed. The grader's verdicts are used when there was no local pass. A
    harness self-report is never a source of correctness.
    """
    # A recorded trial that displaced this row was itself graded, on this
    # exact puzzle set, by a run that finished. Unless that record came from
    # this very run — then the images on disk are its own, and the local read
    # below is a stronger check of the same result than the grader verdicts
    # the record was written from.
    trial = summary.get("trial") or {}
    if trial.get("used") and not trial.get("same_run"):
        return summary["accuracy"], "trial-record"
    local = summary.get("local") or {}
    # A model that produced no image at all is verifiably wrong, not unknown:
    # the puzzle set is known and nothing was answered. So the local pass
    # having *run* is what counts, not whether it found anything to read.
    if local:
        return local["accuracy"], "local-transcription"
    # Only a record that actually carries a verdict was graded; a record whose
    # grader call 404'd proves nothing either way.
    if summary.get("n_verdicts"):
        return summary["accuracy"], "grader"
    return 0.0, "none"


def _difficulty_mix(puzzles):
    mix = {'easy': 0, 'medium': 0, 'hard': 0}
    for p in puzzles:
        mix[p['difficulty']] += 1
    return mix


def _totals(summaries):
    return {
        "models": len(summaries),
        "graded_models": sum(1 for s in summaries if s["graded"]),
        "records": sum(s["n_puzzles"] for s in summaries),
        "correct": sum(s["n_correct"] for s in summaries),
        "errors": sum(s["n_errors"] for s in summaries),
        "pngs_on_disk": sum(s["pngs_on_disk"] for s in summaries),
        "harness_claimed_success": sum(
            (s.get("harness") or {}).get("n_success", 0) for s in summaries
        ),
        "locally_verified_correct": sum(
            (s.get("local") or {}).get("n_correct", 0) for s in summaries
        ),
        "images_read": sum(
            (s.get("local") or {}).get("n_images_read", 0) for s in summaries
        ),
        "input_tokens": sum(s["input_tokens"] for s in summaries),
        "output_tokens": sum(s["output_tokens"] for s in summaries),
        "total_tokens": sum(s["total_tokens"] for s in summaries),
    }


def _pct(x):
    return f"{x * 100:5.1f}%"


def _claimed(summary):
    """`success` count the harness self-reported, as `n/rounds`, or `-`.

    A trailing `!` marks a state file written against a different puzzle set,
    so the count refers to puzzles this results dir no longer holds.
    """
    h = summary.get("harness") or {}
    if not h:
        return "-"
    mark = "!" if h.get("stale") else ""
    return f"{h.get('n_success', 0)}/{h.get('n_rounds', 0)}{mark}"


def _per_difficulty(summary):
    """Per-tier counts from whichever verification the row is ranked on."""
    local = summary.get("local") or {}
    if local:
        return local.get("per_difficulty") or {}
    return summary["per_difficulty"]


def _local_cell(summary):
    """Locally verified correct out of the whole puzzle set, or `-`."""
    local = summary.get("local") or {}
    if not local:
        return "-"
    return f"{local['n_correct']}/{local['n_in_set']}"


def _grader_cell(summary):
    """Grader-verified correct out of its records, or `n/a` if none came back."""
    if not summary.get("n_verdicts"):
        return "n/a"
    return f"{summary['n_correct']}/{summary['n_puzzles']}"


def _verified_counts(summary):
    """(correct, total) from the source this row is ranked on, or None.

    Follows `_verified_accuracy` exactly — trial record, then local
    transcription, then grader — so the score a reader sees is the score the
    ranking used. They diverged before this existed: a partial grader tally
    (an image whose grader call failed counts as neither right nor wrong) was
    displayed as e.g. `2/3` while the local pixel pass had verified 3/3 and
    ranked the row accordingly, putting an apparent 2/3 above real 3/3 rows.
    """
    trial = summary.get("trial") or {}
    if trial.get("used") and not trial.get("same_run"):
        return summary["n_correct"], summary["n_puzzles"]
    local = summary.get("local") or {}
    if local:
        return local["n_correct"], local["n_in_set"]
    if summary.get("n_verdicts"):
        return summary["n_correct"], summary["n_puzzles"]
    return None


def _score_cell(summary):
    """Correct-out-of-total from the verification this row is ranked on.

    A simple ``3/3`` is all a reader needs; ``n/a`` is noise. Which source
    that is comes from `_verified_counts`, so the column and the table's
    order can never tell different stories.

    `*` marks a row this results dir won outright — a result better than
    anything recorded for that model on this puzzle set. Rows shown from the
    record are unmarked, that being the default source.
    """
    counts = _verified_counts(summary)
    if counts is None:
        return "-"
    star = "" if (summary.get("trial") or {}).get("used") else (
        "*" if _beat_trial(summary) else ""
    )
    return f"{counts[0]}/{counts[1]}{star}"


def _is_perfect(summary):
    """True when every puzzle was answered correctly by the best available source."""
    counts = _verified_counts(summary)
    return counts is not None and counts[0] == counts[1] > 0


def _total_cost(summary):
    """Total API spend in dollars, or None when the model has no pricing.

    Cost = (input_tokens × in_price + output_tokens × out_price) / 1e6.
    """
    in_price = summary.get("cost_per_1M_input", 0)
    out_price = summary.get("cost_per_1M_output", 0)
    if in_price == 0 and out_price == 0:
        return None
    return (
        summary["input_tokens"] * in_price
        + summary["output_tokens"] * out_price
    ) / 1_000_000


def _report_rank_key(summary):
    """Report order: score, then time, then cost, then tokens.

    The score is the verified one (`verified_accuracy`, the same figure
    `_score_cell` prints), and it dominates outright — a faster or cheaper row
    never climbs above a more accurate one. Accuracy and time come from
    `trials._rank_key`, the project's one ranking rule, so the report and the
    permanent record cannot drift apart; this only refines its cost step,
    spending dollars where pricing is known instead of tokens as a proxy.

    A row nothing verified has no score to stand on and sorts last whatever
    its time.
    """
    base = _rank_key({**summary, "accuracy": summary["verified_accuracy"]})
    cost = _total_cost(summary)
    return (
        summary["verification_source"] == "none",
        base[0],                                  # -accuracy
        base[1],                                  # avg_elapsed_s, unknown last
        cost if cost is not None else float("inf"),
        base[2],                                  # total_tokens
    )


def _cost_cell(summary):
    """Total API cost in dollars, only for models that earned a perfect score.

    Cost = (input_tokens × in_price + output_tokens × out_price) / 1e6.
    Returns ``-`` for every model that is not perfect, so the column doubles
    as a quick "should I bother?" signal.
    """
    if not _is_perfect(summary):
        return "-"
    cost = _total_cost(summary)
    if cost is None:
        return "-"
    return f"${cost:.2f}"


def _cost_per_hour_cell(summary):
    """Dollars per wall-clock hour: how much the model burns per unit of time.

    total_cost = (tok_in × in_price + tok_out × out_price) / 1e6
    total_sec  = avg_elapsed_s × n_puzzles
    $/h        = total_cost × 3600 / total_sec

    Shown for every model that has both pricing and timing data, regardless
    of score — it's an efficiency metric, not a correctness one.
    """
    in_price = summary.get("cost_per_1M_input", 0)
    out_price = summary.get("cost_per_1M_output", 0)
    if in_price == 0 and out_price == 0:
        return "-"
    if summary["n_puzzles"] == 0:
        return "-"
    total_sec = summary["avg_elapsed_s"] * summary["n_puzzles"]
    if total_sec <= 0:
        return "-"
    total_cost = (
        summary["input_tokens"] * in_price
        + summary["output_tokens"] * out_price
    ) / 1_000_000
    cph = total_cost * 3600 / total_sec
    return f"${cph:.2f}"


def unverified_claims(report):
    """Models the harness called successful that no grading pass confirmed."""
    out = []
    for s in report["summaries"]:
        claimed = (s.get("harness") or {}).get("n_success", 0)
        local = s.get("local") or {}
        verified = local.get("n_correct") if local else s["n_correct"]
        if claimed > verified:
            out.append((s, claimed, verified))
    return out


def format_source(report, *, markdown=False, label=True):
    """One line naming every file the report was built from.

    Printed first by all three formats, because a table that outlives its
    context — pasted into an issue, a commit message, a chat — is only
    interpretable if it says which run it describes — down to which rows came
    from the results dir and which from the permanent trial record. Nothing
    here calls the network.
    """
    t = report["totals"]
    n_state = sum(1 for s in report["summaries"] if s.get("harness"))
    n_legacy = sum(
        1 for s in report["summaries"]
        if s.get("record_path") and Path(s["record_path"]).parent.name != FINAL_SUBDIR
    )
    graded = f"{report['results_dir']}/{FINAL_SUBDIR}/*.json ({t['graded_models']} graded"
    graded += f", {n_legacy} from the pre-final/ layout)" if n_legacy else ")"
    bits = [
        graded,
        f"{report['solutions_dir']}/*/.harness_state.json ({n_state})",
    ]
    if report["n_puzzles"]:
        bits.append(
            f"{report['results_dir']}/puzzles.json ({report['n_puzzles']} puzzles)"
        )
    if report.get("models_file"):
        bits.append(f"{report['models_file']} (model IDs, types, prices)")

    trial = report.get("trial") or {}
    if trial.get("files"):
        detail = f"{trial.get('n_from_trial', 0)} row(s)"
        beaten = trial.get("n_beaten", 0)
        if beaten:
            detail += f", {beaten} beaten by results/"
        same_run = trial.get("n_same_run", 0)
        if same_run:
            detail += f", {same_run} recorded from this very run"
        unrecorded = trial.get("n_unrecorded", 0)
        if unrecorded:
            detail += f", {unrecorded} not in the record"
        bits.append(", ".join(trial["files"]) + f" (recorded best; {detail})")
    elif trial.get("dir"):
        bits.append(f"no trial in {trial['dir']}/ matches these puzzles")
    else:
        bits.append("trials not read")

    check = report.get("image_check")
    if check is None:
        note = "solution PNGs not re-read"
    elif check["all_ok"]:
        note = "solution PNGs re-read locally"
    else:
        note = "local PNG re-read failed its own check"

    body = " + ".join(bits) + f"; {note}."
    if not label:
        return body
    return ("**Source:** " if markdown else "Source: ") + body


def format_text(report, *, hide_errors=False):
    """Human-readable report for a terminal."""
    out = []
    mix = ", ".join(f"{k}: {v}" for k, v in report["difficulty_mix"].items())
    t = report["totals"]
    out.append(format_source(report))
    out.append(
        f"{report['n_puzzles']} puzzles"
        + (f" ({mix})" if mix else " (puzzles.json missing)")
        + f"  |  {t['models']} models ({t['graded_models']} with run records)"
    )
    out.append("")

    check = report.get("image_check")
    if check is not None:
        if check["all_ok"]:
            out.append(
                f"Image transcriber: reproduced all {check['n_checked']} input "
                f"puzzle(s) exactly — solution verdicts below are trustworthy."
            )
        else:
            out.append(
                f"Image transcriber: FAILED its own check ({check['n_ok']}/"
                f"{check['n_checked']} input puzzles reproduced). Local "
                f"verification was disabled; the numbers below fall back to the "
                f"grader."
            )
            for c in check["checks"]:
                if not c["ok"]:
                    detail = c.get("error") or (
                        f"{c.get('cells_matched', 0)}/81 cells, first "
                        f"mismatches {c.get('mismatched_cells')}"
                    )
                    out.append(f"    puzzle {c['puzzle_id']}: {detail}")
        out.append("")

    if not report["summaries"]:
        out.append("No per-model records or harness state found.")
    else:
        if hide_errors:
            out.append(
                f"{'#':>2}  {'model':<38} {'type':>4} {'mw':>3} {'score':>7} {'avg s':>8} {'cost':>8} {'$/h':>8} {'claimed':>8} "
                f"{'imgs':>6} {'tok_in':>11} {'tok_out':>11}"
            )
        else:
            out.append(
                f"{'#':>2}  {'model':<38} {'type':>4} {'mw':>3} {'score':>7} {'avg s':>8} {'cost':>8} {'$/h':>8} {'claimed':>8} "
                f"{'imgs':>6} {'tok_in':>11} {'tok_out':>11} {'errs':>5}"
            )
        for i, s in enumerate(report["summaries"], 1):
            imgs = f"{s['n_output_images']}/{s['n_puzzles']}"
            mw = "yes" if s.get("middleware") else " --"
            # The unit rides on the value, so a few rows pasted without the
            # header still say what the number is.
            avg = f"{s['avg_elapsed_s']:.1f}s"
            row = (
                f"{i:>2}  {s['model']:<38} {s.get('type', 'V'):>4} {mw:>3} {_score_cell(s):>7} "
                f"{avg:>8} {_cost_cell(s):>8} {_cost_per_hour_cell(s):>8} "
                f"{_claimed(s):>8} {imgs:>6} "
                f"{s['input_tokens']:>11,} {s['output_tokens']:>11,}"
            )
            if not hide_errors:
                row += f" {s['n_errors']:>5}"
            out.append(row)
        out.append("")
        out.append(
            "  mw      = the model's rounds carried a transcription as image "
            "alt text (yes), or"
        )
        out.append(
            "            did not — including runs too old to record it (--)."
        )
        out.append(
            "  score   = correct out of total, from the strongest verification "
            "this row has:"
        )
        out.append(
            "            the recorded trial, else the local transcription "
            "check, else the grader."
        )
        if any(_beat_trial(x) for x in report["summaries"]):
            out.append(
                "  *       = this results dir beat the recorded trial for this "
                "puzzle set — a new"
            )
            out.append(
                "            best. Unmarked rows come from the record, which "
                "outlives results/."
            )
        out.append(
            "  cost    = total API spend at published OpenRouter prices, shown "
            "only for models"
        )
        out.append(
            "            that answered every puzzle correctly."
        )
        out.append(
            "  $/h     = cost per wall-clock hour (total_cost × 3600 / total_seconds)."
        )
        out.append(
            "  claimed = the harness's own success count from "
            f"{HARNESS_STATE_FILENAME} — the agent"
        )
        out.append(
            "            finished and wrote an image. Its own word, never checked."
        )
        out.append(
            "  Ranked on `score` first, then avg s, then cost — never on "
            "`claimed`."
        )

    tables = {s["safe_name"]: _per_difficulty(s) for s in report["summaries"]}
    diffs = sorted(
        {d for t in tables.values() for d in t if d is not None}
    )
    if diffs:
        out.append("")
        out.append(
            "Verified by difficulty (correct/n), from "
            + ("local reads where available:" if report.get("image_check")
               else "grader verdicts:")
        )
        out.append(f"    {'model':<40} " + " ".join(f"{d:>9}" for d in diffs))
        for s in report["summaries"]:
            cells = []
            for d in diffs:
                pd = tables[s["safe_name"]].get(d)
                cells.append(f"{pd['correct']}/{pd['n']}" if pd else "-")
            out.append(f"    {s['model']:<40} " + " ".join(f"{c:>9}" for c in cells))

    learners = [
        s for s in report["summaries"]
        if (s.get("learning") or {}).get("warmup_elapsed_s") is not None
    ]
    if learners:
        out.append("")
        out.append("Within-session learning (warmup round -> steady state):")
        for s in learners:
            lrn = s["learning"]
            steady = lrn.get("steady_state_avg_elapsed_s")
            speedup = lrn.get("speedup_after_warmup")
            out.append(
                f"    {s['model']:<40} {lrn['warmup_elapsed_s']:7.1f}s -> "
                + (f"{steady:7.1f}s" if steady is not None else "    n/a")
                + (f"  ({speedup:.2f}x)" if speedup else "")
            )

    local_errors = {}
    for s in report["summaries"]:
        for et, n in ((s.get("local") or {}).get("error_types") or {}).items():
            local_errors[et] = local_errors.get(et, 0) + n
    if local_errors:
        out.append("")
        out.append("Why locally-checked images failed:")
        for et, n in sorted(local_errors.items(), key=lambda kv: -kv[1]):
            out.append(f"    {n:>4} image(s): {et}")

    shaky = [s for s in report["summaries"] if (s.get("local") or {}).get("uncertain")]
    if shaky:
        out.append("")
        out.append("Images with cells the transcriber was unsure of:")
        for s in shaky:
            for pid, cells in sorted(s["local"]["uncertain"].items()):
                reads = ", ".join(
                    f"r{c['cell'][0]}c{c['cell'][1]}={c['read_as']}" for c in cells[:4]
                )
                out.append(f"    {s['model']:<32} puzzle {pid}: {reads}")

    disagree = grader_disagreements(report)
    if disagree:
        out.append("")
        out.append("Grader and local transcription disagree:")
        for s, pid, grader_ok, local_ok in disagree:
            out.append(
                f"    {s['model']:<32} puzzle {pid}: grader said "
                f"{'correct' if grader_ok else 'wrong'}, image reads as "
                f"{'correct' if local_ok else 'wrong'}"
            )
        out.append(
            "    One of the two misread the image; the local grid is in the "
            "JSON report."
        )

    unverified = unverified_claims(report)
    if unverified:
        out.append("")
        out.append("Claimed by the harness, more than anything verified:")
        for s, claimed, verified in unverified:
            out.append(
                f"    {s['model']:<40} claimed {claimed}, verified {verified}"
            )
        out.append(
            "    A self-report is not a score. Where an image exists it was "
            "checked; where it does not, nothing backs the claim."
        )

    stale = [s for s in report["summaries"] if (s.get("harness") or {}).get("stale")]
    if stale:
        out.append("")
        out.append("Harness state written against a different puzzle set:")
        for s in stale:
            out.append(
                f"    {s['model']:<40} its rounds (marked `!` above) refer to "
                f"other puzzles"
            )

    resumed = [
        s for s in report["summaries"] if (s.get("harness") or {}).get("n_resumed")
    ]
    if resumed:
        out.append("")
        out.append("Rounds replayed from a previous run (not re-timed here):")
        for s in resumed:
            h = s["harness"]
            out.append(
                f"    {s['model']:<40} {h['n_resumed']}/{h['n_rounds']} round(s) resumed"
            )

    kinds = {}
    for s in report["summaries"]:
        for kind, info in s["errors"].items():
            k = kinds.setdefault(kind, {"n": 0, "models": 0, "sample": info["sample"]})
            k["n"] += info["n"]
            k["models"] += 1
    if kinds and not hide_errors:
        out.append("")
        out.append("Errors by kind:")
        for kind, info in sorted(kinds.items(), key=lambda kv: -kv[1]["n"]):
            out.append(
                f"    {info['n']:>4} record(s) across {info['models']} model(s): {kind}"
            )
            out.append(f"         e.g. {info['sample']}")

    mismatched = [
        s for s in report["summaries"] if s["pngs_on_disk"] != s["n_output_images"]
    ]
    if mismatched:
        out.append("")
        out.append("PNGs on disk disagree with the records:")
        for s in mismatched:
            out.append(
                f"    {s['model']:<40} disk={s['pngs_on_disk']} "
                f"records={s['n_output_images']}"
            )

    if report["ungraded_models"]:
        out.append("")
        out.append("Solution dirs with neither records nor harness state:")
        for o in report["ungraded_models"]:
            out.append(f"    {o['model']:<40} {o['pngs_on_disk']} png(s)")

    if report["unreadable_files"]:
        out.append("")
        out.append("Unreadable result files:")
        for name, why in report["unreadable_files"]:
            out.append(f"    {name}: {why}")

    out.append("")
    out.append(
        f"Totals: {t['locally_verified_correct']}/{t['images_read']} images "
        f"verified correct locally, {t['correct']} by the grader, "
        f"{t['harness_claimed_success']} claimed by the harness; "
        f"{t['pngs_on_disk']} png(s), "
        f"tok_in {t['input_tokens']:,}, tok_out {t['output_tokens']:,} "
        f"({t['total_tokens']:,} total)"
    )
    return "\n".join(out)


def grader_disagreements(report):
    """Per-puzzle conflicts between the grader's verdict and the local read.

    Both claim to have read the same PNG, so a conflict means one of them is
    wrong — worth surfacing rather than silently preferring either.
    """
    out = []
    for s in report["summaries"]:
        local = (s.get("local") or {}).get("per_puzzle") or {}
        for pid, verdict in sorted(local.items()):
            graded = (s.get("grader_verdicts") or {}).get(pid)
            if graded is None:
                continue
            if bool(graded) != bool(verdict.get("correct")):
                out.append((s, pid, bool(graded), bool(verdict.get("correct"))))
    return out


def format_markdown(report, *, hide_errors=False):
    """Same content as `format_text`, pasteable into a README or an issue."""
    mix = ", ".join(f"{k}: {v}" for k, v in report["difficulty_mix"].items())
    t = report["totals"]
    out = [
        format_source(report, markdown=True),
        "",
        f"{report['n_puzzles']} puzzles"
        + (f" ({mix})" if mix else " (puzzles.json missing)"),
        "",
    ]
    if hide_errors:
        out += [
            "| # | model | type | mw | score | avg s | cost | $/h | images | tok_in | tok_out |",
            "|--:|---|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
    else:
        out += [
            "| # | model | type | mw | score | avg s | cost | $/h | images | tok_in | tok_out | errors |",
            "|--:|---|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
    for i, s in enumerate(report["summaries"], 1):
        mw = "yes" if s.get("middleware") else "--"
        row = (
            f"| {i} | `{s['model']}` | {s.get('type', 'V')} | {mw} | {_score_cell(s)} | "
            f"{s['avg_elapsed_s']:.1f}s | {_cost_cell(s)} | {_cost_per_hour_cell(s)} | "
            f"{s['n_output_images']}/{s['n_puzzles']} | "
            f"{s['input_tokens']:,} | {s['output_tokens']:,}"
        )
        if hide_errors:
            row += " |"
        else:
            row += f" | {s['n_errors']} |"
        out.append(row)
    if any(_beat_trial(s) for s in report["summaries"]):
        out.append("")
        out.append(
            "`*` — this results dir beat the recorded trial for this puzzle "
            "set: a new best. Unmarked rows come from the record."
        )

    check = report.get("image_check")
    if check is not None and not check["all_ok"]:
        out.append("")
        out.append(
            f"**The transcriber failed its own check** ({check['n_ok']}/"
            f"{check['n_checked']} input puzzles reproduced), so local "
            f"verification was disabled."
        )

    if report["ungraded_models"]:
        out += ["", "Solution dirs with neither records nor harness state:", ""]
        for o in report["ungraded_models"]:
            out.append(f"- `{o['model']}` — {o['pngs_on_disk']} png(s)")

    kinds = {}
    for s in report["summaries"]:
        for kind, info in s["errors"].items():
            kinds[kind] = kinds.get(kind, 0) + info["n"]
    if kinds and not hide_errors:
        out += ["", "Errors by kind:", ""]
        for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
            out.append(f"- {n} record(s): {kind}")
    return "\n".join(out)


def leaderboard(report):
    """The `leaderboard.json` payload: ranked, verified per-model summaries.

    Rows nothing verified are left out — an unverified self-report has no place
    in a ranking. `accuracy` carries the verification actually used, and
    `verification_source` says which it was, so a locally-checked entry is
    never mistaken for a grader-checked one. The remaining keys match what
    `Benchmark.run` writes.
    """
    drop = {
        "safe_name", "pngs_on_disk", "errors", "harness", "graded", "source",
        "local", "grader_verdicts", "n_verdicts", "verified_accuracy",
        "record_path",
    }
    out = []
    for s in report["summaries"]:
        if s["verification_source"] == "none":
            continue
        entry = {k: v for k, v in s.items() if k not in drop}
        entry["accuracy"] = s["verified_accuracy"]
        entry["verification_source"] = s["verification_source"]
        entry["grader_accuracy"] = s["accuracy"] if s["n_verdicts"] else None
        local = s.get("local") or {}
        entry["local_accuracy"] = local.get("accuracy") if local else None
        out.append(entry)
    return out
