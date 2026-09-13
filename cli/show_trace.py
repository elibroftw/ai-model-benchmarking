#!/usr/bin/env python3
"""Read back what a model actually did, from the traces the harness wrote.

Every harness session logs each of its agent's steps to `trace.jsonl` beside
that model's `.harness_state.json`: the reply, the code smolagents extracted
from the reply, what the code printed, and the error where there was one. The
state file and the leaderboard say how a round came out; this says how it got
there, which is the difference between "glm-5.3-flash scored 0/3" and "it
answered in prose and never emitted a <code> block, so nothing ran".

Usage:
    uv run cli/show_trace.py                       # one line per model
    uv run cli/show_trace.py glm-5.3-flash         # its last session, step by step
    uv run cli/show_trace.py glm --round 2         # only that round
    uv run cli/show_trace.py glm --full            # no display truncation
    uv run cli/show_trace.py glm --session all     # every session in the file
    uv run cli/show_trace.py glm --format json     # the raw events, filtered

The model argument is a substring, matched against the model IDs found in the
traces; anything unambiguous will do. Fields the harness itself truncated are
marked `(+N more)` — re-run that model with `--trace-max-chars 0` to keep them
whole.
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Written by sudoku-agent-harness beside its state file, one JSON object per
# line. Kept as literals rather than imported: this script reads results from
# whatever harness produced them and must not depend on that package being
# installed.
TRACE_FILENAME = "trace.jsonl"
DEFAULT_RESULTS_DIR = "results"

# Per-field display cap. The trace's own cap is far larger (20k by default),
# so this only decides how much of a step fits on screen; --full lifts it.
DISPLAY_MAX_CHARS = 700

# Order events are rendered in within a round, for the --events filter.
EVENT_KINDS = ("session", "round_start", "step", "note", "round_end",
               "round_resumed", "session_end")


def find_traces(results_dir, harness_id=None):
    """Every `<solutions-*>/<model>/trace.jsonl` under a results dir."""
    root = Path(results_dir)
    pattern = f"solutions-{harness_id}" if harness_id else "solutions-*"
    dirs = sorted(root.glob(pattern))
    if not dirs and (root / "solutions").is_dir():
        dirs = [root / "solutions"]
    return sorted(
        p for d in dirs for p in d.glob(f"*/{TRACE_FILENAME}") if p.is_file()
    )


def load_events(path):
    """Parse a trace file, skipping any line that isn't a JSON object.

    A trace is appended to live, so the last line of a file belonging to a
    running session can be half-written; that is a line to skip, not an error.
    """
    events = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            events.append(rec)
    return events


def split_sessions(events):
    """Group events into [(session_id, events)] in the order they were written.

    Sessions append to one file, so a file holds every run of that model
    against that puzzle set; only the last one describes the results sitting
    in the directory now.
    """
    order = []
    groups = {}
    for e in events:
        sid = e.get("session")
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(e)
    return [(sid, groups[sid]) for sid in order]


def session_model(events):
    """The model ID a session recorded for itself, if it got that far."""
    for e in events:
        if e.get("event") == "session" and e.get("model"):
            return e["model"]
    return None


def trace_model(events, path):
    """The model a trace is about, falling back to the directory name.

    The directory name has `/` replaced, so `z-ai_glm-5.3-flash` is the best
    that can be said about a session that died before writing its header.
    """
    for _, evs in reversed(split_sessions(events)):
        model = session_model(evs)
        if model:
            return model
    return Path(path).parent.name


def session_stats(events):
    """The numbers that make one session comparable with another."""
    steps = [e for e in events if e.get("event") == "step"]
    ends = [e for e in events if e.get("event") == "round_end"]
    end = next(
        (e for e in reversed(events) if e.get("event") == "session_end"), {}
    )
    return {
        "n_rounds": len(ends),
        "n_resumed": sum(
            1 for e in events if e.get("event") == "round_resumed"
        ),
        "n_solved": sum(1 for e in ends if e.get("success")),
        "n_steps": len(steps),
        # Steps where the model replied but nothing ran: no code came out of
        # the reply. A handful is a model finding its footing; a round made
        # entirely of these is a model that never learned the protocol, and
        # scores zero for a reason that has nothing to do with Sudoku.
        "n_no_code": sum(
            1 for e in steps if not e.get("code") and e.get("model_output")
        ),
        "n_step_errors": sum(1 for e in steps if e.get("error")),
        "n_notes": sum(1 for e in events if e.get("event") == "note"),
        "output_tokens": sum(e.get("output_tokens") or 0 for e in steps),
        "interrupted": bool(end.get("interrupted")),
        "started": next((e.get("ts") for e in events), None),
        "temperature": next(
            (e.get("temperature") for e in events
             if e.get("event") == "session"), None
        ),
    }


def _clip(text, limit):
    """Cut display text, reporting what was left out."""
    if text is None:
        return None
    text = text if isinstance(text, str) else json.dumps(text, default=str)
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}\n… (+{len(text) - limit} more)"


def _block(label, text, limit, harness_chars=None, indent="      "):
    """One labelled, indented field of a step."""
    body = _clip(text, limit)
    if not body:
        return []
    note = ""
    if harness_chars:
        # The harness cut this field before it ever reached the file, so no
        # display setting here can recover the rest.
        note = f"  (trace kept {len(text)} of {harness_chars} chars)"
    out = [f"{indent}{label}:{note}"]
    out += [f"{indent}  {ln}" for ln in body.splitlines()]
    return out


def _ts(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "?"


def render_session(sid, events, *, limit, rounds=None, steps=None, kinds=None):
    """One session as text, round by round."""
    out = []
    stats = session_stats(events)
    head = session_model(events) or "?"
    out.append(f"session {sid}  {head}  started {_ts(stats['started'])}")
    bits = [
        f"{stats['n_solved']}/{stats['n_rounds']} solved",
        f"{stats['n_steps']} steps",
    ]
    if stats["n_no_code"]:
        bits.append(f"{stats['n_no_code']} replies with no code")
    if stats["n_step_errors"]:
        bits.append(f"{stats['n_step_errors']} step errors")
    if stats["n_resumed"]:
        bits.append(f"{stats['n_resumed']} replayed")
    if stats["interrupted"]:
        bits.append("INTERRUPTED")
    out.append("  " + ", ".join(bits))

    for e in events:
        kind = e.get("event")
        if kinds and kind not in kinds:
            continue
        rnd = e.get("round")
        if rounds and kind != "session" and rnd not in rounds:
            continue
        if kind == "session":
            out.append("")
            out.append(
                f"  temperature={e.get('temperature')}  "
                f"images={e.get('send_images')}  "
                f"max_steps={e.get('max_steps')}  "
                f"timeout={e.get('timeout')}s  "
                f"inputs={e.get('n_inputs')}  resumed={e.get('n_resumed')}"
            )
        elif kind == "round_start":
            out.append("")
            out.append(
                f"  ===== round {rnd}/{e.get('n_rounds')}: "
                f"{e.get('item_name')} "
                f"(middleware={e.get('middleware')}, "
                f"images={e.get('send_images')}) ====="
            )
            out += _block("prompt", e.get("prompt"), limit,
                          e.get("prompt_chars"), indent="    ")
        elif kind == "round_resumed":
            out.append("")
            out.append(
                f"  ===== round {rnd}: {e.get('item_name')} replayed from "
                f"state (success={e.get('success')}, "
                f"{_fmt_s(e.get('elapsed'))}) ====="
            )
        elif kind == "step":
            n = e.get("step")
            if steps and n not in steps:
                continue
            out.append("")
            flags = []
            if e.get("is_final_answer"):
                flags.append("FINAL")
            if not e.get("code") and e.get("model_output"):
                flags.append("NO CODE")
            if e.get("n_images"):
                flags.append(f"{e['n_images']} image(s)")
            out.append(
                f"    step {n}  {_fmt_s(e.get('duration'))}  "
                f"in {e.get('input_tokens')} / out {e.get('output_tokens')} tok"
                + (f"  [{', '.join(flags)}]" if flags else "")
            )
            out += _block("reply", e.get("model_output"), limit,
                          e.get("model_output_chars"))
            out += _block("code", e.get("code"), limit, e.get("code_chars"))
            out += _block("observations", e.get("observations"), limit,
                          e.get("observations_chars"))
            out += _block("error", e.get("error"), limit)
            if e.get("tool_calls"):
                out += _block("tool_calls", json.dumps(e["tool_calls"]), limit)
        elif kind == "note":
            out.append(
                f"    [{e.get('kind')}] {_clip(e.get('detail'), limit)}"
            )
        elif kind == "round_end":
            out.append(
                f"    --- round {rnd} {'OK' if e.get('success') else 'FAILED'}"
                f" in {_fmt_s(e.get('elapsed'))}, {e.get('n_steps')} step(s),"
                f" attempt {e.get('attempts')}"
            )
            if e.get("error"):
                out.append(f"        error: {_clip(e['error'], limit)}")
            if e.get("final_answer"):
                out.append(f"        answer: {_clip(e['final_answer'], 200)}")
        elif kind == "session_end":
            out.append("")
            out.append(
                f"  session end: {e.get('n_solved')}/{e.get('n_rounds')} "
                f"solved in {_fmt_s(e.get('total_elapsed'))}"
                + (" (interrupted)" if e.get("interrupted") else "")
            )
    return "\n".join(out)


def _fmt_s(x):
    return f"{x:.1f}s" if isinstance(x, (int, float)) else "?"


def render_summary(traces):
    """One line per model: enough to see which trace is worth opening."""
    rows = []
    for path in traces:
        events = load_events(path)
        if not events:
            continue
        sessions = split_sessions(events)
        sid, last = sessions[-1]
        st = session_stats(last)
        rows.append({
            "model": trace_model(events, path),
            "n_sessions": len(sessions),
            "session": sid,
            "path": path,
            "size": path.stat().st_size,
            **st,
        })
    if not rows:
        return "no trace.jsonl files found (only runs since tracing was added have them)"

    rows.sort(key=lambda r: (-r["n_no_code"], -r["n_step_errors"], r["model"]))
    width = max(len(r["model"]) for r in rows)
    out = [
        f"{len(rows)} trace(s). Newest session of each model; "
        f"`show_trace.py <model>` for the steps.",
        "",
        f"{'model':<{width}}  {'sess':>4}  {'solved':>6}  {'steps':>5}  "
        f"{'nocode':>6}  {'errs':>4}  {'notes':>5}  {'out tok':>9}  started",
    ]
    for r in rows:
        out.append(
            f"{r['model']:<{width}}  {r['n_sessions']:>4}  "
            f"{r['n_solved']}/{r['n_rounds']:<4}  {r['n_steps']:>5}  "
            f"{r['n_no_code']:>6}  {r['n_step_errors']:>4}  "
            f"{r['n_notes']:>5}  {r['output_tokens']:>9,}  "
            f"{_ts(r['started'])}"
            + ("  INTERRUPTED" if r["interrupted"] else "")
        )
    out += [
        "",
        "  nocode = steps where the model replied but no <code> block came out",
        "           of it, so nothing ran. A round of nothing but these is a",
        "           model that never followed the protocol, not one that",
        "           failed at Sudoku.",
        "  errs   = steps that ended in an error (parse, interpreter, provider).",
        "  notes  = harness decisions mid-round: retries, image or temperature",
        "           fallbacks, interrupts.",
    ]
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no model argument, prints one line per traced model.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model ID or any unambiguous substring of one. Omit for the "
             "summary table.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Where the solutions dirs live (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--harness-id",
        default=None,
        help="Only read solutions-<id>/. By default every solutions-* dir "
             "under --results-dir is searched.",
    )
    parser.add_argument(
        "--session",
        default="last",
        help="Which session of that model to show: `last` (default), `all`, "
             "or a session id. A trace holds every run of the model against "
             "this puzzle set; only the last describes what is on disk now.",
    )
    parser.add_argument(
        "--round",
        type=int,
        action="append",
        dest="rounds",
        help="Only this round (repeatable).",
    )
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        dest="steps",
        help="Only this step number (repeatable).",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Comma-separated event kinds to keep: "
             f"{', '.join(EVENT_KINDS)}. e.g. --events step,note",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DISPLAY_MAX_CHARS,
        help=f"Cut each field to this many chars on screen (default "
             f"{DISPLAY_MAX_CHARS}; 0 for no cut). Does not affect what the "
             f"file holds.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print every field in full, however long — same as --max-chars 0.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="text (default) or json, which emits the filtered raw events.",
    )
    parser.add_argument("--out", help="Write to this file instead of stdout.")
    args = parser.parse_args(argv)

    traces = find_traces(args.results_dir, args.harness_id)
    if not traces:
        sys.exit(
            f"no {TRACE_FILENAME} under {args.results_dir}/solutions-*/ — "
            f"only runs since tracing was added have one"
        )

    if not args.model:
        text = render_summary(traces)
    else:
        wanted = args.model.lower()
        loaded = [(p, load_events(p)) for p in traces]
        named = [(p, evs, trace_model(evs, p)) for p, evs in loaded if evs]
        matches = [x for x in named if wanted == x[2].lower()]
        if not matches:
            matches = [x for x in named if wanted in x[2].lower()]
        if not matches:
            sys.exit(
                f"no traced model matches {args.model!r}. Traced: "
                + ", ".join(sorted(m for _, _, m in named))
            )
        if len(matches) > 1:
            sys.exit(
                f"{args.model!r} matches several models: "
                + ", ".join(sorted(m for _, _, m in matches))
            )
        path, events, model = matches[0]

        sessions = split_sessions(events)
        if args.session == "last":
            chosen = sessions[-1:]
        elif args.session == "all":
            chosen = sessions
        else:
            chosen = [s for s in sessions if s[0] == args.session]
            if not chosen:
                sys.exit(
                    f"{path} has no session {args.session!r}. It has: "
                    + ", ".join(sid or "?" for sid, _ in sessions)
                )

        limit = None if (args.full or args.max_chars <= 0) else args.max_chars
        kinds = (
            {k.strip() for k in args.events.split(",") if k.strip()}
            if args.events else None
        )
        rounds = set(args.rounds) if args.rounds else None
        steps = set(args.steps) if args.steps else None

        if args.format == "json":
            picked = [
                e for _, evs in chosen for e in evs
                if (not kinds or e.get("event") in kinds)
                and (not rounds or e.get("round") in rounds
                     or e.get("event") in ("session", "session_end"))
                and (not steps or e.get("event") != "step"
                     or e.get("step") in steps)
            ]
            text = json.dumps(picked, indent=2)
        else:
            parts = [f"{path}  ({len(sessions)} session(s), {model})"]
            parts += [
                render_session(sid, evs, limit=limit, rounds=rounds,
                               steps=steps, kinds=kinds)
                for sid, evs in chosen
            ]
            text = "\n\n".join(parts)

    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
