"""Core harness logic: orchestrates one model session through the DSH API.

The harness:
1. Connects to a running DSH instance (or starts one, then stops it)
2. Creates a fresh session and sets the model
3. Runs each input round sequentially through the same session, so the
   agent's conversation history and code-execution state persist across
   rounds (enabling the warmup-vs-steady-state learning measurement)
4. Streams JSONL per-round records to stdout for the benchmarker
5. Handles resume via ``.harness_state.json``
6. Archives the working directory for post-hoc cheat inspection
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from .dsh_api import DshClient, is_dsh_running, spawn_dsh
from .load_task import load_task

DSH_PORT = 3080
DSH_BASE_URL = f"http://127.0.0.1:{DSH_PORT}"
STATE_FILENAME = ".harness_state.json"
OUTPUT_BASENAME = "output.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _puzzle_id_from_name(name: str) -> int | None:
    """Extract NNN from e.g. ``puzzle_042.png``."""
    stem = Path(name).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _fingerprint_inputs(inputs_dir: Path, glob_pattern: str) -> str:
    """Content hash of every input file so resume is safe across puzzle sets."""
    h = hashlib.sha256()
    for p in sorted(inputs_dir.glob(glob_pattern)):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _load_state(output_dir: Path, model_id: str, fingerprint: str) -> dict:
    """Return {puzzle_name: record} for rounds already completed."""
    state_path = output_dir / STATE_FILENAME
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    if state.get("fingerprint") != fingerprint:
        return {}
    if state.get("model") != model_id:
        return {}
    done = {}
    for rec in state.get("rounds", []):
        name = rec.get("puzzle_name")
        if name and rec.get("success") and (output_dir / name).exists():
            done[name] = rec
    return done


def _save_state(
    output_dir: Path, model_id: str, fingerprint: str, rounds: list
) -> None:
    """Persist completed rounds atomically."""
    state_path = output_dir / STATE_FILENAME
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"model": model_id, "fingerprint": fingerprint, "rounds": rounds},
            indent=2,
        )
    )
    os.replace(tmp, state_path)


def _archive_wd(
    wd: Path, archive_dir: Path, model_id: str, rounds: list
) -> None:
    """Snapshot the working directory for post-hoc cheat inspection."""
    safe = model_id.replace("/", "_").replace(":", "_")
    dest = archive_dir.resolve() / safe
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        wd,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", ".pyc"),
    )
    (dest / "session.json").write_text(
        json.dumps({"model": model_id, "rounds": rounds}, indent=2)
    )


def _parse_provider_model(model_id: str) -> tuple[str, str]:
    """Split ``openrouter/anthropic/claude-opus-5`` into (provider, model)."""
    first_slash = model_id.find("/")
    if first_slash == -1:
        return "openrouter", model_id
    return model_id[:first_slash], model_id[first_slash + 1:]


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_harness(
    model_id: str,
    task_path: Path,
    inputs_dir: Path,
    output_dir: Path,
    *,
    timeout: int = 1200,
    archive_dir: Path | None = None,
    no_image: bool = False,
    fresh: bool = False,
    verbose: bool = False,
) -> int:
    """Run the full harness for one model.

    Returns the exit code (0 = all rounds produced output, 1 = some/missing,
    2 = fatal error).
    """
    # ---- Load task spec --------------------------------------------------
    try:
        task = load_task(str(task_path))
    except ValueError as e:
        print(f"[dsh-agent] {e}", file=sys.stderr)
        return 2

    first_prompts = task["prompts"]["first_round"]
    next_prompts = task["prompts"]["next_round"]
    input_glob = task["input_glob"]

    # ---- Discover input files -------------------------------------------
    input_paths = sorted(inputs_dir.glob(input_glob))
    if not input_paths:
        print(
            f"[dsh-agent] no inputs matching '{input_glob}' in {inputs_dir}",
            file=sys.stderr,
        )
        return 2
    n_puzzles = len(input_paths)
    fingerprint = _fingerprint_inputs(inputs_dir, input_glob)

    # ---- Resume state ----------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = {} if fresh else _load_state(output_dir, model_id, fingerprint)
    if completed:
        print(
            f"[dsh-agent] resuming: {len(completed)}/{n_puzzles} already solved",
            file=sys.stderr,
        )

    # ---- Connect to DSH (or spawn it) -----------------------------------
    dsh_proc = None
    if is_dsh_running(DSH_BASE_URL):
        if verbose:
            print("[dsh-agent] using running DSH instance", file=sys.stderr)
    else:
        if verbose:
            print("[dsh-agent] spawning DSH web...", file=sys.stderr)
        try:
            dsh_proc = spawn_dsh(
                port=DSH_PORT,
                timeout=30,
                workdir=str(Path.cwd()),
            )
        except RuntimeError as e:
            print(f"[dsh-agent] {e}", file=sys.stderr)
            return 2

    client = DshClient(base_url=DSH_BASE_URL, timeout=30)

    try:
        # ---- Per-round loop ---------------------------------------------
        rounds: list[dict] = []
        send_images = not no_image
        executed = 0
        interrupted = False
        total_start = time.perf_counter()
        total_input_tokens = 0
        total_output_tokens = 0

        # WORKING DIRECTORY: use a session-scoped dir under the project root
        # so the DSH sandbox (which permits access under the workspace) can
        # read and write to it.  A /tmp dir would be invisible to sandboxed
        # code execution.
        wd_token = uuid.uuid4().hex[:12]
        project_root = Path.cwd()
        workdir = project_root / "dsh-agent" / "workdir" / wd_token
        workdir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[dsh-agent] workdir {workdir}", file=sys.stderr)

        try:
            # Install static assets (verifier scripts, etc.) once.
            for asset_entry in task.get("assets", []):
                src_str = (
                    asset_entry
                    if isinstance(asset_entry, str)
                    else asset_entry.get("path", "")
                )
                if not src_str:
                    continue
                src = Path(src_str)
                if not src.is_absolute():
                    src = task_path.parent / src
                if src.exists():
                    shutil.copy2(src, workdir / src.name)
                    if verbose:
                        print(f"[dsh-agent] installed asset {src.name}", file=sys.stderr)

            # Create session with CWD set to our workdir.  The DSH agent's
            # bash tool starts in this directory, so file I/O from code
            # execution lands where we can read it.
            session_id = client.create_session(cwd=str(workdir))
            provider, model = _parse_provider_model(model_id)
            client.select_model(session_id, provider=provider, model=model)
            if verbose:
                print(
                    f"[dsh-agent] session {session_id}  "
                    f"model={provider}/{model}  cwd={workdir}",
                    file=sys.stderr,
                )

            input_dst = workdir / "input.png"
            output_src = workdir / OUTPUT_BASENAME
            last_seq = 0  # track the latest event seq for incremental polling

            for i, inp_path in enumerate(input_paths):
                round_num = i + 1
                puzzle_id = _puzzle_id_from_name(inp_path.name) or i

                # ---- Resume: skip if already solved ---------------------
                if inp_path.name in completed:
                    prior = completed[inp_path.name]
                    record = {**prior, "resumed": True}
                    rounds.append(record)
                    print(json.dumps(record), flush=True)
                    if verbose:
                        print(
                            f"[dsh-agent] round {round_num}/{n_puzzles}: "
                            f"{inp_path.name} already solved, replaying",
                            file=sys.stderr,
                        )
                    continue

                # ---- Prepare working files ------------------------------
                if output_src.exists():
                    output_src.unlink()
                shutil.copy(inp_path, input_dst)

                # ---- Choose prompt variant ------------------------------
                first_executed = executed == 0
                variants = first_prompts if first_executed else next_prompts

                raw_prompt = (
                    variants["vision"]
                    if send_images
                    else (variants.get("text_only") or variants["vision"])
                )
                prompt_text = raw_prompt.format(
                    n_puzzles=n_puzzles,
                    round=round_num,
                )

                # Tell the agent its working directory paths explicitly.
                agent_prompt = (
                    prompt_text
                    + f"\n\n## Working directory\n"
                    f"Your working directory is: {workdir}\n"
                    f"Read input from: {input_dst}\n"
                    f"Write output to: {output_src}\n"
                    f"Helper files you save here persist across rounds.\n"
                )

                # ---- Submit via DSH API ---------------------------------
                image_bytes = inp_path.read_bytes() if send_images else None

                if send_images and first_executed:
                    try:
                        client.prompt(
                            session_id, agent_prompt, image_data=image_bytes
                        )
                    except RuntimeError as exc:
                        err_msg = str(exc).lower()
                        if not ("image" in err_msg or "unsupported" in err_msg):
                            raise
                        send_images = False
                        image_bytes = None
                        prompt_text = (
                            variants.get("text_only") or variants["vision"]
                        ).format(n_puzzles=n_puzzles, round=round_num)
                        agent_prompt = (
                            prompt_text
                            + f"\n\n## Working directory\n"
                            f"Your working directory is: {workdir}\n"
                            f"Read input from: {input_dst}\n"
                            f"Write output to: {output_src}\n"
                        )
                        client.prompt(session_id, agent_prompt)
                        print(
                            "[dsh-agent] model cannot accept images; "
                            "switched to text-only mode",
                            file=sys.stderr,
                        )
                else:
                    client.prompt(
                        session_id, agent_prompt,
                        image_data=image_bytes if send_images else None,
                    )

                # ---- Wait for turn to finish ----------------------------
                if verbose:
                    print(
                        f"[dsh-agent] round {round_num}/{n_puzzles}: "
                        f"waiting for turn...",
                        file=sys.stderr,
                    )

                start_time = time.perf_counter()
                answer_text, turn_end, last_seq = client.wait_for_turn_end(
                    session_id,
                    timeout=float(timeout),
                    last_seq=last_seq,
                )
                elapsed = time.perf_counter() - start_time
                executed += 1

                # ---- Token projections (cumulative) --------------------
                hist = client.history(session_id)
                proj = hist.get("projections", {}).get("values", {})
                token_usage = proj.get("tokenUsage", {})
                cum_in = token_usage.get("uncachedInputTokens", 0) or 0
                cum_out = token_usage.get("outputTokens", 0) or 0

                # ---- Collect output ------------------------------------
                error = None
                success = output_src.exists()

                if not success:
                    if answer_text:
                        error = (
                            f"Agent did not produce {OUTPUT_BASENAME} "
                            f"(final answer: {answer_text[:200]})"
                        )
                    else:
                        reason_kind = turn_end.get("kind", "unknown")
                        if reason_kind == "error":
                            err_info = turn_end.get("error", {})
                            error = (
                                f"turn error: {err_info.get('code', '?')}: "
                                f"{err_info.get('message', '?')}"
                            )
                        else:
                            error = f"turn ended with reason: {reason_kind}"

                if success:
                    shutil.copy2(output_src, output_dir / inp_path.name)

                record = {
                    "puzzle_id": puzzle_id,
                    "puzzle_name": inp_path.name,
                    "round": round_num,
                    "success": success,
                    "elapsed": elapsed,
                    "final_answer": (answer_text[:2000] if answer_text else None),
                }
                if error:
                    record["error"] = error

                round_in = max(0, cum_in - total_input_tokens)
                round_out = max(0, cum_out - total_output_tokens)
                record["input_tokens"] = round_in
                record["output_tokens"] = round_out
                record["total_tokens"] = round_in + round_out
                total_input_tokens = cum_in
                total_output_tokens = cum_out

                rounds.append(record)
                print(json.dumps(record), flush=True)
                _save_state(output_dir, model_id, fingerprint, rounds)

            # ---- Summary -----------------------------------------------
            total_elapsed = time.perf_counter() - total_start
            n_solved = sum(1 for r in rounds if r["success"])
            summary = {
                "summary": True,
                "n_puzzles": n_puzzles,
                "n_solved": n_solved,
                "interrupted": interrupted,
                "total_elapsed": total_elapsed,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            }
            print(json.dumps(summary), flush=True)

            # ---- Archive -----------------------------------------------
            if archive_dir is not None:
                _archive_wd(workdir, archive_dir, model_id, rounds)
                if verbose:
                    print(
                        f"[dsh-agent] archived working dir -> "
                        f"{archive_dir / model_id.replace('/', '_')}",
                        file=sys.stderr,
                    )

        finally:
            # Clean up the workdir (best-effort).
            shutil.rmtree(workdir, ignore_errors=True)

        return 0 if n_solved == n_puzzles else 1

    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 2
    finally:
        client.close()
        if dsh_proc is not None:
            dsh_proc.terminate()
            try:
                dsh_proc.wait(timeout=10)
            except Exception:
                dsh_proc.kill()