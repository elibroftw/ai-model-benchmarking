"""Minimal async HTTP client for the DSH RPC API.

The DSH Web GUI exposes a JSON-RPC-like API at ``POST /api/<method>``.
Every request uses the same envelope:

.. code-block:: json

    {"type": "client-request", "rpcId": "<id>", "method": "...", "payload": {...}}
"""

from __future__ import annotations

import base64
import json
import sys
import time
import uuid
from pathlib import Path

import httpx


class DshClient:
    """Talk to a running DSH instance over its HTTP API.

    Args:
        base_url: Root of the DSH web server, e.g. ``http://127.0.0.1:3080``.
        timeout:  HTTP request timeout in seconds (not the agent turn timeout).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:3080", timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self._rpc_seq = 0
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout))

    # ------------------------------------------------------------------
    # Low-level RPC call
    # ------------------------------------------------------------------

    def _call(self, method: str, payload: dict, rpc_id: str | None = None) -> dict:
        """Send one RPC and return the full server response dict."""
        if rpc_id is None:
            self._rpc_seq += 1
            rpc_id = str(self._rpc_seq)

        body = {
            "type": "client-request",
            "rpcId": rpc_id,
            "method": method,
            "payload": payload,
        }
        resp = self._http.post(f"/api/{method}", json=body)
        resp.raise_for_status()
        return resp.json()

    def _call_ok(self, method: str, payload: dict) -> dict:
        """Call an RPC and return the ``value`` inside the result envelope.

        Raises RuntimeError if the server returned an error.
        """
        resp = self._call(method, payload)
        result = resp.get("result", {})
        if not result.get("ok"):
            err = result.get("error", {})
            raise RuntimeError(
                f"DSH API {method} failed: {err.get('code', '?')}: "
                f"{err.get('message', '?')}"
            )
        return result.get("value", {})

    # ------------------------------------------------------------------
    # Host
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        """Return host metadata (version, cwd, default provider/model, …)."""
        return self._call_ok("host.describe", {})

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, cwd: str | None = None, agent_preset: str | None = None) -> str:
        """Create a blank session and return its id.

        Args:
            cwd:  Working directory the session starts in.
            agent_preset:  Agent composition preset (e.g. ``standard``).
        """
        payload: dict = {}
        if cwd is not None:
            payload["cwd"] = cwd
        if agent_preset is not None:
            payload["agentPreset"] = agent_preset
        value = self._call_ok("session.create", payload)
        return value["sessionId"]

    def list_sessions(self) -> list[dict]:
        """Return all visible sessions with their metadata."""
        value = self._call_ok("session.list", {})
        return value.get("items", [])

    def select_model(self, session_id: str, provider: str, model: str) -> dict:
        """Set the provider + model for an existing session."""
        return self._call_ok("session.selectModel", {
            "sessionId": session_id,
            "provider": provider,
            "model": model,
        })

    def get_models(self, session_id: str) -> dict:
        """Return the session's current model selection and available model groups."""
        return self._call_ok("session.models", {
            "sessionId": session_id,
        })

    def prompt(self, session_id: str, text: str, image_data: bytes | None = None) -> bool:
        """Submit a user message (text + optional image) to a session.

        Returns True if the server accepted the prompt (it will be processed
        asynchronously).  Does NOT wait for the turn to finish.
        """
        content: list[dict] = [{"type": "text", "text": text}]
        if image_data is not None:
            b64 = base64.b64encode(image_data).decode()
            content.append({
                "type": "image",
                "mediaType": "image/png",
                "data": b64,
            })
        value = self._call_ok("session.prompt", {
            "sessionId": session_id,
            "mode": "queue",
            "content": content,
        })
        return bool(value.get("accepted", False))

    def history(self, session_id: str, before_seq: int | None = None) -> dict:
        """Return the session's event log.

        Args:
            session_id:  The session to query.
            before_seq:  If set, only events strictly before this seq are returned
                         (for pagination / incremental polling).

        Returns the raw value dict which contains an ``events`` list.
        """
        payload: dict = {"sessionId": session_id}
        if before_seq is not None:
            payload["beforeSeq"] = before_seq
        return self._call_ok("session.history", payload)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def wait_for_turn_end(
        self,
        session_id: str,
        timeout: float = 600,
        poll: float = 2.0,
        last_seq: int = 0,
    ) -> tuple[str, dict, int]:
        """Poll the session history until the current turn finishes.

        Args:
            session_id:  Session to poll.
            timeout:     Max wall-clock seconds to wait.
            poll:        Seconds between polls.
            last_seq:    Only consider events after this seq.

        Returns:
            (final_answer_text, turn_end_data, max_seq_seen)

        Raises TimeoutError if the turn doesn't finish within ``timeout``.
        """
        deadline = time.monotonic() + timeout
        max_seq = last_seq
        while time.monotonic() < deadline:
            hist = self.history(session_id)
            events = hist.get("events", [])
            answer = ""
            turn_end = None

            for entry in events:
                ev = entry.get("event", {})
                seq = ev.get("seq", 0)
                if seq > max_seq:
                    max_seq = seq
                if seq <= last_seq:
                    continue
                etype = ev.get("type", "")
                if etype == "assistant/message":
                    blocks = (
                        ev.get("data", {})
                        .get("message", {})
                        .get("content", [])
                    )
                    parts = [
                        b.get("text", "")
                        for b in blocks
                        if b.get("type") == "text"
                    ]
                    if parts:
                        answer = "".join(parts)
                elif etype == "turn/end":
                    turn_end = ev.get("data", {}).get("reason", {})

            if turn_end is not None:
                return answer, turn_end, max_seq

            time.sleep(poll)

        raise TimeoutError(
            f"DSH session {session_id}: turn did not finish within {timeout}s"
        )

    def close(self) -> None:
        self._http.close()


# ---- Module-level helpers ---------------------------------------------------


def is_dsh_running(base_url: str = "http://127.0.0.1:3080") -> bool:
    """Quick check whether a DSH web server is reachable."""
    try:
        resp = httpx.get(f"{base_url}/", timeout=5)
        return resp.status_code == 200 and "dsh" in resp.text.lower()
    except Exception:
        return False


def spawn_dsh(
    port: int = 3080,
    timeout: float = 30,
    profile: str = "web",
    workdir: str | None = None,
) -> "subprocess.Popen[bytes]":
    """Start a DSH web server and wait until it's ready.

    Returns the Popen handle so the caller can terminate it later.

    Requires ``dsh`` on PATH.  Raises RuntimeError if the server doesn't
    become reachable within *timeout* seconds.
    """
    import subprocess

    cmd = ["dsh", "--profile", profile]
    if port != 3080:
        cmd.extend(["--port", str(port)])
    if workdir:
        cmd.extend(["--cwd", workdir])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(base_url, timeout=2)
            if resp.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.5)

    # Timed out – kill and report.
    proc.kill()
    _, stderr = proc.communicate(timeout=5)
    raise RuntimeError(
        f"DSH web did not start within {timeout}s on {base_url}\n"
        f"stderr:\n{stderr.decode(errors='replace')[:2000]}"
    )