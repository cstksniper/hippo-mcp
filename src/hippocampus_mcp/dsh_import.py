# -*- coding: utf-8 -*-
"""Import DeepSeek Harness session logs into a hippocampus memory library.

D-3 (2026-11-22): productized history import for ``hippocampus init --dsh`` —
when a companion gains a memory, its past conversations can come along.

Three layers, all pure-stdlib except optional ``zstandard``:

1. **Locate** — port of DSH's ``projectKey`` (dsh-session-persistence-jsonl):
   the sessions root groups logs by encoded project directory
   (``--E-<work-root>-dsh_lab-ws-dufeiyu--``). :func:`project_key` reproduces
   the encoding so a workspace path finds its logs without guessing.
2. **Extract** — surface user/assistant text from ``session.vN.jsonl[.zstd]``
   events: ms-epoch timestamps, assistant bodies under ``data.message``,
   reasoning blocks and host boilerplate (system-reminder injections)
   filtered out. The archive keeps what was said, not what was thought.
3. **Write** — one TCP exchange per message through the shared daemon
   (HELLO → initialize → tools/call ingest), C1-bound to the workspace
   library; per-message ``idem_key`` gives true replay protection and a
   per-session ledger skips already-imported events. Writes require a
   RUNNING daemon — this module never touches library files directly (the
   dual-backend split of 2026-11-06 is the lesson behind that rule).
"""
from __future__ import annotations

import io
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

__all__ = [
    "project_key", "find_session_logs", "open_session_log", "extract_messages",
    "import_session_logs", "ImportReport",
]

HELLO_PREFIX = "HIPPO-HELLO "
CHECKPOINT_PREFIX = "This is an automatically generated checkpoint"
NOISE_USER_PREFIXES = (
    "<system-reminder>",
    "Time sampled while preparing",
    "This is an automatically generated checkpoint",
)

#: Per-message text cap: long assistant answers are stored truncated; the
#: raw log on disk remains the full-fidelity archive.
DEFAULT_MAX_CHARS = 8000
DEFAULT_IMPORTANCE = 0.35


def project_key(cwd: str) -> str:
    """Port of DSH's ``projectKey`` (session-persistence-jsonl).

    Separators (``/`` ``\\\\`` ``:``) collapse to a single ``-``; safe code
    units ``[A-Za-z0-9._-]`` pass through; everything else becomes a
    ``~XXXX`` uppercase-hex escape; leading dashes are stripped and the key
    is wrapped in ``--`` pairs, bounded to 251 inner chars.
    """
    if not cwd:
        raise ValueError("cannot encode an empty project path")
    readable = ""
    separator_run = False
    for ch in cwd:
        code = ord(ch)
        if ch in ("/", "\\", ":"):
            if not separator_run:
                readable += "-"
            separator_run = True
        elif ch != "~" and re.match(r"[A-Za-z0-9._-]", ch):
            readable += ch
            separator_run = False
        else:
            readable += "~" + format(code, "04X")
            separator_run = False
    inner = readable.lstrip("-") or "root"
    return "--" + inner[:251] + "--"


def find_session_logs(dsh_home: str, workspace: str) -> list[str]:
    """Return session log paths for ``workspace`` under ``dsh_home``.

    Layout: ``<dsh_home>/sessions/<projectKey(workspace)>/session-*/session.v*.jsonl*``.
    A missing directory yields an empty list (fresh installs have no history).
    """
    key = project_key(workspace)
    base = Path(dsh_home) / "sessions" / key
    if not base.is_dir():
        return []
    found: list[str] = []
    for session_dir in sorted(base.glob("session-*")):
        for log in sorted(session_dir.glob("session.v*.jsonl*")):
            found.append(str(log))
    return found


def open_session_log(log_path: str) -> list[dict]:
    """Parse one session log into an event list (zstd handled if available)."""
    if log_path.endswith(".zst") or log_path.endswith(".zstd"):
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError(
                "the log is zstd-compressed; install the optional dependency: "
                "pip install zstandard") from exc
        # Read whole file then decompress: stream_reader on a raw file handle
        # fails with OSError(EINVAL) on some Windows setups, and DSH logs may
        # be multi-frame streams (BytesIO handles both).
        raw = Path(log_path).read_bytes()
        blob = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
        text = blob.decode("utf-8", errors="replace")
    else:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _format_stamp(ts: Any) -> str | None:
    """DSH stamps are epoch milliseconds; guard against out-of-range values
    (Windows localtime raises OSError/EINVAL on absurd epochs)."""
    if not isinstance(ts, (int, float)) or ts <= 0:
        return None
    secs = ts / 1000.0 if ts > 1e11 else ts
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(secs))
    except (OSError, OverflowError, ValueError):
        return None


def extract_messages(events: list[dict], max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    """Pick surface user/assistant text out of the event stream.

    Shapes differ per role: ``user/message`` carries ``data.content`` blocks;
    ``assistant/message`` wraps them in ``data.message`` and also embeds
    ``reasoning`` blocks (the model's internal thinking) which are skipped —
    the archive keeps what was said, not what was thought. Host boilerplate
    injected as user turns (system-reminder blocks, runtime time samples) is
    not the user speaking and is filtered.
    """
    out: list[dict] = []
    for ev in events:
        etype = ev.get("type", "")
        if etype not in ("user/message", "assistant/message"):
            continue
        data = ev.get("data") or {}
        if etype == "assistant/message":
            blocks = ((data.get("message") or {}).get("content")) or []
        else:
            blocks = data.get("content") or []
        text = "\n".join(
            b.get("text", "") for b in blocks
            if isinstance(b, dict) and b.get("type") == "text").strip()
        if not text:
            continue
        if etype == "user/message" and text.startswith(NOISE_USER_PREFIXES):
            continue
        speaker = "user" if etype == "user/message" else "companion"
        stamp = _format_stamp(ev.get("time"))
        seq = ev.get("seq")
        if len(text) > max_chars:
            text = text[:max_chars] + " …[truncated]"
        out.append({"seq": seq, "speaker": speaker, "text": text, "timestamp": stamp})
    return out


# ---------------------------------------------------------------------------
# Daemon write path (stdlib TCP client; same handshake as hook_client)
# ---------------------------------------------------------------------------
def _read_response(sock: socket.socket, buf: bytearray, want_id: int,
                   deadline: float) -> dict:
    """Read lines until the JSON-RPC frame with ``want_id`` arrives.

    The daemon answers line-by-line (initialize response FIRST, tools/call
    response SECOND) — grabbing the first line is the classic bug: you end up
    parsing initialize's ack as the tool result.
    """
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("daemon exchange deadline exceeded")
        sock.settimeout(min(remaining, 5.0))
        idx = buf.find(b"\n")
        if idx < 0:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("daemon closed the connection")
            buf += chunk
            continue
        line = bytes(buf[:idx]).decode("utf-8", errors="replace").strip()
        del buf[:idx + 1]
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("id") == want_id:
            return obj
        # other frames (initialize ack) are skipped


def _exchange(sock: socket.socket, buf: bytearray, *, hello_library: str,
              cwd: str, calls: list[tuple[int, str, dict]],
              deadline: float) -> dict[int, dict]:
    """HELLO → initialize → initialized → tools/call…; returns id→result map."""
    payload = {"library": hello_library, "cwd": cwd, "pid": os.getpid()}
    sock.sendall((HELLO_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
                 .encode("utf-8"))
    sock.sendall((json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "hippo-dsh-import", "version": "1"}},
    }, ensure_ascii=False) + "\n").encode("utf-8"))
    sock.sendall((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
                  + "\n").encode("utf-8"))
    for call_id, name, args in calls:
        sock.sendall((json.dumps({
            "jsonrpc": "2.0", "id": call_id, "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }, ensure_ascii=False) + "\n").encode("utf-8"))
    results: dict[int, dict] = {}
    for call_id, _name, _args in calls:
        resp = _read_response(sock, buf, call_id, deadline)
        if resp.get("error"):
            raise RuntimeError(f"tools/call id={call_id} rejected: "
                               f"{json.dumps(resp['error'], ensure_ascii=False)[:200]}")
        text = ((resp.get("result") or {}).get("content") or [{}])[0].get("text", "")
        try:
            results[call_id] = json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"tools/call id={call_id} returned non-JSON: "
                               f"{str(text)[:120]}") from exc
    return results


def _daemon_port_from_registry() -> int | None:
    """Freshest live daemon entry from the launcher registry (stdlib scan)."""
    import glob as _glob
    registry_dir = os.path.join(os.environ.get("TEMP", os.environ.get("TMP", "/tmp")),
                                "hippocampus_mcp")
    best: tuple[float, int] | None = None
    for path in _glob.glob(os.path.join(registry_dir, "daemon-*.json")):
        try:
            info = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        port = info.get("port")
        alive = info.get("alive_at")
        if not isinstance(port, int) or not isinstance(alive, (int, float)):
            continue
        if time.time() - alive > 90:
            continue
        if best is None or alive > best[0]:
            best = (alive, port)
    return best[1] if best else None


def import_session_logs(*, log_paths: list[str], cwd: str,
                        hello_library: str = "",
                        daemon_port: int | None = None,
                        importance: float = DEFAULT_IMPORTANCE,
                        max_chars: int = DEFAULT_MAX_CHARS,
                        timeout_s: float = 300.0,
                        dry_run: bool = False) -> list[dict[str, Any]]:
    """Import the given session logs into the workspace-bound library.

    Writes go through the shared daemon (a running daemon is REQUIRED unless
    ``dry_run``); the C1 HELLO binding targets the workspace's own library.
    Idempotent twice over: a per-session ledger (``import_ledger.json`` next
    to each log) skips already-imported seqs, and per-message ``idem_key``
    makes a daemon-side replay a no-op. Returns one summary dict per log.
    """
    summaries: list[dict[str, Any]] = []
    pending_jobs: list[tuple[str, list[dict], list[dict]]] = []

    for log_path in log_paths:
        session_dir = Path(log_path).parent
        session_id = session_dir.name.replace("session-", "")
        events = open_session_log(log_path)
        messages = extract_messages(events, max_chars)
        ledger_path = session_dir / "import_ledger.json"
        done: set = set()
        if ledger_path.exists():
            try:
                done = set((json.loads(ledger_path.read_text(encoding="utf-8"))
                            .get("imported_seqs")) or [])
            except Exception:
                done = set()
        pending = [m for m in messages if m["seq"] not in done]
        summary: dict[str, Any] = {
            "log": log_path, "session": session_id,
            "surface_messages": len(messages),
            "already_imported": len(messages) - len(pending),
            "pending": len(pending), "written": 0, "errors": [],
            "dry_run": dry_run,
        }
        summaries.append(summary)
        if pending:
            pending_jobs.append((log_path, pending, summary))

    if dry_run or not pending_jobs:
        return summaries

    port = daemon_port if daemon_port is not None else _daemon_port_from_registry()
    if port is None:
        raise RuntimeError("no live daemon found in the launcher registry; "
                           "start the memory service first (e.g. talk to the "
                           "companion once) or pass daemon_port explicitly")

    deadline = time.monotonic() + timeout_s
    sock = socket.create_connection(("127.0.0.1", port), timeout=10.0)
    buf = bytearray()
    try:
        for log_path, pending, summary in pending_jobs:
            session_id = Path(log_path).parent.name.replace("session-", "")
            # id=2..: one ingest call per message (slim common profile has no
            # batch tool); idem_key makes daemon-side replays no-ops.
            calls: list[tuple[int, str, dict]] = []
            for idx, m in enumerate(pending):
                calls.append((2 + idx, "ingest", {
                    "text": f"[{m['speaker']} @ {m['timestamp'] or 'unknown-time'}] {m['text']}",
                    "speaker": m["speaker"],
                    "importance": importance,
                    "source": f"dsh-session:{session_id}:seq{m['seq']}",
                    "tags": json.dumps({"source": "dsh-import",
                                        "session": session_id[:8],
                                        "seq": str(m["seq"])}, ensure_ascii=False),
                    "idem_key": f"dsh-{session_id}-{m['seq']}",
                    "auto_tags": False,
                }))
            results = _exchange(sock, buf, hello_library=hello_library,
                                cwd=cwd, calls=calls, deadline=deadline)
            written = 0
            imported_seqs: list = []
            errors: list[str] = []
            for m, (call_id, _name, _args) in zip(pending, calls):
                result = results.get(call_id) or {}
                if result.get("error"):
                    errors.append(str(result["error"])[:200])
                    continue
                written += 1
                imported_seqs.append(m["seq"])
            summary["written"] = written
            if errors:
                summary["errors"] = errors
            if imported_seqs and not errors:
                ledger_path = Path(log_path).parent / "import_ledger.json"
                try:
                    old: dict = {}
                    if ledger_path.exists():
                        old = json.loads(ledger_path.read_text(encoding="utf-8"))
                    merged = sorted(set((old.get("imported_seqs") or []) + imported_seqs),
                                    key=lambda x: (x is None, x))
                    ledger_path.write_text(json.dumps({
                        "imported_seqs": merged,
                        "last_import": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }, ensure_ascii=False, indent=1), encoding="utf-8")
                except Exception as exc:
                    summary["errors"].append(f"ledger write failed: {exc}")
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return summaries
