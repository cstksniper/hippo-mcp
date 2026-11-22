# -*- coding: utf-8 -*-
"""Zero-step DeepSeek Harness onboarding: ``hippocampus init --dsh``.

Companion-agent onboarding (2026-11-21, plan:
``docs/design/DSH虚拟人记忆_设计书_20261121.md`` §3.5 — the user-directed
ease-of-use amendment: the companion should be able to install its own
memory; this command is the machine side of that story). One command makes a
DSH workspace memory-ready:

1. bind a workspace to a dedicated library (reuses :func:`run_init` — the
   same manifest/binding primitives as ``init --project``);
2. append the ``dsh-mcp-client`` insert entry to the profile's
   ``cordis.patch.yml`` (idempotent by entry id; DSH hot-reloads the file);
3. write the persona + memory-discipline block into the workspace
   ``AGENTS.md`` (marker block, same idempotency contract as ``init
   --project``);
4. drop ``memory-setup.md`` into the workspace — the self-install/self-check
   guide the companion itself reads.

Idempotent at every step: re-running reports ``already_bound`` /
``already_present`` and only refreshes the text files.

Privacy/scope: writes are limited to the DSH profile patch, the workspace
text files, the new library directory, and the user-level sidecar.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .init_project import run_init, write_agents_snippet
from .library import normpath

__all__ = ["run_init_dsh", "history_summary"]

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_PATCH_ENTRY_ID = "mcp-hippocampus"
#: Model id pinned into the patch env AND used by the end-of-run backend
#: self-check -- the two must stay the same or the check lies.
_DSH_EMB_MODEL = "BAAI/bge-base-zh-v1.5"
_AGENTS_BEGIN = "<!-- hippocampus:dsh-begin"
_AGENTS_END = "<!-- hippocampus:dsh-end -->"
_GUIDE_TEMPLATE = "memory_setup_dsh.md"
_GUIDE_NAME = "memory-setup.md"

_PATCH_TEMPLATE = """- insert:
    - id: {entry_id}
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: hippocampus
        transport: stdio
        command: '{python_cmd}'
        args: ['-m', 'hippocampus_mcp.launcher']
        cwd: '{workspace}'
        toolCallTimeoutMs: 120000
        env:
          PYTHONPATH: '{pythonpath}'
          HIPPOCAMPUS_PERSIST_DIR: '{user_lib}'
          HIPPOCAMPUS_EMB_MODEL: '{emb_model}'
          HIPPOCAMPUS_AUTO_BACKFILL: '0'
          HIPPOCAMPUS_SUPERSEDE: '0'
          HIPPOCAMPUS_STORE_BACKEND: 'sqlite'
          HIPPOCAMPUS_FTS5: '1'
          HIPPOCAMPUS_LAUNCHER: '1'
          HIPPOCAMPUS_TOOL_PROFILE: 'common'
"""


def _dsh_home_default() -> str:
    """DSH home resolution order: --dsh-home > DSH_HOME env > ~/.dsh."""
    return os.environ.get("DSH_HOME") or os.path.join(os.path.expanduser("~"), ".dsh")


def _patch_path(dsh_home: str, profile: str) -> Path:
    return Path(dsh_home) / "profiles" / profile / "cordis.patch.yml"


def _append_patch_entry(patch: Path, *, python_cmd: str, workspace: str,
                        pythonpath: str, user_lib: str,
                        emb_model: str) -> str:
    """Append the MCP client insert entry; idempotent by entry id.

    Returns ``created`` or ``already_present``. The YAML block is appended
    verbatim with single-quoted scalars (safe for Windows backslashes).
    """
    text = patch.read_text(encoding="utf-8") if patch.exists() else ""
    if _PATCH_ENTRY_ID in text:
        return "already_present"
    block = _PATCH_TEMPLATE.format(
        entry_id=_PATCH_ENTRY_ID, python_cmd=python_cmd,
        workspace=workspace.replace("'", "''"),
        pythonpath=pythonpath.replace("'", "''"),
        user_lib=user_lib.replace("'", "''"),
        emb_model=emb_model)
    new = text.rstrip() + "\n" + block if text.strip() else block
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text(new, encoding="utf-8", newline="\n")
    return "created"


def _write_guide(workspace: str) -> str:
    """Write ``memory-setup.md`` (the companion's own self-install guide)."""
    body = (_TEMPLATES_DIR / _GUIDE_TEMPLATE).read_text(encoding="utf-8")
    path = Path(workspace) / _GUIDE_NAME
    existed = path.exists()
    if existed and path.read_text(encoding="utf-8") == body:
        return "unchanged"
    path.write_text(body, encoding="utf-8", newline="\n")
    return "replaced" if existed else "created"


def history_summary(dsh_home: str, workspace: str) -> dict[str, Any]:
    """Read-only discovery of the workspace's past session logs (D-3).

    Reports how many historical session logs exist for this workspace and
    the exact one-command import path — init itself stays zero-daemon and
    read-only here; the import writes only once the memory service is up.
    """
    from .dsh_import import find_session_logs
    logs = find_session_logs(dsh_home, workspace)
    return {
        "found_logs": len(logs),
        "logs": logs,
        "import_command": ("hippo-mcp import-dsh --workspace " +
                           json_quote(workspace)),
    }


def json_quote(value: str) -> str:
    """Shell-quoting helper that survives Windows backslash paths."""
    return '"' + value + '"'


def run_init_dsh(workspace: str, *, dsh_home: str = "", profile: str = "web",
                 name: str = "", root: str = "", user_lib: str = "",
                 agents: bool = True, guide: bool = True,
                 python_cmd: str = "",
                 create_storage: bool = True) -> dict[str, Any]:
    """Execute the DSH onboarding and return a machine-readable report.

    Args:
        workspace: the DSH workspace directory to bind (created if missing —
            a companion gets a fresh home, not a shared coding workspace).
        dsh_home: harness home override (default: ``DSH_HOME`` env, else
            ``~/.dsh``).
        profile: profile name whose patch receives the MCP entry.
        name / root / user_lib / create_storage: forwarded to
            :func:`init_project.run_init`.
        agents: write the persona + discipline block into ``AGENTS.md``.
        guide: write ``memory-setup.md``.
        python_cmd: interpreter for the MCP spawn (default: the running one).
        pythonpath override is derived from this package's location, so both
            pip installs and source checkouts produce a working entry.

    Idempotent end to end; see the module docstring for the four steps.
    """
    workspace = normpath(os.path.abspath(os.path.expanduser(workspace)))
    os.makedirs(workspace, exist_ok=True)
    dsh_home = (os.path.expanduser(dsh_home) if dsh_home
                else _dsh_home_default())
    if not os.path.isdir(dsh_home):
        os.makedirs(os.path.join(dsh_home, "profiles", profile), exist_ok=True)

    patch = _patch_path(dsh_home, profile)
    user_lib_final = os.path.expanduser(user_lib) if user_lib else ""

    # Step 1+2: library + workspace binding (same primitives as init --project).
    base_report = run_init(workspace, name=name, root=root, user_lib=user_lib,
                           agents=False, create_storage=create_storage)
    user_lib_resolved = user_lib_final or os.path.join(
        os.path.expanduser("~"), ".hippocampus_mcp", "user")

    # Step 3: MCP client patch entry.
    if python_cmd:
        python_cmd_resolved = python_cmd
    else:
        python_cmd_resolved = sys.executable
    pythonpath = str(Path(__file__).resolve().parents[1])
    patch_status = _append_patch_entry(
        patch, python_cmd=python_cmd_resolved, workspace=workspace,
        pythonpath=pythonpath, user_lib=user_lib_resolved,
        emb_model=_DSH_EMB_MODEL)

    report: dict[str, Any] = {
        "mode": "dsh",
        "workspace": workspace,
        "dsh_home": dsh_home,
        "profile": profile,
        "patch": patch_status,
        "patch_path": str(patch),
        "library": {k: base_report.get(k) for k in ("status", "name", "target")},
    }

    # Step 4: persona + discipline block (own template, own markers).
    report["agents"] = "skipped"
    if agents:
        block = (_TEMPLATES_DIR / "agents_snippet_dsh.md").read_text(encoding="utf-8").strip()
        report["agents"] = write_agents_snippet(
            workspace, block=block, begin_mark=_AGENTS_BEGIN, end_mark=_AGENTS_END)

    # Step 5: self-install guide for the companion.
    report["guide"] = "skipped"
    if guide:
        report["guide"] = _write_guide(workspace)

    report["next"] = ("select the workspace in the DSH UI; the MCP entry "
                      "hot-reloads (no restart needed); verify with "
                      "`library_whoami` in a companion session")

    # Step 6: embedder-backend self-check (read-only lookahead; no weights
    # load). A fresh machine without artifacts would silently serve the hash
    # embedder -- surface that here, with the exact repair command, instead of
    # letting the fish discover "retrieval is just worse" weeks later.
    try:
        from .engine.embeddings import prospective_embedder_backend
        backend = prospective_embedder_backend(_DSH_EMB_MODEL)
    except Exception as exc:                     # noqa: BLE001 - never fail onboarding
        backend = f"unknown ({type(exc).__name__})"
    report["embedder_backend"] = backend
    if backend == "hash":
        report["next"] = (report["next"] + " IMPORTANT: the embedding model is "
                          "missing on this machine (backend would degrade to "
                          "hash). Run `hippo-mcp models --fetch` now "
                          "(bundle download; no torch needed) before the "
                          "first session.")

    # Step 7 (D-3): history discovery — does this workspace have PAST session
    # logs worth importing? Read-only; the actual import is a separate
    # command that needs the memory service running (zero-daemon init).
    try:
        report["history"] = history_summary(dsh_home, workspace)
        if report["history"]["found_logs"]:
            report["next"] = (report["next"] + " HISTORY: "
                              + str(report["history"]["found_logs"])
                              + " past session log(s) found for this workspace"
                              + " — import them with `"
                              + report["history"]["import_command"]
                              + "` (requires the memory service running).")
    except Exception as exc:                     # noqa: BLE001 - never fail onboarding
        report["history"] = {"error": f"{type(exc).__name__}: {exc}"}
    return report
