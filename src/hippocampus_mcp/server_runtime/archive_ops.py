# -*- coding: utf-8 -*-
"""Agent-facing whole-library archive actions (dispatcher-only, zero env reads).

These two async actions are intentionally **not** ``@server.tool`` decorators:
they are low-frequency admin operations reached through ``hippo_help`` ->
``hippo_extra``, so the declared 46-tool surface (and the many guards that
pin it) stays unchanged. The re-export in ``server.py`` puts them in the
live ``vars(server)`` namespace that ``_dispatch_map`` reads.

Safety boundary: the input archive path may be anywhere the user names, but
**every path the agent writes is confined** -- exports go to the agent's
current workspace (or an explicitly allowed root), imports go under a
configured library root, and existing libraries are never overwritten.
No environment variable is read here: roots and the user library default are
obtained through the library layer's configured accessors.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from hippocampus_mcp.library import default_roots, normpath
from hippocampus_mcp.library.archive import (
    ArchiveError,
    export_archive,
    import_archive,
    inspect_archive,
    safe_name,
)
from hippocampus_mcp.library.task_directory import load_directory

#: The conventional user-level library, matching the CLI import default.
_USER_LIB = os.path.join("~", ".hippocampus_mcp", "user")


def _abs(path: str) -> str:
    """Absolute, user-expanded path (no existence check)."""
    return os.path.abspath(os.path.expanduser(str(path or "")))


def _under(path: str, root: str) -> bool:
    """Containment check that also resolves symlinks (no lexical bypass)."""
    base = os.path.realpath(_abs(root))
    child = os.path.realpath(_abs(path))
    try:
        return os.path.commonpath([child, base]) == base
    except (ValueError, OSError):
        return False


def _roots() -> list[str]:
    """Configured library roots, normalized for comparisons."""
    return [_abs(r) for r in default_roots() if str(r).strip()]


def _user_lib() -> str:
    """Path of the user-level library holding the task directory sidecar."""
    return _abs(_USER_LIB)


def _binding_conflict(workspace: str, target: str) -> str:
    """Why this workspace cannot be rebound, or an empty string if it can.

    A bad sidecar self-reports (never silently treated as empty): the import
    itself still succeeds, but the binding is skipped and the reason is
    returned to the agent.
    """
    data, issue = load_directory(_user_lib())
    if issue:
        return f"task directory unreadable ({issue})"
    for entry in data.get("entries") or []:
        if normpath(str(entry.get("workspace_root") or "")) != normpath(workspace):
            continue
        existing = normpath(str(entry.get("path") or ""))
        if existing and existing != normpath(target):
            return str(entry.get("task") or "another library")
    return ""


def export_for_agent(source: str, out_dir: str = "", name: str = "") -> dict[str, Any]:
    """Snapshot *source* into the agent workspace; never overwrite an archive."""
    workspace = _abs(os.getcwd())
    target_dir = _abs(out_dir) if out_dir else workspace
    if not (_under(target_dir, workspace)
            or any(_under(target_dir, root) for root in _roots())):
        raise ArchiveError("out_dir must be inside the agent workspace or a "
                           "configured library root")
    if not os.path.isdir(target_dir):
        raise ArchiveError(f"out_dir is not an existing directory: {target_dir}")
    base = safe_name(name or os.path.basename(normpath(source)) or "memory")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = ""
    for suffix in range(1, 100):
        candidate = os.path.join(
            target_dir, f"hippo-{base}-{stamp}{'' if suffix == 1 else f'-{suffix}'}.zip")
        if not os.path.exists(candidate):
            out = candidate
            break
    if not out:
        raise ArchiveError("no free archive name in this second (1..99 taken)")
    result = export_archive(source, out, name=(name or None))
    result["ok"] = True
    result["agent_hint"] = ("Tell the user the absolute archive path above; "
                            "the zip contains full memory content, so keep it private.")
    result["scope_notice"] = (
        "This archive covers Hippo memories only (text + vectors). A companion "
        "agent's persona, growth ledger and event stream (e.g. dsh-fish) are "
        "not included; move those with that agent's own backup/transfer.")
    return result


def import_for_agent(archive: str, into: str = "", name: str = "",
                     bind_workspace: str = "") -> dict[str, Any]:
    """Validate then import *archive* into a new library under a configured root."""
    checked = inspect_archive(archive)
    source = checked["manifest"].get("source") or {}
    roots = _roots()
    if not roots:
        raise ArchiveError("no library root is configured for the import target")
    if into:
        target = _abs(into)
        if not any(_under(target, root) for root in roots):
            raise ArchiveError("into must be inside a configured library root")
    else:
        target = os.path.join(roots[0],
                              safe_name(name or source.get("name") or "imported"))
    if os.path.exists(target):
        raise ArchiveError(f"import target already exists: {target} "
                           "(agent import never overwrites a library)")
    workspace = _abs(bind_workspace) if bind_workspace else _abs(os.getcwd())
    conflict = _binding_conflict(workspace, target)
    result = import_archive(
        archive, target,
        bind_workspace=("" if conflict else workspace),
        user_lib=("" if conflict else _user_lib()),
    )
    result["ok"] = True
    result["binding"] = {
        "workspace": "" if conflict else workspace,
        "status": "skipped" if conflict else "bound",
        "reason": conflict,
    }
    result["agent_hint"] = ("The imported library is ready. If binding was skipped, "
                            "ask the user before switching this workspace's library.")
    return result


async def archive_export(out_dir: str = "", library: str = "", name: str = "") -> str:
    """When the user says "pack my memories into a zip so I can take them to another machine" -> this tool.
Exports the current (or named) memory library into a zip the agent can hand back by path.
Nothing is overwritten; the archive contains full memory content.

Args:
    out_dir: Destination directory; omitted means this agent's current working directory.
             Must stay inside that workspace or a configured library root.
    library: Library name/path; omitted means the library this session currently uses.
    name: Optional archive label; omitted derives one from the library name.
    """
    from hippocampus_mcp import server as srv

    try:
        source = srv._library_admin_service().describe(library=library)["path"]
        return json.dumps(export_for_agent(source, out_dir, name),
                          ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 -- tool boundary: self-report, never half-write silently
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


async def archive_import(archive: str, into: str = "", name: str = "",
                         bind_workspace: str = "") -> str:
    """When the user says "here is my memory zip path, restore it on this machine" -> this tool.
Validates the whole archive before writing, creates a new library under a configured root,
and binds the agent's current workspace unless that workspace already points elsewhere.

Args:
    archive: Path to the zip archive the user provided (or the agent exported earlier).
    into: Explicit new library directory; must be inside a configured library root.
          Omitted = a fresh directory named after the archived library.
    name: Optional name for the fresh library directory.
    bind_workspace: Workspace to bind to the imported library; omitted means the agent's cwd.
    """
    try:
        return json.dumps(import_for_agent(archive, into, name, bind_workspace),
                          ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 -- tool boundary: self-report, never half-write silently
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
