# -*- coding: utf-8 -*-
"""Portable whole-library zip archives (export / import across machines).

Scope: this is a **whole library snapshot**, not the text-only
``export_memories`` tool. It carries L1 memories, vectors, derived layers and
every sidecar (task directory, lexicon, ledger, maps, LNN checkpoint), so
recall on the destination machine behaves like the source machine.

Hard rules (all learned from the surrounding persistence code):

1. **Backend is a property of the archive, not of the destination config.**
   The library directory decides the backend (``hippocampus.db`` wins over
   ``memories.json``). A JSON library opened by a sqlite-configured process
   would otherwise look empty -- ``store.backend.needs_migration``.
2. **SQLite needs a consistent copy, not a file copy.** A live daemon keeps
   WAL files and open connections; ``VACUUM INTO`` on an independent
   connection is the only snapshot method already proven in this codebase.
   JSON files are copied with a before/after stamp check because each layer
   is written via tmp + ``os.replace`` and can change mid-copy.
3. **Validate before writing anything.** Import reads and hashes every member
   of the zip, rejects path traversal / unlisted members / version mismatch,
   and only then extracts into a sibling staging directory. A failed import
   leaves no half-written target.
4. **Never silently overwrite.** Export refuses an existing archive unless
   forced; import refuses an existing target unless ``replace=True``, and a
   replaced target is moved to a timestamped sibling backup first.
5. **Absolute paths are machine-local.** ``library.json``'s id is derived
   from the normalized path and ``directory.json``'s ``path`` is its primary
   key; both are rewritten on import. Entries pointing at *other* libraries
   cannot be remapped and are counted in the result instead of being dropped.

No environment reads live here: this module belongs to the constrained
``library`` layer and receives every knob as an explicit argument.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import zipfile
from typing import Any

from . import MANIFEST_NAME, library_id_for, normpath
from .task_directory import DIRECTORY_NAME, load_directory, save_directory

__all__ = [
    "FORMAT",
    "VERSION",
    "ArchiveError",
    "export_archive",
    "inspect_archive",
    "import_archive",
    "safe_name",
]

FORMAT = "hippo-memory-archive"
VERSION = 1
MANIFEST_MEMBER = "hippo-archive.json"
DATA_PREFIX = "data/"

_DB_NAME = "hippocampus.db"
_MEMORIES_NAME = "memories.json"

# Transient / machine-local files. ``.bak`` files are earlier snapshots, not
# part of the current state; WAL/SHM are useless without their live db and are
# replaced by the VACUUM copy; ``.tmp`` / ``.corrupt-*`` are in-flight writes
# or quarantined failures.
_SKIP_DIR_RE = re.compile(r"(\.bak|\.snap|\.corrupt|__pycache__)", re.I)


class ArchiveError(ValueError):
    """Invalid archive, unsafe member path, or a refused overwrite."""


def _skip_rel(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    for part in parts:
        low = part.lower()
        if low in ("", ".", "..", ".ds_store", "thumbs.db"):
            return True
        if _SKIP_DIR_RE.search(low):
            return True
        if low.endswith(("-wal", "-shm", ".lock")):
            return True
        if ".tmp" in low or low.endswith(".bak"):
            return True
        if MANIFEST_MEMBER.lower() == low:
            return True
    return False


def _collect_files(directory: str) -> list[str]:
    """All regular files of a library, as sorted posix relative paths."""
    found: list[str] = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if not _skip_rel(d))
        for name in sorted(files):
            full = os.path.join(root, name)
            if not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, directory).replace(os.sep, "/")
            if not _skip_rel(rel):
                found.append(rel)
    return sorted(found)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_stable(src: str, dst: str, attempts: int = 3) -> None:
    """Copy a live file, retrying if it changed underneath us."""
    for _ in range(attempts):
        before = os.stat(src)
        shutil.copyfile(src, dst)
        after = os.stat(src)
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            return
    raise ArchiveError(f"source file kept changing while snapshotting: {src}")


def _snapshot(directory: str, stage: str, backend: str, rels: list[str]) -> None:
    for rel in rels:
        src = os.path.join(directory, rel.replace("/", os.sep))
        dst = os.path.join(stage, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if backend == "sqlite" and rel == _DB_NAME:
            # Independent connection: VACUUM cannot run inside another
            # connection's transaction, and WAL must not be copied by hand.
            con = sqlite3.connect(src, timeout=30)
            try:
                con.execute("VACUUM INTO ?", (dst,))
            finally:
                con.close()
        else:
            _copy_stable(src, dst)


def _probe(stage: str, backend: str) -> tuple[int, bool]:
    """(memory count, fts5 present) from the *snapshot*, never from env."""
    if backend == "sqlite":
        con = sqlite3.connect(os.path.join(stage, _DB_NAME))
        try:
            tables = {str(row[0]) for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "memories" not in tables:
                raise ArchiveError("archived sqlite database has no memories table")
            count = int(con.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            return count, "memories_fts" in tables
        finally:
            con.close()
    path = os.path.join(stage, _MEMORIES_NAME)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ArchiveError(f"archived memories file is unreadable: {path} ({exc})") from exc
    records = data.get("records") if isinstance(data, dict) else data
    if isinstance(records, dict):
        records = list(records.values())
    if not isinstance(records, list):
        raise ArchiveError(f"archived memories file has an unknown shape: {path}")
    return len(records), False


def _read_meta(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def export_archive(directory: str, out: str, *, force: bool = False,
                   name: str | None = None) -> dict[str, Any]:
    """Export one library to a zip archive. Returns a summary dict.

    Raises ``ArchiveError`` for a non-library directory, an existing output
    (unless *force*), or an unreadable memories store. The output itself is
    written through a temp file + ``os.replace``.
    """
    directory = normpath(os.path.expanduser(str(directory)))
    out = os.path.abspath(os.path.expanduser(str(out)))
    backend = ("sqlite" if os.path.isfile(os.path.join(directory, _DB_NAME))
               else "json" if os.path.isfile(os.path.join(directory, _MEMORIES_NAME))
               else "")
    if not backend:
        raise ArchiveError(f"not a hippocampus library (no {_DB_NAME} / "
                           f"{_MEMORIES_NAME}): {directory}")
    if os.path.exists(out) and not force:
        raise ArchiveError(f"archive already exists: {out} (pass --force to replace it)")
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rels = _collect_files(directory)
    if not rels:
        raise ArchiveError(f"library directory is empty: {directory}")
    stage = tempfile.mkdtemp(prefix=".hippo-export-", dir=parent or None)
    tmp_zip = f"{out}.tmp.{os.getpid()}"
    try:
        _snapshot(directory, stage, backend, rels)
        count, fts = _probe(stage, backend)
        meta = _read_meta(os.path.join(stage, MANIFEST_NAME))
        files: list[dict[str, Any]] = []
        for rel in rels:
            full = os.path.join(stage, rel.replace("/", os.sep))
            files.append({"path": DATA_PREFIX + rel,
                          "size": os.path.getsize(full),
                          "sha256": _sha256(full)})
        manifest: dict[str, Any] = {
            "format": FORMAT,
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": {
                "path": directory,
                "name": str(name or meta.get("name") or
                            os.path.basename(directory.rstrip("\\/"))),
                "backend": backend,
                "fts5": bool(fts),
                "embedder_model": str(meta.get("embedder_model") or ""),
                "memory_count": count,
                "library_id": str(meta.get("library_id") or library_id_for(directory)),
            },
            "files": files,
            "total_bytes": sum(int(f["size"]) for f in files),
        }
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST_MEMBER,
                        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            for item in files:
                zf.write(os.path.join(stage, item["path"][len(DATA_PREFIX):]
                                      .replace("/", os.sep)),
                         arcname=item["path"])
        os.replace(tmp_zip, out)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if os.path.exists(tmp_zip):
            try:
                os.remove(tmp_zip)
            except OSError:
                pass
    return {"archive": out, "backend": backend, "memory_count": count,
            "files": len(files), "bytes": manifest["total_bytes"],
            "name": manifest["source"]["name"], "fts5": fts}


def inspect_archive(archive: str) -> dict[str, Any]:
    """Validate every member (format, traversal, listing, sha256) without
    writing anything. Returns ``{"manifest": ..., "members": [...]}``.
    """
    archive = os.path.abspath(os.path.expanduser(str(archive)))
    if not os.path.isfile(archive):
        raise ArchiveError(f"archive not found: {archive}")
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"not a zip archive: {archive} ({exc})") from exc
    with zf:
        try:
            raw = zf.read(MANIFEST_MEMBER)
        except KeyError as exc:
            raise ArchiveError(f"archive lacks {MANIFEST_MEMBER}: {archive}") from exc
        try:
            manifest = json.loads(raw)
        except ValueError as exc:
            raise ArchiveError(f"archive manifest is not valid JSON ({exc})") from exc
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise ArchiveError(f"not a hippo memory archive: {archive}")
        if manifest.get("version") != VERSION:
            raise ArchiveError(f"unsupported archive version {manifest.get('version')!r} "
                               f"(supported: {VERSION})")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ArchiveError("archive manifest lists no data files")
        members = set(zf.namelist())
        expected: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise ArchiveError("bad file entry in archive manifest")
            arc = item.get("path")
            if not isinstance(arc, str) or not arc.startswith(DATA_PREFIX):
                raise ArchiveError(f"member outside {DATA_PREFIX}: {arc!r}")
            rel = arc[len(DATA_PREFIX):]
            parts = rel.split("/")
            drive = os.path.splitdrive(parts[0])[0]
            if (not rel or "\\" in rel or drive or
                    any(p in ("", ".", "..") for p in parts)):
                raise ArchiveError(f"unsafe member path in archive: {arc!r}")
            if arc in expected:
                raise ArchiveError(f"duplicate member in archive manifest: {arc!r}")
            expected.add(arc)
            if arc not in members:
                raise ArchiveError(f"archive member missing: {arc!r}")
            info = zf.getinfo(arc)
            if isinstance(item.get("size"), int) and info.file_size != item["size"]:
                raise ArchiveError(f"member size mismatch: {arc!r}")
            digest = hashlib.sha256()
            with zf.open(arc) as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item.get("sha256"):
                raise ArchiveError(f"checksum mismatch for member: {arc!r}")
        extras = [n for n in members
                  if n.startswith(DATA_PREFIX) and not n.endswith("/")
                  and n not in expected]
        if extras:
            raise ArchiveError(f"archive contains unlisted member(s): {extras[:3]}")
    return {"manifest": manifest, "members": sorted(expected)}


def _rewrite_paths(stage: str, target: str, manifest: dict[str, Any]) -> dict[str, int]:
    """Rebind this library's path-dependent identity after extraction."""
    manifest_path = os.path.join(stage, MANIFEST_NAME)
    meta = _read_meta(manifest_path)
    if meta:
        meta["library_id"] = library_id_for(target)
        with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    source = str((manifest.get("source") or {}).get("path") or "")
    data, issue = load_directory(stage)
    if issue:
        raise ArchiveError(f"archived {DIRECTORY_NAME} is unreadable ({issue}); "
                           "refusing to import a broken task directory")
    entries = list(data.get("entries") or [])
    remapped = unmapped = 0
    if source and entries:
        key = normpath(source)
        new_id = library_id_for(target)
        for entry in entries:
            if normpath(str(entry.get("path") or "")) == key:
                entry["path"] = target
                entry["library_id"] = new_id
                remapped += 1
            else:
                # Other libraries are out of scope: guessing a replacement
                # path would silently point this task at the wrong memory.
                unmapped += 1
        save_directory(stage, {"schema": data.get("schema"), "entries": entries})
    return {"directory_entries": len(entries),
            "remapped_entries": remapped, "unmapped_entries": unmapped}


def safe_name(raw: str) -> str:
    """Turn an archive's library name into a safe directory basename."""
    name = re.sub(r"[\\/:*?\"<>|]", "_", str(raw or "").strip())
    name = name.strip(". ") or "imported"
    return name[:80]


def import_archive(archive: str, target: str, *, replace: bool = False,
                   bind_workspace: str = "", user_lib: str = "") -> dict[str, Any]:
    """Extract a validated archive into *target*, atomically and reversibly.

    Existing targets are refused unless *replace*; in that case the old
    directory is moved to ``<target>.bak-<timestamp>`` before the staged copy
    takes its place. ``bind_workspace`` registers the imported library for a
    workspace through the managed task-directory write primitive.
    """
    checked = inspect_archive(archive)
    manifest = checked["manifest"]
    target = normpath(os.path.expanduser(str(target)))
    if os.path.exists(target) and not replace:
        raise ArchiveError(f"target already exists: {target} (pass --replace to back it up first)")
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=".hippo-import-", dir=parent)
    backup = ""
    try:
        with zipfile.ZipFile(os.path.abspath(os.path.expanduser(str(archive)))) as zf:
            for arc in checked["members"]:
                rel = arc[len(DATA_PREFIX):]
                dst = os.path.join(stage, rel.replace("/", os.sep))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with zf.open(arc) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
        for item in manifest["files"]:
            rel = item["path"][len(DATA_PREFIX):].replace("/", os.sep)
            if _sha256(os.path.join(stage, rel)) != item["sha256"]:
                raise ArchiveError(f"extracted file failed checksum: {item['path']!r}")
        info = _rewrite_paths(stage, target, manifest)
        if os.path.exists(target):
            backup = f"{target}.bak-{time.strftime('%Y%m%d_%H%M%S')}"
            os.replace(target, backup)
        try:
            os.replace(stage, target)
        except OSError:
            if backup and os.path.exists(backup):
                os.replace(backup, target)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    bound = ""
    if bind_workspace:
        from .task_directory import upsert_entry
        home = os.path.abspath(os.path.expanduser(
            user_lib or os.path.join("~", ".hippocampus_mcp", "user")))
        os.makedirs(home, exist_ok=True)
        task = os.path.basename(normpath(bind_workspace).rstrip("\\/")) or "workspace"
        upsert_entry(home, {"task": task, "path": target,
                            "workspace_root": os.path.abspath(
                                os.path.expanduser(bind_workspace)),
                            "access": "full"})
        bound = bind_workspace

    source = manifest.get("source") or {}
    return {"archive": os.path.abspath(str(archive)), "target": target,
            "backend": str(source.get("backend") or ""),
            "memory_count": source.get("memory_count"),
            "files": len(manifest["files"]), "bytes": manifest.get("total_bytes", 0),
            "name": str(source.get("name") or ""), "backup": backup,
            "bound_workspace": bound, **info}
