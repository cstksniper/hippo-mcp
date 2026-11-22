# -*- coding: utf-8 -*-
"""Lean-model fetch: download onnx model bundles without torch or any HF client.

Why this module exists
----------------------
The lean core ships no ``sentence_transformers`` / ``huggingface_hub``, so a
fresh install physically cannot obtain embedding-model artifacts: the old
``models --fetch`` imported ``sentence_transformers`` (an optional ``[torch]``
extra), and a fresh machine silently degraded to the hash embedder. This
module is the fix: a **stdlib-only** downloader (``urllib.request``) that
fetches pre-exported onnx bundles from the public ``model-bundles`` GitHub
release, verifies sha256, and unpacks them into the regular model cache roots
the embeddings resolver already scans.

URL list order (first success wins):
1. ``HIPPOCAMPUS_FETCH_URLS`` -- comma-separated templates with an ``{asset}``
   placeholder (self-hosted / mirror override; highest priority);
2. the GitHub release direct URL (overseas users, proxied users);
3. the ``ghproxy.net`` mirror of the same URL (mainland-China fallback where
   ``github.com`` itself is blocked but the asset CDN/mirrors are reachable).

Reliability contract:
- HTTP ``Range`` resume via ``.part`` files (interrupted downloads continue);
- bundle sha256 verified against the release's ``SHA256SUMS.txt`` **and** the
  per-file ``MANIFEST.sha256`` inside the zip (both must pass);
- zip-slip guard on extraction; a probe load runs after unpack so failures
  surface here, not at first retrieval.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

__all__ = [
    "fetch_model_bundle",
    "asset_name_for_model",
    "available_bundles",
    "FetchError",
]

#: Rolling release tag holding the bundles (see .github/workflows/model-bundles.yml).
BUNDLE_RELEASE_TAG = "model-bundles"
#: Public repo that publishes the release.
BUNDLE_REPO = "cstksniper/hippocampus-mcp"
#: api.github.com asset route: resolves to the objects CDN without ever
#: touching ``github.com:443`` — on some mainland lines github.com is blocked
#: while api.github.com + the objects CDN pass (hit 2026-11-22). This route
#: is therefore tried before the release-download URL.
BUNDLE_API_BASE = f"https://api.github.com/repos/{BUNDLE_REPO}"
#: Mirrors are third-party services: they are a fallback, never a trust root
#: (sha256 verification is the trust root).
MIRROR_PREFIX = "https://ghproxy.net/https://github.com"
#: Env override: comma-separated URL templates, ``{asset}`` placeholder.
ENV_FETCH_URLS = "HIPPOCAMPUS_FETCH_URLS"
#: Known published bundles (short model name -> upstream repo id for messages).
AVAILABLE_BUNDLES = {
    "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}
#: File published next to the bundles on the release.
SHA256SUMS_ASSET = "SHA256SUMS.txt"
#: Bundle layout version (written by scripts/build_model_bundle.py in its
#: MANIFEST header; fetch refuses other versions rather than guessing).
BUNDLE_VERSION = 1
#: Download tuning.
CHUNK_SIZE = 1024 * 1024
CONNECT_TIMEOUT = 30
MAX_ATTEMPTS_PER_URL = 3
PROGRESS_STEP_BYTES = 32 * 1024 * 1024


class FetchError(RuntimeError):
    """Raised when a bundle cannot be downloaded, verified, or unpacked."""


def asset_name_for_model(model_name: str) -> str:
    """Release asset for a model id: last path segment + ``-onnx.zip``.

    ``BAAI/bge-base-zh-v1.5`` -> ``bge-base-zh-v1.5-onnx.zip``; a bare short
    name passes through unchanged. This matches the cache-root layout the
    embeddings resolver scans (``<root>/<last segment>/``).
    """
    short = os.path.basename(model_name.rstrip("/\\"))
    if not short:
        raise FetchError(f"cannot derive a bundle name from model id {model_name!r}")
    return f"{short}-onnx.zip"


def fetch_urls(asset: str) -> list[str]:
    """Ordered candidate URLs for one asset (env override > direct > mirror).

    Kept for display/diagnostics; the downloader uses :func:`asset_candidates`
    which additionally resolves the api.github.com asset route.
    """
    direct = (f"https://github.com/{BUNDLE_REPO}/releases/download/"
              f"{BUNDLE_RELEASE_TAG}/{asset}")
    urls: list[str] = []
    env = os.environ.get(ENV_FETCH_URLS, "").strip()
    if env:
        for tpl in env.split(","):
            tpl = tpl.strip()
            if tpl:
                urls.append(tpl.replace("{asset}", asset))
    urls.append(direct)
    urls.append(f"{MIRROR_PREFIX}/{BUNDLE_REPO}/releases/download/"
                f"{BUNDLE_RELEASE_TAG}/{asset}")
    return urls


def _resolve_api_asset_url(asset: str) -> str | None:
    """Resolve one asset's api.github.com URL via the release listing.

    Returns None (instead of raising) when the API is unreachable, rate-limited
    or the asset is absent — the candidate list simply moves on to the next
    route. Unauthenticated calls are fine (60/h per IP; a fetch makes two).
    """
    try:
        req = urllib.request.Request(
            f"{BUNDLE_API_BASE}/releases/tags/{BUNDLE_RELEASE_TAG}",
            headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for info in data.get("assets", []):
            if info.get("name") == asset and info.get("url"):
                return info["url"]
    except Exception as exc:  # noqa: BLE001 - a dead API route is not fatal
        print(f"  [fetch] api.github.com route unavailable ({type(exc).__name__}); "
              "falling back to direct/mirror URLs", flush=True)
    return None


def asset_candidates(asset: str) -> list[tuple[str, dict]]:
    """Ordered ``(url, headers)`` candidates for one asset.

    Order: ``HIPPOCAMPUS_FETCH_URLS`` env override > api.github.com asset
    route (bypasses the blocked github.com:443, lands on the objects CDN) >
    github.com release-download direct > ghproxy.net mirror. sha256
    verification is the trust root regardless of the route used.
    """
    cands: list[tuple[str, dict]] = []
    env = os.environ.get(ENV_FETCH_URLS, "").strip()
    if env:
        for tpl in env.split(","):
            tpl = tpl.strip()
            if tpl:
                cands.append((tpl.replace("{asset}", asset), {}))
    api_url = ""
    try:
        api_url = _resolve_api_asset_url(asset) or ""
    except Exception as exc:  # noqa: BLE001 - a dead API route is not fatal
        print(f"  [fetch] api.github.com route unavailable ({type(exc).__name__}); "
              "falling back to direct/mirror URLs", flush=True)
    if api_url:
        cands.append((api_url, {"Accept": "application/octet-stream"}))
    cands.append((f"https://github.com/{BUNDLE_REPO}/releases/download/"
                  f"{BUNDLE_RELEASE_TAG}/{asset}", {}))
    cands.append((f"{MIRROR_PREFIX}/{BUNDLE_REPO}/releases/download/"
                  f"{BUNDLE_RELEASE_TAG}/{asset}", {}))
    return cands


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK_SIZE)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _parse_sha256sums(text: str) -> dict[str, str]:
    """``<sha256>  <name>`` lines; comment lines ignored.

    Names are kept verbatim (stripped of whitespace and the ``*`` binary-mode
    marker): release-level SHA256SUMS entries are flat asset names, while the
    inner MANIFEST carries relative paths like ``onnx/model.onnx`` that must
    survive as-is.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip().lstrip("*").strip()] = parts[0].lower()
    return out


def _download(candidates: list[tuple[str, dict]], dest: str, *,
              expected_sha256: str = "", progress_label: str = "") -> None:
    """Download one file trying the (url, headers) candidates in order, with
    Range resume.

    ``.part`` carries partial state across attempts; a server that ignores
    Range (answers 200 instead of 206) restarts cleanly. Verification happens
    once, after the final byte lands.
    """
    part = dest + ".part"
    for url, headers in candidates:
        for attempt in range(1, MAX_ATTEMPTS_PER_URL + 1):
            try:
                have = os.path.getsize(part) if os.path.isfile(part) else 0
                req = urllib.request.Request(url, headers=headers or {})
                if have > 0:
                    req.add_header("Range", f"bytes={have}-")
                started = time.time()
                with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as resp:
                    status = getattr(resp, "status", 200)
                    if have > 0 and status != 206:
                        # Server ignored the Range header: restart from zero.
                        have = 0
                    mode = "ab" if have > 0 else "wb"
                    total = None
                    length = resp.headers.get("Content-Length")
                    if length:
                        total = int(length) + have
                    next_report = ((have // PROGRESS_STEP_BYTES) + 1) * PROGRESS_STEP_BYTES
                    with open(part, mode) as f:
                        done = have
                        while True:
                            block = resp.read(CHUNK_SIZE)
                            if not block:
                                break
                            f.write(block)
                            done += len(block)
                            if total and done >= next_report:
                                pct = done * 100 // total
                                rate = done / max(time.time() - started, 0.001) / (1024 * 1024)
                                print(f"  [{progress_label or os.path.basename(dest)}] "
                                      f"{pct:3d}% ({done / (1024 * 1024):.0f}/{total / (1024 * 1024):.0f} MB, "
                                      f"{rate:.1f} MB/s)", flush=True)
                                next_report += PROGRESS_STEP_BYTES
                if expected_sha256:
                    got = _sha256_file(part)
                    if got != expected_sha256:
                        # Wrong bytes (truncated/mutated): drop the part so the
                        # next attempt starts clean instead of resuming garbage.
                        os.remove(part)
                        raise FetchError(
                            f"sha256 mismatch for {url}: expected {expected_sha256}, got {got}")
                os.replace(part, dest)
                return
            except FetchError:
                raise
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
                print(f"  [fetch] {url} attempt {attempt}/{MAX_ATTEMPTS_PER_URL} failed: {exc}",
                      flush=True)
                time.sleep(min(2 ** attempt, 8))
    msg = f"all candidate URLs failed for {os.path.basename(dest)}"
    if os.path.isfile(part):
        msg += f"; partial file kept at {part} for a resume attempt"
    raise FetchError(msg)


def _verify_inner_manifest(zip_path: str, dest_dir: str) -> None:
    """Verify the per-file ``MANIFEST.sha256`` carried inside the bundle."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if "MANIFEST.sha256" not in names:
            raise FetchError("bundle has no MANIFEST.sha256 (unknown/corrupt bundle)")
        text = zf.read("MANIFEST.sha256").decode("utf-8")
        header = next(
            (ln for ln in text.splitlines()
             if ln.startswith("# hippocampus model bundle")), "")
        try:
            version = int(header.rsplit(" v", 1)[-1])
        except ValueError:
            version = -1
        if version != BUNDLE_VERSION:
            raise FetchError(f"unsupported bundle format version: {header!r}")
        expected = _parse_sha256sums(text)
        for rel, want in expected.items():
            if rel == "MANIFEST.sha256":
                continue  # self-hash is not carried (chicken-and-egg)
            path = os.path.join(dest_dir, rel.replace("/", os.sep))
            if not os.path.isfile(path):
                raise FetchError(f"bundle member missing after extract: {rel}")
            got = _sha256_file(path)
            if got != want:
                raise FetchError(f"inner manifest mismatch for {rel}: expected {want}, got {got}")


def _safe_extract(zip_path: str, dest_dir: str) -> None:
    """Extract with a zip-slip guard: every member must land inside dest_dir."""
    dest_real = os.path.realpath(dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = os.path.realpath(os.path.join(dest_dir, info.filename))
            if not (target == dest_real or target.startswith(dest_real + os.sep)):
                raise FetchError(f"refusing unsafe zip member: {info.filename!r}")
        zf.extractall(dest_dir)


def fetch_model_bundle(model_name: str, cache_root: str, *, probe: bool = True) -> dict:
    """Fetch, verify and unpack the bundle for ``model_name``; return a report.

    The result lands in ``<cache_root>/<last-segment>/`` -- a location the
    embeddings resolver already scans, so the next attach picks it up without
    any configuration change.
    """
    from .onnx_embedder import find_onnx_artifacts  # local import: probe-only dependency

    asset = asset_name_for_model(model_name)
    short = os.path.basename(model_name.rstrip("/\\"))
    dest_dir = os.path.join(cache_root, short)

    if os.path.isdir(dest_dir) and find_onnx_artifacts(dest_dir) is not None:
        return {"status": "already_local", "model": model_name, "dir": dest_dir,
                "backend": "onnx"}

    tmp_dir = tempfile.mkdtemp(prefix="hippo-fetch-")
    report: dict = {"status": "fetched", "model": model_name, "dir": dest_dir,
                    "backend": "onnx"}
    try:
        # 1) SHA256SUMS (same candidate policy; tiny). Missing sums => refuse
        #    rather than trusting an unverified 237MB payload.
        sums_path = os.path.join(tmp_dir, SHA256SUMS_ASSET)
        _download(asset_candidates(SHA256SUMS_ASSET), sums_path,
                  progress_label=SHA256SUMS_ASSET)
        sums = _parse_sha256sums(open(sums_path, "r", encoding="utf-8").read())
        if asset not in sums:
            raise FetchError(
                f"{SHA256SUMS_ASSET} has no entry for {asset}; available: "
                f"{sorted(k for k in sums if k.endswith('.zip'))}")

        # 2) The bundle itself.
        zip_path = os.path.join(tmp_dir, asset)
        print(f"[fetch] downloading {asset} ...", flush=True)
        _download(asset_candidates(asset), zip_path, expected_sha256=sums[asset],
                  progress_label=asset)

        # 3) Unpack + inner-manifest verification + placement.
        staging = os.path.join(tmp_dir, "unpacked")
        os.makedirs(staging, exist_ok=True)
        _safe_extract(zip_path, staging)
        inner = os.path.join(staging, short)
        if os.path.isdir(inner):
            staging = inner  # tolerate a top-level folder in the zip
        _verify_inner_manifest(zip_path, staging)
        os.makedirs(cache_root, exist_ok=True)
        if os.path.isdir(dest_dir):
            shutil.move(dest_dir, os.path.join(tmp_dir, "old-" + short))
        shutil.move(staging, dest_dir)
        print(f"[fetch] placed model at {dest_dir}", flush=True)

        # 4) Probe load: failures must surface here, not at first retrieval.
        if probe:
            from .onnx_embedder import OnnxEmbedder
            emb = OnnxEmbedder(dest_dir, model_name=model_name)
            vec = emb.embed("warmup")
            emb.release_model()
            report["dim"] = int(len(vec))
            print(f"[fetch] probe OK: dim={report['dim']}", flush=True)
        return report
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def available_bundles() -> dict[str, str]:
    """Short-name -> upstream repo id for bundles known to be published."""
    return dict(AVAILABLE_BUNDLES)


def _main_cli(argv: list[str]) -> int:  # pragma: no cover - exercised via cli.py
    print("use: hippocampus-mcp models --fetch [MODEL]", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main_cli(sys.argv[1:]))
