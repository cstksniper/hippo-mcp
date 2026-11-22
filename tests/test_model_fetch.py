# -*- coding: utf-8 -*-
"""Tests for ``engine/model_fetch.py`` — the lean, torch-free model bundle
downloader (2026-11-22 model-artifact closure).

Coverage: sha256sum parsing, asset naming, URL-list order (env override >
direct > mirror), the local-HTTP-server driven download loop (sha256
verification, URL fallback, Range resume, restart-on-200), zip-slip and
inner-manifest guards, and a full ``fetch_model_bundle`` round trip against a
synthetic bundle (probe disabled — no real onnx weights in unit tests).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hippocampus_mcp.engine import model_fetch
from hippocampus_mcp.engine.model_fetch import (
    FetchError,
    _download,
    _parse_sha256sums,
    _safe_extract,
    _sha256_file,
    _verify_inner_manifest,
    asset_name_for_model,
    fetch_model_bundle,
    fetch_urls,
)

# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_parse_sha256sums_formats() -> None:
    text = ("# comment line\n"
            "abc123  bge.zip\n"
            "def456 * other.zip\n")
    parsed = _parse_sha256sums(text)
    assert parsed == {"bge.zip": "abc123", "other.zip": "def456"}


def test_asset_name_for_model() -> None:
    assert asset_name_for_model("BAAI/bge-base-zh-v1.5") == "bge-base-zh-v1.5-onnx.zip"
    assert asset_name_for_model("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2-onnx.zip"
    with pytest.raises(FetchError):
        asset_name_for_model("///")


def test_fetch_urls_order(monkeypatch: pytest.MonkeyPatch) -> None:
    asset = asset_name_for_model("BAAI/bge-base-zh-v1.5")
    # No env: direct first, mirror second.
    urls = fetch_urls(asset)
    assert urls[0].startswith("https://github.com/")
    assert urls[0].endswith(asset)
    assert urls[1].startswith(model_fetch.MIRROR_PREFIX + "/")
    assert urls[1].endswith(asset)

    # Env override wins and interpolates {asset}.
    monkeypatch.setenv(model_fetch.ENV_FETCH_URLS,
                       "https://m1.example/{asset}, https://m2.example/{asset}")
    urls = fetch_urls(asset)
    assert urls[0] == f"https://m1.example/{asset}"
    assert urls[1] == f"https://m2.example/{asset}"
    assert urls[2].startswith("https://github.com/")


def test_asset_candidates_includes_api_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Order: env override > api.github.com asset route > direct > mirror.

    The API route exists because on some lines github.com:443 is blocked while
    api.github.com + the objects CDN pass (hit 2026-11-22).
    """
    monkeypatch.delenv(model_fetch.ENV_FETCH_URLS, raising=False)
    monkeypatch.setattr(model_fetch, "_resolve_api_asset_url",
                        lambda a: "https://api.github.com/repos/x/releases/assets/1")
    cands = model_fetch.asset_candidates("foo.zip")
    assert cands[0][0].startswith("https://api.github.com/")
    assert cands[0][1] == {"Accept": "application/octet-stream"}
    assert cands[1][0].startswith("https://github.com/")
    assert cands[2][0].startswith(model_fetch.MIRROR_PREFIX + "/")


def test_asset_candidates_api_failure_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead API route degrades to direct+mirror instead of failing."""
    monkeypatch.delenv(model_fetch.ENV_FETCH_URLS, raising=False)

    def boom(_a: str):
        raise TimeoutError("api down")

    monkeypatch.setattr(model_fetch, "_resolve_api_asset_url", boom)
    cands = model_fetch.asset_candidates("foo.zip")
    assert len(cands) == 2  # direct + mirror, no API entry
    assert cands[0][0].startswith("https://github.com/")


# --------------------------------------------------------------------------
# Local HTTP server harness
# --------------------------------------------------------------------------


class _FileServer:
    """Tiny static server over an in-memory dict; records Range usage."""

    def __init__(self, files: dict[str, bytes], *, support_range: bool = True) -> None:
        self.files = files
        self.range_seen: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
                path = self.path.lstrip("/")
                body = self.server.files.get(path)  # type: ignore[attr-defined]
                if body is None:
                    self.send_error(404)
                    return
                rng = self.headers.get("Range")
                if rng and support_range:
                    self.server.range_seen.append(rng)  # type: ignore[attr-defined]
                    have = int(rng.split("=")[1].rstrip("-"))
                    payload = body[have:]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {have}-{len(body) - 1}/{len(body)}")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    if rng:
                        self.server.range_seen.append(rng)  # type: ignore[attr-defined]
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.files = files  # type: ignore[attr-defined]
        self.httpd.range_seen = self.range_seen  # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{name}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def file_server():
    holder: dict[str, _FileServer] = {}

    def make(files: dict[str, bytes], **kw: object) -> _FileServer:
        srv = _FileServer(files, **kw)  # type: ignore[arg-type]
        holder["srv"] = srv
        return srv

    yield make
    if "srv" in holder:
        holder["srv"].close()


# --------------------------------------------------------------------------
# _download behaviour
# --------------------------------------------------------------------------


def test_download_verifies_sha256(file_server) -> None:
    payload = b"x" * (3 * model_fetch.CHUNK_SIZE + 17)
    srv = file_server({"blob.bin": payload})
    dest = Path(_tmpdir()) / "blob2.bin"
    # Wrong expectation must fail and leave no part file behind.
    with pytest.raises(FetchError):
        _download([(srv.url("blob.bin"), {})], str(dest), expected_sha256="0" * 64)
    assert not dest.exists() and not Path(str(dest) + ".part").exists()
    # Correct expectation lands the full content.
    good = hashlib.sha256(payload).hexdigest()
    _download([(srv.url("blob.bin"), {})], str(dest), expected_sha256=good)
    assert dest.read_bytes() == payload


def test_download_falls_back_to_next_url(file_server) -> None:
    payload = b"payload-through-fallback"
    srv = file_server({"real.bin": payload})
    dest = Path(_tmpdir()) / "fallback.bin"
    _download([(srv.url("missing.bin"), {}), (srv.url("real.bin"), {})], str(dest))
    assert dest.read_bytes() == payload


def test_download_resumes_from_part(file_server) -> None:
    payload = bytes(range(256)) * 4096  # 1 MiB
    srv = file_server({"blob.bin": payload})
    dest = Path(_tmpdir()) / "resume.bin"
    part = Path(str(dest) + ".part")
    half = len(payload) // 2
    part.write_bytes(payload[:half])
    _download([(srv.url("blob.bin"), {})], str(dest))
    assert dest.read_bytes() == payload
    assert srv.range_seen and srv.range_seen[0] == f"bytes={half}-"


def test_download_restarts_when_server_ignores_range(file_server) -> None:
    payload = b"fresh" * 100
    srv = file_server({"blob.bin": payload}, support_range=False)
    dest = Path(_tmpdir()) / "restart.bin"
    Path(str(dest) + ".part").write_bytes(b"stale-half")
    _download([(srv.url("blob.bin"), {})], str(dest))
    assert dest.read_bytes() == payload


# --------------------------------------------------------------------------
# Zip guards
# --------------------------------------------------------------------------


def test_safe_extract_rejects_zip_slip(tmp_path: Path) -> None:
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(FetchError):
        _safe_extract(str(zpath), str(dest))
    assert not (tmp_path / "escaped.txt").exists()


def _make_bundle_zip(short: str, tmp_path: Path) -> Path:
    """A minimal valid bundle (content-only; no real onnx weights)."""
    members = {
        "tokenizer.json": b"{}",
        "config.json": b"{}",
        "onnx/model.onnx": b"dummy-weights",
        "onnx/onnx_meta.json": json.dumps(
            {"pooling": "cls", "word_embedding_dimension": 768}).encode(),
    }
    manifest = [f"# hippocampus model bundle v{model_fetch.BUNDLE_VERSION}",
                "# format: <sha256>  <relative-path>"]
    for rel, data in members.items():
        manifest.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    zpath = tmp_path / f"{short}-onnx.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for rel, data in members.items():
            zf.writestr(rel, data)
        zf.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    return zpath


def test_verify_inner_manifest_detects_tamper(tmp_path: Path) -> None:
    short = "tiny-model"
    zpath = _make_bundle_zip(short, tmp_path)
    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    _safe_extract(str(zpath), str(unpacked))
    # Untampered bundle passes.
    _verify_inner_manifest(str(zpath), str(unpacked))
    # Tamper with one member => mismatch.
    (unpacked / "tokenizer.json").write_bytes(b"tampered")
    with pytest.raises(FetchError):
        _verify_inner_manifest(str(zpath), str(unpacked))


# --------------------------------------------------------------------------
# fetch_model_bundle round trip (probe disabled: no real weights in tests)
# --------------------------------------------------------------------------


def test_fetch_model_bundle_round_trip(tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch,
                                       file_server) -> None:
    short = "tiny-model"
    zpath = _make_bundle_zip(short, tmp_path)
    zip_bytes = zpath.read_bytes()
    sums = f"{hashlib.sha256(zip_bytes).hexdigest()}  {zpath.name}\n"
    srv = file_server({
        model_fetch.SHA256SUMS_ASSET: sums.encode(),
        zpath.name: zip_bytes,
    })
    monkeypatch.setenv(model_fetch.ENV_FETCH_URLS, srv.url("{asset}"))

    cache_root = tmp_path / "cache"
    report = fetch_model_bundle(short, str(cache_root), probe=False)
    assert report["status"] == "fetched"
    model_dir = cache_root / short
    assert (model_dir / "onnx" / "model.onnx").read_bytes() == b"dummy-weights"
    assert (model_dir / "tokenizer.json").read_bytes() == b"{}"

    # Second call short-circuits: artifacts already present.
    report2 = fetch_model_bundle(short, str(cache_root), probe=False)
    assert report2["status"] == "already_local"


def test_fetch_model_bundle_refuses_unlisted_sums(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch,
                                                  file_server) -> None:
    short = "tiny-model"
    zpath = _make_bundle_zip(short, tmp_path)
    srv = file_server({
        model_fetch.SHA256SUMS_ASSET: b"deadbeef  other-thing-onnx.zip\n",
        zpath.name: zpath.read_bytes(),
    })
    monkeypatch.setenv(model_fetch.ENV_FETCH_URLS, srv.url("{asset}"))
    with pytest.raises(FetchError, match="no entry"):
        fetch_model_bundle(short, str(tmp_path / "cache"), probe=False)


def test_fetch_model_bundle_probe_runs(tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch,
                                       file_server) -> None:
    """``probe=True`` loads the embedder once and releases it; the reported
    dim comes from the probe (this path caught a real API-name bug locally)."""
    import hippocampus_mcp.engine.onnx_embedder as oe

    calls: list[tuple] = []

    class StubEmbedder:
        def __init__(self, model_dir: str, model_name: str = "") -> None:
            calls.append(("init", model_dir, model_name))

        def embed(self, text: str) -> list[float]:
            calls.append(("embed", text))
            return [0.0] * 384

        def release_model(self) -> bool:
            calls.append(("release",))
            return True

    monkeypatch.setattr(oe, "OnnxEmbedder", StubEmbedder)

    short = "tiny-model"
    zpath = _make_bundle_zip(short, tmp_path)
    zip_bytes = zpath.read_bytes()
    srv = file_server({
        model_fetch.SHA256SUMS_ASSET:
            f"{hashlib.sha256(zip_bytes).hexdigest()}  {zpath.name}\n".encode(),
        zpath.name: zip_bytes,
    })
    monkeypatch.setenv(model_fetch.ENV_FETCH_URLS, srv.url("{asset}"))

    report = fetch_model_bundle(short, str(tmp_path / "cache"), probe=True)
    assert report["dim"] == 384
    assert ("init", str(tmp_path / "cache" / short), short) in calls
    assert ("embed", "warmup") in calls
    assert ("release",) in calls


def _tmpdir() -> str:
    import tempfile
    return tempfile.mkdtemp(prefix="hippo-fetch-test-")
