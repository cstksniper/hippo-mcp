#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package a local sentence-transformers model directory into an onnx bundle zip.

Why this script exists
----------------------
The lean core has no ``sentence_transformers`` / ``huggingface_hub`` dependency,
so a fresh machine cannot obtain embedding-model artifacts by itself. The fix
is a downloadable bundle (published as a GitHub Release asset by the
``model-bundles.yml`` CI workflow). This script is the shared packager: the
CI workflow calls it right after ``export_onnx.py``, and it can also be run
by hand against a locally verified model directory.

Bundle layout (consumed by ``engine/model_fetch.py``; the onnx runtime path
reads exactly ``onnx/model.onnx`` + ``onnx/onnx_meta.json`` + ``tokenizer.json``)::

    <model-name>-onnx.zip
    ├── MANIFEST.sha256        # per-file sha256 + provenance header
    ├── tokenizer.json
    ├── config.json            # integrity reference (not read at runtime)
    ├── LICENSE-README.txt     # provenance + license statement
    └── onnx/
        ├── model.onnx
        └── onnx_meta.json

Usage::

    python scripts/build_model_bundle.py --model-dir /path/to/bge-base-zh-v1.5 --out /tmp/staging
    python scripts/build_model_bundle.py --model-dir <dir> --out <dir> \\
        --source-repo BAAI/bge-base-zh-v1.5 --license-file /path/to/LICENSE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile

#: Files copied from the model dir root (tokenizer is required at runtime;
#: config.json ships for integrity reference only).
ROOT_FILES = ("tokenizer.json", "config.json")
#: Files required under ``onnx/`` (the runtime contract).
ONNX_DIR = "onnx"
ONNX_FILES = ("model.onnx", "onnx_meta.json")
#: Bundle format version, bumped when the layout changes (model_fetch refuses
#: bundles with an unknown version).
BUNDLE_VERSION = 1

_MANIFEST_NAME = "MANIFEST.sha256"
_LICENSE_README_NAME = "LICENSE-README.txt"


def _sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _model_short_name(model_dir: str) -> str:
    """Bundle/asset name = the model directory's last path segment (this is
    also the name the embeddings resolver looks for under a cache root)."""
    base = os.path.basename(os.path.abspath(model_dir).rstrip("/\\"))
    if not base:
        raise SystemExit(f"--model-dir has no usable last segment: {model_dir!r}")
    return base


def _load_onnx_meta(model_dir: str) -> dict:
    meta_path = os.path.join(model_dir, ONNX_DIR, "onnx_meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit(
            f"missing {meta_path}: run scripts/export_onnx.py --model {model_dir} first"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_license_readme(path: str, *, source_repo: str, model_dir: str,
                          license_file: str) -> None:
    lines = [
        "hippo-mcp onnx model bundle",
        "=" * 40,
        "",
        f"Model directory name : {os.path.basename(os.path.abspath(model_dir))}",
        f"Upstream repository  : {source_repo or '(unspecified)'}",
        f"ONNX exported by     : scripts/export_onnx.py (see onnx_meta.json for opset/pooling)",
        "",
        "This bundle redistributes third-party model weights. The upstream license",
        "applies to the weights; this packager adds no restrictions.",
        "",
    ]
    if license_file and os.path.isfile(license_file):
        lines += ["--- upstream license ---", ""]
        with open(license_file, "r", encoding="utf-8", errors="replace") as f:
            lines.append(f.read().rstrip())
        lines.append("")
    else:
        lines += [
            "NOTE: no LICENSE file was embedded at packaging time. Verify the",
            "upstream model card license before redistributing this bundle.",
            "",
        ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def build_bundle(model_dir: str, out_dir: str, *, source_repo: str = "",
                 license_file: str = "") -> str:
    """Assemble the bundle zip; returns the zip path. Idempotent (re-runs overwrite)."""
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise SystemExit(f"model dir not found: {model_dir}")

    meta = _load_onnx_meta(model_dir)
    for name in ONNX_FILES:
        p = os.path.join(model_dir, ONNX_DIR, name)
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}")
    for name in ROOT_FILES:
        p = os.path.join(model_dir, name)
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p} (required bundle member)")

    short = _model_short_name(model_dir)
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, f"{short}-onnx.zip")

    # Stage the file list (relative paths inside the zip; no top-level folder
    # so extraction lands directly in <cache-root>/<short>/). ONNX artifacts
    # MUST keep their onnx/ prefix — that subdirectory is the runtime contract
    # (find_onnx_artifacts looks for <dir>/onnx/{model.onnx,onnx_meta.json}).
    members: list[tuple[str, str]] = []
    for name in ROOT_FILES:
        members.append((name, os.path.join(model_dir, name)))
    for name in ONNX_FILES:
        members.append((f"{ONNX_DIR}/{name}",
                        os.path.join(model_dir, ONNX_DIR, name)))

    _write_license_readme(
        os.path.join(out_dir, _LICENSE_README_NAME),
        source_repo=source_repo, model_dir=model_dir, license_file=license_file)
    members.append((_LICENSE_README_NAME, os.path.join(out_dir, _LICENSE_README_NAME)))

    # MANIFEST is written last: it hashes everything else in the zip.
    manifest_lines = [
        f"# hippocampus model bundle v{BUNDLE_VERSION}",
        f"# model: {short}",
        f"# upstream: {source_repo or '(unspecified)'}",
        f"# exported_by: scripts/export_onnx.py; opset={meta.get('opset', '?')}; "
        f"pooling={meta.get('pooling', '?')}; dim="
        f"{meta.get('word_embedding_dimension') or (meta.get('pooling_config') or {}).get('word_embedding_dimension', '?')}",
        "# format: <sha256>  <relative-path>",
    ]
    for rel, src in members:
        manifest_lines.append(f"{_sha256_file(src)}  {rel}")
    manifest_path = os.path.join(out_dir, _MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(manifest_lines) + "\n")
    members.append((_MANIFEST_NAME, manifest_path))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, src in members:
            zf.write(src, rel)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[Bundle] {zip_path} ({size_mb:.1f} MB)")
    print(f"[Bundle] zip sha256: {_sha256_file(zip_path)}")
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Package a model dir into an onnx bundle zip (see module docstring)")
    ap.add_argument("--model-dir", required=True,
                    help="Local sentence-transformers model directory containing onnx/ artifacts")
    ap.add_argument("--out", required=True, help="Output directory for the zip + manifest")
    ap.add_argument("--source-repo", default="",
                    help="Upstream model repo id recorded in the manifest, e.g. BAAI/bge-base-zh-v1.5")
    ap.add_argument("--license-file", default="",
                    help="Upstream LICENSE file to embed as LICENSE-README.txt content")
    args = ap.parse_args()
    build_bundle(args.model_dir, args.out,
                 source_repo=args.source_repo, license_file=args.license_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
