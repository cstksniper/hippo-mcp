#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify that an onnx export reproduces its sentence-transformers torch model.

The CI ``model-bundles`` workflow runs this gate before publishing: it encodes
probe sentences with the torch model (``SentenceTransformer``) and with the
runtime ``OnnxEmbedder`` (onnxruntime + Rust tokenizer, no torch), and requires
per-sentence cosine similarity >= 0.999 on both the document path and the bge
query-prefix path. An export whose vectors drifted from the torch reference
must never become a downloadable bundle.

Torch is allowed here exactly like in ``scripts/export_onnx.py``: this runs
offline, once, in CI -- never in the lean runtime.

Usage::

    python scripts/verify_onnx_parity.py --model-dir /path/to/bge-base-zh-v1.5
    python scripts/verify_onnx_parity.py --model-dir <dir> --threshold 0.999
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

#: Probe sentences: mixed Chinese/English, short enough to avoid long-tokenizer
#: edge paths (those are covered by the regular test suite, not this gate).
#: Content is deliberately generic household/work filler — no user data.
PROBE_SENTENCES = [
    "周一上午十点有个项目评审会，请提前把材料发出来。",
    "冰箱里的排骨晚上炖了，别忘了买葱和姜。",
    "The quick brown fox jumps over the lazy dog.",
    "Long-term memory for AI agents over the MCP protocol.",
    "今天记得给绿萝浇水，别放在空调出风口下面。",
    "hippocampus stores raw memories and never forgets unless asked.",
    "2026-11-22 是星期日。",
    "检索融合了向量召回与 BM25 词面召回。",
]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate: onnx embedder must reproduce the torch model (cos >= threshold)")
    ap.add_argument("--model-dir", required=True,
                    help="Model directory containing onnx/ artifacts AND torch weights")
    ap.add_argument("--threshold", type=float, default=0.999,
                    help="Minimum per-sentence cosine similarity (default 0.999)")
    args = ap.parse_args()

    model_dir = args.model_dir
    from sentence_transformers import SentenceTransformer
    from hippocampus_mcp.engine.onnx_embedder import BGE_ZH_QUERY_PREFIX, OnnxEmbedder

    print(f"[parity] loading torch model from {model_dir} ...", flush=True)
    st = SentenceTransformer(model_dir)

    print("[parity] building onnx embedder ...", flush=True)
    onnx = OnnxEmbedder(model_dir)

    # Document path: no query prefix on either side.
    st_doc = st.encode(PROBE_SENTENCES, show_progress_bar=False).astype(np.float64)
    onnx_doc = onnx.embed_batch(PROBE_SENTENCES)
    if st_doc.shape != onnx_doc.shape:
        print(f"[parity] FAIL: shape mismatch torch={st_doc.shape} onnx={onnx_doc.shape}")
        return 1

    worst = 1.0
    worst_idx = -1
    for i, (a, b) in enumerate(zip(st_doc, onnx_doc)):
        c = _cos(a, b)
        if c < worst:
            worst, worst_idx = c, i
        flag = "ok" if c >= args.threshold else "FAIL"
        print(f"  doc[{i}] cos={c:.6f} {flag}  {PROBE_SENTENCES[i][:24]!r}")

    # Query path: bge models prepend the official instruction prefix; the onnx
    # embedder must replicate it (embed_query) to match torch retrieval behavior.
    q = PROBE_SENTENCES[0]
    st_q = st.encode([BGE_ZH_QUERY_PREFIX + q], show_progress_bar=False)[0].astype(np.float64)
    onnx_q = onnx.embed_query(q)
    cq = _cos(st_q, onnx_q)
    print(f"  query    cos={cq:.6f} {'ok' if cq >= args.threshold else 'FAIL'}")

    dim_ok = st_doc.shape[1] == onnx_doc.shape[1]
    ok = dim_ok and worst >= args.threshold and cq >= args.threshold
    print(f"[parity] {'PASS' if ok else 'FAIL'}: worst doc cos={worst:.6f} "
          f"(#{worst_idx}), query cos={cq:.6f}, dim={onnx_doc.shape[1]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
