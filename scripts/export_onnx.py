#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export a sentence-transformers model to ONNX (one-time offline operation).

Why this is a separate script
-----------------------------
The point of the export is that the **runtime no longer loads torch /
sentence-transformers** (~280MB saved, measured). So:

* this script **may** import torch / transformers (runs once, offline);
* the produced .onnx is loaded at runtime by onnxruntime — the runtime never
  touches torch.

Never guess the pooling
-----------------------
``1_Pooling/config.json`` of bge-base-zh-v1.5 says **CLS pooling**
(``pooling_mode_cls_token: true``, not mean). This script writes the pooling
config it actually read into the produced ``onnx_meta.json`` and the runtime
follows it — **no guessing**, otherwise vectors diverge from the torch
reference (cos far below 0.9999).

Usage::

    python scripts/export_onnx.py                     # export the default model
    python scripts/export_onnx.py --model /path/to/bge-base-zh-v1.5
    python scripts/export_onnx.py --opset 17
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _default_model_dir() -> str:
    """Default export target: bge-base-zh-v1.5 under <project parent>/libs/models."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = [
        os.path.join(os.path.dirname(root), "libs", "models", "BAAI", "bge-base-zh-v1.5"),
        os.path.join(os.path.dirname(root), "libs", "models", "bge-base-zh-v1.5"),
        os.path.join(root, "libs", "models", "bge-base-zh-v1.5"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[1]


def read_pooling_config(model_dir: str) -> dict:
    """Read the ST pooling config. Raise if unreadable — guessing pooling is
    the single most dangerous mistake here."""
    path = os.path.join(model_dir, "1_Pooling", "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"cannot find {path}: this is not a sentence-transformers model directory, "
            f"so the pooling mode cannot be confirmed (a wrong guess makes vectors "
            f"diverge from the torch reference)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pooling_mode_and_dim(pooling: dict) -> tuple[str, int]:
    """Normalize the pooling mode + embedding dim from **either** schema.

    Two generations exist in the wild and both must work (hit in CI on
    2026-11-22): sentence-transformers 4.x writes boolean flags
    (``pooling_mode_cls_token: true`` + ``word_embedding_dimension``), 5.x
    writes a plain string (``pooling_mode: "cls"`` + ``embedding_dimension``).
    Returning ``(mode, dim)`` keeps this decision in exactly one place.
    """
    if "pooling_mode" in pooling and isinstance(pooling.get("pooling_mode"), str):
        mode = pooling["pooling_mode"].strip().lower()
    else:
        mode = ("cls" if pooling.get("pooling_mode_cls_token")
                else "mean" if pooling.get("pooling_mode_mean_tokens")
                else "max" if pooling.get("pooling_mode_max_tokens")
                else "")
    dim = int(pooling.get("word_embedding_dimension")
              or pooling.get("embedding_dimension") or 0)
    return mode, dim


def _calib_sentences(limit: int) -> list[str]:
    """Calibration corpus: prefer the ``--calib`` file (one sentence per line);
    default samples read-only from **real memory libraries**.

    The calibration set must **represent the production corpus distribution**
    (length / punctuation / proper-noun density), otherwise activation ranges
    are estimated poorly and int8 loses more accuracy than necessary. Sampling
    from real libraries is the closest-to-production choice.

    An earlier version hardcoded ``memories.json`` of one json library —
    **sqlite libraries do not have that file**, so it always raised => the
    script **silently fell back to built-in samples**: the calibration set no
    longer represented production and int8 accuracy dropped for nothing, with
    one "real-library sampling failed" line covering it up. It now walks
    ``data/`` backend-agnostically via ``_lib_access`` and **warns loudly**
    when it cannot gather enough sentences.
    """
    import random

    out: list[str] = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    from _lib_access import has_library_data, load_records  # noqa: E402

    data_dir = os.path.join(root, "data")
    default_lib = ""
    candidates = [default_lib] if has_library_data(default_lib) else []
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            d = os.path.join(data_dir, name)
            # Backups must not enter the production corpus (they are historical
            # snapshots; their distribution may be long outdated).
            if (os.path.isdir(d) and has_library_data(d)
                    and ".bak" not in name and ".snap" not in name
                    and d not in candidates):
                candidates.append(d)

    for lib in candidates:
        try:
            recs = load_records(lib)
            pool = [
                ((r.get("content") or "") if isinstance(r, dict)
                 else str(getattr(r, "content", "") or "")).strip()
                for r in recs
            ]
            pool = [s for s in pool if 10 <= len(s) <= 300]
            if len(pool) >= max(16, limit // 2):
                random.seed(42)
                random.shuffle(pool)
                out = pool[:limit]
                print(f"[calib] calibration corpus source: {lib} "
                      f"({len(pool)} usable sentences, taking {len(out)})",
                      file=sys.stderr)
                break
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to sample calibration corpus from {lib} "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)

    if not candidates:
        print("[calib] no memory library with data found under data/", file=sys.stderr)
    if len(out) < limit // 2:
        # A degraded calibration set must be **visible**: it directly decides
        # how much accuracy int8 loses — it must not be silenced away.
        print(f"[warn] not enough calibration corpus from any library "
              f"({len(out)} sentences); falling back to built-in samples — "
              "the calibration set is not representative, quantized accuracy "
              "may be noticeably worse", file=sys.stderr)
    if len(out) < limit // 2:
        # Deliberately generic filler (mixed lengths/scripts); never embed
        # real user data here — this file ships to the public repository.
        out += [
            "Sample: the calibration set must cover common length and punctuation ranges.",
            "Remember to back up the ledger before running the migration window.",
            "One mu of the field yields 120 kilograms of dried leaves, stored away from light.",
            "I spent the afternoon copying two hours of archive records by hand.",
        ]
        out += [
            f" filler sentence {i} — mixed ascii and Chinese text."
            for i in range(32)
        ]
    return out[:limit]


def quantize_int8(fp32_path: str, out_path: str, model_dir: str) -> None:
    """Static int8 quantization (QDQ + MinMax calibration + per-channel weights,
    MatMul/Gemm only).

    Only MatMul/Gemm are quantized: nearly all BERT weights live in those two
    op types (~90% of the benefit), while LayerNorm/embeddings stay fp32
    (much smaller accuracy loss). The calibration set is sampled from real
    library corpora.
    """
    import numpy as np  # noqa: F401
    from onnxruntime.quantization import (  # noqa: PLC0415
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
        quantize_static,
    )
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_dir)
    sentences = _calib_sentences(256)
    print(f"    calibration corpus: {len(sentences)} sentences (real-library sample)")

    # The fp32 export defaults to external data files (model.onnx.data); the
    # quantizer hits Errno 22 when writing external data on this platform, so
    # inline the tensors into a single file before quantizing.
    import onnx  # noqa: PLC0415

    fp32_inline = out_path + ".fp32-inlined.onnx"
    if not os.path.isfile(fp32_inline):
        m = onnx.load(fp32_path, load_external_data=True)
        onnx.save(m, fp32_inline)
    fp32_path = fp32_inline

    input_names = ["input_ids", "attention_mask", "token_type_ids"]

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._batches = []
            B = 8
            for i in range(0, len(sentences), B):
                enc = tok(sentences[i:i + B], padding="longest", truncation=True,
                          max_length=512, return_tensors="np")
                feed = {
                    "input_ids": enc["input_ids"].astype("int64"),
                    "attention_mask": enc["attention_mask"].astype("int64"),
                }
                if "token_type_ids" in enc:
                    feed["token_type_ids"] = enc["token_type_ids"].astype("int64")
                else:
                    feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
                self._batches.append(feed)
            self._i = 0

        def get_next(self):
            if self._i >= len(self._batches):
                return None
            b = self._batches[self._i]
            self._i += 1
            return b

    quantize_static(
        fp32_path, out_path, _Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["MatMul", "Gemm"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a sentence-transformers model to ONNX")
    ap.add_argument("--model", default="", help="model directory (default bge-base-zh-v1.5)")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset (default 17)")
    ap.add_argument("--out", default="", help="output directory (default <model>/onnx/)")
    ap.add_argument("--int8", action="store_true",
                    help="also export a static int8 quantized copy to <model>/onnx_int8/ (H3)")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model or _default_model_dir())
    if not os.path.isdir(model_dir):
        print(f"[FAIL] model directory not found: {model_dir}", file=sys.stderr)
        return 1

    pooling = read_pooling_config(model_dir)
    mode, dim = pooling_mode_and_dim(pooling)
    if mode not in ("cls", "mean", "max") or dim <= 0:
        print(f"[FAIL] unrecognized pooling config: {pooling}", file=sys.stderr)
        return 1

    out_dir = args.out or os.path.join(model_dir, "onnx")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.onnx")

    print(f"[1/4] loading torch model: {model_dir}")
    import torch                                   # noqa: E402  this script may use torch
    from transformers import AutoModel, AutoTokenizer  # noqa: E402

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()

    print("[2/4] building example input")
    batch = tok(
        ["sample text for the export sanity check", "a second sample text"],
        padding="longest",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    if "token_type_ids" not in batch:
        # Some models have no token_type_ids (e.g. parts of the RoBERTa family)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

    # 🔴 Compatibility trap: transformers 5.x changed BertModel.forward's
    # signature and collides with legacy torch.onnx.export(dynamo=False)
    # positional passing ("got multiple values for argument 'use_cache'").
    # Wrap the model in a **keyword-calling** wrapper; stable on both
    # exporter generations.
    class _KeywordWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.inner(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            return out.last_hidden_state

    export_model = _KeywordWrapper(model)

    example = tuple(batch[n] for n in input_names)

    print(f"[3/4] torch.onnx.export (opset={args.opset}) -> {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            example,
            out_path,
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,  # 🔴 torch 2.13's default dynamo-exported graph is
                           # pathological on ORT (inference RSS 5GB+, attention
                           # decomposed into giant intermediate tensors); the
                           # legacy exporter must be used (the H2-acceptance form).
        )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)

    print("[4/4] writing onnx_meta.json")
    meta = {
        "model_dir": model_dir,
        "pooling": mode,
        "pooling_config": pooling,
        # 🔴 The top-level dim field is mandatory: runtime OnnxEmbedder reads
        # only the top-level ``word_embedding_dimension`` (engine/
        # onnx_embedder.py); when missing it used to fall back ``or 768`` to
        # bge's dim => the LNN core was built for 768 while vectors came out
        # 384, and the first ingest exploded on a matmul shape mismatch
        # (hit in practice). Both ST schemas normalize to it via
        # pooling_mode_and_dim().
        "word_embedding_dimension": dim,
        "max_seq_length": int(
            json.load(
                open(os.path.join(model_dir, "sentence_bert_config.json"), "r", encoding="utf-8")
            ).get("max_seq_length", 512)
        )
        if os.path.isfile(os.path.join(model_dir, "sentence_bert_config.json"))
        else 512,
        "opset": args.opset,
        "input_names": input_names,
        "output_names": ["last_hidden_state"],
        "normalize": True,
        "exported_by": "scripts/export_onnx.py",
    }
    with open(os.path.join(out_dir, "onnx_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] export finished: {out_path}  ({size_mb:.1f} MB)")
    print(f"[OK] pooling={mode}  max_seq_length={meta['max_seq_length']}")

    if args.int8:
        int8_dir = args.out and os.path.join(os.path.dirname(out_path), "onnx_int8") \
            or os.path.join(model_dir, "onnx_int8")
        os.makedirs(int8_dir, exist_ok=True)
        int8_path = os.path.join(int8_dir, "model.onnx")
        print("[5/5] int8 static quantization (QDQ + real-library calibration, "
              "MatMul/Gemm only)...")
        quantize_int8(out_path, int8_path, model_dir)
        meta8 = dict(meta)
        meta8["quant"] = "int8-qdq-minmax-matmul"
        meta8["fp32_reference"] = out_path
        with open(os.path.join(int8_dir, "onnx_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta8, f, ensure_ascii=False, indent=2)
        print(f"[OK] int8 export finished: {int8_path} "
              f"({os.path.getsize(int8_path) / (1024 * 1024):.1f} MB)")

    print("[NEXT] run scripts/verify_onnx.py to compare cosine similarity against "
          "the torch reference (requires >= 0.9999)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
