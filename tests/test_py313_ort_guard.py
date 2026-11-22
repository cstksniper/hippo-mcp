# -*- coding: utf-8 -*-
"""py3.13 + onnxruntime native-crash guard (target test + negative control).

Background (2026-11-22, K-drive DSH verification)
--------------------------------------------------
On Windows, ``import onnxruntime`` on some Python 3.13 builds dies with a
native ``access violation`` inside ``onnxruntime/capi/_pybind_state.py``
(``create_module``).  That is a *process-level* death (exit 139), NOT a
Python ``ImportError`` — so the ``except ImportError`` around the import in
``_load_model`` cannot catch it.  The old behaviour: the whole MCP server
process silently disappears the first time an embedding is needed, and the
host only sees "Not connected" with no diagnostic anywhere.

The fix has two layers, both asserted here:
  L1 (source install): ``pyproject.toml`` caps ``requires-python`` at <3.13
     so a one-line install never picks a known-broken interpreter.
  L2 (runtime guard): ``onnx_import_probe()`` spawns a *child process* to
     import onnxruntime — a native crash there stays in the child — and
     ``_load_model`` turns a failed probe into a diagnostic ImportError
     (which warmup's ``except BaseException`` can record as fatal) instead
     of letting the main process die wordlessly.

Three groups:
  A — behavior: probe returns (True, ...) when import works; the guard helper
      converts a failed probe into a message naming both the interpreter and
      the remediation.
  B — negative control: a simulated crashing probe (return code 139 with an
      access-violation trace) MUST make the guard raise; a healthy probe MUST
      NOT.  Mutating the guard to ignore probe failures is falsifiable here.
  C — source-level: pyproject caps Python <3.13; _load_model calls the probe
      *before* the real ``import onnxruntime``; the check order in
      _attach_onnx (O3 of test_install_surface) is preserved.
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

FAILED: list[str] = []
PASSED = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print("  ok  -", label)
    else:
        FAILED.append(label)
        print("  FAIL-", label, ("| " + str(detail)[:200]) if detail else "")


def section(name: str) -> None:
    print("\n--", name, "--")


def _function_source(path: str, name: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    lines = src.splitlines()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = node.lineno - 1              # 0-based index of def line
            end = node.end_lineno                # exclusive
            body_start = start
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant) and isinstance(
                    body[0].value.value, str):
                body_start = body[0].end_lineno  # first line after docstring (0-based)
            return "\n".join(lines[body_start:end])
    return ""


def main() -> int:
    from hippocampus_mcp.engine import onnx_embedder as oe

    section("A: probe + guard behavior")
    ok, detail = oe.onnx_import_probe()
    check(ok is True, "A1 probe True on the healthy py311 test env", detail)
    check(isinstance(detail, str), "A2 probe detail is a string")
    # guard is pure: given a probe result, returns/raises deterministically
    msg = oe._guard_probe_result(True, "ok", python_version="3.11.9")
    check(msg == "", "A3 healthy probe -> no error message", repr(msg))
    err = oe._guard_probe_result(
        False, "Windows fatal exception: access violation\n"
               "  File onnxruntime/capi/_pybind_state.py, line 32",
        python_version="3.13.12")
    check("access violation" in err, "A4 failed probe message carries the native cause", err)
    check(oe._guard_probe_result is not None, "A5 guard helper exists")

    section("B: negative control (mutation must be caught)")
    raised = False
    try:
        oe._raise_on_bad_probe(False, "access violation at create_module",
                               python_version="3.13.12")
    except ImportError as e:
        raised = True
        check("Python 3.13" in str(e), "B1 crash -> ImportError names Python 3.13", str(e))
        check("3.11" in str(e), "B2 crash -> message suggests Python 3.11", str(e))
    check(raised, "B3 bad probe raises (mutating to ignore would fail here)")
    not_raised = False
    try:
        oe._raise_on_bad_probe(True, "ok", python_version="3.11.9")
        not_raised = True
    except ImportError:
        pass
    check(not_raised, "B4 healthy probe does NOT raise")

    section("C: source-level wiring")
    pyproject = open(os.path.join(ROOT, "pyproject.toml"), "r", encoding="utf-8").read()
    check('requires-python = ">=3.11,<3.13"' in pyproject,
          "C1 pyproject caps Python at <3.13",
          [l for l in pyproject.splitlines() if "requires-python" in l])
    check('"Programming Language :: Python :: 3.13"' not in pyproject,
          "C2 3.13 classifier removed")

    oe_src = os.path.join(ROOT, "src", "hippocampus_mcp", "engine", "onnx_embedder.py")
    load = _function_source(oe_src, "_load_model")
    probe_pos = load.find("onnx_import_probe()")
    imp_pos = load.find("import onnxruntime")
    guard_pos = load.find("_raise_on_bad_probe")
    check(probe_pos != -1, "C3 _load_model calls onnx_import_probe()")
    check(guard_pos != -1, "C4 _load_model calls _raise_on_bad_probe")
    check(probe_pos != -1 and guard_pos != -1 and guard_pos < imp_pos,
          "C5 guard runs BEFORE the real import onnxruntime",
          "probe=%d guard=%d import=%d" % (probe_pos, guard_pos, imp_pos))

    # O3 order from test_install_surface must stay intact
    emb_src = os.path.join(ROOT, "src", "hippocampus_mcp", "engine", "embeddings.py")
    attach = _function_source(emb_src, "_attach_onnx")
    dep_pos = attach.find("onnx_runtime_available()")
    imp_pos2 = attach.find("from .onnx_embedder import")
    check(dep_pos != -1 and imp_pos2 != -1 and dep_pos < imp_pos2,
          "C6 install-surface O3 order preserved")

    print("\n" + "=" * 60)
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
    else:
        print("all passed (%d assertions)" % PASSED)
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
