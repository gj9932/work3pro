"""Smoke tests for Chunk 8 eval CLIs (paper §4.5).

Each test invokes the relevant eval module on the synthetic fixture (or in
TBD-skeleton mode for GPU-only paths) and verifies that:
- the CLI runs to completion without raising
- JSONL output exists and the header row records the right experiment name
- TBD-only paths still emit the table skeleton so collect_tables can render it

The actual numbers are not asserted — they're either synthetic-fixture loop
closure (banner-labelled) or TBD until the GPU box is online.
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
RUNS = ROOT / "runs" / "geodistill" / "eval_smoke"


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(f"command failed:\n  {' '.join(cmd)}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


def _read_jsonl(path: Path) -> list[dict]:
    import json
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def setup_module(_m=None):
    RUNS.mkdir(parents=True, exist_ok=True)


def test_eval_token_probe_with_model_synthetic():
    out = RUNS / "table2.jsonl"
    _run([PY, "-m", "geodistill.eval.eval_token_geometry_probe",
          "--source", "synthetic", "--scenes", "2", "--hidden", "16", "--epochs", "20",
          "--with_model", "--out", str(out)])
    assert out.exists()
    rows = _read_jsonl(out)
    assert rows and rows[0]["experiment"] == "eval_token_geometry_probe"
    assert any(r.get("kind") == "table2_row" for r in rows)


def test_eval_token_probe_no_model_synthetic():
    out = RUNS / "table2_baseline.jsonl"
    _run([PY, "-m", "geodistill.eval.eval_token_geometry_probe",
          "--source", "synthetic", "--scenes", "2", "--hidden", "16", "--epochs", "20",
          "--out", str(out)])
    rows = _read_jsonl(out)
    assert any(r.get("kind") == "table2_row" for r in rows)


def test_eval_relation_probe_synthetic():
    out = RUNS / "table3.jsonl"
    _run([PY, "-m", "geodistill.eval.eval_relation_probe",
          "--source", "synthetic", "--scenes", "2", "--teacher", "hard_quantile",
          "--out", str(out)])
    rows = _read_jsonl(out)
    assert any(r.get("kind") == "table3_row" for r in rows)


def test_eval_efficiency_synthetic():
    out = RUNS / "table6.jsonl"
    _run([PY, "-m", "geodistill.eval.eval_efficiency",
          "--source", "synthetic", "--scenes", "1", "--out", str(out)])
    rows = _read_jsonl(out)
    methods = {r.get("method") for r in rows if r.get("kind") == "table6_row"}
    assert {"Qwen + LPGA", "Qwen2.5-VL"}.issubset(methods)


def test_eval_spatial_qa_skeleton():
    out = RUNS / "table1.jsonl"
    _run([PY, "-m", "geodistill.eval.eval_spatial_qa", "--source", "synthetic", "--out", str(out)])
    rows = _read_jsonl(out)
    qa_rows = [r for r in rows if r.get("kind") == "table1_row"]
    assert qa_rows and all(r["overall_accuracy"] == "TBD" for r in qa_rows)


def test_eval_open_loop_skeleton():
    out = RUNS / "table4.jsonl"
    _run([PY, "-m", "geodistill.eval.eval_open_loop_planning", "--source", "synthetic", "--out", str(out)])
    rows = _read_jsonl(out)
    plan_rows = [r for r in rows if r.get("kind") == "table4_row"]
    assert plan_rows and plan_rows[0].get("L2_avg") == "TBD"


def test_eval_bench2drive_skeleton_main_protocol():
    out = RUNS / "table5.jsonl"
    _run([PY, "-m", "geodistill.eval.eval_bench2drive", "--source", "synthetic", "--out", str(out)])
    rows = _read_jsonl(out)
    b2d_rows = [r for r in rows if r.get("kind") == "table5_row"]
    assert b2d_rows and b2d_rows[0]["protocol"] == "main", \
        "default Bench2Drive protocol must be 'main' (paper §4.5 forbids b2d_vl in main)"


def test_eval_bench2drive_explicit_b2d_vl_tagged():
    out = RUNS / "table5_vl.jsonl"
    _run([PY, "-m", "geodistill.eval.eval_bench2drive", "--source", "synthetic",
          "--allow_b2d_vl", "--out", str(out)])
    rows = _read_jsonl(out)
    b2d_rows = [r for r in rows if r.get("kind") == "table5_row"]
    assert b2d_rows and b2d_rows[0]["protocol"] == "b2d_vl"


def test_eval_robustness_synthetic():
    out = RUNS / "robustness.jsonl"
    _run([PY, "-m", "geodistill.eval.eval_robustness",
          "--source", "synthetic", "--scenes", "2", "--epochs", "20", "--out", str(out)])
    rows = _read_jsonl(out)
    rk = [r.get("corruption") for r in rows if r.get("kind") == "robustness_row"]
    assert "clean" in rk
    assert any(c in rk for c in ("single_camera_drop", "image_occlusion"))


def test_run_shortcut_audit_synthetic():
    out = RUNS / "shortcut.jsonl"
    _run([PY, "-m", "scripts.geodistill.run_shortcut_audit",
          "--source", "synthetic", "--scenes", "2", "--seed", "0", "--teacher", "hard_quantile",
          "--out", str(out)])
    rows = _read_jsonl(out)
    kinds = {r.get("shuffle_kind") for r in rows if r.get("kind") == "shortcut_row"}
    assert "clean" in kinds and len(kinds) >= 4


if __name__ == "__main__":
    failures = 0
    setup_module()
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:                                                       # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
