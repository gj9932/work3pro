"""One-shot P0 smoke test: run R²AC tests + all 3 CLIs on the synthetic fixture +
collect tables, asserting the loop closes end-to-end on CPU.

  .venv_p0/bin/python -m scripts.geodistill.run_p0_smoke

Exits non-zero if any stage fails. Produces JSONL + Markdown under runs/geodistill/.
This proves the pipeline is correct; it does NOT produce paper numbers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/geodistill/p0"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"FAILED ({r.returncode}): {' '.join(cmd)}")
        sys.exit(r.returncode)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. unit tests
    run([PY, "test/geodistill/r2ac_test.py"])
    run([PY, "test/geodistill/teacher_test.py"])
    # 2. teacher stats
    run([PY, "-m", "scripts.geodistill.run_teacher_stats", "--source", "synthetic",
         "--scenes", "8", "--seed", "0", "--out", str(OUT / "teacher_stats.jsonl")])
    # 3. probe
    run([PY, "-m", "scripts.geodistill.run_probe", "--source", "synthetic",
         "--scenes", "16", "--seed", "0", "--hidden", "64", "--epochs", "300",
         "--out", str(OUT / "probe.jsonl")])
    # 4. depth heads
    run([PY, "-m", "scripts.geodistill.run_depth_heads", "--source", "synthetic",
         "--scenes", "16", "--seed", "0", "--teacher", "hard_quantile", "--epochs", "250",
         "--out", str(OUT / "depth_heads.jsonl")])
    # 5. relation probe
    run([PY, "-m", "scripts.geodistill.run_relation_probe", "--source", "synthetic",
         "--scenes", "16", "--seed", "0", "--hidden", "64", "--epochs", "150",
         "--out", str(OUT / "relation_probe.jsonl")])
    # 6. collect tables
    run([PY, "-m", "scripts.geodistill.collect_tables",
         "--in_dir", str(OUT), "--out_dir", str(ROOT / "runs/geodistill/tables")])

    print("\n=== P0 SMOKE OK ===")
    print("tables in runs/geodistill/tables/ (synthetic-labelled). Real numbers need GPU+nuScenes.")


if __name__ == "__main__":
    main()
