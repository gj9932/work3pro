# GeoDistill-VLM — P0 Experiment Harness

P0 validates whether the paper design is worth pursuing, by measuring the three
things a CVPR reviewer would demand first (see `paper/work3pro_cvpr2027_draft5_zh_qwen(1).md`):

1. **Teacher construction** — 3 variants with a shared output contract:
   - `hard_quantile` — hard projection + 10% quantile (baseline)
   - `entropic_ot` — entropic OT, `gamma=0` (feature-only soft coupling)
   - `full_ntl_fgt` — FGW-inspired feature + fixed-structure coupling
2. **Per-teacher statistics** (paper item 2 / Tables 7–8): valid-token ratio, depth
   distribution, 0-10/10-30/30m+ counts, `q_conc`/`q_mass`/`q_teacher` distributions,
   build time, extra memory, candidate anchors/token, Sinkhorn iterations, sparse
   edges/token.
3. **Token geometry probe** (item 3 / Table 2): AbsRel, RMSE, delta1, depth bins,
   Spearman(`q^OT`, |depth error|).
4. **Depth-head model comparison** (item 4 / §4.4): raw / fixed-log / fixed-power-warp
   / R²AC(predicted a) / external-depth surrogate / LPGA.

## Important: what runs where

This repo currently has **no GPU, no nuScenes, no Qwen2.5-VL weights, no `transformers`**.
Therefore the harness has two data sources behind one flag:

- `--source synthetic` (**default, CPU**) — a geometrically self-consistent fixture
  (`geodistill/data/synthetic.py`). It proves the entire loop closes and is correct.
  **Its numbers are NOT paper results** and every generated table is banner-labelled
  `SOURCE: SYNTHETIC FIXTURE`.
- `--source nuscenes` (**GPU box only**) — routes through
  `geodistill/data/nuscenes_adapter.py`, which composes the existing
  `NuScenesQwenDataset` + frozen `QwenVisualFrozen` + LiDAR loader to emit the SAME
  `TokenScene` contract. On this CPU box it fails fast with a clear message.

No numbers are ever fabricated; real cells stay `TBD` until run on the GPU box.

## Quick start (CPU smoke — closes the whole loop)

```bash
.venv_p0/bin/python -m scripts.geodistill.run_p0_smoke
```

This runs the R²AC + teacher unit tests, all 3 CLIs on the synthetic fixture, and
renders the Markdown tables under `runs/geodistill/tables/`.

## Individual CLIs

```bash
# item 2 — teacher statistics (Tables 7-8)
.venv_p0/bin/python -m scripts.geodistill.run_teacher_stats \
    --source synthetic --scenes 8 --seed 0 \
    --out runs/geodistill/p0/teacher_stats.jsonl

# item 3 — token geometry probe (Table 2)
.venv_p0/bin/python -m scripts.geodistill.run_probe \
    --source synthetic --scenes 16 --hidden 64 --epochs 400 \
    --out runs/geodistill/p0/probe.jsonl

# item 4 — depth-head comparison (§4.4)
.venv_p0/bin/python -m scripts.geodistill.run_depth_heads \
    --source synthetic --scenes 16 --teacher hard_quantile --epochs 300 \
    --out runs/geodistill/p0/depth_heads.jsonl

# render tables from JSONL
.venv_p0/bin/python -m scripts.geodistill.collect_tables \
    --in_dir runs/geodistill/p0 --out_dir runs/geodistill/tables
```

## Running for real (GPU box with nuScenes + Qwen2.5-VL)

1. Install `requirements_geodistill.txt` in a Python ≥3.10 env (GPU torch).
2. Edit `configs/geodistill/geodistill_qwen25vl_nuscenes.yaml`: set `dataset.root`,
   `dataset.info_path` (use `tools/gen_info.py` to build the info pickle), and the
   Qwen model id / local path.
3. Add `--source nuscenes --config configs/geodistill/geodistill_qwen25vl_nuscenes.yaml`
   to any CLI above. Everything downstream (teachers, probe, heads, tables) is
   unchanged — only the data factory in `geodistill/data/loader.py` switches.
4. The `unidepth_pe` row in the depth-head comparison is an external-monocular-depth
   **surrogate** on synthetic data; on the GPU box wire it to real UniDepthV2 outputs
   (marked TBD in `geodistill/models/depth_head_eval.py`).

## Layout

```
geodistill/
  core/       r2ac.py risk_field.py camera.py sinkhorn.py   # pure math (R²AC verified)
  data/       contract.py synthetic.py nuscenes_adapter.py loader.py
  teacher/    base.py hard_quantile.py entropic_ot.py full_ntlfgt.py ot_common.py
              stats.py edges.py
  probe/      token_geometry.py metrics.py
  models/     depth_heads.py depth_head_eval.py qwen_visual_frozen.py
  utils/      logging_jsonl.py
scripts/geodistill/   run_teacher_stats.py run_probe.py run_depth_heads.py
                      collect_tables.py run_p0_smoke.py
test/geodistill/      r2ac_test.py teacher_test.py
```

## JSONL log fields

Every run writes a `header` row (envelope: experiment, source, git commit, host,
config) then one row per result. Field names match the table renderers in
`collect_tables.py`. See that file for the exact schema per experiment.
