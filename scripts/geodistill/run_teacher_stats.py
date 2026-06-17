"""P0 CLI: build all 3 teachers and dump per-teacher statistics (paper item 2 / Table 7-8).

Usage:
  .venv_p0/bin/python -m scripts.geodistill.run_teacher_stats \
      --source synthetic --scenes 8 --seed 0 --out runs/geodistill/p0/teacher_stats.jsonl

Outputs one JSONL row per (teacher) aggregated over scenes, plus per-scene rows.
Real numbers only; --source nuscenes routes to the real path (GPU box).
"""

from __future__ import annotations

import argparse
import resource
import statistics

import torch

from geodistill.data.loader import load_scenes
from geodistill.teacher import (
    build_hard_quantile_teacher,
    build_entropic_ot_teacher,
    build_full_ntlfgt_teacher,
    build_candidate_sets,
    OTConfig,
)
from geodistill.teacher.stats import teacher_statistics, _dist
from geodistill.teacher.edges import sample_edges, EdgeBudget
from geodistill.utils import JsonlWriter, run_envelope


def _maxrss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kB
    import sys
    return rss if sys.platform == "darwin" else rss * 1024


def build_all_teachers(scene, ot_cfg: OTConfig):
    cands_hard = build_candidate_sets(scene, dilate_px=0.0)
    cands_ot = build_candidate_sets(scene, dilate_px=ot_cfg.cand_dilate_px)
    return {
        "hard_quantile": build_hard_quantile_teacher(scene, cands=cands_hard),
        "entropic_ot": build_entropic_ot_teacher(scene, ot_cfg, cands=cands_ot),
        "full_ntl_fgt": build_full_ntlfgt_teacher(scene, ot_cfg, cands=cands_ot),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="geodistill yaml (nuscenes source)")
    ap.add_argument("--epsilon_ot", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--out", default="runs/geodistill/p0/teacher_stats.jsonl")
    args = ap.parse_args()

    scenes = load_scenes(args.source, args.scenes, args.seed, args.config)
    ot_cfg = OTConfig(epsilon_ot=args.epsilon_ot, gamma=args.gamma)
    budget = EdgeBudget()

    # aggregate accumulators per teacher
    agg: dict[str, list[dict]] = {"hard_quantile": [], "entropic_ot": [], "full_ntl_fgt": []}

    with JsonlWriter(args.out) as w:
        w.write({**run_envelope("teacher_stats", args.source,
                                {"scenes": args.scenes, "seed": args.seed,
                                 "epsilon_ot": args.epsilon_ot, "gamma": args.gamma}),
                 "kind": "header"})
        rss0 = _maxrss_bytes()
        for si, scene in enumerate(scenes):
            teachers = build_all_teachers(scene, ot_cfg)
            for name, t in teachers.items():
                _edges, epp = sample_edges(scene, t, budget)
                rss = _maxrss_bytes()
                st = teacher_statistics(scene, t, extra_memory_bytes=max(0, rss - rss0),
                                        edges_per_token=epp)
                st.update({"kind": "per_scene", "scene_idx": si, "scene_id": scene.scene_id})
                w.write(st)
                agg[name].append(st)

        # write aggregate rows (mean over scenes for scalar fields)
        for name, rows in agg.items():
            if not rows:
                continue
            agg_row = {"kind": "aggregate", "teacher": name, "n_scenes": len(rows)}
            scalar_keys = ["valid_token_ratio", "build_time_s", "candidate_anchors_per_token",
                           "num_anchors", "sinkhorn_iterations", "transported_mass",
                           "extra_memory_mb", "sparse_edges_per_token"]
            for k in scalar_keys:
                vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and r[k] == r[k]]
                agg_row[k + "_mean"] = statistics.mean(vals) if vals else float("nan")
            # depth bins summed
            for bk in ["count_0_10m", "count_10_30m", "count_30m_plus"]:
                agg_row[bk + "_sum"] = sum(r["depth_bins"][bk] for r in rows)
            # q distributions: mean of per-scene means
            for q in ["q_conc_dist", "q_mass_dist", "q_teacher_dist", "depth_dist"]:
                means = [r[q]["mean"] for r in rows if r[q]["n"] > 0]
                agg_row[q + "_mean"] = statistics.mean(means) if means else float("nan")
            w.write(agg_row)
            print(f"[{name}] valid={agg_row['valid_token_ratio_mean']:.3f} "
                  f"time/scene={agg_row['build_time_s_mean']:.3f}s "
                  f"anchors/tok={agg_row['candidate_anchors_per_token_mean']:.1f} "
                  f"sinkhorn_it={agg_row['sinkhorn_iterations_mean']:.0f} "
                  f"edges/tok={agg_row['sparse_edges_per_token_mean']:.1f}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
