"""Pre-build per-edge relation labels (paper §3.9) on top of the token cache.

Reads ``runs/geodistill/cache/token_labels/*.pt`` (output of
``build_token_label_cache``) and for each sample writes a sibling
``runs/geodistill/cache/relation_labels/<name>.pt`` containing:

    {
      "edges": (E, 2) long,
      "labels": EdgeLabels (paper §3.9 — cross_view / depth_order / topology / occlusion + reliability),
      "meta": {scene_id, n_edges, edge_types}
    }

The trainer can either:
- precache (cheaper, recommended for stage A0/A1 where edges are sampled per epoch), or
- regenerate on the fly via ``geodistill.geometry.edge_sampler_v2.sample_edges_v2``.

Local CPU can run this against the synthetic fixture (the .pt cache lives
wherever you put it); on the GPU box you point it at the real cache dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from geodistill.geometry.cross_view_label import build_edge_labels
from geodistill.geometry.edge_sampler_v2 import EdgeBudgetV2, sample_edges_v2
from geodistill.runtime.config import load_config_with_extends
from geodistill.utils import JsonlWriter, run_envelope


def main():
    ap = argparse.ArgumentParser(description="Build per-edge relation label cache (paper §3.9)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--in_dir", default="runs/geodistill/cache/token_labels")
    ap.add_argument("--out_dir", default="runs/geodistill/cache/relation_labels")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    yaml_cfg = load_config_with_extends(args.config)
    rel_cfg = yaml_cfg.get("relation", {})
    edges_cfg = rel_cfg.get("edge_types", {})
    budget = EdgeBudgetV2(
        P_local=int(edges_cfg.get("P_local", 6)),
        P_ray=int(edges_cfg.get("P_ray", 4)),
        P_cross=int(edges_cfg.get("P_cross", 4)),
        P_hard=int(edges_cfg.get("P_hard", 1)),
        P_far=int(edges_cfg.get("P_far", 1)),
    )
    voxel_size = float(rel_cfg.get("cross_view_voxel", 0.5))
    tau_depth = float(rel_cfg.get("tau_depth", 1.0))

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.pt"))
    if args.limit:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"no .pt found under {in_dir}; run build_token_label_cache first")

    log_path = out_dir / "build_log.jsonl"
    with JsonlWriter(log_path) as w:
        w.write({**run_envelope("build_relation_label_cache", "nuscenes",
                                  {"in_dir": str(in_dir), "out_dir": str(out_dir),
                                   "budget": vars(budget),
                                   "voxel_size": voxel_size, "tau_depth": tau_depth,
                                   "n_inputs": len(files)}),
                 "kind": "header"})
        for i, p in enumerate(files):
            payload = torch.load(p, map_location="cpu")
            scene = payload["scene"]
            teacher = payload["teacher"]
            edges, types, mean_epp = sample_edges_v2(scene, teacher, budget, seed=args.seed + i)
            labels = build_edge_labels(scene, teacher, edges,
                                          voxel_size=voxel_size, tau_depth=tau_depth)

            out_path = out_dir / p.name
            torch.save({"edges": labels.edge_index, "labels": labels,
                          "edge_types": types,
                          "meta": {"src": str(p), "scene_id": scene.scene_id,
                                    "n_edges": int(labels.edge_index.shape[0]),
                                    "edges_per_token": float(mean_epp)}},
                         out_path)
            w.write({"kind": "wrote", "src": str(p), "out": str(out_path),
                      "n_edges": int(labels.edge_index.shape[0]),
                      "edges_per_token": float(mean_epp),
                      "n_cross_view_pos": int(labels.cross_view.sum())})
            if i % 50 == 0:
                print(f"[i={i}/{len(files)}] edges={labels.edge_index.shape[0]} "
                      f"cv_pos={int(labels.cross_view.sum())}")

    print(f"[relation-label-cache] done → {out_dir}")


if __name__ == "__main__":
    main()
