"""P0 CLI: relation recoverability probe for relation-residual screening.

For each teacher, sample sparse token edges and train tiny pair probes for:
  - depth order: whether q is farther than p;
  - cross-view match: whether cross-camera tokens refer to nearby 3D points.

This is a screening metric only. It does not replace downstream spatial QA or
planning evaluation.
"""

from __future__ import annotations

import argparse

from geodistill.data.loader import load_scenes
from geodistill.teacher import (
    build_hard_quantile_teacher,
    build_entropic_ot_teacher,
    build_full_ntlfgt_teacher,
    OTConfig,
)
from geodistill.probe import run_relation_probe, RelationProbeConfig
from geodistill.utils import JsonlWriter, run_envelope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--epsilon_ot", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--depth_margin", type=float, default=0.5)
    ap.add_argument("--cross_match_thresh", type=float, default=1.0)
    ap.add_argument("--out", default="runs/geodistill/p0/relation_probe.jsonl")
    args = ap.parse_args()

    scenes = load_scenes(args.source, args.scenes, args.seed, args.config)
    ot_cfg = OTConfig(epsilon_ot=args.epsilon_ot, gamma=args.gamma)
    builders = {
        "hard_quantile": build_hard_quantile_teacher,
        "entropic_ot": lambda s: build_entropic_ot_teacher(s, ot_cfg),
        "full_ntl_fgt": lambda s: build_full_ntlfgt_teacher(s, ot_cfg),
    }
    pcfg = RelationProbeConfig(
        hidden=args.hidden,
        epochs=args.epochs,
        seed=args.seed,
        depth_margin=args.depth_margin,
        cross_match_thresh=args.cross_match_thresh,
    )

    with JsonlWriter(args.out) as w:
        w.write({**run_envelope("relation_probe", args.source,
                                {"scenes": args.scenes, "hidden": args.hidden,
                                 "epochs": args.epochs, "gamma": args.gamma,
                                 "depth_margin": args.depth_margin,
                                 "cross_match_thresh": args.cross_match_thresh}),
                 "kind": "header"})
        for name, builder in builders.items():
            res = run_relation_probe(scenes, builder, pcfg)
            res["teacher"] = name
            res["kind"] = "relation_probe"
            w.write(res)
            o = res["order"]
            c = res["cross_view"]
            print(f"[{name}] order_acc={o['accuracy']:.3f} "
                  f"order_gain={o['gain']:.3f} n={o['n']} | "
                  f"cross_acc={c['accuracy']:.3f} cross_gain={c['gain']:.3f} n={c['n']} | "
                  f"edges/tok={res['mean_edges_per_token']:.1f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
