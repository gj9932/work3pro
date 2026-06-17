"""P0 CLI: depth-representation model comparison (paper item 4 / §4.4 baselines, Table 2).

Trains/evaluates each depth head on frozen tokens and reports metric depth quality:
  raw / log / power_warp / r2ac (predicted a) / unidepth_pe (external surrogate) / lpga

Usage:
  .venv_p0/bin/python -m scripts.geodistill.run_depth_heads \
      --source synthetic --scenes 12 --teacher hard_quantile \
      --out runs/geodistill/p0/depth_heads.jsonl
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
from geodistill.models.depth_head_eval import train_eval_head, MODEL_VARIANTS
from geodistill.models.depth_heads import HeadTrainConfig
from geodistill.core.risk_field import RiskFieldConfig
from geodistill.utils import JsonlWriter, run_envelope


TEACHERS = {
    "hard_quantile": build_hard_quantile_teacher,
    "entropic_ot": lambda s, cfg: build_entropic_ot_teacher(s, cfg),
    "full_ntl_fgt": lambda s, cfg: build_full_ntlfgt_teacher(s, cfg),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--teacher", default="hard_quantile", choices=list(TEACHERS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--epsilon_ot", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--out", default="runs/geodistill/p0/depth_heads.jsonl")
    args = ap.parse_args()

    scenes = load_scenes(args.source, args.scenes, args.seed, args.config)
    ot_cfg = OTConfig(epsilon_ot=args.epsilon_ot, gamma=args.gamma)
    if args.teacher == "hard_quantile":
        builder = TEACHERS["hard_quantile"]
    else:
        builder = lambda s: TEACHERS[args.teacher](s, ot_cfg)  # noqa: E731

    hcfg = HeadTrainConfig(epochs=args.epochs)
    rcfg = RiskFieldConfig()

    with JsonlWriter(args.out) as w:
        w.write({**run_envelope("depth_head_comparison", args.source,
                                {"scenes": args.scenes, "teacher": args.teacher,
                                 "epochs": args.epochs}),
                 "kind": "header"})
        for h in MODEL_VARIANTS:
            res = train_eval_head(h, scenes, builder, hcfg, rcfg, seed=args.seed)
            res["kind"] = "depth_head"
            res["teacher_used"] = args.teacher
            w.write(res)
            if "error" in res:
                print(f"[{h}] ERROR {res['error']}")
                continue
            o = res["overall"]
            b = res["binned"]
            print(f"[{h:12s}] params={res['trainable_params']:<5d} ext={res['external_depth']!s:<5s} "
                  f"AbsRel={o['AbsRel']:.4f} RMSE={o['RMSE']:.3f} d1={o['delta1']:.3f} "
                  f"| 30+={b['30m_plus']['AbsRel']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
