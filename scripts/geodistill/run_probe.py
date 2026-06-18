"""P0 CLI: token geometry probe across the 3 teachers (paper item 3 / Table 2).

For each teacher, train a frozen-token depth probe and report AbsRel/RMSE/delta1
(overall + 0-10/10-30/30m+ bins) and Spearman(q^OT, |depth error|).

Usage:
  .venv_p0/bin/python -m scripts.geodistill.run_probe \
      --source synthetic --scenes 12 --seed 0 --hidden 64 \
      --out runs/geodistill/p0/probe.jsonl
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
from geodistill.probe import run_token_geometry_probe, ProbeConfig
from geodistill.utils import JsonlWriter, run_envelope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--hidden", type=int, default=0, help="0=linear probe, >0=2-layer")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--epsilon_ot", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--out", default="runs/geodistill/p0/probe.jsonl")
    args = ap.parse_args()

    scenes = load_scenes(args.source, args.scenes, args.seed, args.config)
    ot_cfg = OTConfig(epsilon_ot=args.epsilon_ot, gamma=args.gamma)

    builders = {
        "hard_quantile": build_hard_quantile_teacher,
        "entropic_ot": lambda s: build_entropic_ot_teacher(s, ot_cfg),
        "full_ntl_fgt": lambda s: build_full_ntlfgt_teacher(s, ot_cfg),
    }
    pcfg = ProbeConfig(hidden=args.hidden, epochs=args.epochs, seed=args.seed)

    with JsonlWriter(args.out) as w:
        w.write({**run_envelope("token_geometry_probe", args.source,
                                {"scenes": args.scenes, "hidden": args.hidden,
                                 "epochs": args.epochs, "gamma": args.gamma}),
                 "kind": "header"})
        for name, builder in builders.items():
            res = run_token_geometry_probe(scenes, builder, pcfg)
            res["teacher"] = name
            res["kind"] = "probe"
            w.write(res)
            if "error" in res:
                print(f"[{name}] ERROR {res['error']}")
                continue
            o = res["overall"]
            b = res["binned"]
            print(f"[{name}] AbsRel={o['AbsRel']:.4f} RMSE={o['RMSE']:.3f} d1={o['delta1']:.3f} "
                  f"| 0-10={b['0_10m']['AbsRel']:.3f} 10-30={b['10_30m']['AbsRel']:.3f} "
                  f"30+={b['30m_plus']['AbsRel']:.3f} | rho(qOT,err)={res['spearman_qOT_vs_abserr']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
