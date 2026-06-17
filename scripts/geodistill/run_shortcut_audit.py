"""CLI: shortcut audit (paper §4.7) on the synthetic fixture or nuScenes.

Usage:
  .venv_p0/bin/python -m scripts.geodistill.run_shortcut_audit \\
      --source synthetic --scenes 8 --seed 0 \\
      --teacher hard_quantile \\
      --out runs/geodistill/shortcut/all.jsonl

Each row is one (scene, shuffle_kind) result. ``shuffle_kind=clean`` is the
baseline against which the others are compared.

The metric is intentionally cheap (token-level AbsRel of the teacher against
GT depth, plus mean transported mass / mean q_teacher), because the goal is
just to demonstrate that each shuffle perturbs the supervision channel it
should perturb. Real downstream metrics live in ``eval_*`` (paper §4.5).
"""

from __future__ import annotations

import argparse
from typing import Callable

import torch

from geodistill.data.loader import load_scenes
from geodistill.eval.shortcut_audit import (
    SHUFFLE_KINDS, ShuffleConfig, apply_shuffle, ALLOCATION_KEYS,
)
from geodistill.geometry.cross_view_label import build_edge_labels
from geodistill.geometry.edge_sampler_v2 import EdgeBudgetV2, sample_edges_v2
from geodistill.teacher import (
    build_hard_quantile_teacher, build_entropic_ot_teacher, build_full_ntlfgt_teacher,
    build_candidate_sets, OTConfig,
)
from geodistill.teacher.ot_common import build_ot_teacher
from geodistill.utils import JsonlWriter, run_envelope


TEACHERS = ("hard_quantile", "entropic_ot", "full_ntl_fgt")


def _absrel(d_pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    keep = mask & torch.isfinite(d_pred) & torch.isfinite(gt) & (gt > 1e-3)
    if int(keep.sum()) == 0:
        return float("nan")
    return float(((d_pred[keep] - gt[keep]).abs() / gt[keep]).mean())


def _build_teacher(name: str, scene, ot_cfg: OTConfig, structure_shuffle_seed: int | None = None):
    if name == "hard_quantile":
        return build_hard_quantile_teacher(scene)
    if name == "entropic_ot":
        return build_entropic_ot_teacher(scene, ot_cfg)
    if name == "full_ntl_fgt":
        if structure_shuffle_seed is not None:
            cands = build_candidate_sets(scene, dilate_px=ot_cfg.cand_dilate_px)
            return build_ot_teacher(scene, ot_cfg, name="full_ntl_fgt",
                                    cands=cands, structure_shuffle_seed=structure_shuffle_seed)
        return build_full_ntlfgt_teacher(scene, ot_cfg)
    raise ValueError(f"unknown teacher {name!r}")


def _eval(scene, teacher, gt_depth, gt_valid):
    return {
        "absrel_vs_gt": _absrel(teacher.d_teacher, gt_depth, teacher.m_T & gt_valid),
        "valid_token_ratio": float(teacher.m_T.float().mean()),
        "mean_q_teacher": float(teacher.q_teacher[teacher.m_T].mean()) if int(teacher.m_T.sum()) else float("nan"),
        "tokens_with_label": int(teacher.diag.get("tokens_with_label", int(teacher.m_T.sum()))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--teacher", default="hard_quantile", choices=TEACHERS)
    ap.add_argument("--epsilon_ot", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--allocation_key", default="r_p", choices=list(ALLOCATION_KEYS))
    ap.add_argument("--kinds", nargs="+", default=list(SHUFFLE_KINDS))
    ap.add_argument("--out", default="runs/geodistill/shortcut/all.jsonl")
    args = ap.parse_args()

    scenes = load_scenes(args.source, args.scenes, args.seed, args.config)
    ot_cfg = OTConfig(epsilon_ot=args.epsilon_ot, gamma=args.gamma)

    with JsonlWriter(args.out) as w:
        w.write({**run_envelope("shortcut_audit", args.source,
                                  {"scenes": args.scenes, "seed": args.seed,
                                   "teacher": args.teacher, "kinds": args.kinds,
                                   "allocation_key": args.allocation_key}),
                 "kind": "header"})
        for si, scene in enumerate(scenes):
            gt_depth = scene.gt_depth
            gt_valid = scene.gt_valid
            if gt_depth is None or gt_valid is None:
                continue

            # Clean baseline
            teacher_clean = _build_teacher(args.teacher, scene, ot_cfg)
            row_clean = {"kind": "shortcut_row", "scene_idx": si, "scene_id": scene.scene_id,
                          "teacher": args.teacher, "shuffle_kind": "clean",
                          **_eval(scene, teacher_clean, gt_depth, gt_valid)}
            w.write(row_clean)

            for k in args.kinds:
                cfg = ShuffleConfig(kind=k, seed=args.seed + si,
                                    allocation_key=args.allocation_key)
                # Special path: transport_structure rebuilds the FGT teacher with the hook
                if k == "transport_structure":
                    if args.teacher != "full_ntl_fgt":
                        # only meaningful for FGT structure term
                        teacher_shuf = teacher_clean
                    else:
                        teacher_shuf = _build_teacher(args.teacher, scene, ot_cfg,
                                                      structure_shuffle_seed=args.seed + si)
                    new_scene, _, meta = scene, teacher_shuf, {"kind": k}
                else:
                    new_scene, base_teacher, meta = apply_shuffle(scene, teacher_clean, cfg)
                    # Re-build teacher only for kinds that touch the scene/teacher build path.
                    if k in ("image_embeds", "calibration"):
                        teacher_shuf = _build_teacher(args.teacher, new_scene, ot_cfg)
                    else:
                        teacher_shuf = base_teacher

                row = {"kind": "shortcut_row", "scene_idx": si, "scene_id": scene.scene_id,
                        "teacher": args.teacher, "shuffle_kind": k,
                        "shuffle_meta": {sk: (sv if not isinstance(sv, torch.Tensor) else sv.tolist())
                                          for sk, sv in meta.items()},
                        **_eval(new_scene, teacher_shuf, gt_depth, gt_valid)}
                w.write(row)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
