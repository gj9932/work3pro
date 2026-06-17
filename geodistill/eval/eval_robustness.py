"""Robustness eval (paper §4.5).

For each corruption type, reports the ratio ``Performance_corrupt / Performance_clean``
on whichever metric the user selected (default: token-level AbsRel via
:func:`geodistill.eval.eval_token_geometry_probe.run_token_probe`).

Runs locally on the synthetic fixture; produces real numbers on the GPU box
once a checkpoint is loaded.
"""

from __future__ import annotations

import argparse

from geodistill.data.loader import load_scenes
from geodistill.eval.eval_token_geometry_probe import run_token_probe
from geodistill.probe.token_geometry import ProbeConfig
from geodistill.trainers import StageCConfig, apply_corruption
from geodistill.utils import JsonlWriter, run_envelope, TBD


__all__ = ["CORRUPTIONS", "run_robustness"]


CORRUPTIONS = (
    "single_camera_drop",
    "multi_camera_drop",
    "front_only",
    "image_occlusion",
    "calibration_noise",
    "low_light",
)


def _absrel(scene, teacher_name: str, cfg: ProbeConfig) -> float:
    out = run_token_probe(scene, teacher_name, model=None, cfg=cfg)
    metrics = out.h_img_metrics
    if not isinstance(metrics, dict) or "overall" not in metrics:
        return float("nan")
    return float(metrics["overall"]["AbsRel"])


def run_robustness(scenes, *, teacher_name: str, out_path: str, source: str = "synthetic",
                   probe_cfg: ProbeConfig = ProbeConfig(),
                   corruptions=CORRUPTIONS):
    with JsonlWriter(out_path) as w:
        w.write({**run_envelope("eval_robustness", source,
                                  {"teacher": teacher_name, "corruptions": list(corruptions)}),
                 "kind": "header"})
        clean_scores = []
        for s in scenes:
            clean_scores.append(_absrel(s, teacher_name, probe_cfg))
        clean_mean = sum(v for v in clean_scores if v == v) / max(1, sum(1 for v in clean_scores if v == v))
        w.write({"kind": "robustness_row", "corruption": "clean", "AbsRel_mean": clean_mean})

        for corr in corruptions:
            corr_cfg = StageCConfig(corruption=corr)
            scores = []
            for s in scenes:
                deg = apply_corruption(s, corr_cfg)
                # The token probe needs gt_depth/valid; apply_corruption preserves them.
                scores.append(_absrel(deg, teacher_name, probe_cfg))
            corr_mean = sum(v for v in scores if v == v) / max(1, sum(1 for v in scores if v == v))
            ratio = (clean_mean / corr_mean) if corr_mean and corr_mean == corr_mean else float("nan")
            w.write({"kind": "robustness_row", "corruption": corr,
                      "AbsRel_mean": corr_mean,
                      "robustness_ratio": ratio,
                      "note": "synthetic fixture; real ratios need GPU + nuScenes" if source == "synthetic" else None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--teacher", default="hard_quantile",
                     choices=("hard_quantile", "entropic_ot", "full_ntl_fgt"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="runs/geodistill/eval/robustness.jsonl")
    args = ap.parse_args()
    scenes = load_scenes(args.source, args.scenes, args.seed, args.config)
    run_robustness(scenes, teacher_name=args.teacher, out_path=args.out, source=args.source,
                    probe_cfg=ProbeConfig(epochs=args.epochs))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
