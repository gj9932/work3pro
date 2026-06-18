"""Efficiency eval (paper §4.5 Table 6).

For each model variant report:
- trainable parameter count
- peak GPU memory (TBD locally; ``torch.cuda.max_memory_allocated`` on GPU)
- 6-camera latency
- FLOPs (``fvcore`` if available, otherwise TBD)

Locally we can compute parameter counts of ``GeoDistillVLM`` itself; the
external-depth slot (UniDepthV2-L) is a TBD until the GPU box is online.
"""

from __future__ import annotations

import argparse
import time

import torch

from geodistill.data.loader import load_scenes
from geodistill.trainers import GeoDistillConfig, GeoDistillVLM, trainable_params
from geodistill.utils import JsonlWriter, run_envelope, TBD


__all__ = ["run_table6"]


def _build_lpga(scene) -> GeoDistillVLM:
    from geodistill.trainers.build_lpga import auto_pe_dims
    cfg = GeoDistillConfig(
        hidden_size=int(scene.features.shape[-1]),
        d_bottleneck=64, d_e_cam=8, d_e_uv=16, d_e_calib=16, d_e_ego=8,
        num_cameras=int(scene.num_cameras),
        pe_d_xyz=auto_pe_dims(int(scene.features.shape[-1])),
        rel_d_edge=32, rel_d_hidden=32,
    )
    return GeoDistillVLM(cfg)


def _measure_latency(model: GeoDistillVLM, scene, n: int = 5) -> float:
    """Forward latency over ``n`` scenes (CPU-friendly)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.forward_lpga(
                h_img=scene.features, token_uv=scene.token_uv, token_box=scene.token_box,
                token_camera_id=scene.token_camera_id, K=scene.K, cam_to_ego=scene.cam_to_ego,
                image_hw=scene.image_hw, v_ego=scene.v_ego,
            )
            z = out["g_mono"].new_zeros((scene.num_tokens, model.cfg.d_bottleneck))
            _ = model.forward_inject(scene.features, out["c_hat"], z)
        times.append(time.perf_counter() - t0)
    return float(sum(times) / len(times))


def run_table6(scenes, *, out_path: str, source: str = "synthetic"):
    with JsonlWriter(out_path) as w:
        w.write({**run_envelope("eval_efficiency", source,
                                  {"scenes": len(scenes)}),
                 "kind": "header"})
        s0 = scenes[0]
        # 1) Qwen + LPGA (us)
        model = _build_lpga(s0)
        params = trainable_params(model)
        latency = _measure_latency(model, s0)
        w.write({"kind": "table6_row", "method": "Qwen + LPGA",
                  "external_depth": "none",
                  "added_params": params,
                  "params_relative_to_unidepthv2_l": TBD,
                  "memory_mb": TBD,
                  "latency_s": latency,
                  "flops": TBD,
                  "AbsRel_30m_plus": TBD, "QA_overall": TBD})
        # 2) Qwen baseline (no LPGA)
        w.write({"kind": "table6_row", "method": "Qwen2.5-VL",
                  "external_depth": "none", "added_params": 0,
                  "params_relative_to_unidepthv2_l": "N/A", "memory_mb": TBD,
                  "latency_s": TBD, "flops": TBD,
                  "AbsRel_30m_plus": "N/A", "QA_overall": TBD})
        # 3) SpaceDrive-style (UniDepthV2-L)
        w.write({"kind": "table6_row", "method": "SpaceDrive-style",
                  "external_depth": "UniDepthV2-L",
                  "added_params": TBD,
                  "params_relative_to_unidepthv2_l": "100%",
                  "memory_mb": TBD, "latency_s": TBD, "flops": TBD,
                  "AbsRel_30m_plus": TBD, "QA_overall": TBD,
                  "note": "Run on GPU with third_party/SpaceDrive initialized."})
        # 4) LiDAR-calibrated SpaceDrive
        w.write({"kind": "table6_row", "method": "LiDAR-calibrated SpaceDrive",
                  "external_depth": "UniDepthV2-L + calibration head (512)",
                  "added_params": TBD,
                  "params_relative_to_unidepthv2_l": TBD,
                  "memory_mb": TBD, "latency_s": TBD, "flops": TBD,
                  "AbsRel_30m_plus": TBD, "QA_overall": TBD,
                  "note": "Run on GPU; calibration head bottleneck = LPGA bottleneck (512)."})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/geodistill/eval/table6_efficiency.jsonl")
    args = ap.parse_args()
    scenes = load_scenes(args.source, args.scenes, args.seed, args.config)
    run_table6(scenes, out_path=args.out, source=args.source)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
