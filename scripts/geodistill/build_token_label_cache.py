"""Pre-build per-token LiDAR labels (paper §3.4) for a nuScenes split.

Composes:
- :class:`dataset.geodistill.NuScenesQwenDataset`            (multi-cam + Qwen processor)
- :class:`geodistill.models.QwenVisualFrozen`                 (frozen vision tower + merger)
- :func:`geodistill.runtime.scene_stream.sample_to_token_scene`
- :func:`geodistill.runtime.ego_motion.accumulate`            (multi-sweep LiDAR)
- :func:`geodistill.runtime.teacher_factory.build_teacher_fn`

Each sample produces a ``{scene: TokenScene, teacher: TeacherOutput}`` dict
saved as ``<scene_token>_<sample_idx>.pt`` so
:func:`geodistill.runtime.data_factory.iter_scenes` can stream them with
``--source cached_nuscenes`` during training.

Local CPU has no transformers / nuScenes weights / data, so this CLI raises
immediately when those imports fail. On the GPU box install
``requirements_geodistill.txt`` and supply the yaml's ``dataset.root`` /
``dataset.info_path`` / ``base_vlm.model_id``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from geodistill.utils import JsonlWriter, run_envelope


def main():
    ap = argparse.ArgumentParser(description="Build token-level LiDAR teacher cache (paper §3.4)")
    ap.add_argument("--config", required=True, help="geodistill yaml with dataset.* and base_vlm.*")
    ap.add_argument("--split", default="train", help="logical split tag (record-only)")
    ap.add_argument("--out_dir", default="runs/geodistill/cache/token_labels")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of samples (debug)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--teacher", default=None,
                     help="override the teacher kind (hard_quantile / entropic_ot / full_ntl_fgt). "
                          "If unset, reads ``teacher.kind`` from yaml.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    log_path = out / "build_log.jsonl"

    try:
        import transformers                                                              # noqa: F401
        from transformers import AutoProcessor
        from dataset.geodistill import NuScenesQwenDataset
        from geodistill.models.qwen_visual_frozen import QwenVisualFrozen
        from geodistill.runtime.config import load_config_with_extends
        from geodistill.runtime.scene_stream import sample_to_token_scene
        from geodistill.runtime.ego_motion import accumulate, MultiSweepLoader
        from geodistill.runtime.teacher_factory import build_teacher_fn
    except Exception as exc:                                                       # noqa: BLE001
        raise ImportError(
            "build_token_label_cache requires transformers + nuScenes data + Qwen weights. "
            "Run on the GPU box."
        ) from exc

    yaml_cfg = load_config_with_extends(args.config)
    if args.teacher:
        yaml_cfg.setdefault("teacher", {})["kind"] = args.teacher

    ds_cfg = yaml_cfg["dataset"]
    base = yaml_cfg["base_vlm"]
    processor = AutoProcessor.from_pretrained(base["model_id"], trust_remote_code=True)
    qwen = QwenVisualFrozen(model_id=base["model_id"],
                              torch_dtype=str(base.get("torch_dtype", "bfloat16")))
    if args.device == "cuda" and torch.cuda.is_available():
        qwen.model.to("cuda")

    dataset = NuScenesQwenDataset(
        root=ds_cfg["root"], info_path=ds_cfg["info_path"],
        processor=processor,
        camera_names=ds_cfg.get("camera_names"),
        load_lidar=True,
        load_boxes_3d=ds_cfg.get("load_boxes_3d", True),
        load_map_labels=ds_cfg.get("load_map_labels", False),
    )
    teacher_fn = build_teacher_fn(yaml_cfg)

    sweep = MultiSweepLoader(
        static_sweeps=int(ds_cfg.get("multi_sweep", {}).get("static_sweeps", 4)),
        dynamic_center_only=bool(ds_cfg.get("multi_sweep", {}).get("dynamic_center_only", True)),
    )

    n_total = len(dataset)
    end = min(n_total, args.start + args.limit) if args.limit else n_total
    print(f"[token-label-cache] dataset={n_total} samples; processing [{args.start}:{end}) → {out}")

    with JsonlWriter(log_path) as w:
        w.write({**run_envelope("build_token_label_cache", args.split,
                                  {"config": args.config, "limit": args.limit,
                                   "start": args.start, "teacher": args.teacher,
                                   "out_dir": str(out)}),
                 "kind": "header"})

        for i in range(args.start, end):
            sample = dataset[i]
            lidar_path = sample.get("lidar_path")
            extra_paths = sample.get("extra_lidar_paths", []) or []
            extra_poses = sample.get("extra_lidar_poses", []) or []
            boxes = sample.get("boxes_3d_tensor")
            is_dynamic = sample.get("boxes_3d_is_dynamic")

            if lidar_path is None:
                w.write({"kind": "skipped", "idx": i, "reason": "no lidar"})
                continue

            try:
                points_ego = accumulate(lidar_path, extra_paths, extra_poses,
                                          boxes_3d=boxes, is_dynamic=is_dynamic, cfg=sweep)
                scene = sample_to_token_scene(sample, qwen, points_ego)
                teacher = teacher_fn(scene)
            except Exception as exc:                                                   # noqa: BLE001
                w.write({"kind": "error", "idx": i, "error": str(exc)})
                print(f"[i={i}] error: {exc}")
                continue

            scene_token = scene.scene_id or f"sample{i:08d}"
            ckpt = out / f"{scene_token}_{i:08d}.pt"
            torch.save({"scene": scene, "teacher": teacher,
                          "meta": {"split": args.split, "sample_idx": i,
                                    "scene_token": scene_token,
                                    "teacher_kind": yaml_cfg.get("teacher", {}).get("kind", "hard_quantile")}},
                         ckpt)
            w.write({"kind": "wrote", "idx": i, "ckpt": str(ckpt),
                      "n_tokens": int(scene.num_tokens),
                      "n_valid": int(teacher.m_T.sum()),
                      "transported_mass": float(teacher.diag.get("transported_mass", float("nan")))})
            if (i - args.start) % 50 == 0:
                print(f"[i={i}/{end}] tokens={scene.num_tokens} valid={int(teacher.m_T.sum())}")

    print(f"[token-label-cache] done → {out}")


if __name__ == "__main__":
    main()
