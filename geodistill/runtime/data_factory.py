"""Scene + teacher iterator factory used by the trainers (paper §3.13).

Three sources, one signature:

    iter_scenes(source, cfg, *, scenes=N, seed=S, infinite=True)
        -> Iterator[(TokenScene, TeacherOutput)]

- ``synthetic`` — wraps :mod:`geodistill.data.synthetic`; the teacher is rebuilt
  from yaml via :func:`teacher_factory.build_teacher_fn`. CPU friendly.
- ``cached_nuscenes`` — reads ``.pt`` files written by
  ``scripts/geodistill/build_token_label_cache.py``. Each file holds a dict
  ``{scene: TokenScene, teacher: TeacherOutput}``. Fast; the recommended path
  during training.
- ``live_nuscenes`` — instantiates :class:`NuScenesQwenDataset` +
  :class:`QwenVisualFrozen` and rebuilds the teacher on every batch. Slow
  (mostly used for debugging the data pipeline). Lazy import of transformers.

When ``infinite=True`` (default) the iterator loops forever — the trainer
controls how many steps to run via ``RunnerConfig.max_steps``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from geodistill.data.contract import TokenScene
from geodistill.runtime.teacher_factory import build_teacher_fn
from geodistill.teacher import TeacherOutput


__all__ = ["SceneSource", "iter_scenes"]


@dataclass(frozen=True)
class SceneSource:
    name: str
    config: dict


def _iter_synthetic(cfg: dict, scenes: int | None, seed: int, infinite: bool):
    from geodistill.data.synthetic import make_synthetic_dataset, SyntheticConfig
    sc = cfg.get("synthetic", {}) or {}
    syn_cfg = SyntheticConfig(
        num_cameras=int(sc.get("num_cameras", 2)),
        grid_h=int(sc.get("grid_h", 8)),
        grid_w=int(sc.get("grid_w", 12)),
        feature_dim=int(sc.get("feature_dim", 64)),
        signal_gain=float(sc.get("signal_gain", 2.0)),
        num_surfaces=int(sc.get("num_surfaces", 9)),
    )
    n = scenes or int(sc.get("scenes", 16))
    teacher_fn = build_teacher_fn(cfg)
    pool = make_synthetic_dataset(n, syn_cfg, seed0=seed)
    iterator = itertools.cycle(pool) if infinite else iter(pool)
    for s in iterator:
        yield s, teacher_fn(s)


def _iter_cached_nuscenes(cfg: dict, scenes: int | None, seed: int, infinite: bool):
    cache_dir = Path(cfg.get("dataset", {}).get("cache_dir",
                                                  "runs/geodistill/cache/token_labels"))
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"cache dir {cache_dir} does not exist; run scripts/geodistill/build_token_label_cache.py first"
        )
    files = sorted(cache_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt files under {cache_dir}")
    if scenes:
        files = files[: int(scenes)]
    g = torch.Generator().manual_seed(seed)

    def _shuffle(xs: list[Path]) -> list[Path]:
        idx = torch.randperm(len(xs), generator=g).tolist()
        return [xs[i] for i in idx]

    while True:
        for p in _shuffle(files):
            payload = torch.load(p, map_location="cpu")
            yield payload["scene"], payload["teacher"]
        if not infinite:
            break


def _iter_live_nuscenes(cfg: dict, scenes: int | None, seed: int, infinite: bool):
    try:
        import transformers                                                            # noqa: F401
    except Exception as exc:                                                       # noqa: BLE001
        raise ImportError(
            "live_nuscenes requires transformers + Qwen2.5-VL weights. "
            "Use --source cached_nuscenes after running build_token_label_cache."
        ) from exc

    from transformers import AutoProcessor
    from geodistill.models.qwen_visual_frozen import QwenVisualFrozen
    from geodistill.runtime.scene_stream import sample_to_token_scene
    from geodistill.runtime.ego_motion import accumulate, MultiSweepLoader
    from dataset.geodistill import NuScenesQwenDataset

    ds_cfg = cfg.get("dataset", {})
    base = cfg.get("base_vlm", {})
    processor = AutoProcessor.from_pretrained(base["model_id"], trust_remote_code=True)
    qwen = QwenVisualFrozen(model_id=base["model_id"])
    dataset = NuScenesQwenDataset(
        root=ds_cfg["root"], info_path=ds_cfg["info_path"],
        processor=processor,
        camera_names=ds_cfg.get("camera_names"),
        load_lidar=True, load_boxes_3d=ds_cfg.get("load_boxes_3d", True),
        load_map_labels=ds_cfg.get("load_map_labels", False),
    )
    teacher_fn = build_teacher_fn(cfg)
    sweep_cfg = MultiSweepLoader(
        static_sweeps=int(ds_cfg.get("multi_sweep", {}).get("static_sweeps", 4)),
        dynamic_center_only=bool(ds_cfg.get("multi_sweep", {}).get("dynamic_center_only", True)),
    )
    n = scenes or len(dataset)
    g = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(min(n, len(dataset)), generator=g).tolist()
        for i in order:
            sample = dataset[i]
            lidar_path = sample.get("lidar_path")
            extra_paths = sample.get("extra_lidar_paths", []) or []
            poses_to_center = sample.get("extra_lidar_poses", []) or []
            boxes_3d = sample.get("boxes_3d_tensor")
            is_dynamic = sample.get("boxes_3d_is_dynamic")
            points_ego = accumulate(
                lidar_path, extra_paths, poses_to_center,
                boxes_3d=boxes_3d, is_dynamic=is_dynamic, cfg=sweep_cfg,
            ) if lidar_path else torch.zeros((0, 3))
            scene = sample_to_token_scene(sample, qwen, points_ego)
            yield scene, teacher_fn(scene)
        if not infinite:
            break


def iter_scenes(
    source: str,
    cfg: dict,
    *,
    scenes: int | None = None,
    seed: int = 0,
    infinite: bool = True,
) -> Iterator[tuple[TokenScene, TeacherOutput]]:
    if source == "synthetic":
        yield from _iter_synthetic(cfg, scenes, seed, infinite)
    elif source == "cached_nuscenes":
        yield from _iter_cached_nuscenes(cfg, scenes, seed, infinite)
    elif source == "live_nuscenes":
        yield from _iter_live_nuscenes(cfg, scenes, seed, infinite)
    else:
        raise ValueError(f"unknown source {source!r}; choose synthetic / cached_nuscenes / live_nuscenes")
