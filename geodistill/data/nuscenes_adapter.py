"""Real nuScenes + frozen Qwen2.5-VL -> TokenScene adapter (P0).

This is the production data path. It is wired but NOT runnable in the local CPU
env (no nuScenes, no Qwen weights, no transformers). When run on a GPU box with
data it produces the SAME ``TokenScene`` contract the synthetic fixture produces,
so teacher construction + probe + depth heads run unchanged and emit real numbers.

It composes the existing project pieces rather than reinventing them:
- ``dataset.geodistill.NuScenesQwenDataset``  -> pixel_values, image_grid_thw, calib
- ``geodistill.models.QwenVisualFrozen``       -> frozen image_embeds + token meta
- ``geotoken.geometry.lidar_rasterizer``       -> load + ego-transform LiDAR points

The probe GT depth on real data is dense-LiDAR-derived per token (documented as
such): we aggregate the LiDAR points that fall in each token region and take a
low quantile as the GT camera-z depth. This is the honest, reproducible target;
it is the same signal the teacher tries to recover, so the probe measures how
well a frozen-token head recovers token-level metric depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from .contract import TokenScene
from geodistill.core.camera import project_ego_to_cam, assign_points_to_tokens


def _require(mod_ok: bool, what: str) -> None:
    if not mod_ok:
        raise RuntimeError(
            f"{what} is unavailable in this environment. The nuScenes+Qwen path "
            f"needs a GPU box with nuScenes data, Qwen2.5-VL weights and "
            f"`transformers`. Use the synthetic fixture for CPU smoke tests."
        )


def build_scene_from_nuscenes(
    sample: dict[str, Any],
    qwen_visual,                      # geodistill.models.QwenVisualFrozen
    lidar_points_ego: torch.Tensor,   # (M, >=3) already ego-frame (center frame)
    gt_quantile: float = 0.1,
) -> TokenScene:
    """Convert one ``NuScenesQwenDataset`` sample + frozen Qwen forward into a TokenScene.

    Args:
        sample: item from ``NuScenesQwenDataset`` (pixel_values, image_grid_thw, calib...).
        qwen_visual: an instantiated ``QwenVisualFrozen`` (vision tower + merger).
        lidar_points_ego: LiDAR points already transformed to the center-ego frame.
        gt_quantile: quantile used to define per-token GT depth from in-region LiDAR.
    """
    out = qwen_visual.encode_images(
        sample["pixel_values"],
        sample["image_grid_thw"].reshape(-1, 3),
        camera_widths=[int(w) for w in sample["image_size_hw"][:, 1].tolist()],
        camera_heights=[int(h) for h in sample["image_size_hw"][:, 0].tolist()],
    )
    features = out.image_embeds.float()
    token_uv = out.token_uv.float()
    token_box = out.token_box.float()
    token_camera_id = out.token_camera_id.long()
    cam_offsets = out.cam_offsets.long()

    K = sample["camera_intrinsics"][0].float()          # (Ncam,3,3)
    cam_to_ego = sample["camera_extrinsics"][0].float()  # (Ncam,4,4)
    image_hw = sample["image_size_hw"].long()

    ego_state = sample.get("ego_state")
    v_ego = float(ego_state[0]) if ego_state is not None else 0.0
    yaw_rate = float(ego_state[1]) if ego_state is not None else 0.0

    points_ego = torch.as_tensor(lidar_points_ego, dtype=torch.float32)[:, :3]

    gt_depth, gt_valid, gt_xyz = _per_token_gt_depth(
        points_ego, K, cam_to_ego, token_box, token_camera_id, cam_offsets, gt_quantile
    )

    return TokenScene(
        features=features,
        token_uv=token_uv,
        token_box=token_box,
        token_camera_id=token_camera_id,
        cam_offsets=cam_offsets,
        K=K,
        cam_to_ego=cam_to_ego,
        image_hw=image_hw,
        points_ego=points_ego,
        v_ego=v_ego,
        yaw_rate=yaw_rate,
        gt_depth=gt_depth,
        gt_valid=gt_valid,
        gt_xyz_ego=gt_xyz,
        scene_id=str(sample.get("scene_token", "nuscenes")),
        meta={"source": "nuscenes", "sample_tokens": sample.get("sample_tokens")},
    )


def _per_token_gt_depth(points_ego, K, cam_to_ego, token_box, token_camera_id, cam_offsets, q):
    """Per-token GT depth = low quantile of in-region LiDAR camera-z (eval target)."""
    N = token_box.shape[0]
    gt_depth = torch.full((N,), float("nan"))
    gt_xyz = torch.full((N, 3), float("nan"))
    Ncam = K.shape[0]
    for c in range(Ncam):
        proj = project_ego_to_cam(points_ego, K[c], cam_to_ego[c])
        uv, depth, valid = proj["uv"], proj["depth"], proj["valid"]
        uv_v = uv[valid]
        depth_v = depth[valid]
        pts_v = points_ego[valid]
        sl = slice(int(cam_offsets[c]), int(cam_offsets[c + 1]))
        boxes_c = token_box[sl]
        tok_idx = assign_points_to_tokens(uv_v, boxes_c)
        for local_t in range(boxes_c.shape[0]):
            sel = tok_idx == local_t
            if int(sel.sum()) == 0:
                continue
            d = depth_v[sel]
            dq = torch.quantile(d, q)
            gt_depth[sl][local_t] = dq
            near = sel.nonzero(as_tuple=True)[0][(d <= dq + 1.0)]
            if near.numel() > 0:
                gt_xyz[sl.start + local_t] = pts_v[near].mean(0)
    gt_valid = torch.isfinite(gt_depth)
    return gt_depth, gt_valid, gt_xyz


def load_nuscenes_scenes(config_path: str, limit: int = 8) -> list[TokenScene]:
    """End-to-end loader (GPU box only). Builds dataset+Qwen, returns TokenScenes.

    Guarded so it fails with a clear message in the CPU env.
    """
    try:
        import transformers  # noqa: F401
        have_tf = True
    except Exception:
        have_tf = False
    _require(have_tf, "transformers / Qwen2.5-VL")

    from geotoken.config import load_config
    from transformers import AutoProcessor
    from dataset.geodistill.nuscenes_qwen_dataset import NuScenesQwenDataset
    from geodistill.models.qwen_visual_frozen import QwenVisualFrozen
    from geotoken.geometry.lidar_rasterizer import load_lidar_points, transform_lidar_points_to_ego

    cfg = load_config(config_path)
    ds_cfg = cfg["dataset"]
    model_id = cfg["base_vlm"]["model_id"]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    qwen = QwenVisualFrozen(model_id=model_id)
    dataset = NuScenesQwenDataset(
        root=ds_cfg["root"], info_path=ds_cfg["info_path"], processor=processor,
        camera_names=ds_cfg["camera_names"], load_lidar=True, load_boxes_3d=False,
    )
    scenes = []
    for i in range(min(limit, len(dataset))):
        sample = dataset[i]
        lidar_path = sample.get("lidar_path")
        pts = load_lidar_points(lidar_path) if lidar_path else torch.zeros((0, 3))
        pts_ego = transform_lidar_points_to_ego(pts)
        scenes.append(build_scene_from_nuscenes(sample, qwen, pts_ego, ds_cfg.get("foreground_quantile", 0.1)))
    return scenes
