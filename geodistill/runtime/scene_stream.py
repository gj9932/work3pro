"""Build a TokenScene from a nuScenes sample + frozen Qwen visual output.

The sample comes from :class:`dataset.geodistill.NuScenesQwenDataset` (already
running the Qwen image processor); the visual output comes from
:class:`geodistill.models.QwenVisualFrozen`. This module composes them with
LiDAR ego-motion accumulation + per-token GT depth construction so trainers /
probes consume the same TokenScene contract that the synthetic fixture
produces.

The GT depth on real data is dense-LiDAR-derived (low quantile of in-region
LiDAR). It is the same signal the teacher tries to recover, so the probe
measures how well a frozen-token head recovers token-level metric depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from geodistill.core.camera import project_ego_to_cam, assign_points_to_tokens
from geodistill.data.contract import TokenScene


__all__ = ["sample_to_token_scene"]


def _per_token_gt_depth(
    points_ego: torch.Tensor,           # (M, 3)
    K: torch.Tensor,                    # (Ncam, 3, 3)
    cam_to_ego: torch.Tensor,           # (Ncam, 4, 4)
    token_box: torch.Tensor,            # (N, 4)
    cam_offsets: torch.Tensor,          # (Ncam+1,)
    quantile: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = token_box.shape[0]
    gt_depth = torch.full((N,), float("nan"))
    gt_xyz = torch.full((N, 3), float("nan"))
    Ncam = K.shape[0]
    for c in range(Ncam):
        proj = project_ego_to_cam(points_ego, K[c], cam_to_ego[c])
        valid = proj["valid"]
        uv_v = proj["uv"][valid]
        depth_v = proj["depth"][valid]
        pts_v = points_ego[valid]
        if uv_v.numel() == 0:
            continue
        sl = slice(int(cam_offsets[c]), int(cam_offsets[c + 1]))
        boxes_c = token_box[sl]
        tok_idx = assign_points_to_tokens(uv_v, boxes_c)
        for local_t in range(boxes_c.shape[0]):
            sel = tok_idx == local_t
            if int(sel.sum()) == 0:
                continue
            d = depth_v[sel]
            dq = torch.quantile(d, quantile)
            gt_depth[sl.start + local_t] = dq
            near = sel.nonzero(as_tuple=True)[0][(d <= dq + 1.0)]
            if near.numel() > 0:
                gt_xyz[sl.start + local_t] = pts_v[near].mean(0)
    return gt_depth, torch.isfinite(gt_depth), gt_xyz


def sample_to_token_scene(
    sample: dict[str, Any],
    qwen_visual,                         # QwenVisualFrozen (frozen)
    points_ego: torch.Tensor,            # (M, 3) center-frame, after multi-sweep
    *,
    gt_quantile: float = 0.1,
) -> TokenScene:
    """Convert one ``NuScenesQwenDataset`` sample + frozen Qwen forward into a TokenScene."""
    out = qwen_visual.encode_images(
        sample["pixel_values"],
        sample["image_grid_thw"].reshape(-1, 3),
        camera_widths=[int(w) for w in sample["image_size_hw"][:, 1].tolist()],
        camera_heights=[int(h) for h in sample["image_size_hw"][:, 0].tolist()],
    )
    features = out.image_embeds.float().detach()
    token_uv = out.token_uv.float()
    token_box = out.token_box.float()
    token_camera_id = out.token_camera_id.long()
    cam_offsets = out.cam_offsets.long()

    K = sample["camera_intrinsics"][0].float()
    cam_to_ego = sample["camera_extrinsics"][0].float()
    image_hw = sample["image_size_hw"].long()

    ego_state = sample.get("ego_state")
    v_ego = float(ego_state[0]) if ego_state is not None else 0.0
    yaw_rate = float(ego_state[1]) if ego_state is not None else 0.0

    points_ego = torch.as_tensor(points_ego, dtype=torch.float32)[:, :3]
    gt_depth, gt_valid, gt_xyz = _per_token_gt_depth(
        points_ego, K, cam_to_ego, token_box, cam_offsets, quantile=gt_quantile,
    )

    return TokenScene(
        features=features,
        token_uv=token_uv, token_box=token_box,
        token_camera_id=token_camera_id, cam_offsets=cam_offsets,
        K=K, cam_to_ego=cam_to_ego, image_hw=image_hw,
        points_ego=points_ego, v_ego=v_ego, yaw_rate=yaw_rate,
        gt_depth=gt_depth, gt_valid=gt_valid, gt_xyz_ego=gt_xyz,
        scene_id=str(sample.get("scene_token", "nuscenes")),
        meta={"source": "nuscenes", "sample_tokens": sample.get("sample_tokens"),
              "timestamp": sample.get("timestamp")},
    )
