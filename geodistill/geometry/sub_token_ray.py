"""Sub-token ray decoding + back-projection (paper §3.7).

Predicted ``δ_hat ∈ [-1, 1]^2`` shifts a token's ray pixel from the box center to
the predicted foreground anchor:

    u_bar = u_p + W_R * δ_hat^u / 2
    v_bar = v_p + H_R * δ_hat^v / 2

The back-projection then uses each token's camera-specific intrinsic and the
camera-to-ego transform (selected by ``token_camera_id``):

    c_hat^cam  = d_hat * K^{-1} [u_bar, v_bar, 1]^T
    c_hat^ego  = T_cam->ego  c_hat^cam

This is the routine the geometry injection consumes via ``Φ(c_hat)``.
"""

from __future__ import annotations

import torch

from geodistill.core.camera import unproject_cam_to_ego


__all__ = ["decode_ray", "back_project"]


def decode_ray(token_uv: torch.Tensor, token_box: torch.Tensor, delta_hat: torch.Tensor) -> torch.Tensor:
    """Returns (N, 2) ``(u_bar, v_bar)`` from token center + region size + δ_hat."""
    if token_uv.shape != delta_hat.shape:
        raise ValueError(f"shape mismatch token_uv {tuple(token_uv.shape)} vs delta_hat {tuple(delta_hat.shape)}")
    w = (token_box[:, 2] - token_box[:, 0]).clamp_min(1e-6)
    h = (token_box[:, 3] - token_box[:, 1]).clamp_min(1e-6)
    u_bar = token_uv[:, 0] + 0.5 * w * delta_hat[:, 0]
    v_bar = token_uv[:, 1] + 0.5 * h * delta_hat[:, 1]
    return torch.stack([u_bar, v_bar], dim=-1)


def back_project(
    uv_bar: torch.Tensor,                       # (N, 2)
    d_hat: torch.Tensor,                        # (N,)
    K: torch.Tensor,                            # (Ncam, 3, 3)
    cam_to_ego: torch.Tensor,                   # (Ncam, 4, 4)
    token_camera_id: torch.Tensor,              # (N,)
) -> torch.Tensor:
    """Per-token unprojection to the center-ego frame.

    Implementation note: this loops over cameras (typically 6) and applies the
    cached :func:`geodistill.core.camera.unproject_cam_to_ego` per group. The
    loop is fine for the paper's deployment regime (≤ 6 cameras × O(N) tokens).
    """
    Ncam = int(K.shape[0])
    out = torch.zeros((uv_bar.shape[0], 3), dtype=torch.float32, device=uv_bar.device)
    for c in range(Ncam):
        sel = token_camera_id.long() == c
        if int(sel.sum()) == 0:
            continue
        out[sel] = unproject_cam_to_ego(uv_bar[sel], d_hat[sel], K[c], cam_to_ego[c])
    return out
