"""Sparse Relation Adapter — same-camera vs cross-camera edge encoders + heads (paper §3.10).

The two branches are deliberately separate:

    SameCamEdgeEncoder  : ξ^same  = [δu/W, δv/H, ρ_q^ego - ρ_p^ego]                (R^5)
    CrossCamEdgeEncoder : ξ^cross = [ρ_p^ego, ρ_q^ego, ρ_q - ρ_p, cam_pair_emb]    (R^9 + emb)

Cross-camera MUST NOT receive (δu, δv) because the two image planes are
unrelated — using them would be a shortcut. The encoders' input dim is checked
on every call to make that hard to violate.

Both encoders feed a shared :class:`RelationHeads` that emits

    Δc      (3,)        Huber regression
    order   (3-class)   {0:eq, 1:q farther, 2:p farther}
    cross   (logit)     BCE with cross-view label
    topo    (6-class)   topology
    occ     (3-class)   occlusion

A confidence ``conf_logit`` is produced jointly so the
:class:`ConfidenceAttentionAggregator` can fold edge reliability into ω_pq.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


__all__ = [
    "RelationEdgeConfig",
    "SameCamEdgeEncoder",
    "CrossCamEdgeEncoder",
    "RelationHeads",
    "RelationEdgeEncoders",
]


SAME_XI_DIM = 5      # [δu/W, δv/H, ρ_q-ρ_p]
CROSS_XI_BASE = 9    # [ρ_p(3), ρ_q(3), ρ_q-ρ_p(3)]


@dataclass(frozen=True)
class RelationEdgeConfig:
    d_token: int = 512
    d_hidden: int = 256
    d_out: int = 256              # r_pq dim
    cam_pair_emb_dim: int = 16
    num_cameras: int = 6
    n_topo_classes: int = 6
    n_order_classes: int = 3
    n_occ_classes: int = 3


def _mlp(d_in: int, d_hidden: int, d_out: int, n_layers: int = 2) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d_hidden, d_hidden), nn.GELU()]
    layers.append(nn.Linear(d_hidden, d_out))
    return nn.Sequential(*layers)


def _edge_features(g_p: torch.Tensor, g_q: torch.Tensor) -> torch.Tensor:
    return torch.cat([g_p, g_q, g_p - g_q, g_p * g_q], dim=-1)


class SameCamEdgeEncoder(nn.Module):
    """A_edge^same. Receives ξ^same (R^5)."""

    expected_xi_dim: int = SAME_XI_DIM

    def __init__(self, cfg: RelationEdgeConfig):
        super().__init__()
        self.cfg = cfg
        d_in = 4 * cfg.d_token + SAME_XI_DIM
        self.mlp = _mlp(d_in, cfg.d_hidden, cfg.d_out)

    def forward(self, g_p: torch.Tensor, g_q: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        if xi.shape[-1] != SAME_XI_DIM:
            raise ValueError(
                f"SameCamEdgeEncoder expects ξ_dim={SAME_XI_DIM}, got {xi.shape[-1]} — "
                f"never feed cross-camera ξ here"
            )
        return self.mlp(torch.cat([_edge_features(g_p, g_q), xi], dim=-1))


class CrossCamEdgeEncoder(nn.Module):
    """A_edge^cross. Receives ξ^cross (R^9 + cam_pair_emb)."""

    expected_xi_dim_base: int = CROSS_XI_BASE

    def __init__(self, cfg: RelationEdgeConfig):
        super().__init__()
        self.cfg = cfg
        self.cam_pair_emb = nn.Embedding(cfg.num_cameras * cfg.num_cameras, cfg.cam_pair_emb_dim)
        d_in = 4 * cfg.d_token + CROSS_XI_BASE + cfg.cam_pair_emb_dim
        self.mlp = _mlp(d_in, cfg.d_hidden, cfg.d_out)

    def forward(self, g_p: torch.Tensor, g_q: torch.Tensor, xi: torch.Tensor,
                cam_pair_id: torch.Tensor) -> torch.Tensor:
        if xi.shape[-1] != CROSS_XI_BASE:
            raise ValueError(
                f"CrossCamEdgeEncoder expects ξ_dim={CROSS_XI_BASE}, got {xi.shape[-1]} — "
                f"NEVER feed (δu, δv) into the cross-camera branch (paper §3.10 shortcut guard)"
            )
        emb = self.cam_pair_emb(cam_pair_id.long())
        x = torch.cat([_edge_features(g_p, g_q), xi, emb], dim=-1)
        return self.mlp(x)


class RelationHeads(nn.Module):
    """5 heads + a confidence logit on top of r_pq."""

    def __init__(self, cfg: RelationEdgeConfig):
        super().__init__()
        self.delta = nn.Linear(cfg.d_out, 3)
        self.order = nn.Linear(cfg.d_out, cfg.n_order_classes)
        self.cross = nn.Linear(cfg.d_out, 1)
        self.topo = nn.Linear(cfg.d_out, cfg.n_topo_classes)
        self.occ = nn.Linear(cfg.d_out, cfg.n_occ_classes)
        self.conf = nn.Linear(cfg.d_out, 1)

    def forward(self, r_pq: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "delta_c": self.delta(r_pq),
            "order_logits": self.order(r_pq),
            "cross_logit": self.cross(r_pq).squeeze(-1),
            "topo_logits": self.topo(r_pq),
            "occ_logits": self.occ(r_pq),
            "conf_logit": self.conf(r_pq).squeeze(-1),
        }


class RelationEdgeEncoders(nn.Module):
    """Wrapper that routes an edge to the correct encoder by camera pair.

    Inputs:
        g          (N, d_token)              token features (g_p^geo)
        edge_index (E, 2) long               pairs (p, q)
        token_camera_id (N,) long
        ego_rays   (N, 3)                    ρ^ego per token (already unit-length)
        token_uv   (N, 2)                    pixel center
        image_hw   (Ncam, 2)                 (H, W)

    Returns:
        r_pq (E, d_out)
    """

    def __init__(self, cfg: RelationEdgeConfig):
        super().__init__()
        self.cfg = cfg
        self.same = SameCamEdgeEncoder(cfg)
        self.cross_enc = CrossCamEdgeEncoder(cfg)

    def forward(
        self,
        g: torch.Tensor,
        edge_index: torch.Tensor,
        token_camera_id: torch.Tensor,
        ego_rays: torch.Tensor,
        token_uv: torch.Tensor,
        image_hw: torch.Tensor,
    ) -> torch.Tensor:
        E = edge_index.shape[0]
        if E == 0:
            return g.new_zeros((0, self.cfg.d_out))
        p_idx = edge_index[:, 0].long()
        q_idx = edge_index[:, 1].long()
        cp = token_camera_id[p_idx].long()
        cq = token_camera_id[q_idx].long()
        same_mask = cp == cq

        out = g.new_zeros((E, self.cfg.d_out))

        # Same-camera batch
        if int(same_mask.sum()) > 0:
            sm = same_mask
            uv_p = token_uv[p_idx[sm]]
            uv_q = token_uv[q_idx[sm]]
            H = image_hw[cp[sm], 0].float().clamp_min(1.0)
            W = image_hw[cp[sm], 1].float().clamp_min(1.0)
            du = (uv_q[:, 0] - uv_p[:, 0]) / W
            dv = (uv_q[:, 1] - uv_p[:, 1]) / H
            drho = ego_rays[q_idx[sm]] - ego_rays[p_idx[sm]]                       # (k, 3)
            xi = torch.cat([du.unsqueeze(-1), dv.unsqueeze(-1), drho], dim=-1)     # (k, 5)
            r = self.same(g[p_idx[sm]], g[q_idx[sm]], xi)
            out[sm] = r

        # Cross-camera batch
        if int((~same_mask).sum()) > 0:
            cm = ~same_mask
            rho_p = ego_rays[p_idx[cm]]
            rho_q = ego_rays[q_idx[cm]]
            xi = torch.cat([rho_p, rho_q, rho_q - rho_p], dim=-1)                  # (k, 9)
            cam_pair = cp[cm] * self.cfg.num_cameras + cq[cm]
            r = self.cross_enc(g[p_idx[cm]], g[q_idx[cm]], xi, cam_pair)
            out[cm] = r

        return out
