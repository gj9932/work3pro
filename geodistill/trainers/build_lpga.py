"""GeoDistill-VLM master module assembly (paper §3.3-3.11).

Wires together:
    Qwen frozen vision (lazy)  →  H_img
    TokenEncoder                →  g_mono
    GeoEncoder (g_mono + e_calib) → g_geo
    LPGA heads (zeta_hat, q_hat, delta_hat, r_hat, a_hat, d_hat)
    sub_token_ray decode + back_project → c_hat (ego frame)
    edges (V2 sampler) + RelationEdgeEncoders → r_pq
    RelationHeads → relation predictions + conf_logit
    RelationAggregator → z_p^rel
    Universal3DPE(c_hat) + GeometryInjector → H_geo

Everything is a regular ``nn.Module`` with profile-driven freeze/unfreeze
hooks; the trainers decide which submodules participate per stage. Nothing
here imports transformers — Qwen is plugged in by the caller through
``forward_from_features`` (the synthetic / probe path) or by the GPU runtime
through the dedicated ``forward_from_pixels`` (lazy, raises on CPU).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from geodistill.core.camera import ray_dirs_ego
from geodistill.geometry.pe_3d import Universal3DPE
from geodistill.geometry.sub_token_ray import decode_ray, back_project
from geodistill.models.lpga_token_encoder import TokenEncoder, TokenEncoderConfig, GeoEncoder
from geodistill.models.lpga_heads import LPGA, LPGAConfig
from geodistill.models.relation_edge import RelationEdgeConfig, RelationEdgeEncoders, RelationHeads
from geodistill.models.relation_aggregator import RelationAggregator, RelationAggregatorConfig
from geodistill.models.geometry_injection import GeometryInjector, GeometryInjectorConfig


__all__ = ["GeoDistillConfig", "GeoDistillVLM", "build_from_config", "auto_pe_dims"]


def auto_pe_dims(hidden_size: int) -> tuple[int, int, int]:
    """Pick (d_x, d_y, d_z) that sum to ``hidden_size`` and are each even.

    Used by eval CLIs / tests so a synthetic fixture with arbitrary feature dim
    still satisfies :class:`GeoDistillVLM`'s ``d_x + d_y + d_z == hidden_size``
    requirement. Mirrors the paper choice of nearly-equal partitioning.
    """
    if hidden_size < 6 or hidden_size % 2 != 0:
        raise ValueError(f"auto_pe_dims requires an even hidden_size >= 6 (got {hidden_size})")
    base = hidden_size // 6 * 2                        # each axis at least this many channels
    rem = hidden_size - 3 * base
    # rem is non-negative and a multiple of 2; give all leftover to z to mirror paper.
    return base, base, base + rem


@dataclass
class GeoDistillConfig:
    hidden_size: int = 3584                 # Qwen2.5-VL hidden; override on synthetic
    d_bottleneck: int = 512
    d_e_cam: int = 32
    d_e_uv: int = 64
    d_e_calib: int = 64
    d_e_ego: int = 32
    num_cameras: int = 6
    D_max: float = 80.0
    beta: float = float(torch.log(torch.tensor(16.0)))
    eta: float = 0.5
    a_eps: float = 1e-3
    pe_d_xyz: tuple = (1194, 1194, 1196)
    pe_tau: float = 10000.0
    rel_d_edge: int = 256
    rel_d_hidden: int = 256
    cam_pair_emb_dim: int = 16
    sigma_up: float = 1e-4
    alpha_pe_max: float = 1.0
    alpha_rel_max: float = 1.0
    use_yaw_rate: bool = False


class GeoDistillVLM(nn.Module):
    def __init__(self, cfg: GeoDistillConfig):
        super().__init__()
        self.cfg = cfg

        token_enc_cfg = TokenEncoderConfig(
            d_in=cfg.hidden_size, d_bottleneck=cfg.d_bottleneck,
            d_e_cam=cfg.d_e_cam, d_e_uv=cfg.d_e_uv, d_e_calib=cfg.d_e_calib,
            num_cameras=cfg.num_cameras,
        )
        self.token_encoder = TokenEncoder(token_enc_cfg)
        self.geo_encoder = GeoEncoder(
            d_in=cfg.d_bottleneck, d_e_calib=cfg.d_e_calib, d_out=cfg.d_bottleneck,
            num_cameras=cfg.num_cameras,
        )
        self.lpga = LPGA(LPGAConfig(
            d_mono=cfg.d_bottleneck, d_e_ego=cfg.d_e_ego, d_hidden=cfg.d_bottleneck // 2,
            D_max=cfg.D_max, beta=cfg.beta, eta=cfg.eta, a_eps=cfg.a_eps,
            use_yaw_rate=cfg.use_yaw_rate,
        ))

        d_x, d_y, d_z = cfg.pe_d_xyz
        if d_x + d_y + d_z != cfg.hidden_size:
            raise ValueError(
                f"PE dims sum {d_x + d_y + d_z} != hidden_size {cfg.hidden_size}; "
                f"adjust pe_d_xyz so it matches the LLM hidden size"
            )
        self.pe = Universal3DPE(d_x=d_x, d_y=d_y, d_z=d_z, tau=cfg.pe_tau)

        rel_cfg = RelationEdgeConfig(
            d_token=cfg.d_bottleneck, d_hidden=cfg.rel_d_hidden, d_out=cfg.rel_d_edge,
            cam_pair_emb_dim=cfg.cam_pair_emb_dim, num_cameras=cfg.num_cameras,
        )
        self.relation_encoders = RelationEdgeEncoders(rel_cfg)
        self.relation_heads = RelationHeads(rel_cfg)
        self.relation_aggregator = RelationAggregator(RelationAggregatorConfig(
            d_edge=cfg.rel_d_edge, d_rel=cfg.d_bottleneck,
            mode="confidence_attention",
        ))

        self.injector = GeometryInjector(GeometryInjectorConfig(
            hidden_size=cfg.hidden_size, d_rel=cfg.d_bottleneck,
            alpha_pe_max=cfg.alpha_pe_max, alpha_rel_max=cfg.alpha_rel_max,
            sigma_up=cfg.sigma_up,
        ))

    # ------------------------------------------------------------------ utils
    def freeze_for_stage_a(self) -> None:
        # Force gates to 0 (paper §3.13 stage A): no geometry leaks into the LLM input.
        with torch.no_grad():
            self.injector.beta_pe.data.zero_()
            self.injector.beta_rel.data.zero_()
        self.injector.beta_pe.requires_grad_(False)
        self.injector.beta_rel.requires_grad_(False)
        # W_up not connected: force its grad off too.
        for p in self.injector.W_up.parameters():
            p.requires_grad_(False)

    def unlock_for_stage_b(self) -> None:
        self.injector.beta_pe.requires_grad_(True)
        self.injector.beta_rel.requires_grad_(True)
        for p in self.injector.W_up.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------- core
    def forward_lpga(
        self,
        h_img: torch.Tensor,
        token_uv: torch.Tensor,
        token_box: torch.Tensor,
        token_camera_id: torch.Tensor,
        K: torch.Tensor,
        cam_to_ego: torch.Tensor,
        image_hw: torch.Tensor,
        v_ego: torch.Tensor | float,
        yaw_rate: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        g_mono = self.token_encoder(h_img, token_camera_id, token_uv, image_hw)
        g_geo = self.geo_encoder(g_mono, K, cam_to_ego, image_hw, token_camera_id)
        out = self.lpga(g_mono, v_ego=v_ego, yaw_rate=yaw_rate)
        uv_bar = decode_ray(token_uv, token_box, out["delta_hat"])
        c_hat = back_project(uv_bar, out["d_hat"], K, cam_to_ego, token_camera_id)
        return {**out, "g_mono": g_mono, "g_geo": g_geo, "uv_bar": uv_bar, "c_hat": c_hat}

    def forward_relation(
        self,
        g_geo: torch.Tensor,
        edge_index: torch.Tensor,
        token_camera_id: torch.Tensor,
        token_uv: torch.Tensor,
        image_hw: torch.Tensor,
        K: torch.Tensor,
        cam_to_ego: torch.Tensor,
        e_conf: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        ego_rays = self._build_ego_rays(token_uv, token_camera_id, K, cam_to_ego)
        r_pq = self.relation_encoders(g_geo, edge_index, token_camera_id, ego_rays, token_uv, image_hw)
        preds = self.relation_heads(r_pq)
        N = int(g_geo.shape[0])
        if e_conf is None:
            e_conf = torch.ones(r_pq.shape[0], device=r_pq.device)
        z_rel = self.relation_aggregator(
            r_pq, edge_index, N,
            conf_logit=preds["conf_logit"], e_conf=e_conf,
        )
        return {"r_pq": r_pq, "z_rel": z_rel, **preds}

    def forward_inject(self, h_img: torch.Tensor, c_hat: torch.Tensor, z_rel: torch.Tensor) -> torch.Tensor:
        pe = self.pe(c_hat)
        return self.injector(h_img, pe, z_rel)

    # ------------------------------------------------------------------ ray
    def _build_ego_rays(self, token_uv: torch.Tensor, token_camera_id: torch.Tensor,
                        K: torch.Tensor, cam_to_ego: torch.Tensor) -> torch.Tensor:
        N = token_uv.shape[0]
        out = torch.zeros((N, 3), dtype=torch.float32, device=token_uv.device)
        for c in range(int(K.shape[0])):
            sel = token_camera_id.long() == c
            if int(sel.sum()) == 0:
                continue
            out[sel] = ray_dirs_ego(token_uv[sel], K[c], cam_to_ego[c])
        return out


def build_from_config(yaml_cfg: dict, hidden_size_override: int | None = None) -> GeoDistillVLM:
    """Construct a ``GeoDistillVLM`` from the project YAML.

    Reads only the fields under ``base_vlm.hidden_size``, ``lpga.*``, ``r2ac.*``,
    ``relation.*``, ``injection.*``. Unknown keys are ignored so existing yamls
    remain compatible.

    ``hidden_size_override`` lets the synthetic / smoke path swap in a smaller
    hidden size while reusing the same yaml; ``pe_d_xyz`` is recomputed via
    :func:`auto_pe_dims` so it always sums to the active hidden size.
    """
    base = yaml_cfg.get("base_vlm", {})
    lpga_cfg = yaml_cfg.get("lpga", {})
    r2ac_cfg = yaml_cfg.get("r2ac", {})
    rel_cfg = yaml_cfg.get("relation", {})
    inj_cfg = yaml_cfg.get("injection", {})

    hidden = int(hidden_size_override or base.get("hidden_size", 3584))
    if hidden_size_override is not None:
        pe_dims = auto_pe_dims(hidden)
    else:
        pe_dims = inj_cfg.get("pe_dims", [1194, 1194, 1196])
        if sum(int(v) for v in pe_dims) != hidden:
            pe_dims = auto_pe_dims(hidden)

    cfg = GeoDistillConfig(
        hidden_size=hidden,
        d_bottleneck=int(lpga_cfg.get("bottleneck", 512)),
        D_max=float(r2ac_cfg.get("D_max", 80.0)),
        beta=float(torch.log(torch.tensor(float(r2ac_cfg.get("exp_beta", 16.0))))),
        a_eps=float(r2ac_cfg.get("a_eps", 1e-3)),
        rel_d_edge=int(rel_cfg.get("d_edge", 256)),
        rel_d_hidden=int(rel_cfg.get("d_hidden", 256)),
        cam_pair_emb_dim=int(rel_cfg.get("cam_pair_emb_dim", 16)),
        sigma_up=float(inj_cfg.get("W_up_sigma", 1e-4)),
        alpha_pe_max=float(inj_cfg.get("alpha_pe_max", 1.0)),
        alpha_rel_max=float(inj_cfg.get("alpha_rel_max", 1.0)),
        pe_d_xyz=tuple(int(v) for v in pe_dims),
        pe_tau=float(inj_cfg.get("pe_freq_tau", 10000.0)),
        eta=float(yaml_cfg.get("allocation", {}).get("eta", 0.5)),
        use_yaw_rate=bool(lpga_cfg.get("use_yaw_rate", False)),
    )
    return GeoDistillVLM(cfg)
