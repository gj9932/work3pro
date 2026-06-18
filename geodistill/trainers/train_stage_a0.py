"""Stage A0 trainer: allocation + ray + relation warm-up (paper §3.13).

Frozen:
    - Qwen Vision Encoder + merger (synthetic path uses scene.features directly,
      so this is implicit)
    - GeometryInjector gates β_pe = β_rel = 0
    - W_up not connected (forced via .requires_grad_(False))

Trained:
    - TokenEncoder, GeoEncoder, LPGA (heads + ego_embed)
    - RelationEdgeEncoders, RelationHeads, RelationAggregator (V_rel)

Loss:
    L_A0 = λ_alloc · L_alloc + λ_ray · L_ray + λ_rel · L_rel
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from geodistill.core.risk_field import RiskFieldConfig, soft_risk, oracle_allocation
from geodistill.core.r2ac import r2ac_forward
from geodistill.geometry.cross_view_label import build_edge_labels
from geodistill.geometry.edge_sampler_v2 import EdgeBudgetV2, sample_edges_v2
from geodistill.losses import (
    L_r, L_q, L_alloc, L_ray, RelationLossWeights, relation_loss, compute_w_p,
)
from geodistill.teacher.base import TeacherOutput
from geodistill.trainers.build_lpga import GeoDistillVLM
from geodistill.trainers.common import OptimSpec, build_optimizer


__all__ = ["StageA0Config", "StageA0Step", "stage_a0_step"]


@dataclass
class StageA0Config:
    risk: RiskFieldConfig = field(default_factory=RiskFieldConfig)
    lambda_r: float = 0.5
    lambda_q: float = 0.5
    lambda_alloc: float = 1.0
    lambda_ray: float = 1.0
    lambda_rel: float = 1.0
    edges: EdgeBudgetV2 = field(default_factory=EdgeBudgetV2)
    rel_weights: RelationLossWeights = field(default_factory=RelationLossWeights)
    voxel_size: float = 0.5
    tau_depth: float = 1.0
    tau_eq_depth: float = 0.5


@dataclass
class StageA0Step:
    L: torch.Tensor
    parts: dict
    diag: dict


def _ego_xy(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return c[:, 0], c[:, 1]


def stage_a0_step(
    model: GeoDistillVLM,
    scene,                                  # TokenScene
    teacher: TeacherOutput,
    cfg: StageA0Config,
) -> StageA0Step:
    """One forward + loss assembly. The actual optimizer.step lives in the CLI."""
    model.freeze_for_stage_a()

    out = model.forward_lpga(
        h_img=scene.features,
        token_uv=scene.token_uv,
        token_box=scene.token_box,
        token_camera_id=scene.token_camera_id,
        K=scene.K, cam_to_ego=scene.cam_to_ego, image_hw=scene.image_hw,
        v_ego=scene.v_ego, yaw_rate=scene.yaw_rate if model.cfg.use_yaw_rate else None,
    )

    # Oracle allocation a* from teacher (paper §3.6 — never use predicted a)
    valid = teacher.m_T & torch.isfinite(teacher.d_teacher)
    # ``c_teacher`` has NaN for invalid tokens; replace with zeros so r_p stays finite
    # (the loss is masked out via m, but downstream cdist / soft_risk would still NaN-poison).
    c_safe = torch.nan_to_num(teacher.c_teacher, nan=0.0)
    x_ego, y_ego = _ego_xy(c_safe)
    r_p = soft_risk(x_ego, y_ego, torch.tensor(scene.v_ego), cfg.risk)
    q_teacher = teacher.q_teacher.float()
    # support score s_p ∝ candidate count saturated at n_0=3 (paper §3.4) — use teacher.q_mass as proxy
    s_p = teacher.q_mass.float()
    q_geom = teacher.q_teacher.float()        # for synthetic q_teacher already mixes geom + OT

    # L_r / L_q / L_alloc
    m = valid.float()
    val_L_r = L_r(out["r_hat"], r_p, m)
    val_L_q = L_q(out["q_hat"], q_geom, s_p, m)
    val_L_alloc = L_alloc(val_L_r, val_L_q, cfg.lambda_r, cfg.lambda_q)

    # L_ray
    w_p = compute_w_p(valid, q_teacher, eps_q=0.05)
    val_L_ray = L_ray(out["delta_hat"], teacher.delta_T, w_p)

    # Relation
    edges, _types, mean_epp = sample_edges_v2(scene, teacher, cfg.edges)
    diag: dict = {"edges_per_token": mean_epp, "n_edges": len(edges)}
    val_L_rel = out["delta_hat"].new_zeros(())
    rel_parts = {k: out["delta_hat"].new_zeros(()) for k in ("L_delta", "L_order", "L_cross", "L_topo", "L_occ")}
    if edges:
        labels = build_edge_labels(scene, teacher, edges,
                                   voxel_size=cfg.voxel_size, tau_depth=cfg.tau_depth,
                                   tau_eq_depth=cfg.tau_eq_depth)
        rel_out = model.forward_relation(
            g_geo=out["g_geo"], edge_index=labels.edge_index,
            token_camera_id=scene.token_camera_id, token_uv=scene.token_uv,
            image_hw=scene.image_hw, K=scene.K, cam_to_ego=scene.cam_to_ego,
            e_conf=labels.reliability,
        )
        rel_pred = {k: rel_out[k] for k in ("delta_c", "order_logits", "cross_logit",
                                             "topo_logits", "occ_logits", "conf_logit")}
        rl = relation_loss(rel_pred, labels, cfg.rel_weights)
        val_L_rel = rl.L_rel
        rel_parts = rl.parts
        diag["L_rel_parts"] = {k: float(v.detach()) for k, v in rl.parts.items()}

    L = cfg.lambda_alloc * val_L_alloc + cfg.lambda_ray * val_L_ray + cfg.lambda_rel * val_L_rel

    return StageA0Step(
        L=L,
        parts={"L_alloc": val_L_alloc, "L_r": val_L_r, "L_q": val_L_q,
               "L_ray": val_L_ray, "L_rel": val_L_rel, **{k: v for k, v in rel_parts.items()}},
        diag=diag,
    )


# ====================================================================== CLI
def _cfg_from_yaml(yaml_cfg: dict) -> StageA0Config:
    rcfg = yaml_cfg.get("risk_field", {})
    risk = RiskFieldConfig(
        t_react=float(rcfg.get("t_react", 1.0)),
        a_brake=float(rcfg.get("a_brake", 4.0)),
        d_margin=float(rcfg.get("d_margin", 2.0)),
        tau_r=float(rcfg.get("tau_r", 2.0)),
        tau_front=float(rcfg.get("tau_front", 1.0)),
        w_corridor=float(rcfg.get("w_corridor", 2.0)),
        eta=float(yaml_cfg.get("allocation", {}).get("eta", 0.5)),
    )
    alloc = yaml_cfg.get("allocation", {})
    lw = yaml_cfg.get("loss_weights", {})
    rel_cfg = yaml_cfg.get("relation", {})
    edges = EdgeBudgetV2(
        P_local=int(rel_cfg.get("edge_types", {}).get("P_local", 6)),
        P_ray=int(rel_cfg.get("edge_types", {}).get("P_ray", 4)),
        P_cross=int(rel_cfg.get("edge_types", {}).get("P_cross", 4)),
        P_hard=int(rel_cfg.get("edge_types", {}).get("P_hard", 1)),
        P_far=int(rel_cfg.get("edge_types", {}).get("P_far", 1)),
    )
    rel_w = RelationLossWeights(
        delta=float(lw.get("delta", 1.0)),
        order=float(lw.get("order", 0.5)),
        cross=float(lw.get("cross", 0.5)),
        topo=float(lw.get("topo", 0.0)),
        occ=float(lw.get("occ", 0.0)),
    )
    return StageA0Config(
        risk=risk,
        lambda_r=float(alloc.get("lambda_r", 0.5)),
        lambda_q=float(alloc.get("lambda_q", 0.5)),
        lambda_alloc=float(alloc.get("lambda_alloc", 1.0)),
        lambda_ray=float(lw.get("ray", 1.0)),
        lambda_rel=float(yaml_cfg.get("loss_weights_misc", {}).get("rel", 1.0)),
        edges=edges,
        rel_weights=rel_w,
        voxel_size=float(rel_cfg.get("cross_view_voxel", 0.5)),
        tau_depth=float(rel_cfg.get("tau_depth", 1.0)),
    )


def _make_loss_fn(stage_cfg: StageA0Config):
    def _loss_fn(model, batch, step):
        scene, teacher = batch
        out = stage_a0_step(model, scene, teacher, stage_cfg)
        from geodistill.runtime.runner import StepOutput
        return StepOutput(loss=out.L, parts=out.parts, diag=out.diag)
    return _loss_fn


def main():
    import argparse
    from geodistill.runtime import (
        load_config_with_extends, train_loop, RunnerConfig, iter_scenes,
    )
    from geodistill.runtime.runner import StepOutput   # noqa: F401  (re-export check)
    from geodistill.trainers.build_lpga import build_from_config
    from geodistill.trainers.common import OptimSpec, build_optimizer, synthetic_hidden_size

    ap = argparse.ArgumentParser(description="GeoDistill stage A0 trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="runs/geodistill/stage_a0")
    ap.add_argument("--source", default=None,
                     choices=[None, "synthetic", "cached_nuscenes", "live_nuscenes"])
    ap.add_argument("--scenes", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16"])
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    args = ap.parse_args()

    yaml_cfg = load_config_with_extends(args.config)
    sched = yaml_cfg.get("schedule", {})
    runner_cfg = RunnerConfig(
        out_dir=args.out_dir,
        max_steps=args.max_steps or int(sched.get("stage_a0_steps", 20000)),
        grad_accum_steps=int(sched.get("grad_accum_steps", 1)),
        grad_clip=float(sched.get("grad_clip", 1.0)),
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        amp=(args.amp if args.device == "cuda" else "off"),
        device=args.device,
        seed=args.seed if args.seed is not None else int(yaml_cfg.get("experiment", {}).get("seed", 42)),
        resume=args.resume,
    )

    model = build_from_config(yaml_cfg, hidden_size_override=synthetic_hidden_size(yaml_cfg, args.source))
    model.freeze_for_stage_a()

    spec = OptimSpec(
        lr=float(sched.get("lr_lpga", 2e-4)),
        weight_decay=float(sched.get("weight_decay", 0.01)),
        grad_clip=runner_cfg.grad_clip,
    )
    optim = build_optimizer(model.parameters(), spec)

    stage_cfg = _cfg_from_yaml(yaml_cfg)
    source = args.source or yaml_cfg.get("dataset", {}).get("source", "live_nuscenes")
    batch_iter = iter_scenes(source, yaml_cfg, scenes=args.scenes, seed=runner_cfg.seed,
                              infinite=True)
    train_loop(
        model=model, optim=optim, batch_iter=batch_iter,
        loss_fn=_make_loss_fn(stage_cfg), cfg=runner_cfg,
        extra_envelope={"stage": "A0", "source": source, "config": str(yaml_cfg.get("config_path", ""))},
    )


if __name__ == "__main__":
    main()
