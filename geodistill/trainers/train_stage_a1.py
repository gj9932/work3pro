"""Stage A1 trainer: A0 + metric depth + coord (paper §3.13).

Adds:
    L_comp  on companded ζ (oracle a* target)
    L_rank  on metric depth (d_hat = F_inv(ζ_hat, StopGrad(a_hat)))
    L_coord on c_hat^ego

Stage A1 still keeps the gates at zero (no LLM injection yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from geodistill.core.r2ac import r2ac_forward
from geodistill.core.risk_field import RiskFieldConfig, soft_risk, oracle_allocation
from geodistill.losses import L_comp, L_rank, L_coord, compute_w_p
from geodistill.trainers.train_stage_a0 import StageA0Config, stage_a0_step, StageA0Step


__all__ = ["StageA1Config", "StageA1Step", "stage_a1_step"]


@dataclass
class StageA1Config(StageA0Config):
    lambda_depth: float = 1.0
    lambda_rank: float = 0.1
    lambda_coord: float = 1.0
    tau_d: float = 1.0
    eps_q: float = 0.05
    rank_pairs_per_token: int = 8


def _build_rank_pairs(n_tokens: int, valid: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Random pairs of valid tokens for the rank loss."""
    idx = valid.nonzero(as_tuple=True)[0]
    if idx.numel() < 2:
        return torch.zeros((0, 2), dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    K = max(1, idx.numel() * k // 2)
    a = idx[torch.randint(0, idx.numel(), (K,), generator=g)]
    b = idx[torch.randint(0, idx.numel(), (K,), generator=g)]
    keep = a != b
    return torch.stack([a[keep], b[keep]], dim=-1)


@dataclass
class StageA1Step:
    L: torch.Tensor
    parts: dict
    diag: dict


def stage_a1_step(model, scene, teacher, cfg: StageA1Config, seed: int = 0) -> StageA1Step:
    a0 = stage_a0_step(model, scene, teacher, cfg)

    # We need the same out dict; rebuild via the model (cheap). The duplicate
    # forward keeps the A0 step pure / re-usable; A1 trainer pays for the
    # convenience and still benefits from a single optimizer step.
    out = model.forward_lpga(
        h_img=scene.features,
        token_uv=scene.token_uv,
        token_box=scene.token_box,
        token_camera_id=scene.token_camera_id,
        K=scene.K, cam_to_ego=scene.cam_to_ego, image_hw=scene.image_hw,
        v_ego=scene.v_ego, yaw_rate=scene.yaw_rate if model.cfg.use_yaw_rate else None,
    )

    valid = teacher.m_T & torch.isfinite(teacher.d_teacher)
    if not bool(valid.any()):
        return StageA1Step(L=a0.L, parts={**a0.parts}, diag={**a0.diag})

    # Oracle a*  (use NaN-safe teacher coordinates; invalid tokens are masked anyway)
    c_safe = torch.nan_to_num(teacher.c_teacher, nan=0.0)
    d_safe = torch.nan_to_num(teacher.d_teacher, nan=0.0)
    x = c_safe[:, 0]; y = c_safe[:, 1]
    r_p = soft_risk(x, y, torch.tensor(scene.v_ego), cfg.risk)
    a_star = oracle_allocation(r_p, teacher.q_teacher.float(), cfg.risk.eta)

    # ζ* (paper §3.6: always built from oracle)
    zeta_star = r2ac_forward(d_safe, a_star, model.cfg.D_max, model.cfg.beta, model.cfg.a_eps)
    w_p = compute_w_p(valid, teacher.q_teacher.float(), eps_q=cfg.eps_q)

    val_L_comp = L_comp(out["zeta_hat"], zeta_star, w_p)

    # L_rank on metric domain — d_hat already StopGrad'd inside LPGA.decode_depth
    pairs = _build_rank_pairs(scene.num_tokens, valid, cfg.rank_pairs_per_token, seed=seed)
    pair_w = w_p[pairs[:, 0]] * w_p[pairs[:, 1]] if pairs.numel() else w_p.new_zeros(0)
    pair_w = pair_w.sqrt() if pair_w.numel() else pair_w
    val_L_rank = L_rank(out["d_hat"], d_safe, pairs, pair_w, tau_d=cfg.tau_d)

    # L_coord
    c_T = c_safe.detach()
    val_L_coord = L_coord(out["c_hat"], c_T, w_p)

    L = a0.L + cfg.lambda_depth * val_L_comp + cfg.lambda_rank * val_L_rank + cfg.lambda_coord * val_L_coord
    parts = {**a0.parts, "L_comp": val_L_comp, "L_rank": val_L_rank, "L_coord": val_L_coord}
    return StageA1Step(L=L, parts=parts, diag={**a0.diag, "n_rank_pairs": int(pairs.shape[0])})


# ====================================================================== CLI
def _cfg_from_yaml(yaml_cfg: dict) -> StageA1Config:
    from geodistill.trainers.train_stage_a0 import _cfg_from_yaml as _a0_from_yaml
    a0 = _a0_from_yaml(yaml_cfg)
    lw = yaml_cfg.get("loss_weights", {})
    return StageA1Config(
        risk=a0.risk,
        lambda_r=a0.lambda_r, lambda_q=a0.lambda_q, lambda_alloc=a0.lambda_alloc,
        lambda_ray=a0.lambda_ray, lambda_rel=a0.lambda_rel,
        edges=a0.edges, rel_weights=a0.rel_weights,
        voxel_size=a0.voxel_size, tau_depth=a0.tau_depth, tau_eq_depth=a0.tau_eq_depth,
        lambda_depth=float(lw.get("depth", 1.0)),
        lambda_rank=float(lw.get("rank", 0.1)),
        lambda_coord=float(lw.get("coord", 1.0)),
        tau_d=float(yaml_cfg.get("loss_misc", {}).get("tau_d", 1.0)),
        eps_q=float(yaml_cfg.get("loss_misc", {}).get("eps_q", 0.05)),
    )


def main():
    import argparse
    from geodistill.runtime import (
        load_config_with_extends, train_loop, RunnerConfig, iter_scenes,
    )
    from geodistill.runtime.runner import StepOutput
    from geodistill.trainers.build_lpga import build_from_config
    from geodistill.trainers.common import OptimSpec, build_optimizer, synthetic_hidden_size

    ap = argparse.ArgumentParser(description="GeoDistill stage A1 trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="runs/geodistill/stage_a1")
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
        max_steps=args.max_steps or int(sched.get("stage_a1_steps", 40000)),
        grad_accum_steps=int(sched.get("grad_accum_steps", 1)),
        grad_clip=float(sched.get("grad_clip", 1.0)),
        log_every=args.log_every, ckpt_every=args.ckpt_every,
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

    def _loss_fn(model, batch, step):
        scene, teacher = batch
        out = stage_a1_step(model, scene, teacher, stage_cfg, seed=step)
        return StepOutput(loss=out.L, parts=out.parts, diag=out.diag)

    train_loop(
        model=model, optim=optim, batch_iter=batch_iter,
        loss_fn=_loss_fn, cfg=runner_cfg,
        extra_envelope={"stage": "A1", "source": source, "config": str(yaml_cfg.get("config_path", ""))},
    )


if __name__ == "__main__":
    main()
