"""Stage B trainer: VLM spatial instruction tuning (paper §3.13).

Trains:
    everything from A1 +
    GeometryInjector W_up + α_pe / α_rel gates +
    Qwen LLM rank-16 LoRA (lazy, GPU-side) +
    optional ``<POS>`` coord decoder

Frozen:
    Qwen Vision Encoder + merger.

Step-0 invariant (verified by trainer):
    H_geo == H_img bit-exactly when β_pe = β_rel = 0 (paper §3.11).

Loss:
    L_B = L_LM + λ_plan · L_plan + λ_geo · L_geo + λ_fgt · L_FGT + λ_keep · L_keep + λ_sem · L_sem

L_LM / L_plan / L_FGT live outside this step on the GPU side. The synthetic
path here exercises the geometry-only subset:

    L_geo = L_alloc + L_ray + L_rel + L_comp + L_coord (reuses StageA1Step)

so the step-0 invariant + L_keep/L_sem checks remain testable on CPU.

Two entry points:
    stage_b_step       — minimal geometry + sem loss (kept stable for tests)
    stage_b_full_step  — geometry + real z_rel via relation aggregator + sem loss
                          + returns ``h_geo`` so the trainer can attach L_LM / L_plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from geodistill.core.r2ac import r2ac_forward
from geodistill.core.risk_field import soft_risk, oracle_allocation
from geodistill.geometry.cross_view_label import build_edge_labels
from geodistill.geometry.edge_sampler_v2 import sample_edges_v2
from geodistill.losses import (
    L_r, L_q, L_alloc, L_ray, L_comp, L_rank, L_coord, compute_w_p,
    relation_loss, SemanticPreservingLoss,
)
from geodistill.trainers.build_lpga import GeoDistillVLM
from geodistill.trainers.common import check_step0_identity
from geodistill.trainers.train_stage_a1 import StageA1Config, stage_a1_step, _build_rank_pairs


__all__ = ["StageBConfig", "StageBStep", "stage_b_step", "stage_b_full_step"]


@dataclass
class StageBConfig(StageA1Config):
    lambda_keep: float = 0.1
    lambda_sem: float = 0.1
    enforce_step0_identity: bool = True
    geometry_minibatch: bool = True   # whether L_FGT path is enabled — synthetic always treats False


@dataclass
class StageBStep:
    L: torch.Tensor
    parts: dict
    diag: dict


def stage_b_step(
    model: GeoDistillVLM,
    scene,
    teacher,
    cfg: StageBConfig,
    is_first_step: bool = False,
    seed: int = 0,
) -> StageBStep:
    # Allow gates / W_up to receive grad in stage B
    model.unlock_for_stage_b()

    a1 = stage_a1_step(model, scene, teacher, cfg, seed=seed)

    # Re-run injection so we have H_geo to feed into L_keep / L_sem
    out = model.forward_lpga(
        h_img=scene.features,
        token_uv=scene.token_uv,
        token_box=scene.token_box,
        token_camera_id=scene.token_camera_id,
        K=scene.K, cam_to_ego=scene.cam_to_ego, image_hw=scene.image_hw,
        v_ego=scene.v_ego, yaw_rate=scene.yaw_rate if model.cfg.use_yaw_rate else None,
    )
    # z_rel — synthetic path just zero-pads; the GPU trainer will plug in the
    # relation aggregator output here.
    z_rel = out["g_mono"].new_zeros((scene.num_tokens, model.cfg.d_bottleneck))
    h_geo = model.forward_inject(scene.features, out["c_hat"], z_rel)

    if is_first_step and cfg.enforce_step0_identity:
        check_step0_identity(h_geo, scene.features)

    sem_loss = SemanticPreservingLoss()(h_geo, scene.features, scene.cam_offsets)
    L = a1.L + cfg.lambda_keep * sem_loss["L_keep"] + cfg.lambda_sem * sem_loss["L_sem"]
    parts = {**a1.parts, "L_keep": sem_loss["L_keep"], "L_sem": sem_loss["L_sem"], "cka": sem_loss["cka"]}

    return StageBStep(L=L, parts=parts, diag={**a1.diag, "alpha_pe": float(model.injector.alpha_pe.detach()),
                                              "alpha_rel": float(model.injector.alpha_rel.detach())})


# ============================================================ full Stage B step
@dataclass
class StageBFullOutput:
    L_geo: torch.Tensor
    parts: dict
    diag: dict
    h_geo: torch.Tensor                # (N, hidden_size) — splice into the LLM
    image_grid_thw: torch.Tensor | None
    cam_offsets: torch.Tensor


def stage_b_full_step(
    model: GeoDistillVLM,
    scene,
    teacher,
    cfg: StageBConfig,
    is_first_step: bool = False,
    seed: int = 0,
) -> StageBFullOutput:
    """Full Stage B forward.

    Returns ``h_geo`` so the GPU trainer can plug it into the Qwen LLM forward
    via :func:`geodistill.runtime.qwen_lm.compute_lm_loss`. ``L_LM`` / ``L_plan``
    are added by the caller and not by this function — that keeps this routine
    free of transformers / peft.
    """
    model.unlock_for_stage_b()

    # 1) LPGA forward (depth + ray + back-projection)
    out = model.forward_lpga(
        h_img=scene.features,
        token_uv=scene.token_uv,
        token_box=scene.token_box,
        token_camera_id=scene.token_camera_id,
        K=scene.K, cam_to_ego=scene.cam_to_ego, image_hw=scene.image_hw,
        v_ego=scene.v_ego,
        yaw_rate=scene.yaw_rate if model.cfg.use_yaw_rate else None,
    )

    # 2) Relation pipeline → z_rel (real, not zero-pad)
    edges, _types, mean_epp = sample_edges_v2(scene, teacher, cfg.edges, seed=seed)
    diag: dict = {"edges_per_token": mean_epp, "n_edges": len(edges)}
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
        z_rel = rel_out["z_rel"]
        rel_pred = {k: rel_out[k] for k in ("delta_c", "order_logits", "cross_logit",
                                              "topo_logits", "occ_logits", "conf_logit")}
        rl = relation_loss(rel_pred, labels, cfg.rel_weights)
        L_rel = rl.L_rel
        rel_parts = rl.parts
        diag["L_rel_parts"] = {k: float(v.detach()) for k, v in rl.parts.items()}
    else:
        z_rel = out["g_mono"].new_zeros((scene.num_tokens, model.cfg.d_bottleneck))
        L_rel = z_rel.new_zeros(())
        rel_parts = {k: z_rel.new_zeros(()) for k in ("L_delta", "L_order", "L_cross", "L_topo", "L_occ")}

    # 3) inject geometry — H_geo = H_img + α_pe·Φ(c̃) + α_rel·W_up(z_rel)
    h_geo = model.forward_inject(scene.features, out["c_hat"], z_rel)

    if is_first_step and cfg.enforce_step0_identity:
        check_step0_identity(h_geo, scene.features)

    # 4) geometry losses (allocation + ray + depth + coord) on the same forward
    valid = teacher.m_T & torch.isfinite(teacher.d_teacher)
    c_safe = torch.nan_to_num(teacher.c_teacher, nan=0.0)
    d_safe = torch.nan_to_num(teacher.d_teacher, nan=0.0)
    r_p = soft_risk(c_safe[:, 0], c_safe[:, 1], torch.tensor(scene.v_ego), cfg.risk)
    a_star = oracle_allocation(r_p, teacher.q_teacher.float(), cfg.risk.eta)
    zeta_star = r2ac_forward(d_safe, a_star, model.cfg.D_max, model.cfg.beta, model.cfg.a_eps)
    w_p = compute_w_p(valid, teacher.q_teacher.float(), eps_q=cfg.eps_q)
    s_p = teacher.q_mass.float()
    m_f = valid.float()

    val_L_r = L_r(out["r_hat"], r_p, m_f)
    val_L_q = L_q(out["q_hat"], teacher.q_teacher.float(), s_p, m_f)
    val_L_alloc = L_alloc(val_L_r, val_L_q, cfg.lambda_r, cfg.lambda_q)
    val_L_ray = L_ray(out["delta_hat"], teacher.delta_T, w_p)
    val_L_comp = L_comp(out["zeta_hat"], zeta_star, w_p)
    pairs = _build_rank_pairs(scene.num_tokens, valid, cfg.rank_pairs_per_token, seed=seed)
    pair_w = w_p[pairs[:, 0]] * w_p[pairs[:, 1]] if pairs.numel() else w_p.new_zeros(0)
    pair_w = pair_w.sqrt() if pair_w.numel() else pair_w
    val_L_rank = L_rank(out["d_hat"], d_safe, pairs, pair_w, tau_d=cfg.tau_d)
    val_L_coord = L_coord(out["c_hat"], c_safe.detach(), w_p)

    L_geo = (cfg.lambda_alloc * val_L_alloc
             + cfg.lambda_ray * val_L_ray
             + cfg.lambda_rel * L_rel
             + cfg.lambda_depth * val_L_comp
             + cfg.lambda_rank * val_L_rank
             + cfg.lambda_coord * val_L_coord)

    # 5) sem-preserving (L_keep + L_sem)
    sem = SemanticPreservingLoss()(h_geo, scene.features, scene.cam_offsets)
    L_total = L_geo + cfg.lambda_keep * sem["L_keep"] + cfg.lambda_sem * sem["L_sem"]

    parts = {
        "L_alloc": val_L_alloc, "L_r": val_L_r, "L_q": val_L_q,
        "L_ray": val_L_ray, "L_rel": L_rel,
        **{k: v for k, v in rel_parts.items()},
        "L_comp": val_L_comp, "L_rank": val_L_rank, "L_coord": val_L_coord,
        "L_keep": sem["L_keep"], "L_sem": sem["L_sem"], "cka": sem["cka"],
    }
    diag.update({
        "alpha_pe": float(model.injector.alpha_pe.detach()),
        "alpha_rel": float(model.injector.alpha_rel.detach()),
        "n_rank_pairs": int(pairs.shape[0]),
    })
    return StageBFullOutput(
        L_geo=L_total, parts=parts, diag=diag, h_geo=h_geo,
        image_grid_thw=getattr(scene, "image_grid_thw", None),
        cam_offsets=scene.cam_offsets,
    )


# ====================================================================== CLI
def _cfg_from_yaml(yaml_cfg: dict) -> StageBConfig:
    from geodistill.trainers.train_stage_a1 import _cfg_from_yaml as _a1_from_yaml
    a1 = _a1_from_yaml(yaml_cfg)
    lw = yaml_cfg.get("loss_weights", {})
    return StageBConfig(
        risk=a1.risk,
        lambda_r=a1.lambda_r, lambda_q=a1.lambda_q, lambda_alloc=a1.lambda_alloc,
        lambda_ray=a1.lambda_ray, lambda_rel=a1.lambda_rel,
        edges=a1.edges, rel_weights=a1.rel_weights,
        voxel_size=a1.voxel_size, tau_depth=a1.tau_depth, tau_eq_depth=a1.tau_eq_depth,
        lambda_depth=a1.lambda_depth, lambda_rank=a1.lambda_rank, lambda_coord=a1.lambda_coord,
        tau_d=a1.tau_d, eps_q=a1.eps_q,
        lambda_keep=float(lw.get("keep", 0.1)),
        lambda_sem=float(lw.get("sem", 0.1)),
        enforce_step0_identity=True,
        geometry_minibatch=True,
    )


def main():
    """Stage B trainer entry point.

    Two paths:
    - ``--use_llm`` (default on GPU): real Qwen LLM forward + LoRA + L_LM + L_plan.
    - synthetic / no LLM: only geometry + L_keep + L_sem (CPU-friendly smoke).

    The LLM path needs a JSONL of QA / planning instructions formatted to drop
    image tokens at the right positions. We accept ``--qa_jsonl`` + ``--templates``
    on the GPU box; in the synthetic / smoke path the LLM step is skipped and
    the trainer purely optimizes the geometry-side losses + L_keep / L_sem.
    """
    import argparse
    from pathlib import Path

    from geodistill.runtime import (
        load_config_with_extends, train_loop, RunnerConfig, iter_scenes,
    )
    from geodistill.runtime.runner import StepOutput
    from geodistill.trainers.build_lpga import build_from_config
    from geodistill.trainers.common import OptimSpec, build_optimizer, synthetic_hidden_size

    ap = argparse.ArgumentParser(description="GeoDistill stage B trainer (geometry + LLM LoRA)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="runs/geodistill/stage_b")
    ap.add_argument("--source", default=None,
                     choices=[None, "synthetic", "cached_nuscenes", "live_nuscenes"])
    ap.add_argument("--scenes", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None,
                     help="Resume path. If none, will look for runs/geodistill/stage_a1/last.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16"])
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--use_llm", action="store_true",
                     help="Enable Qwen LLM forward + LoRA + L_LM. Requires transformers + Qwen weights.")
    ap.add_argument("--qa_jsonl", default=None, help="Driving QA jsonl for L_LM (paper §4.5).")
    ap.add_argument("--lambda_lm", type=float, default=1.0)
    ap.add_argument("--lambda_plan", type=float, default=1.0)
    args = ap.parse_args()

    yaml_cfg = load_config_with_extends(args.config)
    sched = yaml_cfg.get("schedule", {})
    runner_cfg = RunnerConfig(
        out_dir=args.out_dir,
        max_steps=args.max_steps or int(sched.get("stage_b_steps", 60000)),
        grad_accum_steps=int(sched.get("grad_accum_steps", 1)),
        grad_clip=float(sched.get("grad_clip", 1.0)),
        log_every=args.log_every, ckpt_every=args.ckpt_every,
        amp=(args.amp if args.device == "cuda" else "off"),
        device=args.device,
        seed=args.seed if args.seed is not None else int(yaml_cfg.get("experiment", {}).get("seed", 42)),
        resume=args.resume or "runs/geodistill/stage_a1/last.pt",
    )

    model = build_from_config(yaml_cfg, hidden_size_override=synthetic_hidden_size(yaml_cfg, args.source))
    spec = OptimSpec(
        lr=float(sched.get("lr_lpga", 2e-4)),
        weight_decay=float(sched.get("weight_decay", 0.01)),
        grad_clip=runner_cfg.grad_clip,
    )
    # Optimizer first; LoRA may add params after we wrap the LLM, in which case
    # we'll re-build the optimizer below.
    optim = build_optimizer(model.parameters(), spec)
    stage_cfg = _cfg_from_yaml(yaml_cfg)
    source = args.source or yaml_cfg.get("dataset", {}).get("source", "live_nuscenes")
    batch_iter = iter_scenes(source, yaml_cfg, scenes=args.scenes, seed=runner_cfg.seed,
                              infinite=True)

    # ---- Optional: Qwen LLM + LoRA (paper §3.12) ---------------------------
    qwen, tokenizer, pos_id, qa_iter = None, None, None, None
    if args.use_llm:
        from geodistill.runtime.qwen_lm import (
            QwenLMConfig, load_qwen_for_training, apply_lora_to_llm,
            compute_lm_loss, extract_pos_hidden,
        )
        from geodistill.trainers.coord_decoder import CoordDecoder, CoordDecoderConfig, waypoint_huber_loss

        base = yaml_cfg.get("base_vlm", {})
        lora_cfg = yaml_cfg.get("lora", {})
        qwen, tokenizer, pos_id = load_qwen_for_training(QwenLMConfig(
            model_id=base.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct"),
            torch_dtype=str(sched.get("amp", "bfloat16")),
        ))
        if lora_cfg.get("enable_in_stage_b", True):
            qwen = apply_lora_to_llm(
                qwen, rank=int(lora_cfg.get("rank", 16)),
                alpha=int(lora_cfg.get("alpha", 32)),
                dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=tuple(lora_cfg.get("target_modules",
                                                    ["q_proj", "k_proj", "v_proj", "o_proj"])),
            )
        coord_dec = None
        if yaml_cfg.get("coord_decoder", {}).get("enable", False):
            cd_cfg = yaml_cfg["coord_decoder"]
            coord_dec = CoordDecoder(CoordDecoderConfig(
                hidden_dim=int(cd_cfg.get("hidden_dim", 1024)),
                waypoint_steps=int(cd_cfg.get("waypoint_steps", 6)),
                waypoint_dim=2,
                in_dim=int(base.get("hidden_size", 3584)),
            ))
        # Rebuild optimizer with all trainable params (LoRA + coord decoder).
        params = list(model.parameters())
        params += [p for p in qwen.parameters() if p.requires_grad]
        if coord_dec is not None:
            params += list(coord_dec.parameters())
        optim = torch.optim.AdamW([p for p in params if p.requires_grad],
                                    lr=spec.lr, weight_decay=spec.weight_decay)

        if args.qa_jsonl:
            from dataset.geodistill.driving_qa_dataset import DrivingQADataset
            qa_ds = DrivingQADataset(args.qa_jsonl, min_reliability=0.5)
            print(f"[stage B] driving QA loaded: {qa_ds.stats()}")
            qa_iter = iter(qa_ds)        # GPU side will re-batch with the tokenizer

    # ----------------------------------------------------------------- loss_fn
    def _loss_fn(model, batch, step):
        scene, teacher = batch
        out = stage_b_full_step(model, scene, teacher, stage_cfg,
                                  is_first_step=(step == 0), seed=step)
        L = out.L_geo

        if args.use_llm:
            # Plumb in L_LM if a QA batch is available; otherwise skip without
            # crashing (so a pure-geometry stage B run is still legal).
            try:
                from geodistill.runtime.qwen_lm import compute_lm_loss, extract_pos_hidden
                if qa_iter is None:
                    raise StopIteration
                qa = next(qa_iter)
                tok = tokenizer(qa["question"], qa.get("answer", ""), return_tensors="pt", padding=True)
                input_ids = tok["input_ids"].to(out.h_geo.device)
                attn = tok["attention_mask"].to(out.h_geo.device)
                labels = input_ids.clone()
                # mask the question portion so loss is computed only on the answer
                qlen = int((tok["token_type_ids"][0] == 0).sum()) if "token_type_ids" in tok else 0
                if qlen:
                    labels[:, :qlen] = -100
                lm_loss, lm_out = compute_lm_loss(qwen, out.h_geo, out.image_grid_thw,
                                                    input_ids, attention_mask=attn, labels=labels)
                L = L + args.lambda_lm * lm_loss
                out.parts["L_LM"] = lm_loss

                # Optional planning Huber on <POS> waypoints
                if "waypoints" in qa and coord_dec is not None:
                    pos_hidden = extract_pos_hidden(lm_out, input_ids, pos_id)
                    if int(pos_hidden.shape[1]) > 0:
                        wp_pred = coord_dec(pos_hidden[:, 0])
                        wp = torch.as_tensor(qa["waypoints"], dtype=wp_pred.dtype,
                                              device=wp_pred.device)
                        L_plan = waypoint_huber_loss(wp_pred, wp)
                        L = L + args.lambda_plan * L_plan
                        out.parts["L_plan"] = L_plan
            except StopIteration:
                pass
            except Exception as exc:                                                       # noqa: BLE001
                print(f"[stage B step={step}] LM-loss path skipped: {exc}")

        return StepOutput(loss=L, parts=out.parts, diag=out.diag)

    train_loop(
        model=model, optim=optim, batch_iter=batch_iter,
        loss_fn=_loss_fn, cfg=runner_cfg,
        extra_envelope={"stage": "B", "source": source,
                          "use_llm": args.use_llm,
                          "config": str(yaml_cfg.get("config_path", ""))},
    )


if __name__ == "__main__":
    main()
