"""Token geometry probe + R²AC calibration eval (paper §4.5 Table 2).

Runs same-capacity probes on three feature views per scene:
- ``H_img``  : raw Qwen merger output (baseline)
- ``H_geo``  : after :class:`geometry_injection.GeometryInjector`
- ``model.d_hat``: model-direct decoded depth (no probe)

Reported per-row metrics:
- AbsRel / RMSE / δ1, plus 0-10 / 10-30 / 30m+ bins
- Risk-stratified AbsRel: bucket by oracle ``a*`` (low / med / high)
- ρ(q^OT, |error|): Spearman correlation between transport reliability and
  per-token absolute depth error (paper §4.3)
- ``r̂/q̂/â`` MAE + Spearman vs oracle (paper §4.5 Table 2)
- Linear CKA(H_geo, H_img) and ``GeoGain/SemanticDrift``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from geodistill.core.r2ac import r2ac_forward
from geodistill.core.risk_field import RiskFieldConfig, soft_risk, oracle_allocation
from geodistill.data.contract import TokenScene
from geodistill.data.loader import load_scenes
from geodistill.models.geometry_injection import linear_cka
from geodistill.probe.metrics import depth_metrics, depth_metrics_binned, spearman
from geodistill.probe.token_geometry import LinearProbe, ProbeConfig
from geodistill.teacher import (
    build_hard_quantile_teacher, build_entropic_ot_teacher, build_full_ntlfgt_teacher, OTConfig,
)
from geodistill.trainers import GeoDistillConfig, GeoDistillVLM
from geodistill.utils import JsonlWriter, run_envelope


__all__ = ["run_token_probe", "run_table2"]


TEACHER_BUILDERS = {
    "hard_quantile": build_hard_quantile_teacher,
    "entropic_ot": lambda s: build_entropic_ot_teacher(s, OTConfig(epsilon_ot=0.05, gamma=0.0)),
    "full_ntl_fgt": lambda s: build_full_ntlfgt_teacher(s, OTConfig(epsilon_ot=0.05, gamma=0.5)),
}


def _train_probe(features: torch.Tensor, target: torch.Tensor, val_idx: torch.Tensor,
                 tr_idx: torch.Tensor, hidden: int, epochs: int, lr: float, D_max: float):
    mu = features[tr_idx].mean(0, keepdim=True)
    sd = features[tr_idx].std(0, keepdim=True).clamp_min(1e-6)
    X = (features - mu) / sd
    y = (target / D_max).clamp(0, 1)

    probe = LinearProbe(features.shape[1], hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.SmoothL1Loss()
    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = probe(X[tr_idx])
        loss = lossf(pred, y[tr_idx])
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pred_val = probe(X[val_idx]).clamp(0, 1) * D_max
    return pred_val


def _stratified_absrel(pred: torch.Tensor, gt: torch.Tensor, a_star: torch.Tensor, q: int = 3) -> dict:
    out: dict[str, dict] = {}
    if pred.numel() == 0:
        return {f"q{i}": {"AbsRel": float("nan"), "n": 0} for i in range(q)}
    quantiles = torch.linspace(0, 1, q + 1)
    edges = torch.quantile(a_star.float(), quantiles)
    for i in range(q):
        sel = (a_star >= edges[i]) & (a_star <= edges[i + 1])
        if i < q - 1:
            sel = (a_star >= edges[i]) & (a_star < edges[i + 1])
        out[f"q{i}"] = depth_metrics(pred[sel], gt[sel])
    return out


@dataclass
class TokenProbeOutput:
    teacher: str
    n_tokens: int
    h_img_metrics: dict
    h_geo_metrics: dict | None
    model_direct_metrics: dict | None
    risk_strat_h_geo: dict | None
    rho_qOT_err: float
    cka: float | None
    geo_gain_over_drift: float | None
    factor_calibration: dict


def run_token_probe(
    scene: TokenScene,
    teacher_name: str = "hard_quantile",
    model: GeoDistillVLM | None = None,
    cfg: ProbeConfig = ProbeConfig(),
    risk_cfg: RiskFieldConfig = RiskFieldConfig(),
) -> TokenProbeOutput:
    teacher = TEACHER_BUILDERS[teacher_name](scene)
    valid = teacher.m_T & torch.isfinite(teacher.d_teacher) & scene.gt_valid & torch.isfinite(scene.gt_depth)
    if int(valid.sum()) < 4:
        return TokenProbeOutput(teacher_name, 0, {"error": "no_valid_tokens"}, None, None, None,
                                 float("nan"), None, None, {})

    n = int(valid.sum())
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(cfg.val_frac * n))
    valid_idx = valid.nonzero(as_tuple=True)[0]
    val_idx = valid_idx[perm[:n_val]]
    tr_idx = valid_idx[perm[n_val:]]

    H_img = scene.features
    pred_img = _train_probe(H_img, teacher.d_teacher, val_idx=val_idx, tr_idx=tr_idx,
                             hidden=cfg.hidden, epochs=cfg.epochs, lr=cfg.lr, D_max=cfg.D_max)
    gt_val = scene.gt_depth[val_idx]
    h_img_metrics = {
        "overall": depth_metrics(pred_img, gt_val),
        "binned": depth_metrics_binned(pred_img, gt_val),
    }

    h_geo_metrics = None
    model_direct_metrics = None
    risk_strat = None
    cka_val: float | None = None
    geo_gain: float | None = None
    factor_calib: dict = {}
    rho = float("nan")

    if model is not None:
        with torch.no_grad():
            out_lpga = model.forward_lpga(
                h_img=scene.features, token_uv=scene.token_uv, token_box=scene.token_box,
                token_camera_id=scene.token_camera_id, K=scene.K, cam_to_ego=scene.cam_to_ego,
                image_hw=scene.image_hw, v_ego=scene.v_ego,
                yaw_rate=scene.yaw_rate if model.cfg.use_yaw_rate else None,
            )
            z_rel = out_lpga["g_mono"].new_zeros((scene.num_tokens, model.cfg.d_bottleneck))
            H_geo = model.forward_inject(scene.features, out_lpga["c_hat"], z_rel)

        pred_geo = _train_probe(H_geo, teacher.d_teacher, val_idx=val_idx, tr_idx=tr_idx,
                                 hidden=cfg.hidden, epochs=cfg.epochs, lr=cfg.lr, D_max=cfg.D_max)
        h_geo_metrics = {
            "overall": depth_metrics(pred_geo, gt_val),
            "binned": depth_metrics_binned(pred_geo, gt_val),
        }
        d_hat_val = out_lpga["d_hat"][val_idx].clamp(0, cfg.D_max)
        model_direct_metrics = {
            "overall": depth_metrics(d_hat_val, gt_val),
            "binned": depth_metrics_binned(d_hat_val, gt_val),
        }

        # Risk-stratified AbsRel using oracle a*
        c_safe = torch.nan_to_num(teacher.c_teacher, nan=0.0)
        r_p = soft_risk(c_safe[:, 0], c_safe[:, 1], torch.tensor(scene.v_ego), risk_cfg)
        a_star = oracle_allocation(r_p, teacher.q_teacher.float(), risk_cfg.eta)
        risk_strat = _stratified_absrel(d_hat_val, gt_val, a_star[val_idx], q=3)

        # ρ(q^OT, |err|)
        q_ot = teacher.q_conc[val_idx] * teacher.q_mass[val_idx]
        err = (d_hat_val - gt_val).abs()
        rho = spearman(q_ot, err)

        # CKA + GeoGain/SemanticDrift
        cka_val = float(linear_cka(H_geo, H_img))
        # GeoGain in higher-is-better terms: relative AbsRel reduction.
        absrel_img = float(h_img_metrics["overall"]["AbsRel"])
        absrel_geo = float(h_geo_metrics["overall"]["AbsRel"])
        if absrel_img > 1e-9 and (1.0 - cka_val) > 1e-9:
            geo_gain = ((absrel_img - absrel_geo) / absrel_img) / (1.0 - cka_val)

        # r̂ / q̂ / â calibration (oracle r_p and q_teacher, composed a*)
        m_calib = valid
        factor_calib = {
            "r_hat_MAE": float((out_lpga["r_hat"][m_calib] - r_p[m_calib]).abs().mean()),
            "r_hat_spearman": spearman(out_lpga["r_hat"][m_calib], r_p[m_calib]),
            "q_hat_MAE": float((out_lpga["q_hat"][m_calib] - teacher.q_teacher[m_calib].float()).abs().mean()),
            "q_hat_spearman": spearman(out_lpga["q_hat"][m_calib], teacher.q_teacher[m_calib].float()),
            "a_hat_MAE": float((out_lpga["a_hat"][m_calib] - a_star[m_calib]).abs().mean()),
            "a_hat_spearman": spearman(out_lpga["a_hat"][m_calib], a_star[m_calib]),
        }

    return TokenProbeOutput(
        teacher=teacher_name, n_tokens=n,
        h_img_metrics=h_img_metrics, h_geo_metrics=h_geo_metrics,
        model_direct_metrics=model_direct_metrics, risk_strat_h_geo=risk_strat,
        rho_qOT_err=rho, cka=cka_val, geo_gain_over_drift=geo_gain,
        factor_calibration=factor_calib,
    )


def run_table2(scenes, *, teacher: str, model: GeoDistillVLM | None, cfg: ProbeConfig, out_path: str,
               source: str = "synthetic") -> None:
    with JsonlWriter(out_path) as w:
        w.write({**run_envelope("eval_token_geometry_probe", source,
                                  {"teacher": teacher, "scenes": len(scenes), "hidden": cfg.hidden,
                                   "epochs": cfg.epochs}),
                 "kind": "header"})
        for si, s in enumerate(scenes):
            r = run_token_probe(s, teacher, model=model, cfg=cfg)
            w.write({"kind": "table2_row", "scene_idx": si, "scene_id": s.scene_id,
                      "teacher": r.teacher, "n_tokens": r.n_tokens,
                      "h_img": r.h_img_metrics, "h_geo": r.h_geo_metrics,
                      "model_direct": r.model_direct_metrics,
                      "risk_strat_h_geo": r.risk_strat_h_geo, "rho_qOT_vs_err": r.rho_qOT_err,
                      "cka": r.cka, "geo_gain_over_drift": r.geo_gain_over_drift,
                      "factor_calibration": r.factor_calibration})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--teacher", default="hard_quantile", choices=list(TEACHER_BUILDERS))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--with_model", action="store_true",
                    help="If set, build a GeoDistillVLM and report H_geo / model-direct rows. "
                         "Without this flag the table only has the H_img baseline column.")
    ap.add_argument("--out", default="runs/geodistill/eval/table2_token_probe.jsonl")
    args = ap.parse_args()

    scenes = load_scenes(args.source, args.scenes, args.seed, args.config)

    model = None
    if args.with_model:
        s0 = scenes[0]
        from geodistill.trainers.build_lpga import auto_pe_dims
        cfg = GeoDistillConfig(
            hidden_size=int(s0.features.shape[-1]),
            d_bottleneck=64, d_e_cam=8, d_e_uv=16, d_e_calib=16, d_e_ego=8,
            num_cameras=int(s0.num_cameras),
            pe_d_xyz=auto_pe_dims(int(s0.features.shape[-1])),
            rel_d_edge=32, rel_d_hidden=32,
        )
        model = GeoDistillVLM(cfg)

    run_table2(scenes, teacher=args.teacher, model=model,
               cfg=ProbeConfig(hidden=args.hidden, epochs=args.epochs),
               out_path=args.out, source=args.source)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
