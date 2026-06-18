"""Common trainer utilities (paper §3.13).

Kept dependency-light: torch only. AMP, grad clipping, optimizer construction,
checkpoint save/resume, and a tiny stage-0 invariant verifier live here so
every stage trainer (A0/A1/B/C) shares the same plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch


__all__ = [
    "OptimSpec", "build_optimizer", "save_ckpt", "load_ckpt",
    "freeze_module", "unfreeze_module", "trainable_params",
    "check_step0_identity", "synthetic_hidden_size",
]


@dataclass
class OptimSpec:
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0


def build_optimizer(params: Iterable[torch.nn.Parameter], spec: OptimSpec):
    return torch.optim.AdamW([p for p in params if p.requires_grad], lr=spec.lr,
                             weight_decay=spec.weight_decay, betas=spec.betas)


def save_ckpt(path: str | Path, model: torch.nn.Module, optim: torch.optim.Optimizer | None = None,
              step: int = 0, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "step": step, "extra": extra or {}}
    if optim is not None:
        payload["optim"] = optim.state_dict()
    torch.save(payload, path)


def load_ckpt(path: str | Path, model: torch.nn.Module,
              optim: torch.optim.Optimizer | None = None, strict: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=strict)
    if optim is not None and "optim" in payload:
        optim.load_state_dict(payload["optim"])
    return {"step": int(payload.get("step", 0)), "extra": payload.get("extra", {})}


def freeze_module(m: torch.nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


def unfreeze_module(m: torch.nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(True)


def trainable_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def check_step0_identity(h_geo: torch.Tensor, h_img: torch.Tensor, atol: float = 1e-6) -> None:
    """Stage B step-0 invariant: H_geo must equal H_img (paper §3.11).

    Raises if violated, with a hint pointing to the gate / W_up init.
    """
    diff = (h_geo.detach() - h_img.detach()).abs().max()
    if float(diff) > atol:
        raise AssertionError(
            f"Stage B step-0 identity violated: ||H_geo - H_img||_∞ = {float(diff):.3e}. "
            f"Check that β_pe = β_rel = 0 at init (paper §3.11)."
        )


def synthetic_hidden_size(yaml_cfg: dict, source: str | None) -> int | None:
    """Return the hidden size override for synthetic / cached_nuscenes paths.

    On the synthetic fixture ``scene.features`` has ``feature_dim`` channels
    (default 64), not Qwen's 3584. We need ``GeoDistillVLM`` built around that
    dim so the LPGA / PE / injector all line up.

    Returns ``None`` when the yaml's ``base_vlm.hidden_size`` should be honored
    verbatim (live nuScenes path on the GPU box).
    """
    if source != "synthetic":
        return None
    syn = (yaml_cfg.get("synthetic") or {})
    return int(syn.get("feature_dim", 64))
