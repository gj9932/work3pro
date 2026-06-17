"""Generic training loop shared by stage A0/A1/B/C trainers.

The loop owns:
- iteration over a scene generator (synthetic / cached nuScenes / live nuScenes)
- AMP autocast (configurable bf16 / fp16 / off)
- gradient accumulation + clipping
- checkpoint save / resume + last-step bookkeeping
- step-0 invariant check (paper §3.11 stage B identity)
- log-every-N JSONL diagnostics + last loss
- optional eval hook every K steps

Each stage trainer plugs in:
- ``loss_fn(model, batch) -> StepOutput`` — the per-step forward + loss assembly
- ``trainable_param_filter`` — which params the optimizer actually updates
- ``step0_invariant`` — callable run at iter==0 (e.g. assert H_geo == H_img)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol

import torch
import torch.nn as nn

from geodistill.utils import JsonlWriter, run_envelope


__all__ = ["RunnerConfig", "StepOutput", "save_ckpt", "load_ckpt", "train_loop"]


@dataclass
class StepOutput:
    """Loss + named parts + free-form diagnostics produced per step."""
    loss: torch.Tensor
    parts: dict[str, torch.Tensor] = field(default_factory=dict)
    diag: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    out_dir: Path
    max_steps: int = 1000
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    log_every: int = 10
    ckpt_every: int = 1000
    amp: str = "off"                       # "off" | "bf16" | "fp16"
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 42
    resume: str | None = None


def _amp_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def save_ckpt(path: str | Path, model: nn.Module, optim: torch.optim.Optimizer | None,
              step: int, extra: dict | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "step": int(step), "extra": extra or {}}
    if optim is not None:
        payload["optim"] = optim.state_dict()
    torch.save(payload, p)


def load_ckpt(path: str | Path, model: nn.Module,
              optim: torch.optim.Optimizer | None = None, strict: bool = False) -> int:
    payload = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    if missing:
        print(f"[load_ckpt] missing keys ({len(missing)}): {missing[:4]}{'...' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"[load_ckpt] unexpected keys ({len(unexpected)}): {unexpected[:4]}")
    if optim is not None and "optim" in payload:
        try:
            optim.load_state_dict(payload["optim"])
        except Exception as exc:                                                       # noqa: BLE001
            print(f"[load_ckpt] optim state mismatch ({exc}); starting fresh optimizer state")
    return int(payload.get("step", 0))


def train_loop(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    batch_iter: Iterator,
    loss_fn: Callable[[nn.Module, Any, int], StepOutput],
    cfg: RunnerConfig,
    *,
    step0_invariant: Callable[[nn.Module, Any], None] | None = None,
    eval_fn: Callable[[nn.Module, int], dict] | None = None,
    eval_every: int = 0,
    extra_envelope: dict | None = None,
) -> int:
    """Run a generic training loop. Returns the final step count.

    ``batch_iter`` must be an infinite or large-enough iterator yielding the
    domain batch type (here typically a TokenScene + teacher pair). ``loss_fn``
    consumes ``(model, batch, step)`` and returns a :class:`StepOutput`. The
    runner is loss-domain agnostic.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model.to(device)

    amp_dtype = _amp_dtype(cfg.amp)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp == "fp16" and device.type == "cuda"))

    start_step = 0
    if cfg.resume and Path(cfg.resume).exists():
        start_step = load_ckpt(cfg.resume, model, optim) + 1
        print(f"[runner] resumed from {cfg.resume} @ step {start_step}")

    torch.manual_seed(cfg.seed + start_step)

    with JsonlWriter(log_path) as w:
        w.write({**run_envelope("train_runner", "nuscenes" if cfg.device == "cuda" else "synthetic",
                                 {**(extra_envelope or {}), "max_steps": cfg.max_steps,
                                  "grad_accum": cfg.grad_accum_steps, "amp": cfg.amp,
                                  "device": str(device), "out_dir": str(out_dir)}),
                 "kind": "header"})

        step = start_step
        running_loss = 0.0
        accum_count = 0
        t_start = time.perf_counter()
        optim.zero_grad(set_to_none=True)

        for batch in batch_iter:
            if step >= cfg.max_steps:
                break

            ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
                   if use_amp else _NullCtx())
            with ctx:
                out = loss_fn(model, batch, step)

            if step == start_step and step0_invariant is not None:
                step0_invariant(model, batch)

            loss = out.loss / max(1, cfg.grad_accum_steps)
            if cfg.amp == "fp16" and device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.detach().item()) * max(1, cfg.grad_accum_steps)
            accum_count += 1

            if accum_count >= cfg.grad_accum_steps:
                if cfg.amp == "fp16" and device.type == "cuda":
                    scaler.unscale_(optim)
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                    )
                if cfg.amp == "fp16" and device.type == "cuda":
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                accum_count = 0

                if step % max(1, cfg.log_every) == 0:
                    parts = {k: float(v.detach().item()) if torch.is_tensor(v) else float(v)
                             for k, v in out.parts.items()}
                    rec = {"kind": "step", "step": step,
                            "loss": running_loss / max(1, cfg.grad_accum_steps),
                            "parts": parts, "diag": _safe_diag(out.diag),
                            "elapsed_s": time.perf_counter() - t_start}
                    w.write(rec)
                    print(_fmt_step(rec))
                    running_loss = 0.0

                if cfg.ckpt_every and step > 0 and step % cfg.ckpt_every == 0:
                    save_ckpt(out_dir / f"step_{step:08d}.pt", model, optim, step,
                                extra={"running_loss": running_loss})
                    save_ckpt(out_dir / "last.pt", model, optim, step)

                if eval_fn is not None and eval_every and step > 0 and step % eval_every == 0:
                    metrics = eval_fn(model, step)
                    w.write({"kind": "eval", "step": step, "metrics": _safe_diag(metrics)})

                step += 1

        # Always save a final ``last.pt`` so the next stage can resume.
        save_ckpt(out_dir / "last.pt", model, optim, step)
        return step


class _NullCtx:
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False


def _fmt_step(rec: dict) -> str:
    parts = rec.get("parts", {})
    parts_str = " ".join(f"{k}={v:.3f}" for k, v in list(parts.items())[:6])
    return f"[step {rec['step']:>8}] L={rec['loss']:.4f}  {parts_str}"


def _safe_diag(diag: Any) -> Any:
    if isinstance(diag, dict):
        return {k: _safe_diag(v) for k, v in diag.items()}
    if isinstance(diag, (list, tuple)):
        return [_safe_diag(v) for v in diag]
    if torch.is_tensor(diag):
        return diag.detach().cpu().tolist() if diag.numel() <= 16 else f"<tensor shape={tuple(diag.shape)}>"
    if isinstance(diag, (str, int, float, bool)) or diag is None:
        return diag
    return str(diag)
