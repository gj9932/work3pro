"""``<POS>`` coordinate decoder (paper §3.12, declared as interface, not contribution).

Given a sequence of LLM hidden states with a designated ``<POS>`` token id, this
extracts the hidden states at those positions and decodes per-step waypoints
via a small MLP. Loss is Huber on (x, y) (or (x, y, z) optionally).

The runtime path requires transformers + Qwen2.5-VL; we only define the small
nn.Module here and a Huber loss helper. The actual hidden-state extraction
happens in the GPU-side trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["CoordDecoderConfig", "CoordDecoder", "waypoint_huber_loss"]


@dataclass(frozen=True)
class CoordDecoderConfig:
    hidden_dim: int = 1024
    waypoint_steps: int = 6
    waypoint_dim: int = 2          # 2 = (x, y); 3 = (x, y, z)
    in_dim: int = 3584             # Qwen2.5-VL hidden


class CoordDecoder(nn.Module):
    def __init__(self, cfg: CoordDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.waypoint_steps * cfg.waypoint_dim),
        )

    def forward(self, pos_hidden: torch.Tensor) -> torch.Tensor:
        """``pos_hidden``: (B, in_dim) hidden state at the ``<POS>`` token.

        Returns (B, T, waypoint_dim) waypoint sequence.
        """
        out = self.net(pos_hidden)
        return out.view(-1, self.cfg.waypoint_steps, self.cfg.waypoint_dim)


def waypoint_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
                        beta: float = 1.0) -> torch.Tensor:
    elem = F.huber_loss(pred, target, reduction="none", delta=beta).sum(dim=-1)
    if mask is None:
        return elem.mean()
    mask = mask.float()
    return (elem * mask).sum() / mask.sum().clamp_min(1e-9)
