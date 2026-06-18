"""Depth-representation heads for the model comparison (paper §4.4 baselines 5-9, 16; Table 2).

All heads read the SAME frozen token features (Qwen merger output / synthetic) and
differ only in how depth is represented/decoded. Each predicts metric depth so the
probe metrics are directly comparable:

  raw          : head -> depth directly (clamped to [0, D_max])
  log          : head -> log-depth; depth = exp(.)
  power_warp   : head -> z in [0,1]; depth = D_max * z^p (fixed power, Barron-style)
  r2ac         : head -> companded z; predicts sensitivity a; depth = F_inv(z; a)
  unidepth_pe  : SURROGATE for "Qwen + external monocular depth": a frozen noisy
                 monocular-depth oracle (NOT trained on these tokens) feeds depth.
                 On synthetic data we simulate the external estimator as GT depth +
                 calibrated bias/noise so the precision/efficiency Pareto has a
                 stand-in; on real data this slot is wired to UniDepthV2 (TBD).
  lpga         : the R²AC head IS the LPGA depth head here (alias) — kept separate
                 so the comparison table has an explicit "Qwen + LPGA" row.

Only raw/log/power_warp/r2ac/lpga are trainable on token features. unidepth_pe is
an external-oracle baseline and is not trained on these features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from geodistill.core.r2ac import r2ac_forward, r2ac_inverse


class _Backbone(nn.Module):
    """Shared tiny trunk over frozen features (the only learned part of a head)."""

    def __init__(self, dim: int, hidden: int = 64, out: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


class RawDepthHead(nn.Module):
    def __init__(self, dim, D_max=80.0):
        super().__init__()
        self.bb = _Backbone(dim)
        self.D_max = D_max

    def depth(self, x):
        return torch.sigmoid(self.bb(x).squeeze(-1)) * self.D_max


class LogDepthHead(nn.Module):
    def __init__(self, dim, D_max=80.0):
        super().__init__()
        self.bb = _Backbone(dim)
        self.D_max = D_max

    def depth(self, x):
        # predict log-depth in a bounded band [log(0.5), log(D_max)]
        lo, hi = torch.log(torch.tensor(0.5)), torch.log(torch.tensor(self.D_max))
        z = torch.sigmoid(self.bb(x).squeeze(-1))
        return torch.exp(lo + (hi - lo) * z)


class PowerWarpHead(nn.Module):
    def __init__(self, dim, D_max=80.0, power=2.0):
        super().__init__()
        self.bb = _Backbone(dim)
        self.D_max = D_max
        self.power = power

    def depth(self, x):
        z = torch.sigmoid(self.bb(x).squeeze(-1))
        return self.D_max * z.clamp(0, 1) ** self.power


class R2ACHead(nn.Module):
    """Predicts companded z AND sensitivity a; decodes via analytic inverse.

    a is predicted from features (the paper's "predicted a"); during decode a is
    detached (StopGrad) so the depth gradient cannot flow back through the inverse
    into the allocation — matching paper §3.6.
    """

    def __init__(self, dim, D_max=80.0, beta=None):
        super().__init__()
        self.bb_z = _Backbone(dim)
        self.bb_a = _Backbone(dim)
        self.D_max = D_max
        self.beta = beta if beta is not None else float(torch.log(torch.tensor(16.0)))

    def z_and_a(self, x):
        z = torch.sigmoid(self.bb_z(x).squeeze(-1))
        a = torch.sigmoid(self.bb_a(x).squeeze(-1))
        return z, a

    def depth(self, x):
        z, a = self.z_and_a(x)
        return r2ac_inverse(z, a.detach(), self.D_max, self.beta)


HEAD_REGISTRY = {
    "raw": RawDepthHead,
    "log": LogDepthHead,
    "power_warp": PowerWarpHead,
    "r2ac": R2ACHead,
    "lpga": R2ACHead,   # "Qwen + LPGA" == R²AC depth head over frozen tokens
}


@dataclass(frozen=True)
class HeadTrainConfig:
    epochs: int = 300
    lr: float = 1e-2
    weight_decay: float = 1e-4
    D_max: float = 80.0
    beta: float = float(torch.log(torch.tensor(16.0)))
