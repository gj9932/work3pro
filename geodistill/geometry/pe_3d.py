"""Universal 3D Positional Encoding (paper §3.8).

For each ego-frame coordinate ``c_p = (x, y, z)``, a per-axis sinusoidal embedding is

    phi_a(a)[2i]   = sin(a / tau^(2i/d_a))
    phi_a(a)[2i+1] = cos(a / tau^(2i/d_a))

Default partitioning ``d_x = d_y = 1194, d_z = 1196`` so the concatenation has
the exact 3584 channels of Qwen2.5-VL-7B's hidden size (the partition was chosen
so each axis dim is even and the total matches the Qwen merger output).

Self-implemented (no SpaceDrive import) so the §4.6 ablation rows
"fixed PE scale / learnable zero-init gate / no L_keep" remain controllable.

Design alignment with SpaceDrive (for review-time reference):
- per-axis sinusoidal,
- frequency base ``tau = 10000``,
- partitioned along channel.
"""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["Universal3DPE"]


class Universal3DPE(nn.Module):
    def __init__(self, d_x: int = 1194, d_y: int = 1194, d_z: int = 1196,
                 tau: float = 10000.0, scale: float = 1.0):
        super().__init__()
        for name, d in (("d_x", d_x), ("d_y", d_y), ("d_z", d_z)):
            if d % 2 != 0:
                raise ValueError(f"{name} must be even (got {d})")
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.tau = tau
        self.scale = scale

        # Pre-compute frequency tables (registered as buffers so they move with
        # the module and are saved with state_dict).
        for name, d in (("freq_x", d_x), ("freq_y", d_y), ("freq_z", d_z)):
            half = d // 2
            freqs = tau ** (-torch.arange(half, dtype=torch.float32) / float(half))
            self.register_buffer(name, freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return self.d_x + self.d_y + self.d_z

    def _phi(self, a: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # a:(N,), freqs:(half,) -> (N, 2*half)
        args = a.unsqueeze(-1) * freqs.to(a.device, a.dtype)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, c_ego: torch.Tensor) -> torch.Tensor:
        """``c_ego``: (N, 3) -> (N, d_x + d_y + d_z)."""
        if c_ego.dim() != 2 or c_ego.shape[-1] != 3:
            raise ValueError(f"Universal3DPE expects (N,3), got {tuple(c_ego.shape)}")
        c = c_ego.float() * self.scale
        x = self._phi(c[:, 0], self.freq_x)
        y = self._phi(c[:, 1], self.freq_y)
        z = self._phi(c[:, 2], self.freq_z)
        return torch.cat([x, y, z], dim=-1)
