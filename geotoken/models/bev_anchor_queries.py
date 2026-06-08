from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # keep package importable before dependencies are installed
    torch = None

    class _MissingTorchModule:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to use BEVAnchorQueries")

    class nn:  # type: ignore[no-redef]
        Module = object
        Embedding = _MissingTorchModule
        Linear = _MissingTorchModule

from geotoken.geometry import BEVAnchorGrid, BEVGridConfig


@dataclass(frozen=True)
class BEVAnchorQueryConfig:
    embed_dim: int = 256
    active_anchor_count: int = 400
    full_anchor_count: int = 1600


class BEVAnchorQueries(nn.Module):
    def __init__(self, config: BEVAnchorQueryConfig | None = None, grid: BEVAnchorGrid | None = None):
        super().__init__()
        self.config = config or BEVAnchorQueryConfig()
        self.grid = grid or BEVAnchorGrid(BEVGridConfig(active_anchor_count=self.config.active_anchor_count))
        if self.grid.num_anchors != self.config.full_anchor_count:
            raise ValueError(f"full_anchor_count={self.config.full_anchor_count} does not match grid size={self.grid.num_anchors}")
        self.query_embed = nn.Embedding(self.config.full_anchor_count, self.config.embed_dim)
        self.coord_proj = nn.Linear(2, self.config.embed_dim)

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        bev_cfg = config.get("bev", {})
        model_cfg = config.get("model", {}).get("bev_anchor_queries", {})
        embed_dim = int(model_cfg.get("embed_dim", config.get("model", {}).get("embed_dim", 256)))
        grid = BEVAnchorGrid.from_config(config)
        return cls(
            BEVAnchorQueryConfig(
                embed_dim=embed_dim,
                active_anchor_count=int(bev_cfg.get("active_anchor_count", 400)),
                full_anchor_count=grid.num_anchors,
            ),
            grid=grid,
        )

    def forward(self, anchor_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if anchor_ids is None:
            anchor_ids = torch.arange(self.config.full_anchor_count, device=self.query_embed.weight.device)
        anchor_ids = anchor_ids.to(device=self.query_embed.weight.device, dtype=torch.long)
        centers = self.grid.centers.to(device=anchor_ids.device, dtype=self.query_embed.weight.dtype)[anchor_ids]
        queries = self.query_embed(anchor_ids) + self.coord_proj(_normalize_centers(centers, self.grid.config))
        return {"anchor_ids": anchor_ids, "anchor_centers": centers, "anchor_queries": queries}


def _normalize_centers(centers: torch.Tensor, config: BEVGridConfig) -> torch.Tensor:
    x_mid = (config.x_range[0] + config.x_range[1]) * 0.5
    y_mid = (config.y_range[0] + config.y_range[1]) * 0.5
    x_scale = max((config.x_range[1] - config.x_range[0]) * 0.5, 1e-6)
    y_scale = max((config.y_range[1] - config.y_range[0]) * 0.5, 1e-6)
    offset = centers.new_tensor([x_mid, y_mid])
    scale = centers.new_tensor([x_scale, y_scale])
    return (centers - offset) / scale
