from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

try:
    import torch
except ImportError:  # keep geometry importable before dependencies are installed
    torch = None

from .bev_grid import BEVAnchorGrid, OCCUPANCY_FREE, OCCUPANCY_OCCUPIED, OCCUPANCY_UNKNOWN


@dataclass(frozen=True)
class RelationLabelSpec:
    distance_bins: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
    direction_classes: tuple[str, ...] = (
        "same-cell",
        "front",
        "front-left",
        "left",
        "rear-left",
        "rear",
        "rear-right",
        "right",
        "front-right",
    )
    topology_classes: tuple[str, ...] = (
        "same-free-space",
        "same-occupied-component",
        "free-to-occupied",
        "occupied-to-free",
        "different-free-components",
        "unknown",
    )
    occlusion_classes: tuple[str, ...] = ("i-before-j", "j-before-i", "same-depth")


class RelationGraphBuilder:
    TOPO_SAME_FREE = 0
    TOPO_SAME_OCCUPIED = 1
    TOPO_FREE_TO_OCCUPIED = 2
    TOPO_OCCUPIED_TO_FREE = 3
    TOPO_DIFFERENT_FREE = 4
    TOPO_UNKNOWN = 5
    OCCLUSION_IGNORE_INDEX = -1

    def __init__(self, grid: BEVAnchorGrid | None = None, spec: RelationLabelSpec | None = None):
        _require_torch()
        self.grid = grid
        self.spec = spec or RelationLabelSpec()

    @classmethod
    def from_config(cls, config: dict[str, Any], grid: BEVAnchorGrid | None = None):
        psrd_cfg = config.get("psrd", {})
        distance_bins = tuple(float(v) for v in psrd_cfg.get("distance_bins", RelationLabelSpec().distance_bins))
        return cls(grid=grid or BEVAnchorGrid.from_config(config), spec=RelationLabelSpec(distance_bins=distance_bins))

    def build(
        self,
        anchors_xy: torch.Tensor | None = None,
        active_anchor_ids: torch.Tensor | list[int] | None = None,
        occupancy_labels: torch.Tensor | list[int] | None = None,
        camera_visible: torch.Tensor | list[bool] | None = None,
        projection: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        centers, anchor_ids = self._resolve_anchors(anchors_xy, active_anchor_ids)
        device = centers.device
        distance = torch.cdist(centers, centers, p=2)
        dx = centers[None, :, 0] - centers[:, None, 0]
        dy = centers[None, :, 1] - centers[:, None, 1]

        occupancy, component_ids = self._active_occupancy(occupancy_labels, anchor_ids, centers.shape[0], device)
        topology_labels = self._topology_labels(occupancy, component_ids)
        relation_mask = self._relation_mask(topology_labels, occupancy_labels is not None)

        if camera_visible is not None:
            visible = self._active_mask(camera_visible, anchor_ids, centers.shape[0], device)
            relation_mask = relation_mask & visible[:, None] & visible[None, :]
        if projection is not None and "camera_visible" in projection:
            visible = self._active_mask(projection["camera_visible"], anchor_ids, centers.shape[0], device)
            relation_mask = relation_mask & visible[:, None] & visible[None, :]

        occlusion_labels, occlusion_mask = self._occlusion_stub(centers.shape[0], device)
        return {
            "anchor_ids": anchor_ids,
            "anchor_centers": centers,
            "distance": distance,
            "distance_labels": self._distance_labels(distance),
            "direction_labels": self._direction_labels(dx, dy, distance),
            "topology_labels": topology_labels,
            "occlusion_labels": occlusion_labels,
            "relation_mask": relation_mask,
            "occlusion_mask": occlusion_mask,
            "m_ij": relation_mask,
            "m_occ_ij": occlusion_mask,
            "occupancy_labels": occupancy,
            "topology_component_ids": component_ids,
            "label_spec": {
                "distance_bins": list(self.spec.distance_bins),
                "direction_classes": list(self.spec.direction_classes),
                "topology_classes": list(self.spec.topology_classes),
                "occlusion_classes": list(self.spec.occlusion_classes),
            },
        }

    def _resolve_anchors(
        self,
        anchors_xy: torch.Tensor | None,
        active_anchor_ids: torch.Tensor | list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchors_xy is None:
            if self.grid is None:
                raise ValueError("anchors_xy is required when RelationGraphBuilder has no BEV grid")
            if active_anchor_ids is None:
                anchor_ids = torch.arange(self.grid.num_anchors, dtype=torch.long)
            else:
                anchor_ids = torch.as_tensor(active_anchor_ids, dtype=torch.long).reshape(-1)
            centers = self.grid.centers[anchor_ids]
            return centers.to(dtype=torch.float32), anchor_ids

        centers = torch.as_tensor(anchors_xy, dtype=torch.float32)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(f"anchors_xy must have shape [L, 2], got {tuple(centers.shape)}")
        if active_anchor_ids is None:
            anchor_ids = torch.arange(centers.shape[0], dtype=torch.long, device=centers.device)
        else:
            anchor_ids = torch.as_tensor(active_anchor_ids, dtype=torch.long, device=centers.device).reshape(-1)
            if anchor_ids.numel() != centers.shape[0]:
                raise ValueError("active_anchor_ids and anchors_xy must have the same length")
        return centers, anchor_ids

    def _distance_labels(self, distance: torch.Tensor) -> torch.Tensor:
        bins = torch.tensor(self.spec.distance_bins[1:], dtype=distance.dtype, device=distance.device)
        return torch.bucketize(distance, bins, right=True).to(torch.long)

    def _direction_labels(self, dx: torch.Tensor, dy: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        sector = torch.remainder(torch.round(torch.atan2(dy, dx) / (math.pi / 4.0)).to(torch.long), 8) + 1
        return torch.where(distance <= 1e-6, torch.zeros_like(sector), sector)

    def _topology_labels(self, occupancy: torch.Tensor, component_ids: torch.Tensor) -> torch.Tensor:
        occ_i = occupancy[:, None]
        occ_j = occupancy[None, :]
        comp_i = component_ids[:, None]
        comp_j = component_ids[None, :]
        labels = torch.full((occupancy.numel(), occupancy.numel()), self.TOPO_UNKNOWN, dtype=torch.long, device=occupancy.device)

        free_i = occ_i == OCCUPANCY_FREE
        free_j = occ_j == OCCUPANCY_FREE
        occupied_i = occ_i == OCCUPANCY_OCCUPIED
        occupied_j = occ_j == OCCUPANCY_OCCUPIED
        same_component = (comp_i >= 0) & (comp_i == comp_j)

        labels[free_i & free_j & same_component] = self.TOPO_SAME_FREE
        labels[free_i & free_j & ~same_component] = self.TOPO_DIFFERENT_FREE
        labels[occupied_i & occupied_j & same_component] = self.TOPO_SAME_OCCUPIED
        labels[free_i & occupied_j] = self.TOPO_FREE_TO_OCCUPIED
        labels[occupied_i & free_j] = self.TOPO_OCCUPIED_TO_FREE
        return labels

    def _relation_mask(self, topology_labels: torch.Tensor, has_occupancy: bool) -> torch.Tensor:
        mask = ~torch.eye(topology_labels.shape[0], dtype=torch.bool, device=topology_labels.device)
        if has_occupancy:
            mask = mask & (topology_labels != self.TOPO_UNKNOWN)
        return mask

    def _active_occupancy(
        self,
        occupancy_labels: torch.Tensor | list[int] | None,
        active_anchor_ids: torch.Tensor,
        active_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if occupancy_labels is None:
            occupancy = torch.full((active_count,), OCCUPANCY_UNKNOWN, dtype=torch.long, device=device)
            component_ids = torch.full_like(occupancy, -1)
            return occupancy, component_ids

        labels = torch.as_tensor(occupancy_labels, dtype=torch.long, device=device).reshape(-1)
        if labels.numel() == active_count:
            occupancy = labels
            component_ids = self._fallback_component_ids(occupancy)
            return occupancy, component_ids

        if self.grid is None:
            raise ValueError("Full occupancy_labels require a BEV grid to index active anchors")
        if labels.numel() != self.grid.num_anchors:
            raise ValueError(f"occupancy_labels must have {active_count} or {self.grid.num_anchors} elements")
        components = self._component_ids(labels)
        return labels[active_anchor_ids.to(device)], components[active_anchor_ids.to(device)]

    def _active_mask(
        self,
        mask: torch.Tensor | list[bool],
        active_anchor_ids: torch.Tensor,
        active_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        values = torch.as_tensor(mask, dtype=torch.bool, device=device).reshape(-1)
        if values.numel() == active_count:
            return values
        if self.grid is not None and values.numel() == self.grid.num_anchors:
            return values[active_anchor_ids.to(device)]
        raise ValueError(f"mask must have {active_count} elements or match the full BEV grid")

    def _component_ids(self, labels: torch.Tensor) -> torch.Tensor:
        if self.grid is None:
            return self._fallback_component_ids(labels)
        nx, ny = self.grid.grid_shape
        if labels.numel() != nx * ny:
            raise ValueError(f"occupancy_labels must match grid shape {nx}x{ny}")
        components = torch.full_like(labels, -1)
        next_component = 0
        for value in (OCCUPANCY_FREE, OCCUPANCY_OCCUPIED):
            for index, label in enumerate(labels.detach().cpu().tolist()):
                if label != value or int(components[index]) >= 0:
                    continue
                stack = [index]
                components[index] = next_component
                while stack:
                    current = stack.pop()
                    ix, iy = divmod(current, ny)
                    for nx_i, ny_i in ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)):
                        if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                            continue
                        neighbor = nx_i * ny + ny_i
                        if labels[neighbor] == value and int(components[neighbor]) < 0:
                            components[neighbor] = next_component
                            stack.append(neighbor)
                next_component += 1
        return components

    def _fallback_component_ids(self, occupancy: torch.Tensor) -> torch.Tensor:
        component_ids = torch.full_like(occupancy, -1)
        component_ids[occupancy == OCCUPANCY_FREE] = 0
        component_ids[occupancy == OCCUPANCY_OCCUPIED] = 1
        return component_ids

    def _occlusion_stub(self, active_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full(
            (active_count, active_count),
            self.OCCLUSION_IGNORE_INDEX,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros((active_count, active_count), dtype=torch.bool, device=device)
        return labels, mask


def build_relation_graph(**kwargs) -> dict[str, Any]:
    return RelationGraphBuilder(grid=kwargs.pop("grid", None), spec=kwargs.pop("spec", None)).build(**kwargs)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for relation graph building")
