from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # keep geometry importable before dependencies are installed
    torch = None


OCCUPANCY_FREE = 0
OCCUPANCY_OCCUPIED = 1
OCCUPANCY_UNKNOWN = -1


@dataclass(frozen=True)
class BEVGridConfig:
    x_range: tuple[float, float] = (-40.0, 40.0)
    y_range: tuple[float, float] = (-40.0, 40.0)
    cell_size: tuple[float, float] = (2.0, 2.0)
    active_anchor_count: int = 400


@dataclass(frozen=True)
class BEVAnchorSet:
    anchor_ids: Any
    centers: Any
    bounds: Any
    anchor_mask: Any


class BEVAnchorGrid:
    def __init__(self, config: BEVGridConfig | None = None):
        self.config = config or BEVGridConfig()
        self._centers_list: list[tuple[float, float]] | None = None
        self._bounds_list: list[tuple[float, float, float, float]] | None = None
        self._centers_tensor = None
        self._bounds_tensor = None

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        bev_cfg = config.get("bev", config)
        return cls(
            BEVGridConfig(
                x_range=tuple(bev_cfg.get("x_range", (-40.0, 40.0))),
                y_range=tuple(bev_cfg.get("y_range", (-40.0, 40.0))),
                cell_size=tuple(bev_cfg.get("cell_size", (2.0, 2.0))),
                active_anchor_count=int(bev_cfg.get("active_anchor_count", 400)),
            )
        )

    @property
    def num_anchors(self) -> int:
        return len(self._centers())

    @property
    def grid_shape(self) -> tuple[int, int]:
        cfg = self.config
        nx = int(round((cfg.x_range[1] - cfg.x_range[0]) / cfg.cell_size[0]))
        ny = int(round((cfg.y_range[1] - cfg.y_range[0]) / cfg.cell_size[1]))
        return nx, ny

    @property
    def centers(self):
        if torch is None:
            return self.build_centers()
        if self._centers_tensor is None:
            self._centers_tensor = torch.tensor(self._centers(), dtype=torch.float32)
        return self._centers_tensor

    @property
    def bounds(self):
        if torch is None:
            return self.build_bounds()
        if self._bounds_tensor is None:
            self._bounds_tensor = torch.tensor(self._bounds(), dtype=torch.float32)
        return self._bounds_tensor

    def build_centers(self) -> list[tuple[float, float]]:
        return list(self._centers())

    def build_bounds(self) -> list[tuple[float, float, float, float]]:
        return list(self._bounds())

    def build_candidate_anchors(
        self,
        camera_visible: Any = None,
        occupancy_labels: Any = None,
    ) -> dict[str, Any]:
        anchor_ids = list(range(self.num_anchors))
        result = {
            "anchors_candidate": self._as_centers_output(self._centers()),
            "anchor_ids": self._as_long_output(anchor_ids),
            "anchor_centers": self._as_centers_output(self._centers()),
            "anchor_bounds": self._as_bounds_output(self._bounds()),
            "anchor_mask": self._as_bool_output([True] * self.num_anchors),
        }
        if camera_visible is not None:
            result["camera_visible"] = self._as_bool_output(_as_bool_mask(camera_visible, self.num_anchors))
        if occupancy_labels is not None:
            result["occupancy_labels"] = self._as_long_output(_as_list(occupancy_labels))
        return result

    def select_active_anchors(
        self,
        mode: str = "train",
        active_anchor_count: int | None = None,
        occupancy_labels: Any = None,
        box_coverage_mask: Any = None,
        free_space_boundary_mask: Any = None,
        camera_visible: Any = None,
        anchor_prior: Any = None,
        full_anchors: bool = False,
    ) -> dict[str, Any]:
        count = self.num_anchors if full_anchors else int(active_anchor_count or self.config.active_anchor_count)
        if count <= 0:
            raise ValueError("active_anchor_count must be positive")
        count = min(count, self.num_anchors)

        if mode not in {"train", "inference"}:
            raise ValueError(f"Unsupported active anchor mode: {mode}")
        if mode == "inference" and (occupancy_labels is not None or box_coverage_mask is not None):
            raise ValueError("Inference active-anchor selection cannot use LiDAR or GT box labels")

        if full_anchors:
            active_ids = list(range(self.num_anchors))
        elif mode == "inference":
            active_ids = self._select_inference(count, camera_visible=camera_visible, anchor_prior=anchor_prior)
        else:
            active_ids = self._select_train(
                count,
                occupancy_labels=occupancy_labels,
                box_coverage_mask=box_coverage_mask,
                free_space_boundary_mask=free_space_boundary_mask,
                camera_visible=camera_visible,
            )

        active_mask = [False] * self.num_anchors
        for anchor_id in active_ids:
            active_mask[anchor_id] = True
        centers = self._centers()
        bounds = self._bounds()
        return {
            "active_anchor_ids": self._as_long_output(active_ids),
            "anchors_active": self._as_centers_output([centers[i] for i in active_ids]),
            "active_anchor_bounds": self._as_bounds_output([bounds[i] for i in active_ids]),
            "anchor_masks": self._as_bool_output(active_mask),
            "active_anchor_mask": self._as_bool_output([True] * len(active_ids)),
        }

    def _select_train(
        self,
        count: int,
        occupancy_labels: Any,
        box_coverage_mask: Any,
        free_space_boundary_mask: Any,
        camera_visible: Any,
    ) -> list[int]:
        masks = []
        if box_coverage_mask is not None:
            masks.append(_as_bool_mask(box_coverage_mask, self.num_anchors))
        if occupancy_labels is not None:
            labels = _as_list(occupancy_labels)
            _check_size(labels, self.num_anchors, "occupancy_labels")
            masks.append([label == OCCUPANCY_OCCUPIED for label in labels])
        if free_space_boundary_mask is not None:
            masks.append(_as_bool_mask(free_space_boundary_mask, self.num_anchors))
        if camera_visible is not None:
            masks.append(_as_bool_mask(camera_visible, self.num_anchors))
        if occupancy_labels is not None:
            labels = _as_list(occupancy_labels)
            masks.append([label == OCCUPANCY_FREE for label in labels])
        masks.append([True] * self.num_anchors)
        return _take_by_priority(masks, count)

    def _select_inference(self, count: int, camera_visible: Any, anchor_prior: Any) -> list[int]:
        if anchor_prior is not None:
            prior = [float(v) for v in _as_list(anchor_prior)]
            _check_size(prior, self.num_anchors, "anchor_prior")
            if camera_visible is not None:
                visible = _as_bool_mask(camera_visible, self.num_anchors)
                prior = [score if is_visible else float("-inf") for score, is_visible in zip(prior, visible)]
            ids = [i for i, score in sorted(enumerate(prior), key=lambda item: item[1], reverse=True) if score != float("-inf")]
            if len(ids) >= count:
                return sorted(ids[:count])
            return sorted(_take_by_priority([[True] * self.num_anchors], count, used=ids))
        masks = []
        if camera_visible is not None:
            masks.append(_as_bool_mask(camera_visible, self.num_anchors))
        masks.append([True] * self.num_anchors)
        return _take_by_priority(masks, count)

    def _centers(self) -> list[tuple[float, float]]:
        if self._centers_list is None:
            cfg = self.config
            xs = _range_centers(cfg.x_range[0], cfg.x_range[1], cfg.cell_size[0])
            ys = _range_centers(cfg.y_range[0], cfg.y_range[1], cfg.cell_size[1])
            self._centers_list = [(x, y) for x in xs for y in ys]
        return self._centers_list

    def _bounds(self) -> list[tuple[float, float, float, float]]:
        if self._bounds_list is None:
            half_x = self.config.cell_size[0] * 0.5
            half_y = self.config.cell_size[1] * 0.5
            self._bounds_list = [(x - half_x, y - half_y, x + half_x, y + half_y) for x, y in self._centers()]
        return self._bounds_list

    @staticmethod
    def _as_long_output(values: list[int]):
        return torch.tensor(values, dtype=torch.long) if torch is not None else values

    @staticmethod
    def _as_bool_output(values: list[bool]):
        return torch.tensor(values, dtype=torch.bool) if torch is not None else values

    @staticmethod
    def _as_centers_output(values: list[tuple[float, float]]):
        return torch.tensor(values, dtype=torch.float32) if torch is not None else values

    @staticmethod
    def _as_bounds_output(values: list[tuple[float, float, float, float]]):
        return torch.tensor(values, dtype=torch.float32) if torch is not None else values


def _range_centers(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    if count <= 0:
        raise ValueError(f"Invalid BEV range: start={start}, stop={stop}, step={step}")
    return [start + step * (i + 0.5) for i in range(count)]


def _take_by_priority(masks: list[list[bool]], count: int, used: list[int] | None = None) -> list[int]:
    selected = [] if used is None else list(dict.fromkeys(int(i) for i in used))
    selected_set = set(selected)
    for mask in masks:
        for anchor_id, enabled in enumerate(mask):
            if enabled and anchor_id not in selected_set:
                selected.append(anchor_id)
                selected_set.add(anchor_id)
                if len(selected) >= count:
                    return selected[:count]
    return selected[:count]


def _as_bool_mask(mask: Any, size: int) -> list[bool]:
    values = [bool(v) for v in _as_list(mask)]
    _check_size(values, size, "mask")
    return values


def _as_list(values: Any) -> list[Any]:
    if torch is not None and isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).tolist()
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        return values.reshape(-1).tolist()
    if isinstance(values, (list, tuple)):
        result = []
        for value in values:
            if isinstance(value, (list, tuple)):
                result.extend(value)
            else:
                result.append(value)
        return result
    return list(values)


def _check_size(values: list[Any], size: int, name: str) -> None:
    if len(values) != size:
        raise ValueError(f"{name} must have {size} elements, got {len(values)}")
