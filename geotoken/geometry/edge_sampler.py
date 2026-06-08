from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # keep geometry importable before dependencies are installed
    torch = None


@dataclass(frozen=True)
class EdgeSamplerConfig:
    edges_per_anchor: int = 32
    dense_edge_mode: bool = False
    near_ratio: float = 0.25
    topology_ratio: float = 0.25
    occlusion_ratio: float = 0.125
    hard_ratio: float = 0.125
    seed: int | None = None


class SparseEdgeSampler:
    EDGE_NEAR = 0
    EDGE_TOPOLOGY = 1
    EDGE_OCCLUSION = 2
    EDGE_HARD = 3
    EDGE_RANDOM = 4
    EDGE_DENSE = 5
    EDGE_TYPE_NAMES = ("near", "topology", "occlusion", "hard", "random", "dense")

    TOPO_SAME_FREE = 0
    TOPO_SAME_OCCUPIED = 1
    TOPO_DIFFERENT_FREE = 4
    TOPO_UNKNOWN = 5

    def __init__(self, config: EdgeSamplerConfig | None = None):
        _require_torch()
        self.config = config or EdgeSamplerConfig()
        if self.config.edges_per_anchor <= 0:
            raise ValueError("edges_per_anchor must be positive")

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        edge_cfg = config.get("edge_sampler", {})
        return cls(
            EdgeSamplerConfig(
                edges_per_anchor=int(edge_cfg.get("edges_per_anchor", 32)),
                dense_edge_mode=bool(edge_cfg.get("dense_edge_mode", False)),
                near_ratio=float(edge_cfg.get("near_ratio", 0.25)),
                topology_ratio=float(edge_cfg.get("topology_ratio", 0.25)),
                occlusion_ratio=float(edge_cfg.get("occlusion_ratio", 0.125)),
                hard_ratio=float(edge_cfg.get("hard_ratio", 0.125)),
                seed=edge_cfg.get("seed", None),
            )
        )

    def sample(
        self,
        relation_graph: dict[str, Any],
        edges_per_anchor: int | None = None,
        dense_edge_mode: bool | None = None,
        visual_prior: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        relation_mask = _as_bool_tensor(relation_graph.get("relation_mask", relation_graph.get("m_ij")))
        if relation_mask.ndim != 2 or relation_mask.shape[0] != relation_mask.shape[1]:
            raise ValueError(f"relation_mask must have shape [L, L], got {tuple(relation_mask.shape)}")

        distance = _as_tensor(relation_graph.get("distance"), device=relation_mask.device, dtype=torch.float32)
        if distance.shape != relation_mask.shape:
            raise ValueError("distance must have the same shape as relation_mask")

        dense = self.config.dense_edge_mode if dense_edge_mode is None else bool(dense_edge_mode)
        if dense:
            return self._dense_edges(relation_graph, relation_mask, distance)

        topology_labels = _optional_long_matrix(relation_graph.get("topology_labels"), relation_mask)
        occlusion_mask = _optional_bool_matrix(
            relation_graph.get("occlusion_mask", relation_graph.get("m_occ_ij")), relation_mask
        )
        visual_prior = self._resolve_visual_prior(visual_prior, relation_mask)
        per_anchor = int(edges_per_anchor or self.config.edges_per_anchor)
        quotas = self._quotas(per_anchor)
        generator = self._generator(relation_mask.device)

        edge_pairs: list[torch.Tensor] = []
        edge_types: list[torch.Tensor] = []
        for anchor_id in range(relation_mask.shape[0]):
            selected = torch.zeros(relation_mask.shape[1], dtype=torch.bool, device=relation_mask.device)
            selected_count = 0
            pairs_i: list[torch.Tensor] = []
            types_i: list[torch.Tensor] = []

            def take(candidates: torch.Tensor, count: int, edge_type: int, scores: torch.Tensor | None = None) -> None:
                nonlocal selected_count
                if count <= 0 or selected_count >= per_anchor:
                    return
                remaining = per_anchor - selected_count
                count_i = min(count, remaining)
                candidates_i = candidates & relation_mask[anchor_id] & ~selected
                candidate_ids = candidates_i.nonzero(as_tuple=False).reshape(-1)
                if candidate_ids.numel() == 0:
                    return
                if scores is None:
                    ordered = candidate_ids
                else:
                    _, order = torch.sort(scores[candidate_ids], descending=True, stable=True)
                    ordered = candidate_ids[order]
                chosen = ordered[:count_i]
                selected[chosen] = True
                selected_count += int(chosen.numel())
                pairs_i.append(torch.stack((torch.full_like(chosen, anchor_id), chosen), dim=1))
                types_i.append(torch.full((chosen.numel(),), edge_type, dtype=torch.long, device=relation_mask.device))

            valid = relation_mask[anchor_id]
            take(valid, quotas[self.EDGE_NEAR], self.EDGE_NEAR, scores=-distance[anchor_id])
            take(
                self._topology_candidates(topology_labels, anchor_id, relation_mask),
                quotas[self.EDGE_TOPOLOGY],
                self.EDGE_TOPOLOGY,
            )
            take(occlusion_mask[anchor_id], quotas[self.EDGE_OCCLUSION], self.EDGE_OCCLUSION)
            take(
                self._hard_candidates(topology_labels, anchor_id, visual_prior, relation_mask),
                quotas[self.EDGE_HARD],
                self.EDGE_HARD,
                scores=self._hard_scores(distance, visual_prior, anchor_id),
            )
            take(
                self._random_candidates(distance, relation_mask, anchor_id),
                per_anchor,
                self.EDGE_RANDOM,
                scores=self._random_scores(relation_mask.shape[1], relation_mask.device, generator),
            )
            take(valid, per_anchor, self.EDGE_RANDOM, scores=self._random_scores(relation_mask.shape[1], relation_mask.device, generator))

            if pairs_i:
                edge_pairs.append(torch.cat(pairs_i, dim=0))
                edge_types.append(torch.cat(types_i, dim=0))

        if edge_pairs:
            edge_index = torch.cat(edge_pairs, dim=0).to(torch.long)
            edge_type = torch.cat(edge_types, dim=0).to(torch.long)
        else:
            edge_index = torch.empty((0, 2), dtype=torch.long, device=relation_mask.device)
            edge_type = torch.empty((0,), dtype=torch.long, device=relation_mask.device)
        return self._pack_edges(relation_graph, edge_index, edge_type, relation_mask, distance)

    def _dense_edges(
        self,
        relation_graph: dict[str, Any],
        relation_mask: torch.Tensor,
        distance: torch.Tensor,
    ) -> dict[str, Any]:
        edge_index = relation_mask.nonzero(as_tuple=False).to(torch.long)
        edge_type = torch.full((edge_index.shape[0],), self.EDGE_DENSE, dtype=torch.long, device=relation_mask.device)
        return self._pack_edges(relation_graph, edge_index, edge_type, relation_mask, distance)

    def _pack_edges(
        self,
        relation_graph: dict[str, Any],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        relation_mask: torch.Tensor,
        distance: torch.Tensor,
    ) -> dict[str, Any]:
        result = {
            "edge_index": edge_index,
            "edge_type": edge_type,
            "edge_type_names": list(self.EDGE_TYPE_NAMES),
            "edge_masks": self._gather_matrix(relation_mask, edge_index, dtype=torch.bool),
            "edge_distance": self._gather_matrix(distance, edge_index, dtype=torch.float32),
        }
        for source_key, target_key in (
            ("distance_labels", "distance_labels"),
            ("direction_labels", "direction_labels"),
            ("topology_labels", "topology_labels"),
            ("occlusion_labels", "occlusion_labels"),
            ("occlusion_mask", "occlusion_masks"),
            ("m_occ_ij", "m_occ_ij"),
        ):
            if source_key in relation_graph:
                result[target_key] = self._gather_matrix(relation_graph[source_key], edge_index)
        result["edge_labels"] = {
            key: result[key]
            for key in ("distance_labels", "direction_labels", "topology_labels", "occlusion_labels")
            if key in result
        }
        result["num_edges_per_anchor_max"] = self.config.edges_per_anchor
        result["edge_count"] = int(edge_index.shape[0])
        return result

    def _gather_matrix(self, values: Any, edge_index: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor = _as_tensor(values, device=edge_index.device, dtype=dtype)
        if edge_index.numel() == 0:
            return torch.empty((0,), dtype=tensor.dtype, device=edge_index.device)
        return tensor[edge_index[:, 0], edge_index[:, 1]]

    def _quotas(self, edges_per_anchor: int) -> dict[int, int]:
        fixed = {
            self.EDGE_NEAR: int(round(edges_per_anchor * self.config.near_ratio)),
            self.EDGE_TOPOLOGY: int(round(edges_per_anchor * self.config.topology_ratio)),
            self.EDGE_OCCLUSION: int(round(edges_per_anchor * self.config.occlusion_ratio)),
            self.EDGE_HARD: int(round(edges_per_anchor * self.config.hard_ratio)),
        }
        used = sum(max(0, count) for count in fixed.values())
        fixed[self.EDGE_RANDOM] = max(0, edges_per_anchor - used)
        while sum(fixed.values()) > edges_per_anchor:
            for edge_type in (self.EDGE_RANDOM, self.EDGE_TOPOLOGY, self.EDGE_NEAR, self.EDGE_HARD, self.EDGE_OCCLUSION):
                if fixed[edge_type] > 0 and sum(fixed.values()) > edges_per_anchor:
                    fixed[edge_type] -= 1
        return fixed

    def _topology_candidates(
        self,
        topology_labels: torch.Tensor | None,
        anchor_id: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if topology_labels is None:
            return torch.zeros((reference.shape[1],), dtype=torch.bool, device=reference.device)
        return (topology_labels[anchor_id] == self.TOPO_SAME_FREE) | (
            topology_labels[anchor_id] == self.TOPO_SAME_OCCUPIED
        )

    def _hard_candidates(
        self,
        topology_labels: torch.Tensor | None,
        anchor_id: int,
        visual_prior: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if topology_labels is None:
            if visual_prior is None:
                return torch.zeros((reference.shape[1],), dtype=torch.bool, device=reference.device)
            return torch.ones((visual_prior.shape[-1],), dtype=torch.bool, device=visual_prior.device)
        labels = topology_labels[anchor_id]
        return (labels != self.TOPO_UNKNOWN) & (labels != self.TOPO_SAME_FREE) & (labels != self.TOPO_SAME_OCCUPIED)

    def _hard_scores(self, distance: torch.Tensor, visual_prior: torch.Tensor | None, anchor_id: int) -> torch.Tensor:
        if visual_prior is not None:
            return visual_prior[anchor_id]
        return -distance[anchor_id]

    def _random_candidates(self, distance: torch.Tensor, relation_mask: torch.Tensor, anchor_id: int) -> torch.Tensor:
        valid_distance = distance[anchor_id][relation_mask[anchor_id]]
        if valid_distance.numel() == 0:
            return torch.zeros_like(relation_mask[anchor_id])
        threshold = torch.quantile(valid_distance, 0.5)
        return distance[anchor_id] >= threshold

    def _random_scores(
        self,
        count: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if generator is not None and device.type == "cpu":
            return torch.rand((count,), device=device, generator=generator)
        return torch.rand((count,), device=device)

    def _resolve_visual_prior(self, visual_prior: torch.Tensor | None, relation_mask: torch.Tensor) -> torch.Tensor | None:
        if visual_prior is None:
            return None
        prior = _as_tensor(visual_prior, device=relation_mask.device, dtype=torch.float32)
        if prior.ndim == 2 and prior.shape == relation_mask.shape:
            return prior
        if prior.ndim == 2 and prior.shape[0] == relation_mask.shape[0]:
            prior = torch.nn.functional.normalize(prior, dim=-1)
            return prior @ prior.T
        raise ValueError("visual_prior must have shape [L, L] or [L, C]")

    def _generator(self, device: torch.device) -> torch.Generator | None:
        if self.config.seed is None:
            return None
        if device.type != "cpu":
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self.config.seed))
        return generator


def sample_sparse_edges(relation_graph: dict[str, Any], **kwargs) -> dict[str, Any]:
    return SparseEdgeSampler(config=kwargs.pop("config", None)).sample(relation_graph, **kwargs)


def _as_tensor(values: Any, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    if values is None:
        raise ValueError("Expected tensor-like value, got None")
    if isinstance(values, torch.Tensor):
        tensor = values.to(device=device) if device is not None else values
        return tensor.to(dtype=dtype) if dtype is not None else tensor
    return torch.as_tensor(values, device=device, dtype=dtype)


def _as_bool_tensor(values: Any) -> torch.Tensor:
    return _as_tensor(values, dtype=torch.bool)


def _optional_long_matrix(values: Any, reference: torch.Tensor) -> torch.Tensor | None:
    if values is None:
        return None
    tensor = _as_tensor(values, device=reference.device, dtype=torch.long)
    if tensor.shape != reference.shape:
        raise ValueError("label matrix must match relation_mask shape")
    return tensor


def _optional_bool_matrix(values: Any, reference: torch.Tensor) -> torch.Tensor:
    if values is None:
        return torch.zeros_like(reference, dtype=torch.bool)
    tensor = _as_tensor(values, device=reference.device, dtype=torch.bool)
    if tensor.shape != reference.shape:
        raise ValueError("mask matrix must match relation_mask shape")
    return tensor


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for sparse edge sampling")
