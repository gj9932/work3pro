"""Sparse-relation losses (paper §3.10).

Each per-edge loss is reliability-weighted; the final L_rel is a configurable
linear combination. ``L_topo / L_occ`` are off by default in stage A0
(enabled when the ray-quality sanity check passes — paper §3.10).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


__all__ = ["RelationLossWeights", "RelationLossOutput", "relation_loss"]


@dataclass(frozen=True)
class RelationLossWeights:
    delta: float = 1.0
    order: float = 0.5
    cross: float = 0.5
    topo: float = 0.0
    occ: float = 0.0

    def total(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.delta * parts["L_delta"]
            + self.order * parts["L_order"]
            + self.cross * parts["L_cross"]
            + self.topo * parts["L_topo"]
            + self.occ * parts["L_occ"]
        )


@dataclass
class RelationLossOutput:
    L_rel: torch.Tensor
    parts: dict[str, torch.Tensor]
    n_used: dict[str, int]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    weights = weights.float()
    denom = weights.sum().clamp_min(eps)
    return (values * weights).sum() / denom


def relation_loss(
    pred: dict[str, torch.Tensor],            # output of RelationHeads.forward
    labels,                                    # EdgeLabels (cross_view_label.EdgeLabels)
    weights: RelationLossWeights = RelationLossWeights(),
    huber_beta: float = 1.0,
    occ_ignore: int = -1,
) -> RelationLossOutput:
    rel = labels.reliability.float() * labels.valid_mask.float()
    parts: dict[str, torch.Tensor] = {}
    n_used: dict[str, int] = {}

    # Δc: Huber regression
    if labels.delta_c.numel() > 0:
        elem = F.huber_loss(pred["delta_c"], labels.delta_c, reduction="none", delta=huber_beta).sum(dim=-1)
        parts["L_delta"] = _weighted_mean(elem, rel)
    else:
        parts["L_delta"] = pred["delta_c"].new_zeros(())
    n_used["L_delta"] = int((rel > 0).sum())

    # depth-order: 3-class CE (eq=0, q farther=1, p farther=2)
    if labels.depth_order.numel() > 0:
        ce = F.cross_entropy(pred["order_logits"], labels.depth_order.long(), reduction="none")
        parts["L_order"] = _weighted_mean(ce, rel)
    else:
        parts["L_order"] = pred["order_logits"].new_zeros(())
    n_used["L_order"] = int((rel > 0).sum())

    # cross-view: BCE only on cross-camera edges (others have label 0 by construction)
    if labels.cross_view.numel() > 0:
        cv_target = labels.cross_view.float()
        bce = F.binary_cross_entropy_with_logits(pred["cross_logit"], cv_target, reduction="none")
        parts["L_cross"] = _weighted_mean(bce, rel)
    else:
        parts["L_cross"] = pred["cross_logit"].new_zeros(())
    n_used["L_cross"] = int((rel > 0).sum())

    # topology: 6-class CE; mask out UNKNOWN(=5) in addition to reliability
    if labels.topology.numel() > 0:
        topo_w = rel * (labels.topology != 5).float()
        ce = F.cross_entropy(pred["topo_logits"], labels.topology.long(), reduction="none")
        parts["L_topo"] = _weighted_mean(ce, topo_w)
    else:
        parts["L_topo"] = pred["topo_logits"].new_zeros(())
    n_used["L_topo"] = int((rel * (labels.topology != 5).float() > 0).sum()) if labels.topology.numel() else 0

    # occlusion: 3-class CE; mask out same-camera-equal-depth (=2) for clearer signal
    if labels.occlusion.numel() > 0:
        occ_w = rel * (labels.occlusion != occ_ignore).float() if occ_ignore >= 0 else rel
        ce = F.cross_entropy(pred["occ_logits"], labels.occlusion.long(), reduction="none")
        parts["L_occ"] = _weighted_mean(ce, occ_w)
    else:
        parts["L_occ"] = pred["occ_logits"].new_zeros(())
    n_used["L_occ"] = int((rel > 0).sum())

    L_rel = weights.total(parts)
    return RelationLossOutput(L_rel=L_rel, parts=parts, n_used=n_used)
