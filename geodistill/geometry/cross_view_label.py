"""Cross-view / depth-order / topology / occlusion edge labels (paper §3.9).

For each (p, q) edge sampled by :mod:`geodistill.geometry.edge_sampler_v2`, this
builds:

    delta_c   (3,)     c_T[q] - c_T[p]                               (Huber target)
    depth_order int     {0: equal, 1: q farther, 2: p farther}       (CE 3-class)
    cross_view  bool    1 iff (a) same ego voxel + |Δd|<τ_depth, or
                              (b) same 3D object id + bidirectional camera visibility
    topology    int     6-class (paper §3.9): 0 same-free, 1 same-occupied,
                        2 free→occupied, 3 occupied→free, 4 different-free, 5 unknown
    occlusion   int     {0: p before q, 1: q before p, 2: same depth}
    reliability float   q_p^teacher · q_q^teacher · voxel_overlap · depth_consistency

Topology labels require a free/occupied/unknown classification per token; if the
caller does not provide one (the most common case from synthetic / nuScenes
without map labels), all topology values are set to UNKNOWN(=5) and the
reliability mask zeros them out.

Hard negatives (appearance-similar but 3D-far) are supplied by
``edge_sampler_v2.sample_edges`` as ``P_hard``; this builder does not regenerate
them — it just labels whatever (p, q) pairs come in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from geodistill.data.contract import TokenScene
from geodistill.teacher.base import TeacherOutput


__all__ = ["EdgeLabels", "build_edge_labels"]


TOPO_SAME_FREE = 0
TOPO_SAME_OCCUPIED = 1
TOPO_FREE_TO_OCCUPIED = 2
TOPO_OCCUPIED_TO_FREE = 3
TOPO_DIFFERENT_FREE = 4
TOPO_UNKNOWN = 5
NUM_TOPO_CLASSES = 6


@dataclass
class EdgeLabels:
    edge_index: torch.Tensor          # (E, 2) long
    delta_c: torch.Tensor             # (E, 3)
    depth_order: torch.Tensor         # (E,)
    cross_view: torch.Tensor          # (E,) {0,1}
    topology: torch.Tensor            # (E,) {0..5}
    occlusion: torch.Tensor           # (E,) {0,1,2}
    reliability: torch.Tensor         # (E,)
    valid_mask: torch.Tensor          # (E,) bool — for use in losses


def _voxel_key(c_ego: torch.Tensor, voxel: float) -> torch.Tensor:
    return torch.floor(c_ego / voxel).long()


def _depth_order(d_p: float, d_q: float, eps: float = 0.5) -> int:
    diff = d_q - d_p
    if abs(diff) < eps:
        return 0
    return 1 if diff > 0 else 2


def build_edge_labels(
    scene: TokenScene,
    teacher: TeacherOutput,
    edges: Sequence[tuple[int, int]],
    voxel_size: float = 0.5,
    tau_depth: float = 1.0,
    tau_eq_depth: float = 0.5,
    object_ids: torch.Tensor | None = None,    # (N,) long; -1 = no object
    occupancy_label: torch.Tensor | None = None,  # (N,) {free=0, occupied=1, unknown=-1}
) -> EdgeLabels:
    if not edges:
        zero = torch.zeros((0,), dtype=torch.long)
        return EdgeLabels(
            edge_index=torch.zeros((0, 2), dtype=torch.long),
            delta_c=torch.zeros((0, 3)), depth_order=zero, cross_view=zero,
            topology=zero, occlusion=zero, reliability=torch.zeros(0),
            valid_mask=torch.zeros(0, dtype=torch.bool),
        )

    edge_index = torch.tensor(list(edges), dtype=torch.long)
    p_idx = edge_index[:, 0]
    q_idx = edge_index[:, 1]
    valid = teacher.m_T & torch.isfinite(teacher.d_teacher)
    keep = valid[p_idx] & valid[q_idx]

    c = teacher.c_teacher
    d = teacher.d_teacher
    delta_c = c[q_idx] - c[p_idx]

    # depth order
    depth_order = torch.zeros(len(edges), dtype=torch.long)
    for i, (p, q) in enumerate(edges):
        if not bool(keep[i]):
            continue
        depth_order[i] = _depth_order(float(d[p]), float(d[q]), tau_eq_depth)

    # cross-view: voxel + depth, or shared object id with camera visibility
    cross_view = torch.zeros(len(edges), dtype=torch.long)
    keys = _voxel_key(c, voxel_size)
    has_obj = object_ids is not None
    for i, (p, q) in enumerate(edges):
        if not bool(keep[i]):
            continue
        if int(scene.token_camera_id[p]) == int(scene.token_camera_id[q]):
            # same camera: cross-view label only meaningful as ablation (mark 0)
            continue
        same_voxel = bool((keys[p] == keys[q]).all().item())
        depth_close = abs(float(d[p]) - float(d[q])) < tau_depth
        same_obj = bool(has_obj and int(object_ids[p]) >= 0 and int(object_ids[p]) == int(object_ids[q]))
        if (same_voxel and depth_close) or same_obj:
            cross_view[i] = 1

    # topology: requires occupancy_label (paper §3.9). Without labels => UNKNOWN.
    if occupancy_label is None:
        topology = torch.full((len(edges),), TOPO_UNKNOWN, dtype=torch.long)
    else:
        topology = torch.full((len(edges),), TOPO_UNKNOWN, dtype=torch.long)
        for i, (p, q) in enumerate(edges):
            if not bool(keep[i]):
                continue
            occ_p = int(occupancy_label[p])
            occ_q = int(occupancy_label[q])
            if occ_p < 0 or occ_q < 0:
                continue
            if occ_p == 0 and occ_q == 0:
                # same free vs different free: same camera -> assume same component;
                # cross camera with shared voxel -> same; otherwise different.
                same_voxel = bool((keys[p] == keys[q]).all().item())
                topology[i] = TOPO_SAME_FREE if same_voxel else TOPO_DIFFERENT_FREE
            elif occ_p == 1 and occ_q == 1:
                topology[i] = TOPO_SAME_OCCUPIED
            elif occ_p == 0 and occ_q == 1:
                topology[i] = TOPO_FREE_TO_OCCUPIED
            elif occ_p == 1 and occ_q == 0:
                topology[i] = TOPO_OCCUPIED_TO_FREE

    # occlusion: only meaningful within same camera; otherwise mark "same depth" (=2).
    # Only retain real labels when both tokens are in the same camera AND |Δd|>τ_eq.
    occlusion = torch.full((len(edges),), 2, dtype=torch.long)
    for i, (p, q) in enumerate(edges):
        if not bool(keep[i]):
            continue
        if int(scene.token_camera_id[p]) != int(scene.token_camera_id[q]):
            continue
        diff = float(d[q]) - float(d[p])
        if abs(diff) < tau_eq_depth:
            occlusion[i] = 2
        elif diff > 0:
            occlusion[i] = 1   # q is behind p -> p occludes q
        else:
            occlusion[i] = 0   # p is behind q -> q occludes p

    # reliability = q_p · q_q · voxel_overlap · depth_consistency
    qt = teacher.q_teacher
    voxel_overlap = (keys[p_idx] == keys[q_idx]).all(dim=-1).float()
    depth_consistency = torch.exp(-(d[q_idx] - d[p_idx]).abs() / max(tau_depth, 1e-6))
    reliability = qt[p_idx] * qt[q_idx] * (0.5 + 0.5 * voxel_overlap) * depth_consistency
    reliability = reliability * keep.float()

    valid_mask = keep
    return EdgeLabels(
        edge_index=edge_index, delta_c=delta_c, depth_order=depth_order,
        cross_view=cross_view, topology=topology, occlusion=occlusion,
        reliability=reliability, valid_mask=valid_mask,
    )
