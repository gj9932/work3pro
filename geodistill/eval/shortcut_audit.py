"""Shortcut audit (paper §4.7).

Six (+ one) controlled shuffles. The goal is to verify that the model's
benefit really comes from the supervision the paper claims, not from a
dataset / coordinate / camera-id prior. If a shuffle does NOT degrade the
metric the corresponding contribution is suspect.

Each shuffle is a TokenScene/Teacher transformation:

    1. ``image_embeds``         shuffle scene.features within the batch
    2. ``calibration``          swap K + cam_to_ego across cameras
    3. ``depth_label``          shuffle teacher d_T within the batch
    4. ``allocation_label``     shuffle one of (r_p, q_geom, q_OT, q_teacher, a*)
    5. ``transport_structure``  shuffle the LiDAR typed-relation rows in the FGT
                                 structure cost (covered by an upstream hook in
                                 ``geodistill.teacher.ot_common`` — pass the
                                 ``structure_shuffle`` argument when building the
                                 OT teacher)
    6. ``relation_label``       shuffle (cross_view, depth_order, topology, occ)
                                 in EdgeLabels
    7. ``constant_depth``       overwrite teacher.d_teacher with a single value

The implementation is dependency-light: every shuffle returns a *new* TokenScene
or TeacherOutput so the original objects remain immutable. Tests assert (a)
the shuffle leaves untouched fields bit-exact equal and (b) the touched fields
become detectably different.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import torch

from geodistill.data.contract import TokenScene
from geodistill.geometry.cross_view_label import EdgeLabels
from geodistill.teacher.base import TeacherOutput


__all__ = [
    "SHUFFLE_KINDS",
    "ALLOCATION_KEYS",
    "ShuffleConfig",
    "apply_shuffle",
    "shuffle_image_embeds",
    "shuffle_calibration",
    "shuffle_depth_label",
    "shuffle_allocation_label",
    "shuffle_relation_label",
    "constant_depth",
    "shortcut_audit_step",
]


SHUFFLE_KINDS = (
    "image_embeds",
    "calibration",
    "depth_label",
    "allocation_label",
    "transport_structure",
    "relation_label",
    "constant_depth",
)
ALLOCATION_KEYS = ("r_p", "q_geom", "q_OT", "q_teacher", "a_star")


@dataclass
class ShuffleConfig:
    kind: str = "image_embeds"
    seed: int = 0
    allocation_key: str = "r_p"             # only for kind="allocation_label"
    constant_depth_value: float = 30.0


# ----------------------------------------------------------------------- helpers
def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _perm(n: int, seed: int) -> torch.Tensor:
    return torch.randperm(n, generator=_gen(seed))


# ----------------------------------------------------------------------- shuffles
def shuffle_image_embeds(scene: TokenScene, seed: int = 0) -> TokenScene:
    """Shuffle scene.features within the batch; LiDAR / calib / GT untouched."""
    new = copy.copy(scene)
    perm = _perm(scene.num_tokens, seed)
    new.features = scene.features[perm].clone()
    return new


def shuffle_calibration(scene: TokenScene, seed: int = 0) -> TokenScene:
    """Permute the camera order in K, cam_to_ego, image_hw.

    Tokens still claim ``token_camera_id == c``, but K[c] / cam_to_ego[c] now
    point at a different camera, so feature_cost / barycenter / cross-view
    supervision should degrade.
    """
    new = copy.copy(scene)
    Ncam = int(scene.num_cameras)
    if Ncam < 2:
        return new
    cam_perm = _perm(Ncam, seed)
    new.K = scene.K[cam_perm].clone()
    new.cam_to_ego = scene.cam_to_ego[cam_perm].clone()
    new.image_hw = scene.image_hw[cam_perm].clone()
    return new


def shuffle_depth_label(teacher: TeacherOutput, seed: int = 0) -> TeacherOutput:
    new = copy.copy(teacher)
    n = teacher.d_teacher.numel()
    perm = _perm(n, seed)
    new.d_teacher = teacher.d_teacher[perm].clone()
    new.c_teacher = teacher.c_teacher[perm].clone()
    return new


def shuffle_allocation_label(
    teacher: TeacherOutput,
    r_p: torch.Tensor | None = None,
    q_geom: torch.Tensor | None = None,
    q_OT: torch.Tensor | None = None,
    a_star: torch.Tensor | None = None,
    key: str = "r_p",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Return a dict with the permuted version of the requested allocation field.

    The teacher object itself is NOT mutated — callers can fold the permuted
    field into the loss assembly.
    """
    if key not in ALLOCATION_KEYS:
        raise ValueError(f"unknown allocation key {key!r}; choose from {ALLOCATION_KEYS}")
    n = int(teacher.q_teacher.numel())
    perm = _perm(n, seed)
    out = {}
    if key == "r_p" and r_p is not None:
        out["r_p"] = r_p[perm].clone()
    elif key == "q_geom" and q_geom is not None:
        out["q_geom"] = q_geom[perm].clone()
    elif key == "q_OT" and q_OT is not None:
        out["q_OT"] = q_OT[perm].clone()
    elif key == "q_teacher":
        out["q_teacher"] = teacher.q_teacher[perm].clone()
    elif key == "a_star" and a_star is not None:
        out["a_star"] = a_star[perm].clone()
    else:
        raise ValueError(f"caller must pass tensor matching key={key!r}")
    return out


def shuffle_relation_label(labels: EdgeLabels, seed: int = 0) -> EdgeLabels:
    """Shuffle (cross_view, depth_order, topology, occ) jointly across edges.

    Δc and reliability remain untouched so the shuffle isolates the relation
    label channel.
    """
    n = int(labels.edge_index.shape[0])
    if n == 0:
        return labels
    perm = _perm(n, seed)
    return EdgeLabels(
        edge_index=labels.edge_index,
        delta_c=labels.delta_c,
        depth_order=labels.depth_order[perm].clone(),
        cross_view=labels.cross_view[perm].clone(),
        topology=labels.topology[perm].clone(),
        occlusion=labels.occlusion[perm].clone(),
        reliability=labels.reliability,
        valid_mask=labels.valid_mask,
    )


def constant_depth(teacher: TeacherOutput, value: float = 30.0) -> TeacherOutput:
    """Replace teacher depth with a single constant — checks that the model
    is not just consuming a coarse frustum prior (paper §4.7)."""
    new = copy.copy(teacher)
    new.d_teacher = torch.full_like(teacher.d_teacher, value)
    # Project a fake constant 3D coordinate along the original ray direction.
    # We simply scale the original c_teacher to the new depth ratio when finite.
    finite = torch.isfinite(teacher.d_teacher) & (teacher.d_teacher.abs() > 1e-3)
    scale = torch.where(finite, value / teacher.d_teacher, torch.zeros_like(teacher.d_teacher))
    new.c_teacher = teacher.c_teacher * scale.unsqueeze(-1)
    return new


# ----------------------------------------------------------------------- master
def apply_shuffle(scene: TokenScene, teacher: TeacherOutput,
                  cfg: ShuffleConfig) -> tuple[TokenScene, TeacherOutput, dict]:
    """Apply ``cfg.kind`` and return the modified (scene, teacher) pair plus a meta dict.

    For ``allocation_label``, the modification is exposed via the meta dict
    rather than mutating the teacher (the caller folds it into loss assembly).
    For ``transport_structure``, this function is a no-op and the meta dict
    instructs the OT teacher builder to enable its ``structure_shuffle`` flag
    on the next teacher build (covered by tests in
    ``test/geodistill/shortcut_test.py``).
    For ``relation_label``, the modification is exposed via the meta dict
    keyed ``shuffle_relation_label`` and the caller threads it through
    ``build_edge_labels`` post-hoc.
    """
    kind = cfg.kind
    meta: dict = {"kind": kind, "seed": cfg.seed}
    if kind == "image_embeds":
        return shuffle_image_embeds(scene, cfg.seed), teacher, meta
    if kind == "calibration":
        return shuffle_calibration(scene, cfg.seed), teacher, meta
    if kind == "depth_label":
        return scene, shuffle_depth_label(teacher, cfg.seed), meta
    if kind == "allocation_label":
        meta["allocation_key"] = cfg.allocation_key
        meta["allocation_perm"] = _perm(int(teacher.q_teacher.numel()), cfg.seed)
        return scene, teacher, meta
    if kind == "transport_structure":
        meta["structure_shuffle"] = True
        return scene, teacher, meta
    if kind == "relation_label":
        meta["shuffle_relation_label"] = True
        return scene, teacher, meta
    if kind == "constant_depth":
        return scene, constant_depth(teacher, cfg.constant_depth_value), meta
    raise ValueError(f"unknown shuffle kind {kind!r}")


# ----------------------------------------------------------------------- audit
def shortcut_audit_step(
    eval_fn,                             # callable: (scene, teacher, meta) -> dict of metrics
    scene: TokenScene,
    teacher: TeacherOutput,
    kinds: Iterable[str] = SHUFFLE_KINDS,
    seed: int = 0,
) -> dict[str, dict]:
    """Run ``eval_fn`` once on the clean batch and once per requested shuffle.

    Returns ``{kind: metrics}`` plus a special ``"clean"`` key for the baseline.
    Callers (CLI / tests) compare each kind to ``clean`` and assert degradation.
    """
    out: dict[str, dict] = {"clean": eval_fn(scene, teacher, {"kind": "clean"})}
    for k in kinds:
        new_scene, new_teacher, meta = apply_shuffle(scene, teacher, ShuffleConfig(kind=k, seed=seed))
        out[k] = eval_fn(new_scene, new_teacher, meta)
    return out
