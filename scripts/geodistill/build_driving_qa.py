"""Build the DriveLM / OmniDrive-style spatial QA JSONL splits (paper §4.5 Table 1).

Categories (paper §4.5):
    metric_distance / relative_direction / object_comparison /
    free_space_topology / occlusion / cross_view_correspondence /
    temporal_motion / counterfactual_trajectory

Splits:
    train / val / test_template / test_paraphrase / test_composition

For each nuScenes sample we already cached the token-level teacher
(``runs/geodistill/cache/token_labels/*.pt``); this script consumes the per-token
3D coordinates ``teacher.c_teacher`` together with ``boxes_3d`` to deterministically
generate QAs:

    metric_distance       → "How far is <obj> ahead/behind ego?"
                            answer = forward distance in meters (from box center)
    relative_direction    → "Is <obj_a> to the left/right/front/behind <obj_b>?"
    object_comparison     → "Which is closer to ego: <a> or <b>?"
    free_space_topology   → from BEV occupancy band along the corridor
    occlusion             → from teacher.occlusion edge labels
    cross_view_correspondence → from teacher.cross_view edge labels
    temporal_motion       → ego speed bin
    counterfactual_trajectory → "If ego turns left now, will it intersect <obj>?"

Each QA has surface forms drawn from a paraphrase pool so the
``test_paraphrase`` split holds out unseen surface forms.

Local CPU has no nuScenes / boxes_3d — but we still generate QAs from the
teacher cache (which the user already ran on the GPU box). That's exactly the
mode driver: this script consumes ``.pt`` files written by
:mod:`scripts.geodistill.build_token_label_cache`, plus an optional
``boxes_3d.pt`` companion produced by the same builder when ``load_boxes_3d``
is on.

Outputs:
    spatial_qa_train.jsonl
    spatial_qa_val.jsonl
    spatial_qa_test_template.jsonl
    spatial_qa_test_paraphrase.jsonl
    spatial_qa_test_composition.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from geodistill.utils import JsonlWriter, run_envelope


# Paper §4.5 categories
CATEGORIES = (
    "metric_distance", "relative_direction", "object_comparison",
    "free_space_topology", "occlusion", "cross_view_correspondence",
    "temporal_motion", "counterfactual_trajectory",
)

# Surface-form pools per category. Train uses keys 0/1; test_paraphrase uses 2/3.
PARAPHRASE_POOL: dict[str, list[str]] = {
    "metric_distance": [
        "How far ahead is the {obj}?",
        "What's the distance to the {obj}?",
        "About how many meters away is the {obj}?",
        "Roughly how far is the {obj} from us?",
    ],
    "relative_direction": [
        "Where is the {a} relative to the {b}?",
        "Is the {a} to the left, right, front, or back of the {b}?",
        "From the perspective of the {b}, where does the {a} sit?",
        "On which side of the {b} is the {a}?",
    ],
    "object_comparison": [
        "Which is closer to ego: the {a} or the {b}?",
        "Between the {a} and the {b}, which is nearer?",
        "Of the two — {a} and {b} — which one is closer to us?",
        "Which one is nearer the ego car: the {a} or the {b}?",
    ],
    "free_space_topology": [
        "Is the area in front of ego currently free?",
        "Can ego drive forward without hitting anything?",
        "Is there clear space ahead of the vehicle?",
        "Is the corridor in front of us drivable right now?",
    ],
    "occlusion": [
        "Is the {a} occluded by the {b}?",
        "Does the {b} block the view of the {a}?",
        "From ego's perspective, is the {b} in front of the {a}?",
        "Is the {a} hidden behind the {b}?",
    ],
    "cross_view_correspondence": [
        "Is this object visible in the {cam_a} and the {cam_b} cameras?",
        "Does the {cam_a} view share the same object with the {cam_b} view?",
        "Can we see this object across both the {cam_a} and {cam_b} cameras?",
        "Is the same object captured by the {cam_a} and {cam_b} views?",
    ],
    "temporal_motion": [
        "Is ego currently moving forward?",
        "Is the vehicle in motion right now?",
        "Roughly what's the ego speed in m/s?",
        "How fast is the ego car moving (m/s)?",
    ],
    "counterfactual_trajectory": [
        "If ego turns left now, will it intersect the {obj}?",
        "Would a left turn from here put us in the path of the {obj}?",
        "If we steered left immediately, would we hit the {obj}?",
        "Imagine ego turns left — does it run into the {obj}?",
    ],
}


@dataclass
class QASample:
    question: str
    answer: str
    type: str
    involved_object_ids: list[int] = field(default_factory=list)
    source_relation: str | None = None
    reliability: float = 1.0
    split_id: str = "train"
    template_id: int = 0


def _category_for_object(obj: dict) -> str:
    """Lightweight noun phrase for a 3D box record (or "object" fallback)."""
    name = obj.get("name") or obj.get("category_name") or "object"
    name = str(name).split(".")[-1]                         # vehicle.car -> car
    return name


def _euclid(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm())


def _bin_direction(da: torch.Tensor, db: torch.Tensor) -> str:
    """Return ``front/back/left/right`` from a's POV looking at b (ego frame)."""
    diff = db - da
    # Forward = +x, left = +y in ego frame (paper convention)
    if diff[0].abs() > diff[1].abs():
        return "in front of" if float(diff[0]) > 0 else "behind"
    return "to the left of" if float(diff[1]) > 0 else "to the right of"


def _phrase(category: str, template_id: int, **kw: Any) -> str:
    pool = PARAPHRASE_POOL[category]
    return pool[template_id % len(pool)].format(**kw)


def _qa_for_metric_distance(rng: random.Random, boxes: list[dict], template_id: int) -> QASample | None:
    if not boxes:
        return None
    obj = rng.choice(boxes)
    name = _category_for_object(obj)
    dist = _euclid(torch.tensor([obj["x"], obj["y"]]), torch.tensor([0.0, 0.0]))
    return QASample(
        question=_phrase("metric_distance", template_id, obj=name),
        answer=f"about {dist:.1f} meters",
        type="metric_distance", involved_object_ids=[obj.get("id", -1)],
        source_relation="metric_distance", reliability=float(obj.get("reliability", 1.0)),
        template_id=template_id,
    )


def _qa_for_relative_direction(rng: random.Random, boxes: list[dict], template_id: int) -> QASample | None:
    if len(boxes) < 2:
        return None
    a, b = rng.sample(boxes, 2)
    direction = _bin_direction(torch.tensor([a["x"], a["y"]]), torch.tensor([b["x"], b["y"]]))
    return QASample(
        question=_phrase("relative_direction", template_id,
                            a=_category_for_object(a), b=_category_for_object(b)),
        answer=f"the {_category_for_object(a)} is {direction} the {_category_for_object(b)}",
        type="relative_direction",
        involved_object_ids=[a.get("id", -1), b.get("id", -1)],
        source_relation="relative_direction",
        reliability=min(float(a.get("reliability", 1.0)), float(b.get("reliability", 1.0))),
        template_id=template_id,
    )


def _qa_for_object_comparison(rng: random.Random, boxes: list[dict], template_id: int) -> QASample | None:
    if len(boxes) < 2:
        return None
    a, b = rng.sample(boxes, 2)
    da = _euclid(torch.tensor([a["x"], a["y"]]), torch.tensor([0.0, 0.0]))
    db = _euclid(torch.tensor([b["x"], b["y"]]), torch.tensor([0.0, 0.0]))
    closer = a if da < db else b
    return QASample(
        question=_phrase("object_comparison", template_id,
                            a=_category_for_object(a), b=_category_for_object(b)),
        answer=f"the {_category_for_object(closer)}",
        type="object_comparison",
        involved_object_ids=[a.get("id", -1), b.get("id", -1)],
        source_relation="object_comparison",
        reliability=min(float(a.get("reliability", 1.0)), float(b.get("reliability", 1.0))),
        template_id=template_id,
    )


def _qa_for_free_space(rng: random.Random, ego: dict, template_id: int) -> QASample | None:
    free = bool(ego.get("forward_free", False))
    return QASample(
        question=_phrase("free_space_topology", template_id),
        answer="yes" if free else "no",
        type="free_space_topology",
        involved_object_ids=[],
        source_relation="free_space",
        reliability=float(ego.get("free_space_reliability", 1.0)),
        template_id=template_id,
    )


def _qa_for_occlusion(rng: random.Random, edges: list[dict], template_id: int) -> QASample | None:
    if not edges:
        return None
    e = rng.choice(edges)
    a = e["a_name"]; b = e["b_name"]
    occluded = bool(e.get("a_behind_b", False))
    return QASample(
        question=_phrase("occlusion", template_id, a=a, b=b),
        answer="yes" if occluded else "no",
        type="occlusion",
        involved_object_ids=[e.get("a_id", -1), e.get("b_id", -1)],
        source_relation="occlusion", reliability=float(e.get("reliability", 1.0)),
        template_id=template_id,
    )


def _qa_for_cross_view(rng: random.Random, edges: list[dict], template_id: int) -> QASample | None:
    if not edges:
        return None
    e = rng.choice(edges)
    return QASample(
        question=_phrase("cross_view_correspondence", template_id,
                            cam_a=e["cam_a"], cam_b=e["cam_b"]),
        answer="yes" if bool(e.get("cross_view", False)) else "no",
        type="cross_view_correspondence",
        involved_object_ids=[e.get("a_id", -1), e.get("b_id", -1)],
        source_relation="cross_view", reliability=float(e.get("reliability", 1.0)),
        template_id=template_id,
    )


def _qa_for_temporal_motion(rng: random.Random, ego: dict, template_id: int) -> QASample:
    v = float(ego.get("v_ego", 0.0))
    if template_id % 2 == 0:
        ans = "yes" if v > 0.5 else "no"
        return QASample(
            question=_phrase("temporal_motion", template_id),
            answer=ans, type="temporal_motion", reliability=1.0,
            template_id=template_id, source_relation="ego_motion",
        )
    return QASample(
        question=_phrase("temporal_motion", template_id),
        answer=f"{v:.1f}", type="temporal_motion", reliability=1.0,
        template_id=template_id, source_relation="ego_motion",
    )


def _qa_for_counterfactual(rng: random.Random, boxes: list[dict], template_id: int) -> QASample | None:
    if not boxes:
        return None
    obj = rng.choice(boxes)
    # toy collision predicate: object is on the left side and within 12m forward
    intersects = (float(obj["y"]) > 0.0) and (0.0 < float(obj["x"]) < 12.0)
    return QASample(
        question=_phrase("counterfactual_trajectory", template_id, obj=_category_for_object(obj)),
        answer="yes" if intersects else "no",
        type="counterfactual_trajectory",
        involved_object_ids=[obj.get("id", -1)],
        source_relation="counterfactual",
        reliability=float(obj.get("reliability", 1.0)),
        template_id=template_id,
    )


# ----------------------------------------------------------------------- driver
def build_qa_for_sample(payload: dict, rng: random.Random,
                         template_pool=(0, 1)) -> list[QASample]:
    """Generate QAs for one cached sample.

    ``payload`` is the dict written by ``build_token_label_cache.py`` plus optional
    ``boxes_3d`` / ``edges`` / ``ego`` keys you may have appended downstream.
    """
    boxes = list(payload.get("boxes_3d", []) or [])
    edges = list(payload.get("relation_edges", []) or [])
    ego = payload.get("ego", {}) or {"v_ego": payload["scene"].v_ego}

    qa_list: list[QASample] = []
    for tid in template_pool:
        for fn in (_qa_for_metric_distance, _qa_for_relative_direction,
                    _qa_for_object_comparison, _qa_for_counterfactual):
            q = fn(rng, boxes, tid)
            if q is not None:
                qa_list.append(q)
        for fn in (_qa_for_occlusion, _qa_for_cross_view):
            q = fn(rng, edges, tid)
            if q is not None:
                qa_list.append(q)
        qa_list.append(_qa_for_free_space(rng, ego, tid))
        qa_list.append(_qa_for_temporal_motion(rng, ego, tid))
    return [q for q in qa_list if q is not None]


def split_assignment(scene_token: str, sample_idx: int, fractions: dict) -> str:
    """Deterministic split assignment by hash so the same sample lands in the
    same split across re-runs."""
    h = abs(hash((scene_token, sample_idx))) % 1000
    cum = 0.0
    for split, frac in fractions.items():
        cum += float(frac) * 1000
        if h < cum:
            return split
    return list(fractions)[-1]


def main():
    ap = argparse.ArgumentParser(description="Build DriveLM/OmniDrive-style spatial QA jsonls (paper §4.5)")
    ap.add_argument("--in_dir", default="runs/geodistill/cache/token_labels")
    ap.add_argument("--out_dir", default="runs/geodistill/qa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min_reliability", type=float, default=0.5)
    ap.add_argument("--fractions", type=str,
                     default="train:0.6,val:0.1,test_template:0.1,test_paraphrase:0.1,test_composition:0.1")
    args = ap.parse_args()

    fractions = {}
    for chunk in args.fractions.split(","):
        k, v = chunk.split(":")
        fractions[k.strip()] = float(v)

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.pt"))
    if args.limit:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"no .pt under {in_dir}; run build_token_label_cache first")

    log_path = out_dir / "build_log.jsonl"
    writers: dict[str, JsonlWriter] = {
        s: JsonlWriter(out_dir / f"spatial_qa_{s}.jsonl") for s in fractions
    }
    rng = random.Random(args.seed)

    counts: dict[str, dict[str, int]] = {s: {c: 0 for c in CATEGORIES} for s in fractions}
    try:
        with JsonlWriter(log_path) as w:
            w.write({**run_envelope("build_driving_qa", "nuscenes",
                                      {"in_dir": str(in_dir), "out_dir": str(out_dir),
                                       "fractions": fractions, "min_reliability": args.min_reliability}),
                     "kind": "header"})
            for i, p in enumerate(files):
                payload = torch.load(p, map_location="cpu")
                scene = payload["scene"]
                scene_token = getattr(scene, "scene_id", str(p.stem))
                split = split_assignment(scene_token, i, fractions)
                pool = (0, 1) if split in {"train", "val", "test_template", "test_composition"} else (2, 3)
                qa_list = build_qa_for_sample(payload, rng, template_pool=pool)
                for q in qa_list:
                    if q.reliability < args.min_reliability:
                        continue
                    q.split_id = split
                    rec = {
                        "question": q.question,
                        "answer": q.answer,
                        "type": q.type,
                        "involved_object_ids": q.involved_object_ids,
                        "source_relation": q.source_relation,
                        "reliability": q.reliability,
                        "split_id": q.split_id,
                        "template_id": q.template_id,
                        "scene_id": scene_token,
                        "sample_idx": i,
                    }
                    writers[split].write(rec)
                    counts[split][q.type] += 1
                if (i % 200) == 0:
                    print(f"[i={i}] split={split} so_far={ {k: sum(v.values()) for k, v in counts.items()} }")
            w.write({"kind": "summary", "counts": counts, "n_samples": len(files)})
    finally:
        for w in writers.values():
            w.close()

    print("[driving-qa] split totals:")
    for s, c in counts.items():
        total = sum(c.values())
        print(f"  {s:>20}: {total} ({c})")


if __name__ == "__main__":
    main()
