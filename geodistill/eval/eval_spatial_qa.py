"""Spatial QA eval (paper §4.5 Table 1, 7 categories + hard paraphrase + composition).

Lazy GPU-only runtime: ``transformers``, Qwen2.5-VL weights, and the
``DrivingQA`` dataset (paper §4.5 + ``dataset/geodistill/driving_qa_dataset.py``)
must be available. On CPU this module imports fine but ``run_table1`` raises
the standard ``ImportError`` hint.

The CPU contract: produce a JSONL header + a row per QA category with
``answer = "TBD"`` so :mod:`scripts.geodistill.collect_tables` can render the
table layout offline.
"""

from __future__ import annotations

import argparse

from geodistill.utils import JsonlWriter, run_envelope, TBD


__all__ = ["run_table1", "QA_CATEGORIES", "QA_TBD_HINT"]


QA_CATEGORIES = (
    "metric_distance",
    "relative_direction",
    "object_comparison",
    "free_space_topology",
    "occlusion",
    "cross_view_correspondence",
    "temporal_motion",
    "counterfactual_trajectory",
)
QA_SPLITS = ("val", "test_template", "test_paraphrase", "test_composition")
QA_TBD_HINT = (
    "Spatial QA evaluation requires Qwen2.5-VL weights + transformers + the "
    "DrivingQA dataset. Run on the GPU box; locally only the table skeleton is emitted."
)


def _gpu_eval(model, tokenizer, dataset, splits, categories) -> list[dict]:
    """The real evaluation loop. Importable only when transformers is installed."""
    raise ImportError(QA_TBD_HINT)


def run_table1(*, out_path: str, ckpt_path: str | None = None, model_id: str | None = None,
               dataset_path: str | None = None, splits=QA_SPLITS, categories=QA_CATEGORIES,
               source: str = "synthetic"):
    """Write the Table 1 skeleton; on GPU box, replace ``TBD`` with real numbers."""
    real_run = source == "nuscenes" and ckpt_path is not None
    with JsonlWriter(out_path) as w:
        w.write({**run_envelope("eval_spatial_qa", source,
                                  {"ckpt": ckpt_path, "model_id": model_id, "dataset": dataset_path,
                                   "splits": list(splits), "categories": list(categories)}),
                 "kind": "header"})
        if not real_run:
            for split in splits:
                row = {"kind": "table1_row", "split": split,
                        "overall_accuracy": TBD,
                        "metric_tolerance_accuracy": TBD,
                        "categories": {c: TBD for c in categories},
                        "note": QA_TBD_HINT}
                w.write(row)
            return
        try:
            import transformers  # noqa: F401
        except Exception as exc:                                                       # noqa: BLE001
            raise ImportError(QA_TBD_HINT) from exc
        # GPU side: load Qwen + LoRA, run dataset, dump rows. Implementation lives there.
        rows = _gpu_eval(model_id, ckpt_path, dataset_path, splits, categories)
        for r in rows:
            w.write(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="runs/geodistill/eval/table1_spatial_qa.jsonl")
    args = ap.parse_args()
    run_table1(out_path=args.out, ckpt_path=args.ckpt, model_id=args.model_id,
                dataset_path=args.dataset, source=args.source)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
