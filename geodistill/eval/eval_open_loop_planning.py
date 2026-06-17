"""nuScenes open-loop planning eval (paper §4.5 Table 4).

Reports ``L2 1s/2s/3s``, ``avg L2``, ``collision``, ``intersection`` over the
6-waypoint trajectory predicted by the coordinate decoder (paper §3.12).

Like ``eval_spatial_qa``, this module is lazy: real evaluation requires the
nuScenes planning split + Qwen2.5-VL weights. On CPU the CLI emits a TBD
skeleton so the collect_tables renderer can still render the layout.
"""

from __future__ import annotations

import argparse

from geodistill.utils import JsonlWriter, run_envelope, TBD


__all__ = ["run_table4", "L2_HORIZONS"]


L2_HORIZONS = (1, 2, 3)
HINT = (
    "Open-loop planning eval requires Qwen2.5-VL weights + nuScenes planning split. "
    "Run on the GPU box; locally only the table skeleton is emitted."
)


def run_table4(*, out_path: str, ckpt_path: str | None = None, model_id: str | None = None,
               dataset_path: str | None = None, source: str = "synthetic"):
    real_run = source == "nuscenes" and ckpt_path is not None
    with JsonlWriter(out_path) as w:
        w.write({**run_envelope("eval_open_loop_planning", source,
                                  {"ckpt": ckpt_path, "model_id": model_id, "dataset": dataset_path,
                                   "horizons_s": list(L2_HORIZONS)}),
                 "kind": "header"})
        if not real_run:
            row = {"kind": "table4_row",
                    "L2_per_horizon": {f"L2_{h}s": TBD for h in L2_HORIZONS},
                    "L2_avg": TBD, "collision_rate": TBD, "intersection_rate": TBD,
                    "note": HINT}
            w.write(row)
            return
        try:
            import transformers  # noqa: F401
        except Exception as exc:                                                       # noqa: BLE001
            raise ImportError(HINT) from exc
        raise NotImplementedError(HINT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="runs/geodistill/eval/table4_open_loop.jsonl")
    args = ap.parse_args()
    run_table4(out_path=args.out, ckpt_path=args.ckpt, model_id=args.model_id,
                dataset_path=args.dataset, source=args.source)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
