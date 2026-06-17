"""Bench2Drive closed-loop eval (paper §4.5 Table 5).

Reports Driving Score, Success Rate, Route Completion, Infraction Score, plus
collision / off-road / red-light frequency.

The eval CLI of SpaceDrive (pinned commit) is reused on the GPU box.
``--allow_b2d_vl=false`` is the default — paper §4.5 forbids using
Bench2Drive-VL extra language data in the main protocol; rows that *do* use it
are emitted with ``protocol="b2d_vl"`` so collect_tables renders them on a
separate row.
"""

from __future__ import annotations

import argparse

from geodistill.utils import JsonlWriter, run_envelope, TBD


__all__ = ["run_table5"]


METRICS = ("driving_score", "success_rate", "route_completion", "infraction_score",
           "collision", "off_road", "red_light")
HINT = (
    "Bench2Drive eval requires the SpaceDrive submodule (init_third_party.sh) + "
    "Bench2Drive simulator. Run on the GPU box; locally only the table skeleton is emitted."
)


def run_table5(*, out_path: str, ckpt_path: str | None = None, model_id: str | None = None,
               dataset_path: str | None = None, source: str = "synthetic",
               allow_b2d_vl: bool = False):
    real_run = source == "nuscenes" and ckpt_path is not None
    with JsonlWriter(out_path) as w:
        w.write({**run_envelope("eval_bench2drive", source,
                                  {"ckpt": ckpt_path, "model_id": model_id, "dataset": dataset_path,
                                   "allow_b2d_vl": allow_b2d_vl, "protocol": "b2d_vl" if allow_b2d_vl else "main"}),
                 "kind": "header"})
        if not real_run:
            row = {"kind": "table5_row", "protocol": "b2d_vl" if allow_b2d_vl else "main"}
            row.update({k: TBD for k in METRICS})
            row["note"] = HINT
            w.write(row)
            return
        try:
            from geodistill.baselines import SpaceDriveStyleAdapter  # triggers third_party import
            _ = SpaceDriveStyleAdapter()
        except ImportError as exc:
            raise ImportError(HINT) from exc
        raise NotImplementedError(HINT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--allow_b2d_vl", action="store_true",
                     help="Use Bench2Drive-VL extra language data — paper §4.5 forbids this in the main "
                          "protocol; the row is tagged 'b2d_vl' and rendered separately.")
    ap.add_argument("--out", default="runs/geodistill/eval/table5_bench2drive.jsonl")
    args = ap.parse_args()
    run_table5(out_path=args.out, ckpt_path=args.ckpt, model_id=args.model_id,
                dataset_path=args.dataset, source=args.source, allow_b2d_vl=args.allow_b2d_vl)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
