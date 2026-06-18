"""Run P0 ablations and judge which GeoDistill modules deserve main-paper status.

This is a screening tool, not an automatic code remover. It reuses the existing
P0 experiment CLIs, reads their JSONL outputs, and writes a concise decision
report:

  python -m scripts.geodistill.judge_modules --source synthetic --scenes 8 --epochs 80

For real decisions, run with ``--source nuscenes`` on the GPU box. Synthetic
results only prove that the loop closes; the report is labelled accordingly.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from geodistill.utils import read_jsonl, run_envelope


ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

TEACHERS = ("hard_quantile", "entropic_ot", "full_ntl_fgt")
DEPTH_BASELINES = ("raw", "log", "power_warp")


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _get(d: dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _rel_gain(new: float | None, base: float | None, lower_is_better: bool = True) -> float | None:
    if not (_finite(new) and _finite(base)) or abs(float(base)) < 1e-12:
        return None
    if lower_is_better:
        return (float(base) - float(new)) / abs(float(base))
    return (float(new) - float(base)) / abs(float(base))


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{100.0 * v:+.1f}%"


def _fmt_num(v: Any, nd: int = 4) -> str:
    if not _finite(v):
        return "n/a"
    return f"{float(v):.{nd}f}"


def _run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def _run_p0(args: argparse.Namespace, p0_dir: Path) -> None:
    p0_dir.mkdir(parents=True, exist_ok=True)
    common = ["--source", args.source, "--scenes", str(args.scenes), "--seed", str(args.seed)]
    if args.config:
        common += ["--config", args.config]

    _run([
        PY, "-m", "scripts.geodistill.run_teacher_stats",
        *common,
        "--epsilon_ot", str(args.epsilon_ot),
        "--gamma", str(args.gamma),
        "--out", str(p0_dir / "teacher_stats.jsonl"),
    ])
    _run([
        PY, "-m", "scripts.geodistill.run_probe",
        *common,
        "--hidden", str(args.hidden),
        "--epochs", str(args.epochs),
        "--epsilon_ot", str(args.epsilon_ot),
        "--gamma", str(args.gamma),
        "--out", str(p0_dir / "probe.jsonl"),
    ])
    _run([
        PY, "-m", "scripts.geodistill.run_depth_heads",
        *common,
        "--teacher", args.depth_teacher,
        "--epochs", str(args.epochs),
        "--epsilon_ot", str(args.epsilon_ot),
        "--gamma", str(args.gamma),
        "--out", str(p0_dir / "depth_heads.jsonl"),
    ])
    _run([
        PY, "-m", "scripts.geodistill.run_relation_probe",
        *common,
        "--hidden", str(args.hidden),
        "--epochs", str(args.relation_epochs),
        "--epsilon_ot", str(args.epsilon_ot),
        "--gamma", str(args.gamma),
        "--out", str(p0_dir / "relation_probe.jsonl"),
    ])
    if not args.skip_tables:
        _run([
            PY, "-m", "scripts.geodistill.collect_tables",
            "--in_dir", str(p0_dir),
            "--out_dir", str(args.out_dir / "tables"),
        ])


def _load_inputs(p0_dir: Path) -> dict[str, list[dict]]:
    paths = {
        "teacher_stats": p0_dir / "teacher_stats.jsonl",
        "probe": p0_dir / "probe.jsonl",
        "depth_heads": p0_dir / "depth_heads.jsonl",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required JSONL files: " + ", ".join(missing))
    inputs = {name: read_jsonl(path) for name, path in paths.items()}
    relation_path = p0_dir / "relation_probe.jsonl"
    inputs["relation_probe"] = read_jsonl(relation_path) if relation_path.exists() else []
    return inputs


def _header(rows: list[dict]) -> dict:
    for row in rows:
        if row.get("kind") == "header":
            return row
    return {}


def _probe_rows(rows: list[dict]) -> dict[str, dict]:
    return {r["teacher"]: r for r in rows if r.get("kind") == "probe" and "error" not in r}


def _teacher_agg(rows: list[dict]) -> dict[str, dict]:
    return {r["teacher"]: r for r in rows if r.get("kind") == "aggregate"}


def _head_rows(rows: list[dict]) -> dict[str, dict]:
    return {r["head"]: r for r in rows if r.get("kind") == "depth_head" and "error" not in r}


def _relation_rows(rows: list[dict]) -> dict[str, dict]:
    return {r["teacher"]: r for r in rows if r.get("kind") == "relation_probe"}


def _best_probe_baseline(probes: dict[str, dict]) -> tuple[str | None, dict | None]:
    candidates = [(name, probes.get(name)) for name in ("hard_quantile", "entropic_ot")]
    candidates = [(name, row) for name, row in candidates if row and _finite(_get(row, "overall.AbsRel"))]
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: float(_get(item[1], "overall.AbsRel")))


def _best_depth_baseline(heads: dict[str, dict], metric: str = "overall.AbsRel") -> tuple[str | None, dict | None]:
    candidates = [(name, heads.get(name)) for name in DEPTH_BASELINES]
    candidates = [(name, row) for name, row in candidates if row and _finite(_get(row, metric))]
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: float(_get(item[1], metric)))


def _source_label(inputs: dict[str, list[dict]]) -> str:
    for rows in inputs.values():
        h = _header(rows)
        if h.get("source"):
            return h["source"]
    return "unknown"


def _confidence(source: str) -> str:
    if source == "nuscenes":
        return "real"
    if source == "synthetic":
        return "smoke_only"
    return "unknown"


def judge_ntl_fgt(inputs: dict[str, list[dict]], args: argparse.Namespace) -> dict:
    probes = _probe_rows(inputs["probe"])
    stats = _teacher_agg(inputs["teacher_stats"])
    full = probes.get("full_ntl_fgt")
    base_name, base = _best_probe_baseline(probes)
    source = _source_label(inputs)

    if not full or not base:
        return {
            "module": "NTL-FGT",
            "decision": "NEEDS_EVAL",
            "confidence": _confidence(source),
            "reason": "Missing full_ntl_fgt or hard/OT probe rows.",
            "metrics": {},
        }

    full_abs = _get(full, "overall.AbsRel")
    base_abs = _get(base, "overall.AbsRel")
    full_far = _get(full, "binned.30m_plus.AbsRel")
    base_far = _get(base, "binned.30m_plus.AbsRel")
    full_far_n = _get(full, "binned.30m_plus.n", 0)
    base_far_n = _get(base, "binned.30m_plus.n", 0)
    overall_gain = _rel_gain(full_abs, base_abs)
    far_gain = _rel_gain(full_far, base_far)

    full_stat = stats.get("full_ntl_fgt", {})
    ot_stat = stats.get("entropic_ot", {})
    time_ratio = None
    if _finite(full_stat.get("build_time_s_mean")) and _finite(ot_stat.get("build_time_s_mean")):
        denom = max(float(ot_stat["build_time_s_mean"]), 1e-12)
        time_ratio = float(full_stat["build_time_s_mean"]) / denom

    enough_far = min(int(full_far_n or 0), int(base_far_n or 0)) >= args.min_far_tokens
    useful = (
        (overall_gain is not None and overall_gain >= args.keep_delta)
        or (enough_far and far_gain is not None and far_gain >= args.keep_delta)
    )
    harmful = (
        (overall_gain is not None and overall_gain <= -args.remove_delta)
        and (not enough_far or far_gain is None or far_gain <= args.keep_delta)
    )
    overhead_warning = time_ratio is not None and time_ratio > args.overhead_warn_ratio

    if useful:
        decision = "KEEP"
        reason = "Full NTL-FGT improves token geometry probe over the best simple teacher."
        if overhead_warning:
            reason += " Training overhead is high, so keep only if real downstream gains also hold."
    elif harmful:
        decision = "REMOVE"
        reason = "Full NTL-FGT is worse than the best simple teacher on probe metrics."
    else:
        decision = "DEMOTE"
        reason = "Full NTL-FGT does not clear the keep threshold; treat it as an ablation/appendix candidate."

    if not enough_far:
        reason += f" Far-bin evidence is weak (min n={min(int(full_far_n or 0), int(base_far_n or 0))})."

    return {
        "module": "NTL-FGT",
        "decision": decision,
        "confidence": _confidence(source),
        "reason": reason,
        "metrics": {
            "baseline": base_name,
            "full_absrel": full_abs,
            "baseline_absrel": base_abs,
            "overall_rel_gain": overall_gain,
            "full_30m_absrel": full_far,
            "baseline_30m_absrel": base_far,
            "far_rel_gain": far_gain,
            "full_30m_n": full_far_n,
            "baseline_30m_n": base_far_n,
            "time_ratio_vs_entropic_ot": time_ratio,
        },
    }


def judge_r2ac(inputs: dict[str, list[dict]], args: argparse.Namespace) -> dict:
    heads = _head_rows(inputs["depth_heads"])
    r2ac = heads.get("r2ac")
    base_name, base = _best_depth_baseline(heads)
    source = _source_label(inputs)

    if not r2ac or not base:
        return {
            "module": "R²AC",
            "decision": "NEEDS_EVAL",
            "confidence": _confidence(source),
            "reason": "Missing r2ac or raw/log/power_warp depth-head rows.",
            "metrics": {},
        }

    r_abs = _get(r2ac, "overall.AbsRel")
    b_abs = _get(base, "overall.AbsRel")
    r_far = _get(r2ac, "binned.30m_plus.AbsRel")
    b_far = _get(base, "binned.30m_plus.AbsRel")
    r_far_n = _get(r2ac, "binned.30m_plus.n", 0)
    b_far_n = _get(base, "binned.30m_plus.n", 0)
    overall_gain = _rel_gain(r_abs, b_abs)
    far_gain = _rel_gain(r_far, b_far)
    enough_far = min(int(r_far_n or 0), int(b_far_n or 0)) >= args.min_far_tokens

    useful = (
        (overall_gain is not None and overall_gain >= args.keep_delta)
        or (enough_far and far_gain is not None and far_gain >= args.keep_delta)
    )
    harmful = (
        overall_gain is not None
        and overall_gain <= -args.remove_delta
        and (not enough_far or far_gain is None or far_gain <= args.keep_delta)
    )

    if useful:
        decision = "KEEP"
        reason = "R²AC beats the best fixed depth transform enough to justify main-text space."
    elif harmful:
        decision = "REMOVE"
        reason = "R²AC underperforms simpler fixed depth representations."
    else:
        decision = "DEMOTE"
        reason = "R²AC does not clearly beat simpler depth transforms; keep as optional/appendix."
    if not enough_far:
        reason += f" Far-bin evidence is weak (min n={min(int(r_far_n or 0), int(b_far_n or 0))})."

    return {
        "module": "R²AC",
        "decision": decision,
        "confidence": _confidence(source),
        "reason": reason,
        "metrics": {
            "baseline": base_name,
            "r2ac_absrel": r_abs,
            "baseline_absrel": b_abs,
            "overall_rel_gain": overall_gain,
            "r2ac_30m_absrel": r_far,
            "baseline_30m_absrel": b_far,
            "far_rel_gain": far_gain,
            "r2ac_30m_n": r_far_n,
            "baseline_30m_n": b_far_n,
            "r2ac_params": r2ac.get("trainable_params"),
            "baseline_params": base.get("trainable_params"),
        },
    }


def judge_lpga(inputs: dict[str, list[dict]], args: argparse.Namespace) -> dict:
    heads = _head_rows(inputs["depth_heads"])
    lpga = heads.get("lpga")
    ext = heads.get("unidepth_pe")
    source = _source_label(inputs)

    if not lpga or not ext:
        return {
            "module": "LPGA camera-only adapter",
            "decision": "NEEDS_EVAL",
            "confidence": _confidence(source),
            "reason": "Missing lpga or external-depth comparison row.",
            "metrics": {},
        }

    l_abs = _get(lpga, "overall.AbsRel")
    e_abs = _get(ext, "overall.AbsRel")
    l_far = _get(lpga, "binned.30m_plus.AbsRel")
    e_far = _get(ext, "binned.30m_plus.AbsRel")
    overall_gap = _rel_gain(l_abs, e_abs)
    far_gap = _rel_gain(l_far, e_far)
    synthetic_surrogate = source == "synthetic" or "SURROGATE" in str(ext.get("note", ""))

    if synthetic_surrogate:
        decision = "NEEDS_EVAL"
        reason = "External-depth row is a synthetic surrogate; LPGA needs real UniDepth latency/memory/accuracy Pareto."
    elif (
        overall_gap is not None
        and overall_gap >= -args.pareto_slack
        and (far_gap is None or far_gap >= -args.pareto_slack)
    ):
        decision = "KEEP"
        reason = "LPGA is within the allowed accuracy slack versus external depth; verify latency/memory Pareto."
    else:
        decision = "NEEDS_EVAL"
        reason = "LPGA accuracy is not yet enough by itself; needs efficiency/downstream Pareto evidence."

    return {
        "module": "LPGA camera-only adapter",
        "decision": decision,
        "confidence": _confidence(source),
        "reason": reason,
        "metrics": {
            "lpga_absrel": l_abs,
            "external_absrel": e_abs,
            "overall_rel_gap_vs_external": overall_gap,
            "lpga_30m_absrel": l_far,
            "external_30m_absrel": e_far,
            "far_rel_gap_vs_external": far_gap,
            "lpga_params": lpga.get("trainable_params"),
            "external_depth_note": ext.get("note"),
        },
    }


def judge_relation_residual(inputs: dict[str, list[dict]], args: argparse.Namespace) -> dict:
    rows = _relation_rows(inputs.get("relation_probe", []))
    source = _source_label(inputs)
    full = rows.get("full_ntl_fgt")
    if not full:
        return {
            "module": "relation residual",
            "decision": "NEEDS_EVAL",
            "confidence": _confidence(source),
            "reason": "No relation_probe row is available for full_ntl_fgt; run run_relation_probe or full judge.",
            "metrics": {},
        }

    order_gain = _get(full, "order.gain")
    cross_gain = _get(full, "cross_view.gain")
    order_n = int(_get(full, "order.n", 0) or 0)
    cross_n = int(_get(full, "cross_view.n", 0) or 0)

    enough_order = order_n >= args.min_relation_pairs
    enough_cross = cross_n >= args.min_relation_pairs
    useful_order = enough_order and _finite(order_gain) and float(order_gain) >= args.relation_keep_gain
    useful_cross = enough_cross and _finite(cross_gain) and float(cross_gain) >= args.relation_keep_gain
    weak_order = (not enough_order) or (not _finite(order_gain)) or float(order_gain) < args.relation_demote_gain
    weak_cross = (not enough_cross) or (not _finite(cross_gain)) or float(cross_gain) < args.relation_demote_gain

    if useful_order or useful_cross:
        decision = "KEEP"
        reason = "Relation probe shows recoverable pairwise structure beyond majority baseline."
    elif weak_order and weak_cross:
        decision = "REMOVE"
        reason = "Relation probe does not show useful recoverability for order or cross-view relations."
    else:
        decision = "DEMOTE"
        reason = "Relation signal is marginal; keep as optional/downstream ablation, not a main contribution."

    if not (enough_order and enough_cross):
        reason += f" Pair evidence is limited (order n={order_n}, cross n={cross_n})."

    return {
        "module": "relation residual",
        "decision": decision,
        "confidence": _confidence(source),
        "reason": reason,
        "metrics": {
            "teacher": "full_ntl_fgt",
            "order_gain": order_gain,
            "order_accuracy": _get(full, "order.accuracy"),
            "order_majority_accuracy": _get(full, "order.majority_accuracy"),
            "order_n": order_n,
            "cross_view_gain": cross_gain,
            "cross_view_accuracy": _get(full, "cross_view.accuracy"),
            "cross_view_majority_accuracy": _get(full, "cross_view.majority_accuracy"),
            "cross_view_n": cross_n,
            "mean_edges_per_token": full.get("mean_edges_per_token"),
        },
    }


def static_downstream_decisions(source: str) -> list[dict]:
    confidence = _confidence(source)
    return [
        {
            "module": "semantic constraint + zero-init gate",
            "decision": "DEMOTE",
            "confidence": confidence,
            "reason": "Treat as a training-stability detail unless CKA/downstream regression proves it is central.",
            "metrics": {},
        },
    ]


def build_report(inputs: dict[str, list[dict]], args: argparse.Namespace) -> dict:
    source = _source_label(inputs)
    decisions = [
        judge_ntl_fgt(inputs, args),
        judge_r2ac(inputs, args),
        judge_lpga(inputs, args),
        judge_relation_residual(inputs, args),
        *static_downstream_decisions(source),
    ]
    return {
        **run_envelope(
            "module_judge",
            source,
            {
                "keep_delta": args.keep_delta,
                "remove_delta": args.remove_delta,
                "pareto_slack": args.pareto_slack,
                "min_far_tokens": args.min_far_tokens,
                "input_dir": str(args.input_dir),
            },
        ),
        "kind": "module_decisions",
        "confidence": _confidence(source),
        "decisions": decisions,
    }


def render_markdown(report: dict) -> str:
    source = report.get("source", "unknown")
    confidence = report.get("confidence", "unknown")
    lines = [
        "# GeoDistill Module Judge",
        "",
        f"- Source: `{source}`",
        f"- Confidence: `{confidence}`",
        f"- Generated: `{report.get('timestamp', 'unknown')}`",
        "",
    ]
    if source == "synthetic":
        lines += [
            "> Synthetic results are only a smoke/screening signal. Do not use these decisions as paper claims.",
            "",
        ]

    lines += [
        "## Decisions",
        "",
        "| Module | Decision | Reason | Key metric |",
        "|---|---|---|---|",
    ]
    for d in report["decisions"]:
        m = d.get("metrics", {})
        key = "n/a"
        if d["module"] == "NTL-FGT":
            key = (
                f"AbsRel gain {_fmt_pct(m.get('overall_rel_gain'))}; "
                f"30m+ gain {_fmt_pct(m.get('far_rel_gain'))}; "
                f"time x{_fmt_num(m.get('time_ratio_vs_entropic_ot'), 2)}"
            )
        elif d["module"] == "R²AC":
            key = f"AbsRel gain {_fmt_pct(m.get('overall_rel_gain'))}; 30m+ gain {_fmt_pct(m.get('far_rel_gain'))}"
        elif d["module"] == "LPGA camera-only adapter":
            key = (
                f"vs external AbsRel gap {_fmt_pct(m.get('overall_rel_gap_vs_external'))}; "
                f"30m+ gap {_fmt_pct(m.get('far_rel_gap_vs_external'))}"
            )
        elif d["module"] == "relation residual":
            key = (
                f"order gain {_fmt_pct(m.get('order_gain'))}; "
                f"cross gain {_fmt_pct(m.get('cross_view_gain'))}"
            )
        lines.append(f"| {d['module']} | `{d['decision']}` | {d['reason']} | {key} |")

    lines += ["", "## Raw Metrics", ""]
    for d in report["decisions"]:
        lines += [f"### {d['module']}", "", f"- Decision: `{d['decision']}`", f"- Reason: {d['reason']}"]
        metrics = d.get("metrics") or {}
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"- `{k}`: {_fmt_num(v, 6)}")
                else:
                    lines.append(f"- `{k}`: {v}")
        else:
            lines.append("- Metrics: n/a")
        lines.append("")
    return "\n".join(lines)


def write_outputs(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "module_decisions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "module_decisions.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out_dir / 'module_decisions.json'}")
    print(f"wrote {out_dir / 'module_decisions.md'}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--relation_epochs", type=int, default=200)
    ap.add_argument("--epsilon_ot", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--depth_teacher", default="hard_quantile", choices=TEACHERS)
    ap.add_argument("--out_dir", type=Path, default=Path("runs/geodistill/module_judge"))
    ap.add_argument("--input_dir", type=Path, default=None,
                    help="Read existing P0 JSONL files from this directory. Defaults to out_dir/p0.")
    ap.add_argument("--skip_run", action="store_true",
                    help="Only read existing JSONL files and write decisions.")
    ap.add_argument("--skip_tables", action="store_true")
    ap.add_argument("--keep_delta", type=float, default=0.03,
                    help="Relative improvement needed for KEEP on lower-is-better metrics.")
    ap.add_argument("--remove_delta", type=float, default=0.03,
                    help="Relative regression threshold for REMOVE.")
    ap.add_argument("--pareto_slack", type=float, default=0.10,
                    help="Allowed relative accuracy gap for deployment Pareto candidates.")
    ap.add_argument("--min_far_tokens", type=int, default=20)
    ap.add_argument("--min_relation_pairs", type=int, default=50)
    ap.add_argument("--relation_keep_gain", type=float, default=0.05)
    ap.add_argument("--relation_demote_gain", type=float, default=0.01)
    ap.add_argument("--overhead_warn_ratio", type=float, default=8.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = (ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    if args.input_dir is None:
        args.input_dir = args.out_dir / "p0"
    else:
        args.input_dir = (ROOT / args.input_dir).resolve() if not args.input_dir.is_absolute() else args.input_dir

    if not args.skip_run:
        _run_p0(args, args.input_dir)
    inputs = _load_inputs(args.input_dir)
    report = build_report(inputs, args)
    write_outputs(report, args.out_dir)

    print("\n=== MODULE JUDGE SUMMARY ===")
    for d in report["decisions"]:
        print(f"{d['module']}: {d['decision']} - {d['reason']}")


if __name__ == "__main__":
    main()
