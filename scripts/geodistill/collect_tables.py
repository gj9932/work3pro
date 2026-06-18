"""Collect P0 + P1 JSONL outputs into Markdown table templates (paper §4.5).

P0 tables (already rendered, untouched):
  - table7_teacher_overhead.md   (paper Table 7)
  - table8_teacher_diagnostics.md
  - table2_token_probe.md         (P0 subset)
  - depth_head_comparison.md      (P0 subset of §4.4)

P1 tables (added by Chunk 8):
  - table1_spatial_qa.md          (Driving spatial QA, paper Table 1)
  - table2_token_probe_full.md    (Full Table 2 with H_geo + risk strat + CKA)
  - table3_relation_probe.md      (Paper Table 3)
  - table4_open_loop.md           (nuScenes open-loop planning, paper Table 4)
  - table5_bench2drive.md         (Bench2Drive closed-loop, paper Table 5)
  - table6_efficiency.md          (Efficiency Pareto, paper Table 6)
  - shortcut_audit.md             (paper §4.7 shuffles)

Cells that need real nuScenes+Qwen and were not produced are rendered as ``TBD``.
A banner notes whether the numbers came from synthetic fixture or nuScenes, so
synthetic rows are never mistaken for paper results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geodistill.utils import read_jsonl

TBD = "TBD"


def _fmt(v, nd=4):
    if v is None:
        return TBD
    if isinstance(v, str):
        return v
    if isinstance(v, float) and v != v:  # NaN
        return TBD
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _source_of(rows):
    for r in rows:
        if r.get("kind") == "header":
            return r.get("source", "unknown")
    return "unknown"


def _banner(source):
    if source == "nuscenes":
        return "<!-- source: nuScenes + frozen Qwen2.5-VL (real) -->\n"
    return ("> **SOURCE: SYNTHETIC FIXTURE (CPU smoke).** These are loop-closure "
            "numbers, NOT paper results. Real cells stay TBD until run on "
            "nuScenes + Qwen2.5-VL.\n")


def table7(stats_rows) -> str:
    src = _source_of(stats_rows)
    agg = {r["teacher"]: r for r in stats_rows if r.get("kind") == "aggregate"}
    order = ["hard_quantile", "entropic_ot", "full_ntl_fgt"]
    labels = {"hard_quantile": "Hard quantile teacher",
              "entropic_ot": "Entropic OT, `gamma=0`",
              "full_ntl_fgt": "Full NTL-FGT"}
    lines = ["## Table 7 — Training-time teacher construction overhead", "", _banner(src), "",
             "| Teacher construction | Time / scene (s) | Extra mem (MB) | Cand. anchors / token | Sinkhorn iters | Sparse edges / token |",
             "|---|---:|---:|---:|---:|---:|"]
    for k in order:
        r = agg.get(k)
        if not r:
            lines.append(f"| {labels[k]} | {TBD} | {TBD} | {TBD} | {TBD} | {TBD} |")
            continue
        lines.append(
            f"| {labels[k]} | {_fmt(r.get('build_time_s_mean'),3)} | {_fmt(r.get('extra_memory_mb_mean'),1)} "
            f"| {_fmt(r.get('candidate_anchors_per_token_mean'),1)} | {_fmt(r.get('sinkhorn_iterations_mean'),0)} "
            f"| {_fmt(r.get('sparse_edges_per_token_mean'),1)} |"
        )
    return "\n".join(lines) + "\n"


def table8(stats_rows) -> str:
    src = _source_of(stats_rows)
    agg = {r["teacher"]: r for r in stats_rows if r.get("kind") == "aggregate"}
    order = ["hard_quantile", "entropic_ot", "full_ntl_fgt"]
    labels = {"hard_quantile": "Hard quantile", "entropic_ot": "Entropic OT (γ=0)", "full_ntl_fgt": "Full NTL-FGT"}
    lines = ["## Table 8 (P0 subset) — teacher reliability/coverage diagnostics", "", _banner(src), "",
             "| Teacher | Valid token ratio | depth mean | q_conc mean | q_mass mean | q_teacher mean | 0-10m | 10-30m | 30m+ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in order:
        r = agg.get(k)
        if not r:
            lines.append(f"| {labels[k]} |" + " TBD |" * 8)
            continue
        lines.append(
            f"| {labels[k]} | {_fmt(r.get('valid_token_ratio_mean'),3)} | {_fmt(r.get('depth_dist_mean'),2)} "
            f"| {_fmt(r.get('q_conc_dist_mean'),3)} | {_fmt(r.get('q_mass_dist_mean'),3)} | {_fmt(r.get('q_teacher_dist_mean'),3)} "
            f"| {r.get('count_0_10m_sum', TBD)} | {r.get('count_10_30m_sum', TBD)} | {r.get('count_30m_plus_sum', TBD)} |"
        )
    return "\n".join(lines) + "\n"


def table2(probe_rows) -> str:
    src = _source_of(probe_rows)
    rows = {r["teacher"]: r for r in probe_rows if r.get("kind") == "probe"}
    order = ["hard_quantile", "entropic_ot", "full_ntl_fgt"]
    lines = ["## Table 2 (P0 subset) — frozen token geometry probe", "", _banner(src), "",
             "| Teacher | AbsRel ↓ | RMSE ↓ | delta1 ↑ | 0-10m ↓ | 10-30m ↓ | 30m+ ↓ | rho(q^OT, err) ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in order:
        r = rows.get(k)
        if not r or "error" in r:
            lines.append(f"| {k} |" + " TBD |" * 7)
            continue
        o, b = r["overall"], r["binned"]
        lines.append(
            f"| {k} | {_fmt(o['AbsRel'])} | {_fmt(o['RMSE'],3)} | {_fmt(o['delta1'],3)} "
            f"| {_fmt(b['0_10m']['AbsRel'])} | {_fmt(b['10_30m']['AbsRel'])} | {_fmt(b['30m_plus']['AbsRel'])} "
            f"| {_fmt(r['spearman_qOT_vs_abserr'],3)} |"
        )
    return "\n".join(lines) + "\n"


def table_depth_heads(head_rows) -> str:
    src = _source_of(head_rows)
    rows = [r for r in head_rows if r.get("kind") == "depth_head"]
    lines = ["## Depth representation comparison (P0 subset of §4.4)", "", _banner(src), "",
             "| Model | External depth | Trainable params | AbsRel ↓ | RMSE ↓ | delta1 ↑ | 30m+ AbsRel ↓ |",
             "|---|:--:|---:|---:|---:|---:|---:|"]
    label = {"raw": "Qwen + raw depth", "log": "Qwen + fixed log depth",
             "power_warp": "Qwen + fixed power-warp", "r2ac": "Qwen + R²AC (pred a)",
             "unidepth_pe": "Qwen + UniDepth + 3D PE", "lpga": "Qwen + LPGA"}
    for r in rows:
        if "error" in r:
            lines.append(f"| {label.get(r['head'], r['head'])} | | TBD | TBD | TBD | TBD | TBD |")
            continue
        o, b = r["overall"], r["binned"]
        ext = "yes" if r.get("external_depth") else "no"
        lines.append(
            f"| {label.get(r['head'], r['head'])} | {ext} | {r['trainable_params']} "
            f"| {_fmt(o['AbsRel'])} | {_fmt(o['RMSE'],3)} | {_fmt(o['delta1'],3)} | {_fmt(b['30m_plus']['AbsRel'])} |"
        )
    return "\n".join(lines) + "\n"


# ====================================================================== P1
def table1_spatial_qa(rows) -> str:
    src = _source_of(rows)
    qrows = [r for r in rows if r.get("kind") == "table1_row"]
    cats = []
    for r in qrows:
        for c in r.get("categories", {}):
            if c not in cats:
                cats.append(c)
    cats = cats or ["metric_distance", "relative_direction", "object_comparison",
                     "free_space_topology", "occlusion", "cross_view_correspondence",
                     "temporal_motion", "counterfactual_trajectory"]
    head = "| Split | Overall ↑ | Metric tol. ↑ | " + " | ".join(cats) + " |"
    sep = "|---|---:|---:|" + "---:|" * len(cats)
    lines = ["## Table 1 — Driving spatial QA (paper §4.5)", "", _banner(src), "", head, sep]
    for r in qrows:
        cells = [r.get("split", TBD), _fmt(r.get("overall_accuracy")),
                 _fmt(r.get("metric_tolerance_accuracy"))]
        for c in cats:
            cells.append(_fmt(r.get("categories", {}).get(c, TBD)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def table2_full(rows) -> str:
    src = _source_of(rows)
    drows = [r for r in rows if r.get("kind") == "table2_row"]
    lines = ["## Table 2 — Frozen token geometry probe (full, paper §4.5)", "", _banner(src), "",
             "| Scene | Teacher | View | AbsRel ↓ | RMSE ↓ | δ1 ↑ | 30m+ ↓ | ρ(qOT,err) ↓ | r̂ MAE | q̂ MAE | â MAE | CKA ↑ | GeoGain/Drift ↑ |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in drows:
        scene = r.get("scene_id", TBD)
        teacher = r.get("teacher", TBD)
        for view, key in (("H_img", "h_img"), ("H_geo", "h_geo"), ("model d_hat", "model_direct")):
            m = r.get(key)
            if not m:
                continue
            lines.append(
                f"| {scene} | {teacher} | {view} "
                f"| {_fmt(m['overall']['AbsRel'])} | {_fmt(m['overall']['RMSE'],3)} | {_fmt(m['overall']['delta1'],3)} "
                f"| {_fmt(m['binned']['30m_plus']['AbsRel'])} "
                f"| {_fmt(r.get('rho_qOT_vs_err'),3)} "
                f"| {_fmt(r.get('factor_calibration', {}).get('r_hat_MAE'),3)} "
                f"| {_fmt(r.get('factor_calibration', {}).get('q_hat_MAE'),3)} "
                f"| {_fmt(r.get('factor_calibration', {}).get('a_hat_MAE'),3)} "
                f"| {_fmt(r.get('cka'),3)} | {_fmt(r.get('geo_gain_over_drift'),3)} |"
            )
    return "\n".join(lines) + "\n"


def table3_relation_probe(rows) -> str:
    src = _source_of(rows)
    drows = [r for r in rows if r.get("kind") == "table3_row"]
    lines = ["## Table 3 — Frozen relation probe (paper §4.5)", "", _banner(src), "",
             "| Scene | Teacher | n_edges | Δ-3D MAE ↓ | Order acc ↑ | Cross-view acc ↑ | Topology acc ↑ | Occlusion acc ↑ |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in drows:
        if "error" in r:
            lines.append(f"| {r.get('scene_id', TBD)} | {r.get('teacher', TBD)} | TBD | TBD | TBD | TBD | TBD | TBD |")
            continue
        lines.append(
            f"| {r.get('scene_id', TBD)} | {r.get('teacher', TBD)} | {r.get('n_edges', TBD)} "
            f"| {_fmt(r.get('delta_3d', {}).get('MAE'),3)} "
            f"| {_fmt(r.get('order', {}).get('accuracy'),3)} "
            f"| {_fmt(r.get('cross_view', {}).get('accuracy'),3)} "
            f"| {_fmt(r.get('topology', {}).get('accuracy'),3)} "
            f"| {_fmt(r.get('occlusion', {}).get('accuracy'),3)} |"
        )
    return "\n".join(lines) + "\n"


def table4_open_loop(rows) -> str:
    src = _source_of(rows)
    drows = [r for r in rows if r.get("kind") == "table4_row"]
    lines = ["## Table 4 — nuScenes open-loop planning (paper §4.5)", "", _banner(src), "",
             "| L2 1s ↓ | L2 2s ↓ | L2 3s ↓ | Avg L2 ↓ | Collision ↓ | Intersection ↓ | Note |",
             "|---:|---:|---:|---:|---:|---:|---|"]
    for r in drows:
        per = r.get("L2_per_horizon", {})
        lines.append(
            f"| {_fmt(per.get('L2_1s'),3)} | {_fmt(per.get('L2_2s'),3)} | {_fmt(per.get('L2_3s'),3)} "
            f"| {_fmt(r.get('L2_avg'),3)} "
            f"| {_fmt(r.get('collision_rate'),3)} | {_fmt(r.get('intersection_rate'),3)} "
            f"| {r.get('note', '')} |"
        )
    return "\n".join(lines) + "\n"


def table5_bench2drive(rows) -> str:
    src = _source_of(rows)
    drows = [r for r in rows if r.get("kind") == "table5_row"]
    lines = ["## Table 5 — Bench2Drive closed-loop (paper §4.5)", "", _banner(src), "",
             "| Protocol | Driving Score ↑ | Success ↑ | Route ↑ | Infraction ↑ | Collision ↓ | Off-road ↓ | Red-light ↓ | Note |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in drows:
        lines.append(
            f"| {r.get('protocol', TBD)} | {_fmt(r.get('driving_score'),2)} | {_fmt(r.get('success_rate'),2)} "
            f"| {_fmt(r.get('route_completion'),2)} | {_fmt(r.get('infraction_score'),2)} "
            f"| {_fmt(r.get('collision'),2)} | {_fmt(r.get('off_road'),2)} | {_fmt(r.get('red_light'),2)} "
            f"| {r.get('note', '')} |"
        )
    return "\n".join(lines) + "\n"


def table6_efficiency(rows) -> str:
    src = _source_of(rows)
    drows = [r for r in rows if r.get("kind") == "table6_row"]
    lines = ["## Table 6 — Inference efficiency (paper §4.5)", "", _banner(src), "",
             "| Method | External depth | Added params | Rel. UniDepthV2-L | Memory (MB) | Latency (s) | FLOPs | 30m+ AbsRel ↓ | QA ↑ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in drows:
        lines.append(
            f"| {r.get('method', TBD)} | {r.get('external_depth', TBD)} "
            f"| {r.get('added_params', TBD)} | {r.get('params_relative_to_unidepthv2_l', TBD)} "
            f"| {r.get('memory_mb', TBD)} | {_fmt(r.get('latency_s'),4)} | {r.get('flops', TBD)} "
            f"| {_fmt(r.get('AbsRel_30m_plus'))} | {_fmt(r.get('QA_overall'))} |"
        )
    return "\n".join(lines) + "\n"


def shortcut_audit_table(rows) -> str:
    src = _source_of(rows)
    audit = [r for r in rows if r.get("kind") == "shortcut_row"]
    by_kind: dict[str, list[float]] = {}
    for r in audit:
        k = r.get("shuffle_kind", TBD)
        v = r.get("absrel_vs_gt")
        if isinstance(v, (int, float)) and v == v:
            by_kind.setdefault(k, []).append(float(v))

    def _mean(vs):
        return sum(vs) / len(vs) if vs else float("nan")

    clean = _mean(by_kind.get("clean", []))
    lines = ["## Shortcut audit (paper §4.7)", "", _banner(src), "",
             "Each row reports mean teacher AbsRel vs GT depth on the synthetic fixture.",
             "A genuine supervision channel should make AbsRel rise after its shuffle.",
             "",
             "| Shuffle | Mean AbsRel | Δ vs clean | Degraded? |",
             "|---|---:|---:|:--:|"]
    order = ["clean", "image_embeds", "calibration", "depth_label", "allocation_label",
              "transport_structure", "relation_label", "constant_depth"]
    for k in order:
        vs = by_kind.get(k)
        if not vs:
            lines.append(f"| {k} | TBD | TBD |  |")
            continue
        mean = _mean(vs)
        delta = mean - clean if clean == clean else float("nan")
        deg = "✓" if (k != "clean" and delta == delta and delta > 0) else ("—" if k == "clean" else "✗")
        lines.append(f"| {k} | {_fmt(mean)} | {_fmt(delta)} | {deg} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="runs/geodistill/p0")
    ap.add_argument("--out_dir", default="runs/geodistill/tables")
    ap.add_argument("--p1_dir", default="runs/geodistill/eval",
                     help="Directory holding P1 (Chunk 8) eval JSONL files.")
    ap.add_argument("--shortcut_dir", default="runs/geodistill/shortcut",
                     help="Directory holding shortcut audit JSONL files.")
    args = ap.parse_args()
    ind = Path(args.in_dir)
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("teacher_stats.jsonl", [("table7_teacher_overhead.md", table7),
                                 ("table8_teacher_diagnostics.md", table8)]),
        ("probe.jsonl", [("table2_token_probe.md", table2)]),
        ("depth_heads.jsonl", [("depth_head_comparison.md", table_depth_heads)]),
    ]
    for fname, renderers in jobs:
        path = ind / fname
        if not path.exists():
            print(f"skip {fname} (not found)")
            continue
        rows = read_jsonl(path)
        for out_name, fn in renderers:
            md = fn(rows)
            (outd / out_name).write_text(md, encoding="utf-8")
            print(f"wrote {outd / out_name}")

    # ---- P1 (Chunk 8) tables ----------------------------------------------
    p1d = Path(args.p1_dir)
    p1_jobs = [
        ("table1_spatial_qa.jsonl",  ("table1_spatial_qa.md", table1_spatial_qa)),
        ("table2_token_probe.jsonl", ("table2_token_probe_full.md", table2_full)),
        ("table3_relation_probe.jsonl", ("table3_relation_probe.md", table3_relation_probe)),
        ("table4_open_loop.jsonl",   ("table4_open_loop.md", table4_open_loop)),
        ("table5_bench2drive.jsonl", ("table5_bench2drive.md", table5_bench2drive)),
        ("table6_efficiency.jsonl",  ("table6_efficiency.md", table6_efficiency)),
    ]
    for fname, (out_name, fn) in p1_jobs:
        path = p1d / fname
        if not path.exists():
            print(f"skip P1 {fname} (not found)")
            continue
        rows = read_jsonl(path)
        md = fn(rows)
        (outd / out_name).write_text(md, encoding="utf-8")
        print(f"wrote {outd / out_name}")

    # ---- Shortcut audit ----------------------------------------------------
    sd = Path(args.shortcut_dir)
    if sd.exists():
        all_rows: list[dict] = []
        for p in sorted(sd.glob("*.jsonl")):
            all_rows.extend(read_jsonl(p))
        if all_rows:
            md = shortcut_audit_table(all_rows)
            (outd / "shortcut_audit.md").write_text(md, encoding="utf-8")
            print(f"wrote {outd / 'shortcut_audit.md'}")


if __name__ == "__main__":
    main()
