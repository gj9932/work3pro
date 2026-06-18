#!/usr/bin/env bash
# Run paper §4.5 evaluations end-to-end. By default runs against the synthetic
# fixture (TBD-skeleton for GPU-only evaluations); pass --source nuscenes on
# the GPU box together with --ckpt path to produce real numbers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SOURCE="${SOURCE:-synthetic}"
CKPT="${CKPT:-}"
SCENES="${SCENES:-8}"
OUT="${OUT:-runs/geodistill/eval}"
mkdir -p "${OUT}"

PY="${PY:-.venv_p0/bin/python}"
COMMON="--source ${SOURCE} --scenes ${SCENES}"

set -x
${PY} -m geodistill.eval.eval_token_geometry_probe ${COMMON} --hidden 64 --epochs 200 --with_model \
    --out "${OUT}/table2_token_probe.jsonl"
${PY} -m geodistill.eval.eval_relation_probe ${COMMON} --teacher hard_quantile \
    --out "${OUT}/table3_relation_probe.jsonl"
${PY} -m geodistill.eval.eval_spatial_qa --source "${SOURCE}" ${CKPT:+--ckpt "${CKPT}"} \
    --out "${OUT}/table1_spatial_qa.jsonl"
${PY} -m geodistill.eval.eval_open_loop_planning --source "${SOURCE}" ${CKPT:+--ckpt "${CKPT}"} \
    --out "${OUT}/table4_open_loop.jsonl"
${PY} -m geodistill.eval.eval_bench2drive --source "${SOURCE}" ${CKPT:+--ckpt "${CKPT}"} \
    --out "${OUT}/table5_bench2drive.jsonl"
${PY} -m geodistill.eval.eval_efficiency ${COMMON} --out "${OUT}/table6_efficiency.jsonl"
${PY} -m geodistill.eval.eval_robustness ${COMMON} --teacher hard_quantile \
    --out "${OUT}/robustness.jsonl"
${PY} -m scripts.geodistill.run_shortcut_audit ${COMMON} --teacher full_ntl_fgt \
    --out "${OUT}/shortcut/all.jsonl"
set +x

${PY} -m scripts.geodistill.collect_tables \
    --in_dir runs/geodistill/p0 \
    --p1_dir "${OUT}" \
    --shortcut_dir "${OUT}/shortcut" \
    --out_dir runs/geodistill/tables

echo "All Tables 1-6 + shortcut audit rendered under runs/geodistill/tables/"
