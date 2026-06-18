#!/usr/bin/env bash
# Stage A1: A0 + metric depth + coord (paper §3.13).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
CFG="${CFG:-configs/geodistill/geodistill_qwen25vl_nuscenes.yaml}"
OUT="${OUT:-runs/geodistill/stage_a1}"
RESUME="${RESUME:-runs/geodistill/stage_a0/last.pt}"
mkdir -p "${OUT}"
exec python -m geodistill.trainers.train_stage_a1 --config "${CFG}" --out_dir "${OUT}" --resume "${RESUME}" "$@"
