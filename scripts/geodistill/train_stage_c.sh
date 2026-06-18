#!/usr/bin/env bash
# Stage C: robustness distillation (paper §3.13).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
CFG="${CFG:-configs/geodistill/geodistill_qwen25vl_nuscenes.yaml}"
OUT="${OUT:-runs/geodistill/stage_c}"
RESUME="${RESUME:-runs/geodistill/stage_b/last.pt}"
mkdir -p "${OUT}"
exec python -m geodistill.trainers.train_stage_c --config "${CFG}" --out_dir "${OUT}" --resume "${RESUME}" "$@"
