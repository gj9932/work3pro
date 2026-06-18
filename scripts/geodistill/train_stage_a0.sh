#!/usr/bin/env bash
# Stage A0: allocation + ray + relation warm-up (paper §3.13).
# GPU box only — uses Qwen2.5-VL features through nuScenes_qwen_dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
CFG="${CFG:-configs/geodistill/geodistill_qwen25vl_nuscenes.yaml}"
OUT="${OUT:-runs/geodistill/stage_a0}"
mkdir -p "${OUT}"
exec python -m geodistill.trainers.train_stage_a0 --config "${CFG}" --out_dir "${OUT}" "$@"
