#!/usr/bin/env bash
# Run every invariant test for paper §3.3-3.13 contracts.
# These are CPU-only and must pass on every commit before training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PY="${PY:-.venv_p0/bin/python}"

TESTS=(
  test/geodistill/r2ac_test.py
  test/geodistill/teacher_test.py
  test/geodistill/lpga_test.py
  test/geodistill/injection_test.py
  test/geodistill/relation_test.py
  test/geodistill/losses_test.py
  test/geodistill/trainers_test.py
  test/geodistill/baselines_test.py
  test/geodistill/shortcut_test.py
  test/geodistill/eval_smoke_test.py
)

failures=0
for t in "${TESTS[@]}"; do
  echo "===== ${t} ====="
  if ! ${PY} "${t}"; then
    failures=$((failures + 1))
  fi
done

if [[ ${failures} -gt 0 ]]; then
  echo "FAILED: ${failures} test file(s)"
  exit 1
fi
echo "OK: all paper invariants verified."
