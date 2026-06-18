#!/usr/bin/env bash
# Initialize third_party/SpaceDrive (+ nested unidepth submodule) and write
# THIRD_PARTY_LICENSES.md. Run this ONCE on the GPU box before training
# baselines 3 / 4 / 18.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d "third_party/SpaceDrive/.git" ]]; then
  echo "third_party/SpaceDrive already present; updating submodules…"
  git submodule update --init --recursive third_party/SpaceDrive
else
  echo "Adding third_party/SpaceDrive as a git submodule…"
  git submodule add https://github.com/zhenghao2519/SpaceDrive third_party/SpaceDrive
  git submodule update --init --recursive third_party/SpaceDrive
fi

SPACEDRIVE_COMMIT="$(git -C third_party/SpaceDrive rev-parse --short HEAD)"
UNIDEPTH_COMMIT="$(git -C third_party/SpaceDrive/unidepth rev-parse --short HEAD 2>/dev/null || echo 'n/a')"

cat > THIRD_PARTY_LICENSES.md <<EOF
# Third-party code

| Project | Commit | License | Used for |
|---|---|---|---|
| SpaceDrive | ${SPACEDRIVE_COMMIT} | MIT | Baseline 3 / 4 / 18 (paper §4.4); Bench2Drive eval (paper §4.5) |
| UniDepth (nested submodule) | ${UNIDEPTH_COMMIT} | CC BY-NC 4.0 | Frozen monocular depth in baselines 3 / 4 / 12 |

These dependencies are pulled as submodules and pinned to the commits above. We
do not modify upstream code; baseline wrappers live under
\`geodistill/baselines/\`.

Source URLs:
- SpaceDrive: https://github.com/zhenghao2519/SpaceDrive
- UniDepth: https://github.com/lpiccinelli-eth/UniDepth
EOF

echo "Wrote THIRD_PARTY_LICENSES.md (SpaceDrive=${SPACEDRIVE_COMMIT}, UniDepth=${UNIDEPTH_COMMIT})."
