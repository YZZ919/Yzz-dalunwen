#!/usr/bin/env bash
set -euo pipefail

# Run this script inside WSL.  The target lives inside the project so the
# Windows host and WSL share the exact same training environment without
# touching a system-wide Python installation.
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${workspace}/.deps/torch_cuda"
mkdir -p "${target}"

python3 -m pip install \
  --target "${target}" \
  --upgrade \
  --no-cache-dir \
  --progress-bar off \
  --index-url https://download.pytorch.org/whl/cu124 \
  'torch==2.4.1'

PYTHONPATH="${target}" python3 - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
PY
