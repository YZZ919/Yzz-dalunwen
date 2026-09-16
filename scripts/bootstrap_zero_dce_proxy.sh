#!/usr/bin/env bash
set -euo pipefail

# Install the optional CPU inference dependency entirely inside this workspace.
# The VINS-Mono baseline and C++ BNR library do not require PyTorch.
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${workspace}/.deps/zero_dce_cpu"

if PYTHONPATH="${target}" python3 -c 'import torch; assert torch.__version__ == "2.4.1+cpu"' 2>/dev/null; then
    echo "Zero-DCE proxy dependency already available at ${target}"
    exit 0
fi

python3 -m pip install \
    --target "${target}" \
    --no-cache-dir \
    --progress-bar off \
    --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.4.1+cpu'

PYTHONPATH="${target}" python3 -c 'import torch; print("installed torch", torch.__version__)'
