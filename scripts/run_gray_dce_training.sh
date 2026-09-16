#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
torch_path="${workspace}/.deps/torch_cuda"
if [[ ! -d "${torch_path}" ]]; then
  torch_path="${workspace}/.deps/zero_dce_cpu"
fi

data_root="${workspace}/datasets/sice_gray"
output_dir="${workspace}/checkpoints/gray_dce_cleanroom/gray_official_mirror"
log_path="${workspace}/logs/gray_dce_official_mirror_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "${output_dir}" "$(dirname "${log_path}")"
python3 -u "${workspace}/scripts/train_gray_dce_cleanroom.py" \
  --data-root "${data_root}" \
  --output-dir "${output_dir}" \
  --loss-profile official \
  --epochs 100 \
  --batch-size 8 \
  --crop-size 512 \
  --downsample-scale 16 \
  --lr 1e-4 \
  --exposure-target 0.6 \
  --save-every 5 \
  --torch-path "${torch_path}" \
  --device auto \
  --amp 2>&1 | tee "${log_path}"

