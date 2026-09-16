#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
label="${1:-dual_gray_dce_reference_official_11seq_tracking}"
log_path="logs/brightness_${label}_20260911.txt"
sequences=(
  MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult
  V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult
)
for sequence in "${sequences[@]}"; do
  python3 scripts/evaluate_temporal_brightness.py \
    --sequence-root "datasets/${sequence}" \
    --enhanced-dir "results/${sequence}/enhanced_${label}" \
    --output "results/${sequence}/${label}_brightness.json" \
    >>"${log_path}" 2>&1
done
