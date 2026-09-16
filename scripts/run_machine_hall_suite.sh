#!/usr/bin/env bash
set -euo pipefail

work="/mnt/d/大论文实验/IR-VIO的复现"
data_root="${EUROC_DATA_ROOT:-$work/datasets}"
sequences=("$@")
if [[ ${#sequences[@]} -eq 0 ]]; then
  sequences=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult)
fi

for sequence in "${sequences[@]}"; do
  sequence_root="$data_root/$sequence"
  result_root="$work/results/$sequence"
  groundtruth="$sequence_root/mav0/state_groundtruth_estimate0/data.csv"
  camera_csv="$sequence_root/mav0/cam0/data.csv"
  mkdir -p "$result_root"

  if [[ ! -s "$result_root/vins_mono_asl.csv" ]]; then
    bash "$work/scripts/run_sequence_asl.sh" \
      "$sequence" "$sequence_root" 1.0 0 \
      "$work/config/vins_mono_euroc_no_loop.yaml" vins_mono_asl
  fi
  python3 "$work/scripts/evaluate_ate.py" \
    --estimate "$result_root/vins_mono_asl.csv" \
    --groundtruth "$groundtruth" --camera-csv "$camera_csv" --interpolate \
    --output "$result_root/vins_mono_asl_metrics.json"

  if [[ ! -s "$result_root/irvio_weighting_asl.csv" ]]; then
    bash "$work/scripts/run_sequence_asl.sh" \
      "$sequence" "$sequence_root" 1.0 0 \
      "$work/config/irvio_weighting_euroc_no_loop.yaml" irvio_weighting_asl
  fi
  python3 "$work/scripts/evaluate_ate.py" \
    --estimate "$result_root/irvio_weighting_asl.csv" \
    --groundtruth "$groundtruth" --camera-csv "$camera_csv" --interpolate \
    --output "$result_root/irvio_weighting_asl_metrics.json"
done
