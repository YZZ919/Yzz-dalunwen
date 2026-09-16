#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: $0 SEQUENCE_NAME /absolute/path/to/asl_sequence [speed] [duration_sec] [config_path] [run_label]" >&2
  exit 2
fi

sequence="$1"
sequence_root="$2"
speed="${3:-1.0}"
duration="${4:-0}"
work="/mnt/d/大论文实验/IR-VIO的复现"
config_path="${5:-$work/config/vins_mono_euroc_no_loop.yaml}"
run_label="${6:-vins_mono}"
run_dir="$work/results/$sequence"
mkdir -p "$run_dir"

if [[ ! -f "$sequence_root/mav0/cam0/data.csv" || ! -f "$sequence_root/mav0/imu0/data.csv" ]]; then
  echo "ASL sequence not found or incomplete: $sequence_root" >&2
  exit 2
fi

source /opt/ros/noetic/setup.bash
source "$work/devel/setup.bash"

cleanup() {
  [[ -n "${launch_pid:-}" ]] && kill "$launch_pid" 2>/dev/null || true
  [[ -n "${core_pid:-}" ]] && kill "$core_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

rm -f "$work/results/vins_result.csv" "$work/results/adaptive_weights.csv"
roscore >"$run_dir/${run_label}_roscore.log" 2>&1 &
core_pid=$!
sleep 3

roslaunch vins_estimator euroc.launch \
  config_path:="$config_path" \
  vins_path:="$work" \
  >"$run_dir/${run_label}.log" 2>&1 &
launch_pid=$!
sleep 5

set +e
python3 "$work/scripts/publish_euroc_asl.py" "$sequence_root" \
  --speed "$speed" --duration "$duration" \
  >"$run_dir/${run_label}_replay.log" 2>&1
replay_status=$?
set -e
sleep 10

if [[ "$replay_status" -ne 0 ]]; then
  echo "ASL replay failed with status $replay_status" >&2
  exit "$replay_status"
fi
if [[ ! -s "$work/results/vins_result.csv" ]]; then
  echo "Estimator produced no trajectory." >&2
  exit 1
fi

cp "$work/results/vins_result.csv" "$run_dir/${run_label}.csv"
if [[ -s "$work/results/adaptive_weights.csv" ]]; then
  cp "$work/results/adaptive_weights.csv" "$run_dir/${run_label}_weights.csv"
fi
printf 'sequence=%s\nspeed=%s\nduration=%s\nconfig=%s\nlabel=%s\n' \
  "$sequence" "$speed" "$duration" "$config_path" "$run_label" >"$run_dir/${run_label}_run_meta.txt"
echo "Saved $run_dir/${run_label}.csv"
