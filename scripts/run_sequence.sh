#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 SEQUENCE_NAME /absolute/path/to/sequence.bag [config_path] [run_label]" >&2
  exit 2
fi

sequence="$1"
bag_path="$2"
work="/mnt/d/大论文实验/IR-VIO的复现"
config_path="${3:-$work/config/vins_mono_euroc_no_loop.yaml}"
run_label="${4:-vins_mono}"
run_dir="$work/results/$sequence"
mkdir -p "$run_dir"

if [[ ! -f "$bag_path" ]]; then
  echo "Bag not found: $bag_path" >&2
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

rosbag play --quiet "$bag_path" >"$run_dir/${run_label}_rosbag_play.log" 2>&1
sleep 8

if [[ ! -s "$work/results/vins_result.csv" ]]; then
  echo "Estimator produced no trajectory." >&2
  exit 1
fi

cp "$work/results/vins_result.csv" "$run_dir/${run_label}.csv"
if [[ -s "$work/results/adaptive_weights.csv" ]]; then
  cp "$work/results/adaptive_weights.csv" "$run_dir/${run_label}_weights.csv"
fi
echo "Saved $run_dir/${run_label}.csv"
