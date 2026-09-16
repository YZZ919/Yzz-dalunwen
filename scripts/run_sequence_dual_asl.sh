#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 3 || $# -gt 7 ]]; then
  echo "Usage: $0 SEQUENCE_NAME /absolute/path/to/asl_sequence /absolute/path/to/enhanced_images [speed] [duration_sec] [config_path] [run_label]" >&2
  exit 2
fi

sequence="$1"
sequence_root="$2"
enhanced_dir="$3"
speed="${4:-1.0}"
duration="${5:-0}"
work="/mnt/d/大论文实验/IR-VIO的复现"
config_path="${6:-$work/config/irvio_dual_proxy_euroc_no_loop.yaml}"
run_label="${7:-irvio_dual_proxy}"
run_dir="$work/results/$sequence"
mkdir -p "$run_dir"

if [[ ! -f "$sequence_root/mav0/cam0/data.csv" || ! -f "$sequence_root/mav0/imu0/data.csv" ]]; then
  echo "ASL sequence not found or incomplete: $sequence_root" >&2
  exit 2
fi
if [[ ! -d "$enhanced_dir" ]]; then
  echo "Enhanced image directory not found: $enhanced_dir" >&2
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

rm -f "$work/results/vins_result.csv" "$work/results/adaptive_weights.csv" "$work/results/bnr_frames.csv" "$work/results/tracking_stats.csv"
if grep -q '^tracking_stats_log_path:' "$config_path" 2>/dev/null; then
  printf 'timestamp,raw_flow_attempts,raw_flow_success,raw_flow_after_f,enhanced_flow_attempts,enhanced_flow_success,enhanced_flow_after_f,enhanced_before_bnr,enhanced_removed_bnr,enhanced_after_bnr,published\\n' \
    > "$work/results/tracking_stats.csv"
fi
roscore >"$run_dir/${run_label}_roscore.log" 2>&1 &
core_pid=$!
sleep 3

roslaunch vins_estimator euroc_dual.launch \
  config_path:="$config_path" \
  vins_path:="$work" \
  >"$run_dir/${run_label}.log" 2>&1 &
launch_pid=$!
sleep 5

# Do not release the first frame until both image subscribers have completed
# the ROS handshake.  A fixed sleep alone can occasionally drop one frame at
# startup, which changes the feature-track path on difficult sequences.
for topic in /cam0/image_raw /cam0/image_enhanced; do
  ready=0
  for attempt in $(seq 1 30); do
    if rostopic info "$topic" 2>/dev/null | grep -q "dual_feature_tracker"; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "Timed out waiting for dual_feature_tracker subscriber on $topic" >&2
    exit 1
  fi
done

set +e
python3 "$work/scripts/publish_euroc_asl.py" "$sequence_root" \
  --enhanced-dir "$enhanced_dir" --speed "$speed" --duration "$duration" \
  >"$run_dir/${run_label}_replay.log" 2>&1
replay_status=$?
set -e
sleep 10

# Stop the launch before copying logs so buffered estimator diagnostics are
# flushed to disk. The trajectory itself is written incrementally, but the
# adaptive-weight stream remains open for the lifetime of the node.
kill "$launch_pid" 2>/dev/null || true
wait "$launch_pid" 2>/dev/null || true
launch_pid=""

if [[ "$replay_status" -ne 0 ]]; then
  echo "ASL dual replay failed with status $replay_status" >&2
  exit "$replay_status"
fi
if [[ ! -s "$work/results/vins_result.csv" ]]; then
  echo "Estimator produced no trajectory." >&2
  exit 1
fi

cp "$work/results/vins_result.csv" "$run_dir/${run_label}.csv"
for artifact in adaptive_weights bnr_frames tracking_stats; do
  if [[ -s "$work/results/${artifact}.csv" ]]; then
    cp "$work/results/${artifact}.csv" "$run_dir/${run_label}_${artifact}.csv"
  fi
done
printf 'sequence=%s\nspeed=%s\nduration=%s\nconfig=%s\nlabel=%s\nenhanced_dir=%s\n' \
  "$sequence" "$speed" "$duration" "$config_path" "$run_label" "$enhanced_dir" \
  >"$run_dir/${run_label}_run_meta.txt"
echo "Saved $run_dir/${run_label}.csv"
