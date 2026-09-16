#!/usr/bin/env bash
set -eo pipefail

work="/mnt/d/大论文实验/IR-VIO的复现"
wheel_dir="$work/tmp/fast_inference_wheels"
onnx_dir="$work/.deps/onnx_linux"
mkdir -p "$wheel_dir" "$onnx_dir"

python3 -m pip download --only-binary=:all: --no-deps \
  onnx==1.16.2 protobuf==5.27.5 -d "$wheel_dir"

for wheel in "$wheel_dir"/*.whl; do
  python3 -m zipfile -e "$wheel" "$onnx_dir"
done

python3 "$work/scripts/export_gray_dce_onnx.py"
echo "Exported $work/.deps/gray_dce_cleanroom_480x752.onnx"
