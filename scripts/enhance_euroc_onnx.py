#!/usr/bin/env python3
"""Enhance a EuRoC camera interval with the clean-room grayscale DCE ONNX proxy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np


def camera_rows(sequence: Path):
    csv_path = sequence / "mav0" / "cam0" / "data.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row in reader:
            if row and not row[0].lstrip().startswith("#"):
                yield int(row[0]), row[1]


def read_gray(path: Path):
    """Read through a byte buffer so OpenCV works with Windows Unicode paths."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def write_png(path: Path, image: np.ndarray) -> None:
    encoded, data = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError(f"cannot encode {path}")
    data.tofile(path)


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--model", type=Path,
        default=workspace / ".deps" / "gray_dce_cleanroom_480x752.onnx",
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--backend", choices=("opencv", "onnxruntime", "openvino"), default="opencv"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = list(camera_rows(args.sequence))
    if not rows:
        raise SystemExit("empty EuRoC camera stream")
    start_ns = rows[0][0] + int(max(args.start, 0.0) * 1e9)
    end_ns = start_ns + int(args.duration * 1e9) if args.duration > 0 else None
    selected = [row for row in rows if row[0] >= start_ns and (end_ns is None or row[0] <= end_ns)]
    if not selected:
        raise SystemExit("selected interval contains no images")

    model_bytes = args.model.read_bytes()
    if args.backend == "onnxruntime":
        import onnxruntime as ort  # pylint: disable=import-outside-toplevel

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        network = ort.InferenceSession(
            model_bytes,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        input_name = network.get_inputs()[0].name
    elif args.backend == "openvino":
        import openvino as ov  # pylint: disable=import-outside-toplevel

        core = ov.Core()
        ov_model = core.read_model(model=model_bytes)
        network = core.compile_model(ov_model, "CPU")
        input_name = network.input(0)
        output_name = network.output(0)
    else:
        model_buffer = np.frombuffer(model_bytes, dtype=np.uint8)
        network = cv2.dnn.readNetFromONNX(model_buffer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.sequence / "mav0" / "cam0" / "data"
    timings = []
    input_means = []
    output_means = []
    skipped_existing = 0
    previous_report = None
    if args.report and args.report.is_file():
        try:
            previous_report = json.loads(args.report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_report = None
    started_all = time.perf_counter()
    for index, (_, filename) in enumerate(selected, 1):
        output_path = args.output_dir / filename
        if args.skip_existing and output_path.is_file():
            skipped_existing += 1
            raw = read_gray(input_dir / filename)
            output = read_gray(output_path)
            if raw is None or output is None:
                raise RuntimeError(f"cannot read resumed pair {input_dir / filename} / {output_path}")
            input_means.append(float(raw.mean()))
            output_means.append(float(output.mean()))
            continue
        raw = read_gray(input_dir / filename)
        if raw is None:
            raise RuntimeError(f"cannot read {input_dir / filename}")
        blob = raw.astype(np.float32)[None, None, :, :] / 255.0
        started = time.perf_counter()
        if args.backend == "onnxruntime":
            enhanced = network.run(None, {input_name: blob})[0][0, 0]
        elif args.backend == "openvino":
            enhanced = network([blob])[output_name][0, 0]
        else:
            network.setInput(blob)
            enhanced = network.forward()[0, 0]
        timings.append(time.perf_counter() - started)
        output = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        write_png(output_path, output)
        input_means.append(float(raw.mean()))
        output_means.append(float(output.mean()))
        if index % 100 == 0 or index == len(selected):
            print(f"enhanced {index}/{len(selected)}")

    report = {
        "status": "cleanroom_grayscale_proxy_not_irvio",
        "sequence": str(args.sequence.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "model": str(args.model.resolve()),
        "backend": args.backend,
        "start_sec": args.start,
        "duration_sec": args.duration,
        "frames": len(selected),
        "processed_frames": len(timings),
        "skipped_existing": skipped_existing,
        # Inference timing covers frames processed in this invocation.  When a
        # fully resumed report processes no new frames, retain the prior timing
        # so the aggregate remains comparable while the intensity means below
        # still cover every selected input/output pair.
        "mean_inference_seconds": (
            float(np.mean(timings))
            if timings
            else float((previous_report or {}).get("mean_inference_seconds", 0.0))
        ),
        "timing_scope": "processed_frames_this_run",
        "statistics_scope": "all_selected_frames",
        "total_seconds": time.perf_counter() - started_all,
        "mean_input_intensity": float(np.mean(input_means)),
        "mean_output_intensity": float(np.mean(output_means)),
        "warning": "Clean-room proxy only; the IR-VIO enhancement checkpoint is unavailable.",
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
