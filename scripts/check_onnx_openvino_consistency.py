#!/usr/bin/env python3
"""Compare ONNXRuntime and OpenVINO outputs for one fixed input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help=".npy tensor in NCHW float32 format")
    parser.add_argument("--output", type=Path, default=workspace / "results" / "onnx_openvino_consistency.json")
    args = parser.parse_args()
    try:
        import onnxruntime as ort  # pylint: disable=import-outside-toplevel
        import openvino as ov  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise SystemExit("Set PYTHONPATH to .deps/onnxruntime_windows;.deps/openvino_windows") from exc
    tensor = np.asarray(np.load(args.input), dtype=np.float32)
    ort_session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    ort_output = np.asarray(ort_session.run(None, {input_name: tensor})[0], dtype=np.float32)
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(args.model)), "CPU")
    ov_result = compiled({compiled.input(0): tensor})
    ov_output = np.asarray(next(iter(ov_result.values())), dtype=np.float32)
    difference = np.abs(ort_output - ov_output)
    report = {
        "status": "onnx_openvino_consistency_pass",
        "model": str(args.model.resolve()),
        "input": str(args.input.resolve()),
        "shape": list(ort_output.shape),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "ort_min": float(ort_output.min()),
        "ort_max": float(ort_output.max()),
        "openvino_min": float(ov_output.min()),
        "openvino_max": float(ov_output.max()),
        "tolerance_max_abs": 1e-4,
    }
    if report["max_abs"] > report["tolerance_max_abs"]:
        report["status"] = "onnx_openvino_consistency_fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"].endswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
