#!/usr/bin/env python3
"""Compare an ONNXRuntime output with a saved PyTorch reference tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    image = np.load(args.input).astype(np.float32)
    reference = np.load(args.reference).astype(np.float32)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: image})[0]
    difference = np.abs(output - reference)
    report = {
        "status": "onnxruntime_consistency_pass",
        "onnx": str(args.onnx.resolve()),
        "input": str(args.input.resolve()),
        "reference": str(args.reference.resolve()),
        "providers": session.get_providers(),
        "input_shape": list(image.shape),
        "output_shape": list(output.shape),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "reference_mean": float(reference.mean()),
        "onnx_mean": float(output.mean()),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "onnx_min": float(output.min()),
        "onnx_max": float(output.max()),
    }
    # The exported graph and the PyTorch reference are expected to agree well
    # below image quantization error; fail loudly if that contract breaks.
    if report["max_abs_difference"] > 1e-5:
        report["status"] = "onnxruntime_consistency_fail"
        code = 1
    else:
        code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
