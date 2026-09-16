#!/usr/bin/env python3
"""Run the official RGB Zero-DCE checkpoint as a labelled grayscale proxy.

This is an integration aid, not the unpublished single-channel enhancement
network described by IR-VIO.  A grayscale frame is copied to three channels,
processed by official Zero-DCE, then averaged back to one channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_network(torch_path: Path, model_file: Path, checkpoint: Path):
    sys.path.insert(0, str(torch_path))
    sys.path.insert(0, str(model_file.parent))
    import torch  # pylint: disable=import-outside-toplevel
    import model  # pylint: disable=import-outside-toplevel

    network = model.enhance_net_nopool().cpu().eval()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    network.load_state_dict(state)
    return torch, network


def enhance_gray(torch, network, gray: np.ndarray):
    rgb_like = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_like).permute(2, 0, 1).unsqueeze(0)
    started = time.perf_counter()
    with torch.no_grad():
        _, enhanced, _ = network(tensor)
    elapsed = time.perf_counter() - started
    enhanced_array = enhanced.squeeze(0).permute(1, 2, 0).numpy()
    enhanced_gray = np.clip(enhanced_array.mean(axis=2) * 255.0, 0, 255).astype(np.uint8)
    return enhanced_gray, elapsed


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--torch-path", type=Path, default=workspace / ".deps" / "zero_dce_cpu"
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=workspace / "references" / "third_party" / "Zero-DCE" / "model.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=workspace
        / "references"
        / "third_party"
        / "Zero-DCE"
        / "Epoch99.pth",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    gray = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read {args.input}")
    torch, network = load_network(args.torch_path, args.model_file, args.checkpoint)
    enhanced_gray, elapsed = enhance_gray(torch, network, gray)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), enhanced_gray):
        raise RuntimeError(f"cannot write {args.output}")

    report = {
        "status": "proxy_only_not_irvio",
        "method": "official RGB Zero-DCE checkpoint; grayscale replicated to RGB and averaged back",
        "input": str(args.input),
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": "cpu",
        "inference_seconds": elapsed,
        "input_mean": float(gray.mean()),
        "output_mean": float(enhanced_gray.mean()),
        "warning": (
            "IR-VIO's modified single-channel architecture and checkpoint are not public; "
            "this output must not be reported as IR-VIO enhancement."
        ),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
