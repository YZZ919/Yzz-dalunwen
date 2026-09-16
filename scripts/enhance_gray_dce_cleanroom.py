#!/usr/bin/env python3
"""Reconstruct a single-channel DCE network from the IR-VIO description.

IR-VIO does not publish its architecture details or checkpoint.  This script
uses the official seven-layer Zero-DCE topology, changes 3 input channels to 1
and 24 output curve channels to 8, then deterministically folds the official
RGB weights.  The result is a documented clean-room proxy, not author code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from enhance_zero_dce_proxy import sha256


def build_network(torch):
    nn = torch.nn
    functional = torch.nn.functional

    class GrayEnhanceNet(nn.Module):
        def __init__(self):
            super().__init__()
            channels = 32
            self.e_conv1 = nn.Conv2d(1, channels, 3, 1, 1, bias=True)
            self.e_conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
            self.e_conv3 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
            self.e_conv4 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
            self.e_conv5 = nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=True)
            self.e_conv6 = nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=True)
            self.e_conv7 = nn.Conv2d(channels * 2, 8, 3, 1, 1, bias=True)

        def forward_curve_map(self, image):
            """Estimate the eight pixel-wise curve parameter maps.

            Keeping curve estimation separate from curve application lets the
            training/export path implement IR-VIO's described downsampling
            scale without changing the historical full-resolution proxy
            behavior used by the existing EuRoC results.
            """
            x1 = functional.relu(self.e_conv1(image), inplace=True)
            x2 = functional.relu(self.e_conv2(x1), inplace=True)
            x3 = functional.relu(self.e_conv3(x2), inplace=True)
            x4 = functional.relu(self.e_conv4(x3), inplace=True)
            x5 = functional.relu(self.e_conv5(torch.cat([x3, x4], 1)), inplace=True)
            x6 = functional.relu(self.e_conv6(torch.cat([x2, x5], 1)), inplace=True)
            return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        @staticmethod
        def apply_curve_map(image, curve_map):
            """Apply eight recursive Zero-DCE curves to a grayscale image."""
            curves = torch.chunk(curve_map, 8, dim=1)
            enhanced = image
            for curve in curves:
                enhanced = enhanced + curve * (enhanced.pow(2) - enhanced)
            return enhanced

        def forward(self, image):
            curve_map = self.forward_curve_map(image)
            return self.apply_curve_map(image, curve_map), curve_map

    return GrayEnhanceNet()


def build_downsampled_network(torch, scale=16):
    """Build the clean-room network with low-resolution curve estimation.

    ``scale=16`` follows the IR-VIO description: the feature/curve network
    runs on a 1/16-resolution image, its eight curve maps are bilinearly
    upsampled to the original image size, and the recursive curves are then
    applied at full resolution.  A scale of one is exactly the historical
    full-resolution proxy path.
    """
    import math

    functional = torch.nn.functional
    inner = build_network(torch)
    scale = max(1, int(scale))

    class DownsampledGrayDCE(torch.nn.Module):
        def __init__(self, network, downsample_scale):
            super().__init__()
            self.inner = network
            self.downsample_scale = int(downsample_scale)

        def forward(self, image):
            if self.downsample_scale <= 1:
                return self.inner(image)
            height, width = image.shape[-2:]
            small_height = max(1, math.ceil(height / self.downsample_scale))
            small_width = max(1, math.ceil(width / self.downsample_scale))
            small = functional.interpolate(
                image,
                size=(small_height, small_width),
                mode="bilinear",
                align_corners=False,
            )
            low_curve_map = self.inner.forward_curve_map(small)
            curve_map = functional.interpolate(
                low_curve_map,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            enhanced = self.inner.apply_curve_map(image, curve_map)
            return enhanced, curve_map

        def forward_with_low_curve(self, image):
            """Return enhanced image, full-resolution curves and low curves.

            The extra view is used only by the training audit to test whether
            TV should be evaluated before or after the scale-16 upsample.  The
            regular ``forward`` return signature remains unchanged for all
            historical inference/export scripts.
            """
            if self.downsample_scale <= 1:
                enhanced, curve_map = self.inner(image)
                return enhanced, curve_map, curve_map
            height, width = image.shape[-2:]
            small_height = max(1, math.ceil(height / self.downsample_scale))
            small_width = max(1, math.ceil(width / self.downsample_scale))
            small = functional.interpolate(
                image,
                size=(small_height, small_width),
                mode="bilinear",
                align_corners=False,
            )
            low_curve_map = self.inner.forward_curve_map(small)
            curve_map = functional.interpolate(
                low_curve_map,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            enhanced = self.inner.apply_curve_map(image, curve_map)
            return enhanced, curve_map, low_curve_map

    return DownsampledGrayDCE(inner, scale)


def fold_rgb_checkpoint(torch, network, checkpoint: Path):
    rgb = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    # ``build_downsampled_network`` wraps the same module so the low-resolution
    # curve estimator can be reused.  Fold weights into the actual DCE module
    # in either case; returning the original object preserves the wrapper.
    target = getattr(network, "inner", network)
    gray = target.state_dict()
    for layer in range(2, 7):
        gray[f"e_conv{layer}.weight"] = rgb[f"e_conv{layer}.weight"]
        gray[f"e_conv{layer}.bias"] = rgb[f"e_conv{layer}.bias"]

    # Replicating grayscale to RGB makes the first convolution equal to the
    # sum of its three input kernels.
    gray["e_conv1.weight"] = rgb["e_conv1.weight"].sum(dim=1, keepdim=True)
    gray["e_conv1.bias"] = rgb["e_conv1.bias"]

    # Official output order is eight consecutive RGB curve groups.  Collapse
    # each group to one curve kernel and bias.  Averaging is an explicit proxy
    # assumption; IR-VIO does not publish how its final layer was trained.
    output_weight = rgb["e_conv7.weight"]
    output_bias = rgb["e_conv7.bias"]
    gray["e_conv7.weight"] = output_weight.reshape(8, 3, *output_weight.shape[1:]).mean(dim=1)
    gray["e_conv7.bias"] = output_bias.reshape(8, 3).mean(dim=1)
    target.load_state_dict(gray)
    return network.eval()


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--torch-path", type=Path, default=workspace / ".deps" / "zero_dce_cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=workspace / "references" / "third_party" / "Zero-DCE" / "Epoch99.pth",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.torch_path))
    import torch  # pylint: disable=import-outside-toplevel

    raw = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"cannot read {args.input}")
    network = fold_rgb_checkpoint(torch, build_network(torch), args.checkpoint)
    tensor = torch.from_numpy(raw.astype(np.float32) / 255.0)[None, None, :, :]
    started = time.perf_counter()
    with torch.no_grad():
        enhanced, curve_map = network(tensor)
    elapsed = time.perf_counter() - started
    output = np.clip(enhanced[0, 0].numpy() * 255.0, 0, 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise RuntimeError(f"cannot write {args.output}")

    report = {
        "status": "cleanroom_grayscale_proxy_not_irvio",
        "method": "single-channel seven-layer DCE with eight scalar curve maps",
        "weight_initialization": {
            "conv1": "sum official RGB input-channel kernels",
            "conv2_to_conv6": "copy official kernels and biases",
            "conv7": "mean each consecutive RGB output group into one scalar curve",
        },
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "input": str(args.input),
        "output": str(args.output),
        "input_mean": float(raw.mean()),
        "output_mean": float(output.mean()),
        "curve_map_shape": list(curve_map.shape),
        "inference_seconds": elapsed,
        "device": "cpu",
        "warning": (
            "The topology follows a minimal reading of the paper, but IR-VIO's "
            "training procedure and checkpoint are unavailable. Exclude from official results."
        ),
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
