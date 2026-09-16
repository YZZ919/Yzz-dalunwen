#!/usr/bin/env python3
"""Evaluate enhancement range, saturation and texture retention on a split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def image_paths(root: Path):
    return sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def gradient_energy(torch, image):
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    return float((dx.square().mean() + dy.square().mean()).detach().cpu())


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--downsample-scale", type=int, default=16)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-path", type=Path, default=workspace / ".deps" / "torch_cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.torch_path))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch  # pylint: disable=import-outside-toplevel
    from enhance_gray_dce_cleanroom import build_downsampled_network  # pylint: disable=import-outside-toplevel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    model = build_downsampled_network(torch, args.downsample_scale).to(device).eval()
    model.load_state_dict(state_dict)
    paths = image_paths(args.data_root / args.split)
    if not paths:
        raise SystemExit(f"no images under {args.data_root / args.split}")
    input_means = []
    output_means = []
    input_stds = []
    output_stds = []
    input_gradients = []
    output_gradients = []
    input_low = []
    input_high = []
    output_low = []
    output_high = []
    curve_abs_max = []
    finite = True
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]
            images = []
            for path in batch_paths:
                image = Image.open(path).convert("L").resize((args.resize, args.resize), RESAMPLE_BILINEAR)
                images.append(np.asarray(image, dtype=np.float32) / 255.0)
            tensor = torch.from_numpy(np.stack(images)[:, None]).to(device)
            enhanced, curve = model(tensor)
            finite = finite and bool(torch.isfinite(enhanced).all().item()) and bool(torch.isfinite(curve).all().item())
            input_means.extend(tensor.mean(dim=(1, 2, 3)).cpu().tolist())
            output_means.extend(enhanced.mean(dim=(1, 2, 3)).cpu().tolist())
            input_stds.extend(tensor.std(dim=(1, 2, 3)).cpu().tolist())
            output_stds.extend(enhanced.std(dim=(1, 2, 3)).cpu().tolist())
            input_gradients.extend([gradient_energy(torch, item[None]) for item in tensor])
            output_gradients.extend([gradient_energy(torch, item[None]) for item in enhanced])
            input_low.extend((tensor <= 1.0 / 255.0).float().mean(dim=(1, 2, 3)).cpu().tolist())
            input_high.extend((tensor >= 254.0 / 255.0).float().mean(dim=(1, 2, 3)).cpu().tolist())
            output_low.extend((enhanced <= 1.0 / 255.0).float().mean(dim=(1, 2, 3)).cpu().tolist())
            output_high.extend((enhanced >= 254.0 / 255.0).float().mean(dim=(1, 2, 3)).cpu().tolist())
            curve_abs_max.extend(curve.abs().amax(dim=(1, 2, 3)).cpu().tolist())

    def mean(values):
        return float(np.mean(values)) if values else None

    report = {
        "status": "gray_checkpoint_range_audit_complete" if finite else "gray_checkpoint_range_audit_fail",
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "images": len(paths),
        "resize": [args.resize, args.resize],
        "device": str(device),
        "finite": finite,
        "input_mean": mean(input_means),
        "output_mean": mean(output_means),
        "input_std": mean(input_stds),
        "output_std": mean(output_stds),
        "input_gradient_energy": mean(input_gradients),
        "output_gradient_energy": mean(output_gradients),
        "gradient_energy_ratio": mean(output_gradients) / mean(input_gradients) if mean(input_gradients) else None,
        "input_low_saturation_fraction": mean(input_low),
        "input_high_saturation_fraction": mean(input_high),
        "output_low_saturation_fraction": mean(output_low),
        "output_high_saturation_fraction": mean(output_high),
        "curve_abs_max_mean": mean(curve_abs_max),
        "curve_abs_max_max": float(np.max(curve_abs_max)) if curve_abs_max else None,
        "checkpoint_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "checkpoint_val_loss": payload.get("val_loss") if isinstance(payload, dict) else None,
        "warning": "These are split-level enhancement diagnostics, not trajectory metrics or evidence of author IR-VIO weights.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
