#!/usr/bin/env python3
"""Load both trained gray-DCE checkpoints and run a shape/range smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=workspace / "checkpoints" / "gray_dce_cleanroom")
    parser.add_argument("--torch-path", type=Path,
                        default=workspace / ".deps" / "torch_cuda")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(args.torch_path))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch  # pylint: disable=import-outside-toplevel
    from enhance_gray_dce_cleanroom import build_downsampled_network  # pylint: disable=import-outside-toplevel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    sample = torch.rand(2, 1, 64, 80, device=device)
    print({"torch": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available()),
           "device": str(device)})
    for profile in ("official", "paper"):
        checkpoint = args.checkpoint_root / f"gray_{profile}_mirror" / "best_val.pth"
        payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
        model = build_downsampled_network(torch, 16).to(device).eval()
        model.load_state_dict(payload["model"])
        with torch.no_grad():
            enhanced, curve_map = model(sample)
        finite = bool(torch.isfinite(enhanced).all() and torch.isfinite(curve_map).all())
        print({"profile": profile, "epoch": payload.get("epoch"),
               "enhanced_shape": list(enhanced.shape), "curve_shape": list(curve_map.shape),
               "enhanced_min": float(enhanced.min()), "enhanced_max": float(enhanced.max()),
               "curve_min": float(curve_map.min()), "curve_max": float(curve_map.max()),
               "finite": finite})
        if not finite:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
