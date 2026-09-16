#!/usr/bin/env python3
"""Export the documented clean-room grayscale DCE proxy to fixed-size ONNX.

The historical default is full-resolution curve estimation (scale 1).  The
training clean-room path can be exported with ``--downsample-scale 16`` and a
trained ``--weights`` checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path,
        nargs="?", default=workspace / ".deps" / "gray_dce_cleanroom_480x752.onnx",
        help="output ONNX file",
    )
    parser.add_argument(
        "--torch-path", type=Path, default=workspace / ".deps" / "zero_dce_cpu"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=workspace / "references" / "third_party" / "Zero-DCE" / "Epoch99.pth",
    )
    parser.add_argument(
        "--weights", type=Path,
        help="optional clean-room checkpoint containing a model state_dict",
    )
    parser.add_argument(
        "--downsample-scale", type=int, default=1,
        help="curve-estimation downsampling factor; use 16 for the training variant",
    )
    args = parser.parse_args()
    # ONNX is kept in its own directory because installing into the large
    # PyTorch target on an NTFS-mounted WSL workspace can leave partial files.
    sys.path.insert(0, str(workspace / ".deps" / "onnx_linux"))
    sys.path.insert(1, str(args.torch_path))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch  # pylint: disable=import-outside-toplevel
    from enhance_gray_dce_cleanroom import (
        build_downsampled_network,
        fold_rgb_checkpoint,
    )

    class EnhancedOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, image):
            enhanced, _ = self.inner(image)
            return enhanced

    network = build_downsampled_network(torch, args.downsample_scale)
    if args.weights:
        # Reference checkpoints include Python/NumPy/CUDA RNG states for
        # resume reproducibility, which are intentionally not accepted by
        # PyTorch's restrictive weights-only unpickler.  These are local
        # artifacts, so load the complete trusted payload here and extract the
        # model state dict.
        payload = torch.load(str(args.weights), map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        network.load_state_dict(state_dict)
    else:
        fold_rgb_checkpoint(torch, network, args.checkpoint)
    network = EnhancedOnly(network).eval()
    dummy = torch.zeros((1, 1, 480, 752), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        network,
        dummy,
        str(args.output),
        input_names=["image"],
        output_names=["enhanced"],
        opset_version=11,
        do_constant_folding=True,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
