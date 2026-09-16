#!/usr/bin/env python3
"""Audit the numerical scale of the grayscale clean-room DCE losses.

The public Zero-DCE implementation returns a four-direction spatial tensor
and a TV value summed over curve channels.  The first clean-room trainer
averaged both quantities over those axes.  This script uses fixed tensors to
make that difference measurable, including gradients and CUDA AMP versus
FP32 behavior.  It is deliberately read-only with respect to existing
checkpoints; the report is written to a caller-selected new file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


OFFICIAL_URLS = {
    "model.py": "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/model.py",
    "Myloss.py": "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/Myloss.py",
    "lowlight_train.py": "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/lowlight_train.py",
    "dataloader.py": "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/dataloader.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_torch(torch_path: Path):
    if torch_path is not None:
        sys.path.insert(0, str(torch_path))
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn.functional as functional  # pylint: disable=import-outside-toplevel
    return torch, functional


def _kernels(torch):
    return torch.tensor(
        [
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        ],
        dtype=torch.float32,
    ).unsqueeze(1)


def spatial_local(torch, enhanced, original):
    functional = torch.nn.functional
    kernel = _kernels(torch).to(enhanced.device)
    pool = torch.nn.AvgPool2d(4).to(enhanced.device)
    e = pool(enhanced)
    o = pool(original)
    ge = functional.conv2d(e, kernel, padding=1)
    go = functional.conv2d(o, kernel, padding=1)
    return ((ge - go) ** 2).mean()


def spatial_official(torch, enhanced, original):
    functional = torch.nn.functional
    """Myloss.L_spa followed by the caller's torch.mean (gray C=1)."""
    kernel = _kernels(torch).to(enhanced.device)
    pool = torch.nn.AvgPool2d(4).to(enhanced.device)
    # Official code first averages channels.  For C=1 this is a no-op.
    e = pool(torch.mean(enhanced, dim=1, keepdim=True))
    o = pool(torch.mean(original, dim=1, keepdim=True))
    terms = []
    for index in range(4):
        ge = functional.conv2d(e, kernel[index:index + 1], padding=1)
        go = functional.conv2d(o, kernel[index:index + 1], padding=1)
        terms.append((go - ge) ** 2)
    # Myloss returns a tensor with the four terms added; lowlight_train then
    # applies torch.mean.  The factor of four is therefore intentional.
    return torch.stack(terms, dim=0).sum(dim=0).mean()


def exposure_local(torch, enhanced, target=0.6, patch_size=16):
    pool = torch.nn.AvgPool2d(patch_size).to(enhanced.device)
    return ((pool(enhanced) - target) ** 2).mean()


def exposure_official(torch, enhanced, target=0.6, patch_size=16):
    pool = torch.nn.AvgPool2d(patch_size).to(enhanced.device)
    gray = torch.mean(enhanced, dim=1, keepdim=True)
    return torch.mean((pool(gray) - target) ** 2)


def tv_local(torch, curve):
    dh = (curve[:, :, 1:, :] - curve[:, :, :-1, :]) ** 2
    dw = (curve[:, :, :, 1:] - curve[:, :, :, :-1]) ** 2
    return 2.0 * (dh.mean() + dw.mean())


def tv_official(torch, curve):
    batch = curve.shape[0]
    h = curve.shape[2]
    w = curve.shape[3]
    count_h = (h - 1) * w
    count_w = h * (w - 1)
    h_tv = (curve[:, :, 1:, :] - curve[:, :, :h - 1, :]).pow(2).sum()
    w_tv = (curve[:, :, :, 1:] - curve[:, :, :, :w - 1]).pow(2).sum()
    return 2.0 * (h_tv / count_h + w_tv / count_w) / batch


def _grad_norm(torch, tensors):
    total = 0.0
    for tensor in tensors:
        if tensor.grad is not None:
            total += float(tensor.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def _scalar(value) -> float:
    return float(value.detach().float().cpu())


def _ratio(a: float, b: float) -> float | None:
    if abs(b) < 1e-30:
        return None
    return a / b


def _loss_pair(torch, fn_local, fn_official, *args):
    left = [arg.detach().clone().requires_grad_(True) for arg in args]
    right = [arg.detach().clone().requires_grad_(True) for arg in args]
    local = fn_local(torch, *left)
    official = fn_official(torch, *right)
    local.backward()
    official.backward()
    return {
        "local": _scalar(local),
        "official": _scalar(official),
        "ratio_official_over_local": _ratio(_scalar(official), _scalar(local)),
        "local_grad_norm": _grad_norm(torch, left),
        "official_grad_norm": _grad_norm(torch, right),
        "ratio_grad_norm": _ratio(_grad_norm(torch, right), _grad_norm(torch, left)),
    }


def _amp_report(torch, functional, seed: int):
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda.is_available() is false"}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    # Import only after the CUDA check so CPU-only environments remain useful.
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from enhance_gray_dce_cleanroom import build_downsampled_network  # pylint: disable=import-outside-toplevel

    image = torch.rand((1, 1, 65, 79), device=device)
    base = build_downsampled_network(torch, 16).to(device)
    base.eval()
    state = {key: value.detach().clone() for key, value in base.state_dict().items()}

    def run(amp: bool, strict_loss_fp32: bool = False):
        model = build_downsampled_network(torch, 16).to(device)
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            enhanced, curve = model(image)
            if strict_loss_fp32:
                with torch.autocast(device_type="cuda", enabled=False):
                    spa = spatial_local(torch, enhanced.float(), image.float())
                    exp = exposure_local(torch, enhanced.float())
                    tv = tv_local(torch, curve.float())
                    total = spa + 10.0 * exp + 200.0 * tv
            else:
                spa = spatial_local(torch, enhanced, image)
                exp = exposure_local(torch, enhanced)
                tv = tv_local(torch, curve)
                total = spa + 10.0 * exp + 200.0 * tv
        total.backward()
        grads = [parameter for parameter in model.parameters()]
        return {
            "total": _scalar(total),
            "spa": _scalar(spa),
            "exp": _scalar(exp),
            "tv": _scalar(tv),
            "grad_norm": _grad_norm(torch, grads),
            "enhanced_mean": _scalar(enhanced.mean()),
            "curve_abs_max": _scalar(curve.abs().max()),
        }

    fp32 = run(False)
    amp = run(True)
    amp_strict = run(True, strict_loss_fp32=True)
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "fp32": fp32,
        "amp_all_ops": amp,
        "amp_strict_fp32_loss": amp_strict,
        "relative_total_difference": abs(amp["total"] - fp32["total"]) / max(abs(fp32["total"]), 1e-12),
        "relative_grad_norm_difference": abs(amp["grad_norm"] - fp32["grad_norm"]) / max(abs(fp32["grad_norm"]), 1e-12),
        "strict_loss_relative_total_difference": abs(amp_strict["total"] - fp32["total"]) / max(abs(fp32["total"]), 1e-12),
        "strict_loss_relative_grad_norm_difference": abs(amp_strict["grad_norm"] - fp32["grad_norm"]) / max(abs(fp32["grad_norm"]), 1e-12),
        "note": "The trainer keeps the forward pass under AMP but disables autocast for loss reductions when --loss-compute-fp32 is set.",
    }


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-path", type=Path, default=workspace / ".deps" / "torch_cuda")
    parser.add_argument("--output", type=Path, default=workspace / "results" / "loss_scale_audit_20260911.json")
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    torch, functional = _load_torch(args.torch_path)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Draw on CPU so the fixed tensors are identical on CPU and CUDA runs;
    # moving the already-created values avoids a generator/device mismatch.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    original = torch.rand((2, 1, 64, 80), generator=generator).to(device)
    enhanced = torch.rand((2, 1, 64, 80), generator=generator).to(device)
    curve = (torch.rand((2, 8, 10, 13), generator=generator) * 2.0 - 1.0).to(device)

    spatial = _loss_pair(torch, spatial_local, spatial_official, enhanced, original)
    exposure = _loss_pair(torch, exposure_local, exposure_official, enhanced)
    smooth = _loss_pair(torch, tv_local, tv_official, curve)

    components = {
        "paper": {
            "spa": spatial["local"],
            "exp": exposure["local"],
            "tv": smooth["local"],
            "weights": {"spa": 1.0, "exp": 1.0, "tv": 20.0},
        },
        "official_scaling": {
            "spa": spatial["local"],
            "exp": exposure["local"],
            "tv": smooth["local"],
            "weights": {"spa": 1.0, "exp": 10.0, "tv": 200.0},
        },
        "official_code_scale_with_paper_written_weights": {
            "spa": spatial["official"],
            "exp": exposure["official"],
            "tv": smooth["official"],
            "weights": {"spa": 1.0, "exp": 1.0, "tv": 20.0},
        },
    }
    for profile in components.values():
        profile["weighted_contributions"] = {
            name: profile[name] * profile["weights"][name]
            for name in ("spa", "exp", "tv")
        }
        profile["weighted_total"] = sum(profile["weighted_contributions"].values())

    local_files = {}
    for relative in (
        "scripts/enhance_gray_dce_cleanroom.py",
        "scripts/train_gray_dce_cleanroom.py",
        "references/third_party/Zero-DCE/model.py",
    ):
        path = workspace / relative
        if path.exists():
            local_files[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}

    report: dict[str, Any] = {
        "status": "loss_scale_audit_complete",
        "timestamp_local": "2026-09-11",
        "seed": args.seed,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "fixed_shapes": {
            "original": list(original.shape),
            "enhanced": list(enhanced.shape),
            "curve": list(curve.shape),
        },
        "spatial": spatial,
        "exposure": exposure,
        "tv": smooth,
        "weighted_profiles": components,
        "amp_vs_fp32": _amp_report(torch, functional, args.seed),
        "official_source_urls": OFFICIAL_URLS,
        "local_file_fingerprints": local_files,
        "interpretation": {
            "spatial": "For grayscale C=1, official L_spa returns four directional terms and the trainer applies mean; the local mean over the four stacked directions is therefore 4x smaller.",
            "exposure": "For grayscale C=1, local and official exposure losses are algebraically equivalent.",
            "tv": "Official L_TV sums over curve channels before spatial normalization; averaging over 8 channels locally makes it 8x smaller for an 8-channel map.",
            "scale16": "The supplied IR-VIO main PDF visibly specifies single-channel input and pixelwise curve maps but does not expose a downsample=16 implementation detail; scale=16 remains a clean-room hypothesis pending supplementary/source evidence.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
