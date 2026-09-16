#!/usr/bin/env python3
"""Train the clean-room grayscale DCE model described by IR-VIO.

This is an independent implementation of the published architectural
description (one grayscale input, eight curve maps, curve-estimation scale
16).  It is not author code and must not be labelled as an official IR-VIO
checkpoint.  The default ``official`` loss profile follows the numerical
scaling used by the public Zero-DCE implementation; ``paper`` keeps the
weights written in the paper discussion.

The script intentionally keeps the EuRoC data out of training.  It scans a
``train`` and ``val`` directory (or a paired SICE mirror's ``train`` and
``test`` directories), converts every selected image to one grayscale channel
and trains with the zero-reference losses only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def _load_torch(torch_path: Path | None):
    """Import the workspace-local PyTorch installation after path injection."""

    if torch_path is not None:
        sys.path.insert(0, str(torch_path))
    try:
        import torch  # pylint: disable=import-outside-toplevel
        import torch.nn.functional as functional  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - message is for local setup
        raise RuntimeError(
            "无法导入 PyTorch。请在 WSL 中安装 CUDA/CPU PyTorch，或用 "
            "--torch-path 指向工作目录下的 .deps/zero_dce_cpu。"
        ) from exc
    return torch, functional


def _find_split(root: Path, split: str) -> Path | None:
    candidates = [
        root / split,
        root / split.lower(),
        root / split.upper(),
    ]
    if split == "val":
        candidates.extend([root / "valid", root / "validation", root / "test"])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_images(split_root: Path) -> list[Path]:
    """Collect inputs from flat SICE images or paired ``sample_*`` folders.

    The Hugging Face SICE mirror stores ``label.jpg`` plus several low-light
    images in each sample directory.  A single deterministic ``low1`` image is
    selected per sample so the default run does not silently multiply samples
    or train on the reference label.  Flat directories include every image
    except obvious reference/label files.
    """

    sample_dirs = sorted(
        path
        for path in split_root.rglob("*")
        if path.is_dir() and path.name.lower().startswith("sample_")
    )
    if sample_dirs:
        selected: list[Path] = []
        for sample_dir in sample_dirs:
            images = _image_files(sample_dir)
            low = [
                path
                for path in images
                if path.stem.lower().startswith(("low", "input", "dark"))
            ]
            if low:
                selected.append(sorted(low)[0])
            else:
                fallback = [
                    path
                    for path in images
                    if path.stem.lower() not in {"label", "gt", "groundtruth", "reference"}
                ]
                if fallback:
                    selected.append(sorted(fallback)[0])
        return selected

    return [
        path
        for path in _image_files(split_root)
        if path.stem.lower() not in {"label", "gt", "groundtruth", "reference"}
    ]


class GrayImageDataset:
    """Small dependency-free dataset adapter used by the training loop."""

    def __init__(
        self,
        torch,
        paths: Sequence[Path],
        crop_size: int,
        train: bool,
        seed: int,
        preprocess: str = "legacy_crop",
    ):
        self.torch = torch
        self.paths = list(paths)
        self.crop_size = int(crop_size)
        self.train = bool(train)
        self.seed = int(seed)
        self.preprocess = str(preprocess)

    def __len__(self) -> int:
        return len(self.paths)

    def _crop_or_resize(self, image: Image.Image, index: int) -> Image.Image:
        size = self.crop_size
        width, height = image.size
        if self.preprocess == "official_resize":
            # Zero-DCE's public dataloader resizes every image to its fixed
            # ``size`` (256 in the released code), with no random crop.
            return image.resize((size, size), RESAMPLE_BILINEAR)
        if self.train and width >= size and height >= size:
            rng = random.Random(self.seed + index * 1009)
            left = rng.randint(0, width - size)
            top = rng.randint(0, height - size)
            return image.crop((left, top, left + size, top + size))
        if width != size or height != size:
            return image.resize((size, size), RESAMPLE_BILINEAR)
        return image

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        with Image.open(image_path) as source:
            image = source.convert("L")
            image = self._crop_or_resize(image, index)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(array).unsqueeze(0)
        return tensor, str(image_path)


def _stack_batch(torch, samples: Iterable[tuple[object, str]]):
    tensors: list[object] = []
    names: list[str] = []
    for tensor, name in samples:
        tensors.append(tensor)
        names.append(name)
    if not tensors:
        return None, names
    return torch.stack(tensors, dim=0), names


class SpatialConsistencyLoss:
    def __init__(self, torch, functional):
        self.torch = torch
        self.functional = functional
        kernels = torch.tensor(
            [
                [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
                [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.kernels = kernels
        self.pool = torch.nn.AvgPool2d(4)

    def to(self, device):
        self.kernels = self.kernels.to(device)
        self.pool = self.pool.to(device)
        return self

    def __call__(self, enhanced, original):
        enhanced_small = self.pool(enhanced)
        original_small = self.pool(original)
        grad_enhanced = self.functional.conv2d(enhanced_small, self.kernels, padding=1)
        grad_original = self.functional.conv2d(original_small, self.kernels, padding=1)
        return ((grad_enhanced - grad_original) ** 2).mean()


class OfficialSpatialConsistencyLoss(SpatialConsistencyLoss):
    """Numerical form of public Zero-DCE ``Myloss.L_spa`` for grayscale.

    The released loss returns the four directional squared tensors summed
    together; ``lowlight_train.py`` applies ``torch.mean`` afterwards.  The
    historical clean-room class averaged across the four directions as well,
    which is a factor of four smaller.
    """

    def __call__(self, enhanced, original):
        enhanced_small = self.pool(self.torch.mean(enhanced, dim=1, keepdim=True))
        original_small = self.pool(self.torch.mean(original, dim=1, keepdim=True))
        terms = []
        for index in range(4):
            grad_enhanced = self.functional.conv2d(
                enhanced_small, self.kernels[index:index + 1], padding=1
            )
            grad_original = self.functional.conv2d(
                original_small, self.kernels[index:index + 1], padding=1
            )
            terms.append((grad_original - grad_enhanced) ** 2)
        return self.torch.stack(terms, dim=0).sum(dim=0).mean()


class ExposureLoss:
    def __init__(self, torch, patch_size: int = 16, target: float = 0.6):
        self.torch = torch
        self.pool = torch.nn.AvgPool2d(int(patch_size))
        self.target = float(target)

    def to(self, device):
        self.pool = self.pool.to(device)
        return self

    def __call__(self, enhanced):
        mean = self.pool(enhanced)
        return ((mean - self.target) ** 2).mean()


class IlluminationSmoothnessLoss:
    def __call__(self, curve_map):
        dh = (curve_map[:, :, 1:, :] - curve_map[:, :, :-1, :]) ** 2
        dw = (curve_map[:, :, :, 1:] - curve_map[:, :, :, :-1]) ** 2
        return 2.0 * (dh.mean() + dw.mean())


class OfficialIlluminationSmoothnessLoss:
    """Numerical form of public Zero-DCE ``Myloss.L_TV``.

    Spatial counts are normalized per batch, while the eight curve channels
    remain summed.  This is eight times the historical channel-mean form for
    an 8-channel parameter map.
    """

    def __init__(self, torch):
        self.torch = torch

    def __call__(self, curve_map):
        batch_size = curve_map.shape[0]
        height = curve_map.shape[2]
        width = curve_map.shape[3]
        count_h = (height - 1) * width
        count_w = height * (width - 1)
        h_tv = (curve_map[:, :, 1:, :] - curve_map[:, :, :height - 1, :]).pow(2).sum()
        w_tv = (curve_map[:, :, :, 1:] - curve_map[:, :, :, :width - 1]).pow(2).sum()
        return 2.0 * (h_tv / count_h + w_tv / count_w) / batch_size


def _profile_weights(name: str) -> tuple[float, float, float]:
    if name == "paper":
        return 1.0, 1.0, 20.0
    if name == "official":
        return 1.0, 10.0, 200.0
    raise ValueError(f"unknown loss profile: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_all(torch, seed: int, deterministic: bool = False):
    """Seed every RNG used by this dependency-light training loop."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def _official_weights_init(torch, module):
    """Match the public Zero-DCE ``weights_init`` for Conv layers."""
    if module.__class__.__name__.find("Conv") != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif module.__class__.__name__.find("BatchNorm") != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


def _model_forward(model, images, tv_resolution: str = "full"):
    if tv_resolution == "low" and hasattr(model, "forward_with_low_curve"):
        enhanced, curve_map, tv_curve = model.forward_with_low_curve(images)
        return enhanced, curve_map, tv_curve
    enhanced, curve_map = model(images)
    return enhanced, curve_map, curve_map


def _loss_components(
    torch,
    spatial,
    exposure,
    smooth,
    model,
    images,
    weights,
    loss_compute_fp32: bool = False,
    tv_resolution: str = "full",
):
    enhanced, curve_map, tv_curve = _model_forward(model, images, tv_resolution)
    loss_images = enhanced.float() if loss_compute_fp32 else enhanced
    loss_original = images.float() if loss_compute_fp32 else images
    loss_curve = tv_curve.float() if loss_compute_fp32 else tv_curve
    # ``.float()`` alone does not disable CUDA autocast: convolution/pooling
    # inside the loss modules can still be dispatched in FP16 when the caller
    # is inside the model's autocast context.  Keep the forward pass mixed
    # precision, but make the small-difference reductions unambiguously FP32.
    if loss_compute_fp32:
        with torch.autocast(device_type=loss_images.device.type, enabled=False):
            loss_spa = spatial(loss_images, loss_original)
            loss_exp = exposure(loss_images)
            loss_tv = smooth(loss_curve)
            loss = weights[0] * loss_spa + weights[1] * loss_exp + weights[2] * loss_tv
    else:
        loss_spa = spatial(loss_images, loss_original)
        loss_exp = exposure(loss_images)
        loss_tv = smooth(loss_curve)
        loss = weights[0] * loss_spa + weights[1] * loss_exp + weights[2] * loss_tv
    return loss, loss_spa, loss_exp, loss_tv, enhanced, curve_map


def _autocast_context(torch, enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _rng_state(torch):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(torch, state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _data_manifest(paths: Sequence[Path], workspace: Path):
    manifest = []
    for path in paths:
        try:
            relative = str(path.resolve().relative_to(workspace.resolve()))
        except ValueError:
            relative = str(path.resolve())
        manifest.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return manifest


def _epoch_order(length: int, epoch: int, seed: int, shuffle: bool):
    order = list(range(length))
    if shuffle:
        # A per-epoch local RNG makes the order reproducible and independent
        # of validation/preview calls or the number of workers.
        random.Random(int(seed) + int(epoch) * 1000003).shuffle(order)
    return order


def _save_checkpoint(
    torch,
    path: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    metrics: dict,
    config: dict,
    best_val_loss: float,
):
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "val_loss": float(metrics["val_loss"]),
        "loss_spa": float(metrics["val_spa"]),
        "loss_exp": float(metrics["val_exp"]),
        "loss_tv": float(metrics["val_tv"]),
        "best_val_loss": float(best_val_loss),
        "rng_state": _rng_state(torch),
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _save_preview(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(clipped, mode="L").save(path)


def _evaluate(
    torch,
    model,
    paths: Sequence[Path],
    crop_size: int,
    batch_size: int,
    spatial,
    exposure,
    smooth,
    weights,
    device,
    amp_enabled: bool,
    seed: int,
    preprocess: str = "legacy_crop",
    loss_compute_fp32: bool = False,
    tv_resolution: str = "full",
):
    model.eval()
    totals = {"loss": 0.0, "spa": 0.0, "exp": 0.0, "tv": 0.0, "count": 0}
    preview = None
    dataset = GrayImageDataset(
        torch, paths, crop_size, train=False, seed=seed, preprocess=preprocess
    )
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            images, names = _stack_batch(torch, samples)
            if images is None:
                continue
            images = images.to(device, non_blocking=True)
            with _autocast_context(torch, amp_enabled):
                loss, loss_spa, loss_exp, loss_tv, enhanced, curve_map = _loss_components(
                    torch,
                    spatial,
                    exposure,
                    smooth,
                    model,
                    images,
                    weights,
                    loss_compute_fp32=loss_compute_fp32,
                    tv_resolution=tv_resolution,
                )
            count = len(names)
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["spa"] += float(loss_spa.detach().cpu()) * count
            totals["exp"] += float(loss_exp.detach().cpu()) * count
            totals["tv"] += float(loss_tv.detach().cpu()) * count
            totals["count"] += count
            if preview is None:
                preview = {
                    "input": images[0, 0].detach().float().cpu().numpy(),
                    "enhanced": enhanced[0, 0].detach().float().cpu().numpy(),
                    "curve0": curve_map[0, 0].detach().float().cpu().numpy(),
                    "source": names[0],
                }
    count = max(1, totals["count"])
    return {
        "val_loss": totals["loss"] / count,
        "val_spa": totals["spa"] / count,
        "val_exp": totals["exp"] / count,
        "val_tv": totals["tv"] / count,
        "preview": preview,
    }


def _parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent.parent
    default_torch = workspace / ".deps" / "torch_cuda"
    if not default_torch.exists():
        default_torch = workspace / ".deps" / "zero_dce_cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=workspace / "datasets" / "sice_gray")
    parser.add_argument(
        "--output-dir", type=Path,
        default=workspace / "checkpoints" / "gray_dce_cleanroom" / "gray_official",
    )
    parser.add_argument("--loss-profile", choices=["paper", "official"], default="official")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--downsample-scale", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--exposure-target", type=float, default=0.6)
    parser.add_argument("--exposure-patch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-path", type=Path, default=default_torch)
    parser.add_argument("--init-rgb-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--amp", action="store_true", help="enable CUDA float16 autocast")
    parser.add_argument(
        "--implementation-profile",
        choices=["legacy", "reference"],
        default="legacy",
        help="legacy preserves historical clean-room behavior; reference enables audited reproducibility controls",
    )
    parser.add_argument(
        "--loss-normalization",
        choices=["legacy", "official"],
        default="legacy",
        help="legacy averages directional/channel axes; official matches public Myloss.py scale",
    )
    parser.add_argument(
        "--preprocess",
        choices=["legacy_crop", "official_resize"],
        default="legacy_crop",
        help="legacy random crop-or-resize, or public Zero-DCE fixed resize",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.1)
    parser.add_argument(
        "--initialization",
        choices=["torch_default", "zerodce_official"],
        default="torch_default",
    )
    parser.add_argument(
        "--tv-resolution",
        choices=["full", "low"],
        default="full",
        help="evaluate curve TV after upsample, or on the scale-reduced map",
    )
    parser.add_argument(
        "--reproducible",
        action="store_true",
        help="seed Python/NumPy/PyTorch and shuffle the training order per epoch",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="also request deterministic CUDA kernels (may reduce throughput)",
    )
    parser.add_argument(
        "--loss-compute-fp32",
        action="store_true",
        help="compute zero-reference losses in FP32 when forward uses AMP",
    )
    parser.add_argument(
        "--allow-nonfinite-grad",
        action="store_true",
        help="skip a batch with non-finite gradients instead of failing (diagnostic only)",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="allow a new run to write into a non-empty output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    torch, functional = _load_torch(args.torch_path)
    reproducible = bool(args.reproducible or args.implementation_profile == "reference")
    if reproducible:
        _seed_all(torch, args.seed, deterministic=args.deterministic)
    else:
        # Preserve the historical script's behavior when explicitly using the
        # legacy profile, while still making Python/NumPy deterministic.
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume and not args.allow_overwrite:
        raise FileExistsError(
            f"输出目录非空，为避免覆盖历史实验请换新目录或显式传 --allow-overwrite: {args.output_dir}"
        )

    workspace = Path(__file__).resolve().parent.parent
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from enhance_gray_dce_cleanroom import (  # pylint: disable=import-outside-toplevel
        build_downsampled_network,
        fold_rgb_checkpoint,
    )

    train_root = _find_split(args.data_root, "train")
    val_root = _find_split(args.data_root, "val")
    if train_root is None or val_root is None:
        raise FileNotFoundError(
            f"需要 {args.data_root}/train 和 {args.data_root}/val（也接受 val→test）。"
        )
    train_paths = collect_images(train_root)
    val_paths = collect_images(val_root)
    if args.limit_train:
        train_paths = train_paths[: max(1, args.limit_train)]
    if args.limit_val:
        val_paths = val_paths[: max(1, args.limit_val)]
    if not train_paths or not val_paths:
        raise FileNotFoundError(
            f"数据目录中没有可用图像：train={len(train_paths)}, val={len(val_paths)}"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求 CUDA，但当前 PyTorch 看不到 CUDA；请改用 --device cpu 检查环境。")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        print("warning: --amp 仅在 CUDA 上启用，当前改为关闭 AMP")

    model = build_downsampled_network(torch, args.downsample_scale).to(device)
    if args.initialization == "zerodce_official":
        model.apply(lambda module: _official_weights_init(torch, module))
    if args.init_rgb_checkpoint:
        fold_rgb_checkpoint(torch, model, args.init_rgb_checkpoint)
        model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay)
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    elif hasattr(torch.cuda, "amp"):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    else:
        scaler = None
    if scaler is None:
        amp_enabled = False

    start_epoch = 1
    best_val = math.inf
    if args.resume:
        payload = torch.load(str(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val = float(payload.get("best_val_loss", payload.get("val_loss", math.inf)))
        _restore_rng_state(torch, payload.get("rng_state"))

    weights = _profile_weights(args.loss_profile)
    if args.loss_normalization == "official":
        spatial = OfficialSpatialConsistencyLoss(torch, functional).to(device)
        smooth = OfficialIlluminationSmoothnessLoss(torch)
    else:
        spatial = SpatialConsistencyLoss(torch, functional).to(device)
        smooth = IlluminationSmoothnessLoss()
    exposure = ExposureLoss(torch, args.exposure_patch_size, args.exposure_target).to(device)
    train_dataset = GrayImageDataset(
        torch,
        train_paths,
        args.crop_size,
        train=True,
        seed=args.seed,
        preprocess=args.preprocess,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "status": "cleanroom_grayscale_training_not_official_irvio",
        "data_root": str(args.data_root),
        "train_root": str(train_root),
        "val_root": str(val_root),
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "loss_profile": args.loss_profile,
        "implementation_profile": args.implementation_profile,
        "loss_normalization": args.loss_normalization,
        "loss_weights": {"spa": weights[0], "exp": weights[1], "tv": weights[2]},
        "input_channels": 1,
        "curve_channels": 8,
        "curve_iterations": 8,
        "downsample_scale": int(args.downsample_scale),
        "crop_size": int(args.crop_size),
        "preprocess": args.preprocess,
        "batch_size": int(args.batch_size),
        "optimizer": "Adam",
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip_norm": float(args.grad_clip_norm),
        "initialization": args.initialization,
        "tv_resolution": args.tv_resolution,
        "epochs_requested": int(args.epochs),
        "exposure_target": float(args.exposure_target),
        "exposure_patch_size": int(args.exposure_patch_size),
        "device": str(device),
        "amp": amp_enabled,
        "loss_compute_fp32": bool(args.loss_compute_fp32),
        "allow_nonfinite_grad": bool(args.allow_nonfinite_grad),
        "seed": int(args.seed),
        "reproducible": reproducible,
        "deterministic": bool(args.deterministic),
        "shuffle_each_epoch": reproducible,
        "init_rgb_checkpoint": str(args.init_rgb_checkpoint) if args.init_rgb_checkpoint else None,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "script_fingerprints": {
            "train_gray_dce_cleanroom.py": _sha256(Path(__file__).resolve()),
            "enhance_gray_dce_cleanroom.py": _sha256(scripts_dir / "enhance_gray_dce_cleanroom.py"),
        },
        "data_manifest": {
            "train": _data_manifest(train_paths, workspace),
            "val": _data_manifest(val_paths, workspace),
        },
    }
    config_path = args.output_dir / "training_config.json"
    if not config_path.exists() or not args.resume:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    history_path = args.output_dir / "history.csv"
    history_exists = history_path.exists() and start_epoch > 1
    history_file = history_path.open("a" if history_exists else "w", newline="", encoding="utf-8")
    history_writer = csv.DictWriter(
        history_file,
        fieldnames=[
            "epoch", "train_loss", "train_spa", "train_exp", "train_tv",
            "train_weighted_spa", "train_weighted_exp", "train_weighted_tv",
            "grad_norm_mean", "grad_norm_max", "nonfinite_grad_batches",
            "val_loss", "val_spa", "val_exp", "val_tv", "seconds",
        ],
    )
    if not history_exists:
        history_writer.writeheader()

    print(json.dumps({"status": "started", **config}, ensure_ascii=False))
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            train_totals = {
                "loss": 0.0,
                "spa": 0.0,
                "exp": 0.0,
                "tv": 0.0,
                "count": 0,
                "grad_sum": 0.0,
                "grad_max": 0.0,
                "grad_count": 0,
                "nonfinite_grad_batches": 0,
            }
            epoch_started = __import__("time").perf_counter()
            order = _epoch_order(
                len(train_dataset), epoch, args.seed, shuffle=reproducible
            )
            for start in range(0, len(train_dataset), args.batch_size):
                batch_indices = order[start:min(start + args.batch_size, len(train_dataset))]
                samples = [
                    train_dataset[index]
                    for index in batch_indices
                ]
                images, names = _stack_batch(torch, samples)
                if images is None:
                    continue
                images = images.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(torch, amp_enabled):
                    loss, loss_spa, loss_exp, loss_tv, enhanced, curve_map = _loss_components(
                        torch,
                        spatial,
                        exposure,
                        smooth,
                        model,
                        images,
                        weights,
                        loss_compute_fp32=args.loss_compute_fp32,
                        tv_resolution=args.tv_resolution,
                    )
                if not bool(torch.isfinite(loss).item()) or not bool(torch.isfinite(enhanced).all().item()) or not bool(torch.isfinite(curve_map).all().item()):
                    raise FloatingPointError(f"epoch {epoch}: non-finite loss on {names[0]}")
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradients_finite = all(
                        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                        for parameter in model.parameters()
                    )
                    if not gradients_finite:
                        train_totals["nonfinite_grad_batches"] += 1
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        if not args.allow_nonfinite_grad:
                            raise FloatingPointError(
                                f"epoch {epoch}: non-finite gradient on {names[0]}"
                            )
                        continue
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm
                    )
                    if not bool(torch.isfinite(grad_norm).item()):
                        train_totals["nonfinite_grad_batches"] += 1
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        if not args.allow_nonfinite_grad:
                            raise FloatingPointError(
                                f"epoch {epoch}: non-finite gradient norm on {names[0]}"
                            )
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    gradients_finite = all(
                        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                        for parameter in model.parameters()
                    )
                    if not gradients_finite:
                        train_totals["nonfinite_grad_batches"] += 1
                        optimizer.zero_grad(set_to_none=True)
                        if not args.allow_nonfinite_grad:
                            raise FloatingPointError(
                                f"epoch {epoch}: non-finite gradient on {names[0]}"
                            )
                        continue
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm
                    )
                    if not bool(torch.isfinite(grad_norm).item()):
                        train_totals["nonfinite_grad_batches"] += 1
                        optimizer.zero_grad(set_to_none=True)
                        if not args.allow_nonfinite_grad:
                            raise FloatingPointError(
                                f"epoch {epoch}: non-finite gradient norm on {names[0]}"
                            )
                        continue
                    optimizer.step()
                grad_norm_value = float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm)
                count = len(names)
                train_totals["loss"] += float(loss.detach().cpu()) * count
                train_totals["spa"] += float(loss_spa.detach().cpu()) * count
                train_totals["exp"] += float(loss_exp.detach().cpu()) * count
                train_totals["tv"] += float(loss_tv.detach().cpu()) * count
                train_totals["count"] += count
                train_totals["grad_sum"] += grad_norm_value
                train_totals["grad_max"] = max(train_totals["grad_max"], grad_norm_value)
                train_totals["grad_count"] += 1

            train_count = max(1, train_totals["count"])
            validation = _evaluate(
                torch, model, val_paths, args.crop_size, args.batch_size,
                spatial, exposure, smooth, weights, device, amp_enabled, args.seed,
                preprocess=args.preprocess,
                loss_compute_fp32=args.loss_compute_fp32,
                tv_resolution=args.tv_resolution,
            )
            train_metrics = {
                "train_loss": train_totals["loss"] / train_count,
                "train_spa": train_totals["spa"] / train_count,
                "train_exp": train_totals["exp"] / train_count,
                "train_tv": train_totals["tv"] / train_count,
                "train_weighted_spa": train_totals["spa"] / train_count * weights[0],
                "train_weighted_exp": train_totals["exp"] / train_count * weights[1],
                "train_weighted_tv": train_totals["tv"] / train_count * weights[2],
                "grad_norm_mean": train_totals["grad_sum"] / max(1, train_totals["grad_count"]),
                "grad_norm_max": train_totals["grad_max"],
                "nonfinite_grad_batches": train_totals["nonfinite_grad_batches"],
            }
            row = {
                "epoch": epoch,
                **train_metrics,
                "val_loss": validation["val_loss"],
                "val_spa": validation["val_spa"],
                "val_exp": validation["val_exp"],
                "val_tv": validation["val_tv"],
                "seconds": __import__("time").perf_counter() - epoch_started,
            }
            history_writer.writerow(row)
            history_file.flush()

            metrics = {**row}
            is_best = validation["val_loss"] < best_val
            if is_best:
                best_val = validation["val_loss"]
            if epoch % max(1, args.save_every) == 0 or epoch == args.epochs:
                _save_checkpoint(
                    torch, args.output_dir / f"epoch_{epoch:03d}.pth",
                    model, optimizer, scaler, epoch, metrics, config, best_val,
                )
            if is_best:
                _save_checkpoint(
                    torch, args.output_dir / "best_val.pth",
                    model, optimizer, scaler, epoch, metrics, config, best_val,
                )
                if validation["preview"]:
                    preview = validation["preview"]
                    _save_preview(args.output_dir / "preview_best_input.png", preview["input"])
                    _save_preview(args.output_dir / "preview_best_enhanced.png", preview["enhanced"])
                    curve = (preview["curve0"] + 1.0) / 2.0
                    _save_preview(args.output_dir / "preview_best_curve0.png", curve)
            print(json.dumps({"status": "epoch", **row, "best_val": best_val}, ensure_ascii=False))
    finally:
        history_file.close()
    print(json.dumps({"status": "complete", "best_val": best_val, "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
