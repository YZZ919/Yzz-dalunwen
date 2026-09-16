#!/usr/bin/env python3
"""Prepare a grayscale SICE-style tree for clean-room DCE training.

The official SICE download is commonly distributed as flat image folders,
while public mirrors may use paired ``sample_*/label.jpg`` and ``low*.jpg``
folders.  This script selects one deterministic low-light input per paired
sample (``low1`` when present), converts it to 8-bit grayscale, and writes a
portable flat tree with a manifest.  It never edits the source tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def select_inputs(source: Path, selection: str = "low1") -> list[Path]:
    sample_dirs = sorted(
        path
        for path in source.rglob("*")
        if path.is_dir() and path.name.lower().startswith("sample_")
    )
    if sample_dirs:
        selected: list[Path] = []
        for sample_dir in sample_dirs:
            images = image_files(sample_dir)
            low = [
                path
                for path in images
                if path.stem.lower().startswith(("low", "input", "dark"))
            ]
            if selection == "all_low":
                selected.extend(sorted(low))
                continue
            low1 = [path for path in images if path.stem.lower() == "low1"]
            candidates = low1 or sorted(low)
            if not candidates:
                candidates = [
                    path
                    for path in images
                    if path.stem.lower() not in {"label", "gt", "groundtruth", "reference"}
                ]
            if candidates:
                selected.append(sorted(candidates)[0])
        return selected
    return [
        path
        for path in image_files(source)
        if path.stem.lower() not in {"label", "gt", "groundtruth", "reference"}
    ]


def choose_source(root: Path, split: str, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    names = [split]
    if split == "val":
        names.extend(["test", "valid", "validation"])
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"在 {root} 下找不到 {split}/（或 test/）源目录")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_split(
    source: Path,
    target: Path,
    split: str,
    limit: int | None,
    overwrite: bool,
    selection: str,
):
    inputs = select_inputs(source, selection=selection)
    if limit:
        inputs = inputs[: max(1, int(limit))]
    if not inputs:
        raise FileNotFoundError(f"源目录没有可选图像：{source}")
    target.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    for index, source_path in enumerate(inputs, start=1):
        output_path = target / f"{index:05d}.jpg"
        if overwrite or not output_path.exists():
            with Image.open(source_path) as source_image:
                gray = source_image.convert("L")
                gray.save(output_path, quality=95, optimize=True)
        with Image.open(output_path) as converted:
            width, height = converted.size
        manifest_rows.append(
            {
                "split": split,
                "index": index,
                "source": str(source_path),
                "output": str(output_path),
                "width": width,
                "height": height,
                "sha256": sha256(output_path),
            }
        )
    return manifest_rows


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=workspace / "datasets" / "sice_gray")
    parser.add_argument("--train-source", type=Path)
    parser.add_argument("--val-source", type=Path)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument(
        "--selection",
        choices=("low1", "all_low"),
        default="low1",
        help="one low1 image per sample (default) or all low/input/dark images",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    train_source = choose_source(source_root, "train", args.train_source)
    val_source = choose_source(source_root, "val", args.val_source)
    train_rows = convert_split(
        train_source, args.output_root / "train", "train", args.limit_train,
        args.overwrite, args.selection
    )
    val_rows = convert_split(
        val_source, args.output_root / "val", "val", args.limit_val,
        args.overwrite, args.selection
    )
    rows = train_rows + val_rows
    manifest_path = args.output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "status": "sice_gray_prepared_for_cleanroom_training",
        "source_root": str(source_root),
        "train_source": str(train_source.resolve()),
        "val_source": str(val_source.resolve()),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "selection": (
            "one low1 image per sample_* directory; flat sources keep all non-label images"
            if args.selection == "low1"
            else "all low/input/dark images per sample_* directory; flat sources keep all non-label images"
        ),
        "conversion": "PIL grayscale L, saved as JPEG quality 95",
        "warning": "A mirror-derived split is not automatically identical to the original SICE Part1 split.",
    }
    (args.output_root / "dataset_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
