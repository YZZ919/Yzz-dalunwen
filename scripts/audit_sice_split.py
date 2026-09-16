#!/usr/bin/env python3
"""Audit scene/sample separation in the locally materialized SICE mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def samples(root: Path, split: str):
    sample_root = root / split / "samples"
    result = []
    if not sample_root.exists():
        return result
    for directory in sorted(sample_root.iterdir()):
        if not directory.is_dir():
            continue
        images = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        result.append({
            "sample_id": directory.name,
            "relative_images": [str(p.relative_to(root)).replace("\\", "/") for p in images],
            "image_names": [p.name for p in images],
            "bytes": sum(p.stat().st_size for p in images),
            "sha256": {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in images},
        })
    return result


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=workspace / "datasets" / "sice_mirror_low1")
    parser.add_argument("--output", type=Path, default=workspace / "results" / "sice_split_audit.json")
    args = parser.parse_args()
    train = samples(args.root, "train")
    test = samples(args.root, "test")
    train_ids = {item["sample_id"] for item in train}
    test_ids = {item["sample_id"] for item in test}
    train_images = [image for item in train for image in item["relative_images"]]
    test_images = [image for item in test for image in item["relative_images"]]
    report = {
        "status": "sice_split_audit_complete",
        "root": str(args.root.resolve()),
        "train_samples": len(train),
        "test_samples": len(test),
        "train_images": len(train_images),
        "test_images": len(test_images),
        "overlapping_sample_ids": sorted(train_ids & test_ids),
        "duplicate_image_paths": sorted(set(train_images) & set(test_images)),
        "train_image_name_counts": {name: sum(name in item["image_names"] for item in train) for name in sorted({name for item in train for name in item["image_names"]})},
        "test_image_name_counts": {name: sum(name in item["image_names"] for item in test) for name in sorted({name for item in test for name in item["image_names"]})},
        "train": train,
        "test": test,
        "warning": "This checks the local mirror only; it cannot establish equivalence to the official SICE Part1 2422/600 scene split.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "train_samples", "test_samples", "train_images", "test_images",
        "overlapping_sample_ids", "duplicate_image_paths",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
