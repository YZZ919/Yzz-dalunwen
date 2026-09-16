#!/usr/bin/env python3
"""Measure adjacent-frame brightness change for raw and enhanced EuRoC images."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2
import numpy as np


def image_paths(sequence_root: Path):
    csv_path = sequence_root / "mav0" / "cam0" / "data.csv"
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            try:
                timestamp = int(row[0])
            except (ValueError, IndexError):
                continue
            rows.append((timestamp, sequence_root / "mav0" / "cam0" / "data" / row[1]))
    return rows


def load_mean(path: Path) -> float | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(image.mean()) if image is not None else None


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "p95": None}
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "count": len(values),
        "mean": float(np.mean(absolute)),
        "median": float(np.median(absolute)),
        "std": float(np.std(absolute)),
        "p95": float(np.percentile(absolute, 95)),
    }


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--enhanced-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = image_paths(args.sequence_root)
    raw_means = []
    enhanced_means = []
    timestamps = []
    for timestamp, raw_path in pairs:
        enhanced_path = args.enhanced_dir / raw_path.name
        raw_mean = load_mean(raw_path)
        enhanced_mean = load_mean(enhanced_path)
        if raw_mean is None or enhanced_mean is None:
            continue
        timestamps.append(timestamp)
        raw_means.append(raw_mean)
        enhanced_means.append(enhanced_mean)
    raw_delta = [raw_means[i] - raw_means[i - 1] for i in range(1, len(raw_means))]
    enhanced_delta = [enhanced_means[i] - enhanced_means[i - 1] for i in range(1, len(enhanced_means))]
    report = {
        "status": "temporal_brightness_complete",
        "sequence_root": str(args.sequence_root.resolve()),
        "enhanced_dir": str(args.enhanced_dir.resolve()),
        "frames_compared": len(raw_means),
        "raw_mean_intensity": float(statistics.mean(raw_means)) if raw_means else None,
        "enhanced_mean_intensity": float(statistics.mean(enhanced_means)) if enhanced_means else None,
        "raw_adjacent_delta_abs": summarize(raw_delta),
        "enhanced_adjacent_delta_abs": summarize(enhanced_delta),
        "raw_adjacent_delta_signed_mean": float(statistics.mean(raw_delta)) if raw_delta else None,
        "enhanced_adjacent_delta_signed_mean": float(statistics.mean(enhanced_delta)) if enhanced_delta else None,
        "warning": "Brightness change is a global frame-mean diagnostic; it is not the paper's TSR definition.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
