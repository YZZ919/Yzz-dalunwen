#!/usr/bin/env python3
"""Run a labelled Zero-DCE+BNR component sample over all EuRoC sequences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np

from block_noise_removal import (
    _detect_points,
    _draw_result,
    block_noise_ratios,
    select_features,
)
from enhance_zero_dce_proxy import enhance_gray, load_network, sha256


SEQUENCES = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]


def find_darkest_sample(sequence_dir: Path, stride: int):
    files = sorted((sequence_dir / "mav0" / "cam0" / "data").glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no camera images under {sequence_dir}")
    candidates = files[::stride]
    best = None
    for path in candidates:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        item = (float(image.mean()), path, image)
        if best is None or item[0] < best[0]:
            best = item
    if best is None:
        raise RuntimeError(f"no readable image under {sequence_dir}")
    return best, len(candidates)


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=workspace / "datasets")
    parser.add_argument(
        "--output", type=Path, default=workspace / "results" / "bnr_proxy_sample_suite"
    )
    parser.add_argument("--scan-stride", type=int, default=20)
    parser.add_argument("--detect-max", type=int, default=150)
    parser.add_argument("--torch-path", type=Path, default=workspace / ".deps" / "zero_dce_cpu")
    parser.add_argument(
        "--model-file",
        type=Path,
        default=workspace / "references" / "third_party" / "Zero-DCE" / "model.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=workspace / "references" / "third_party" / "Zero-DCE" / "Epoch99.pth",
    )
    args = parser.parse_args()
    if args.scan_stride <= 0:
        parser.error("--scan-stride must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    torch, network = load_network(args.torch_path, args.model_file, args.checkpoint)
    records = []
    started = time.perf_counter()
    for sequence in SEQUENCES:
        (raw_mean, raw_path, raw), scanned = find_darkest_sample(
            args.datasets / sequence, args.scan_stride
        )
        enhanced, inference_seconds = enhance_gray(torch, network, raw)
        enhanced_path = args.output / f"{sequence}_dark_sample_enhanced.png"
        cv2.imwrite(str(enhanced_path), enhanced)
        ratios, _, _ = block_noise_ratios(raw, enhanced, 6, 8)
        points = _detect_points(enhanced, args.detect_max)
        keep, _ = select_features(points, ratios, raw.shape, 10.8, 8)
        visualization = _draw_result(raw, enhanced, points, keep, 6, 8)
        visualization_path = args.output / f"{sequence}_dark_sample_bnr.png"
        cv2.imwrite(str(visualization_path), visualization)
        records.append(
            {
                "sequence": sequence,
                "sampled_frames": scanned,
                "selected_frame": raw_path.name,
                "raw_mean": raw_mean,
                "enhanced_mean": float(enhanced.mean()),
                "inference_seconds": inference_seconds,
                "blocks_over_threshold": int(np.count_nonzero(ratios > 10.8)),
                "maximum_noise_ratio": float(np.max(ratios)),
                "features_before": int(len(points)),
                "features_removed": int(np.count_nonzero(~keep)),
                "features_retained": int(np.count_nonzero(keep)),
            }
        )
        print(sequence, records[-1], flush=True)

    summary = {
        "status": "proxy_only_not_irvio",
        "method": "darkest 1-in-20 sampled frame; official RGB Zero-DCE proxy; paper BNR",
        "checkpoint_sha256": sha256(args.checkpoint),
        "grid": "6x8",
        "noise_ratio_threshold": 10.8,
        "minimum_retained_features": 8,
        "sequences": len(records),
        "sequences_with_bnr_removal": sum(r["features_removed"] > 0 for r in records),
        "total_features_before": sum(r["features_before"] for r in records),
        "total_features_removed": sum(r["features_removed"] for r in records),
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
        "warning": "Component proxy only; excluded from IR-VIO trajectory metrics.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({k: summary[k] for k in (
        "status", "sequences", "sequences_with_bnr_removal",
        "total_features_before", "total_features_removed", "elapsed_seconds"
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
