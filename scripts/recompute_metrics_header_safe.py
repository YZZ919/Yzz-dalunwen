#!/usr/bin/env python3
"""Recompute all stored ATE JSON files with header-safe trajectory loading."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SEQUENCES = (
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
    "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
)


def first_existing(directory: Path, candidates: tuple[tuple[str, str], ...]):
    for estimate_name, metric_name in candidates:
        estimate = directory / estimate_name
        metric = directory / metric_name
        if estimate.is_file() and metric.is_file():
            return estimate, metric
    return None


def evaluate(evaluator: Path, estimate: Path, groundtruth: Path, camera: Path, output: Path):
    command = [
        sys.executable,
        str(evaluator),
        "--estimate", str(estimate),
        "--groundtruth", str(groundtruth),
        "--camera-csv", str(camera),
        "--interpolate",
        "--output", str(output),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    evaluator = workspace / "scripts" / "evaluate_ate.py"
    for sequence in SEQUENCES:
        sequence_dir = workspace / "results" / sequence
        dataset_dir = workspace / "datasets" / sequence / "mav0"
        groundtruth = dataset_dir / "state_groundtruth_estimate0" / "data.csv"
        camera = dataset_dir / "cam0" / "data.csv"
        if not groundtruth.is_file() or not camera.is_file():
            raise SystemExit(f"missing ground truth/camera data for {sequence}")

        baseline = first_existing(
            sequence_dir,
            (("vins_mono_asl.csv", "vins_mono_asl_metrics.json"),
             ("vins_mono.csv", "metrics.json")),
        )
        weighting = first_existing(
            sequence_dir,
            (("irvio_weighting_asl.csv", "irvio_weighting_asl_metrics.json"),
             ("irvio_weighting.csv", "irvio_weighting_metrics.json")),
        )
        dual = (sequence_dir / "dual_gray_dce_full.csv",
                sequence_dir / "dual_gray_dce_full_ate.json")
        for name, pair in (("baseline", baseline), ("weighting", weighting), ("dual", dual)):
            if pair is None or not pair[0].is_file():
                raise SystemExit(f"missing {name} trajectory for {sequence}")
            evaluate(evaluator, pair[0], groundtruth, camera, pair[1])

        repeat_estimate = sequence_dir / "dual_gray_dce_repeat_20260911.csv"
        repeat_metrics = sequence_dir / "dual_gray_dce_repeat_20260911_ate.json"
        if repeat_estimate.is_file():
            evaluate(evaluator, repeat_estimate, groundtruth, camera, repeat_metrics)
        print(f"recomputed {sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
