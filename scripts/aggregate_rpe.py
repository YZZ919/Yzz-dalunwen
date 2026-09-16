#!/usr/bin/env python3
"""Aggregate fixed-interval RPE for several labels and EuRoC sequences."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from evaluate_rpe import (
    align_se3,
    interpolate_groundtruth,
    load_pose_csv,
    quat_to_matrix,
    rotation_error_angle,
)


def compute(estimate_path: Path, gt_path: Path, delta_sec: float, max_pair_dt_sec: float) -> dict:
    estimate = load_pose_csv(estimate_path)
    groundtruth = load_pose_csv(gt_path)
    gt_times = groundtruth[:, 0]
    keep = (estimate[:, 0] >= gt_times[0]) & (estimate[:, 0] <= gt_times[-1])
    estimate = estimate[keep]
    if len(estimate) < 3:
        raise ValueError("fewer than three estimate poses overlap ground truth")
    gt_positions, gt_quaternions = interpolate_groundtruth(groundtruth, estimate[:, 0])
    aligned_positions, alignment_rotation = align_se3(estimate[:, 1:4], gt_positions)
    estimate_rotations = np.asarray([quat_to_matrix(row[4:8]) for row in estimate])
    gt_rotations = np.asarray([quat_to_matrix(row) for row in gt_quaternions])
    aligned_rotations = np.asarray([alignment_rotation @ matrix for matrix in estimate_rotations])
    times = estimate[:, 0]
    translation_errors = []
    rotation_errors = []
    for index, time in enumerate(times):
        target_time = time + delta_sec
        right = int(np.searchsorted(times, target_time, side="left"))
        candidates = [candidate for candidate in (right - 1, right) if 0 <= candidate < len(times)]
        if not candidates:
            continue
        target_index = min(candidates, key=lambda candidate: abs(times[candidate] - target_time))
        pair_error = abs(times[target_index] - target_time)
        if target_index <= index or pair_error > max_pair_dt_sec:
            continue
        est_t = aligned_rotations[index].T @ (aligned_positions[target_index] - aligned_positions[index])
        gt_t = gt_rotations[index].T @ (gt_positions[target_index] - gt_positions[index])
        translation_errors.append(float(np.linalg.norm(est_t - gt_t)))
        est_r = aligned_rotations[index].T @ aligned_rotations[target_index]
        gt_r = gt_rotations[index].T @ gt_rotations[target_index]
        rotation_errors.append(rotation_error_angle(gt_r.T @ est_r))
    if not translation_errors:
        raise ValueError("no valid pairs")
    translation_errors = np.asarray(translation_errors)
    rotation_errors = np.asarray(rotation_errors)
    return {
        "status": "complete",
        "pairs": int(len(translation_errors)),
        "overlap_estimate_poses": int(len(estimate)),
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_rpe_mean_m": float(np.mean(translation_errors)),
        "translation_rpe_median_m": float(np.median(translation_errors)),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(rotation_errors ** 2))),
        "rotation_rpe_mean_rad": float(np.mean(rotation_errors)),
        "rotation_rpe_rmse_deg": float(np.degrees(np.sqrt(np.mean(rotation_errors ** 2)))),
        "rotation_rpe_mean_deg": float(np.degrees(np.mean(rotation_errors))),
    }


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=workspace / "results")
    parser.add_argument("--datasets-root", type=Path, default=workspace / "datasets")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--delta-sec", type=float, default=1.0)
    parser.add_argument("--max-pair-dt-sec", type=float, default=0.06)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for label in args.labels:
        for sequence in args.sequences:
            estimate = args.results_root / sequence / f"{label}.csv"
            gt = args.datasets_root / sequence / "mav0" / "state_groundtruth_estimate0" / "data.csv"
            item = {"label": label, "sequence": sequence, "estimate": str(estimate.resolve())}
            if not estimate.exists() or not gt.exists():
                item.update({"status": "missing", "error": "estimate or groundtruth not found"})
            else:
                try:
                    item.update(compute(estimate, gt, args.delta_sec, args.max_pair_dt_sec))
                except Exception as exc:  # keep all sequence failures in the report
                    item.update({"status": "failed", "error": str(exc)})
            rows.append(item)

    summaries = []
    for label in args.labels:
        complete = [row for row in rows if row["label"] == label and row["status"] == "complete"]
        t = [row["translation_rpe_rmse_m"] for row in complete]
        r = [row["rotation_rpe_rmse_deg"] for row in complete]
        summaries.append({
            "label": label,
            "complete": len(complete),
            "requested": len(args.sequences),
            "mean_translation_rpe_rmse_m": statistics.mean(t) if t else None,
            "std_translation_rpe_rmse_m": statistics.pstdev(t) if len(t) > 1 else (0.0 if t else None),
            "mean_rotation_rpe_rmse_deg": statistics.mean(r) if r else None,
            "std_rotation_rpe_rmse_deg": statistics.pstdev(r) if len(r) > 1 else (0.0 if r else None),
        })
    report = {
        "status": "rpe_aggregate_complete",
        "delta_sec": args.delta_sec,
        "max_pair_dt_sec": args.max_pair_dt_sec,
        "alignment": "single SE(3) Umeyama alignment per sequence before relative errors",
        "warning": "RPE is computed from the local VINS CSV convention; the paper's TSR definition remains unspecified.",
        "summaries": summaries,
        "rows": rows,
    }
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = sorted({key for row in rows for key in row})
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
