#!/usr/bin/env python3
"""Evaluate fixed-interval translational and rotational RPE on VINS CSVs.

The estimator CSVs in this workspace contain timestamp, position, quaternion
(w,x,y,z), and velocity.  Ground truth is interpolated to each estimate time;
positions are SE(3)-aligned once before relative errors are formed.  The
interval is explicit (default 1.0 s), so this is not a hidden frame-count
metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_pose_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                values = [float(row[index]) for index in range(8)]
            except (ValueError, IndexError):
                continue
            rows.append(values)
    if not rows:
        raise ValueError(f"no timestamp/pose rows in {path}")
    data = np.asarray(rows, dtype=np.float64)
    data[:, 0] *= 1e-9
    order = np.argsort(data[:, 0], kind="stable")
    return data[order]


def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_to_matrix(q):
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(q0, q1, amount):
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(q0 + amount * (q1 - q0))
    theta = math.acos(dot)
    sine = math.sin(theta)
    return (math.sin((1.0 - amount) * theta) * q0 + math.sin(amount * theta) * q1) / sine


def interpolate_groundtruth(gt, query_times):
    times = gt[:, 0]
    positions = np.column_stack([np.interp(query_times, times, gt[:, column]) for column in (1, 2, 3)])
    quaternions = []
    for query in query_times:
        right = int(np.searchsorted(times, query, side="left"))
        if right <= 0:
            quaternions.append(normalize_quaternion(gt[0, 4:8]))
        elif right >= len(times):
            quaternions.append(normalize_quaternion(gt[-1, 4:8]))
        else:
            left = right - 1
            denominator = max(times[right] - times[left], 1e-15)
            amount = float((query - times[left]) / denominator)
            quaternions.append(slerp(gt[left, 4:8], gt[right, 4:8], amount))
    return positions, np.asarray(quaternions)


def align_se3(estimate, target):
    mu_e = estimate.mean(axis=0)
    mu_t = target.mean(axis=0)
    xe = estimate - mu_e
    xt = target - mu_t
    u, _, vt = np.linalg.svd(xe.T @ xt)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = mu_t - rotation @ mu_e
    return (rotation @ estimate.T).T + translation, rotation


def rotation_error_angle(rotation):
    cosine = min(1.0, max(-1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    return math.acos(cosine)


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--delta-sec", type=float, default=1.0)
    parser.add_argument("--max-pair-dt-sec", type=float, default=0.06)
    parser.add_argument("--output", type=Path, default=workspace / "results" / "rpe.json")
    args = parser.parse_args()
    if args.delta_sec <= 0:
        raise SystemExit("--delta-sec must be positive")

    estimate = load_pose_csv(args.estimate)
    groundtruth = load_pose_csv(args.groundtruth)
    gt_times = groundtruth[:, 0]
    keep = (estimate[:, 0] >= gt_times[0]) & (estimate[:, 0] <= gt_times[-1])
    estimate = estimate[keep]
    if len(estimate) < 3:
        raise SystemExit("fewer than three estimate poses overlap ground truth")
    gt_positions, gt_quaternions = interpolate_groundtruth(groundtruth, estimate[:, 0])
    aligned_positions, alignment_rotation = align_se3(estimate[:, 1:4], gt_positions)
    estimate_rotations = np.asarray([quat_to_matrix(row[4:8]) for row in estimate])
    gt_rotations = np.asarray([quat_to_matrix(row) for row in gt_quaternions])
    # Apply the same global SE(3) alignment to estimate orientations.
    aligned_rotations = np.asarray([alignment_rotation @ matrix for matrix in estimate_rotations])

    times = estimate[:, 0]
    translation_errors = []
    rotation_errors = []
    pair_time_errors = []
    for index, time in enumerate(times):
        target_time = time + args.delta_sec
        right = int(np.searchsorted(times, target_time, side="left"))
        candidates = [candidate for candidate in (right - 1, right) if 0 <= candidate < len(times)]
        if not candidates:
            continue
        target_index = min(candidates, key=lambda candidate: abs(times[candidate] - target_time))
        pair_error = abs(times[target_index] - target_time)
        if target_index <= index or pair_error > args.max_pair_dt_sec:
            continue
        pair_time_errors.append(pair_error)
        est_relative_t = aligned_rotations[index].T @ (aligned_positions[target_index] - aligned_positions[index])
        gt_relative_t = gt_rotations[index].T @ (gt_positions[target_index] - gt_positions[index])
        # Both relative translations are expressed in the source pose frame;
        # alignment is applied to the estimate positions, while GT remains in
        # its own frame.  The global alignment rotation makes the frames agree.
        translation_errors.append(float(np.linalg.norm(est_relative_t - gt_relative_t)))
        est_relative_r = aligned_rotations[index].T @ aligned_rotations[target_index]
        gt_relative_r = gt_rotations[index].T @ gt_rotations[target_index]
        rotation_errors.append(rotation_error_angle(gt_relative_r.T @ est_relative_r))

    if not translation_errors:
        raise SystemExit("no valid pairs; increase --max-pair-dt-sec or check timestamps")
    translation_errors = np.asarray(translation_errors)
    rotation_errors = np.asarray(rotation_errors)
    report = {
        "status": "rpe_complete",
        "estimate": str(args.estimate.resolve()),
        "groundtruth": str(args.groundtruth.resolve()),
        "delta_sec": float(args.delta_sec),
        "max_pair_dt_sec": float(args.max_pair_dt_sec),
        "pairs": int(len(translation_errors)),
        "overlap_estimate_poses": int(len(estimate)),
        "pair_time_error_max_sec": float(max(pair_time_errors)),
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_rpe_mean_m": float(np.mean(translation_errors)),
        "translation_rpe_median_m": float(np.median(translation_errors)),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(rotation_errors ** 2))),
        "rotation_rpe_mean_rad": float(np.mean(rotation_errors)),
        "rotation_rpe_rmse_deg": float(np.degrees(np.sqrt(np.mean(rotation_errors ** 2)))),
        "rotation_rpe_mean_deg": float(np.degrees(np.mean(rotation_errors))),
        "alignment": "single SE(3) Umeyama alignment on overlapping positions before relative errors",
        "warning": "This is translational/rotational RPE from the local CSV convention; it is not the paper's unspecified TSR definition.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
