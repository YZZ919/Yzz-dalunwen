#!/usr/bin/env python3
"""Compute EuRoC ATE for a VINS-Mono CSV and ASL ground truth.

The estimate is associated to the nearest ground-truth timestamp, then an
SE(3) alignment (rotation + translation, no scale correction) is applied.
This matches the paper's ATE/RMSE comparison more closely than comparing the
arbitrary VINS and EuRoC world-frame origins directly.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np


def load_csv(path, cols):
    out = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            # EuRoC ground truth has a header, while VINS trajectory CSVs in
            # this workspace are headerless.  Attempting to parse every row
            # handles both forms without discarding the first valid pose.
            try:
                out.append([float(row[i]) for i in cols])
            except (ValueError, IndexError):
                continue
    return np.asarray(out, dtype=float)


def align_se3(est, gt):
    mu_e = est.mean(axis=0)
    mu_g = gt.mean(axis=0)
    xe = est - mu_e
    xg = gt - mu_g
    u, _, vt = np.linalg.svd(xe.T @ xg)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = mu_g - r @ mu_e
    return (r @ est.T).T + t, r, t


def align_sim3(est, gt):
    mu_e = est.mean(axis=0)
    mu_g = gt.mean(axis=0)
    xe = est - mu_e
    xg = gt - mu_g
    u, svals, vt = np.linalg.svd(xe.T @ xg)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        svals[-1] *= -1.0
        r = vt.T @ u.T
    scale = float(svals.sum() / max(np.sum(xe * xe), 1e-15))
    t = mu_g - scale * (r @ mu_e)
    return (scale * (r @ est.T)).T + t, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", required=True, help="VINS result CSV")
    ap.add_argument("--groundtruth", required=True, help="ASL state_groundtruth_estimate0/data.csv")
    ap.add_argument("--camera-csv", help="optional cam0/data.csv for time coverage")
    ap.add_argument("--max-dt", type=float, default=0.02, help="max timestamp association error (s)")
    ap.add_argument("--interpolate", action="store_true",
                    help="linearly interpolate GT positions (useful for sparse Leica bag topics)")
    ap.add_argument("--output", help="write JSON metrics")
    args = ap.parse_args()

    # VINS: timestamp, tx, ty, tz; ASL GT: timestamp, px, py, pz.
    est = load_csv(args.estimate, [0, 1, 2, 3])
    gt = load_csv(args.groundtruth, [0, 1, 2, 3])
    if len(est) == 0 or len(gt) == 0:
        raise SystemExit("empty estimate or ground truth")
    est[:, 0] *= 1e-9
    gt[:, 0] *= 1e-9
    gt = gt[np.argsort(gt[:, 0])]

    if args.interpolate:
        keep = (est[:, 0] >= gt[0, 0]) & (est[:, 0] <= gt[-1, 0])
        est_m = est[keep, 1:4]
        gt_m = np.column_stack([np.interp(est[keep, 0], gt[:, 0], gt[:, j]) for j in range(1, 4)])
        matched_dt = np.zeros(len(est_m), dtype=float)
    else:
        idx = np.searchsorted(gt[:, 0], est[:, 0])
        idx = np.clip(idx, 0, len(gt) - 1)
        left = np.maximum(idx - 1, 0)
        choose_left = np.abs(gt[left, 0] - est[:, 0]) < np.abs(gt[idx, 0] - est[:, 0])
        idx[choose_left] = left[choose_left]
        dt = np.abs(gt[idx, 0] - est[:, 0])
        keep = dt <= args.max_dt
        est_m = est[keep, 1:4]
        gt_m = gt[idx[keep], 1:4]
        matched_dt = dt[keep]
    if len(est_m) < 3:
        raise SystemExit("fewer than three associated poses")

    aligned, _, _ = align_se3(est_m, gt_m)
    sim_aligned, scale = align_sim3(est_m, gt_m)
    err = np.linalg.norm(aligned - gt_m, axis=1)
    sim_err = np.linalg.norm(sim_aligned - gt_m, axis=1)
    gt_span = float(gt[-1, 0] - gt[0, 0])
    est_span = float(est[-1, 0] - est[0, 0])
    coverage = est_span / gt_span if gt_span > 0 else float("nan")
    if args.camera_csv:
        cam = load_csv(args.camera_csv, [0])[:, 0] * 1e-9
        if len(cam) > 1:
            coverage = est_span / float(cam[-1] - cam[0])

    metrics = {
        "estimate": os.path.abspath(args.estimate),
        "groundtruth": os.path.abspath(args.groundtruth),
        "association_max_dt_sec": args.max_dt,
        "association": "linear_interpolation" if args.interpolate else "nearest_groundtruth",
        "estimate_poses": int(len(est)),
        "matched_poses": int(len(est_m)),
        "match_fraction": float(len(est_m) / len(est)),
        "max_association_dt_sec": float(np.max(matched_dt)) if len(matched_dt) else None,
        "estimate_span_sec": est_span,
        "groundtruth_span_sec": gt_span,
        "time_coverage": coverage,
        "ate_rmse_se3_m": float(math.sqrt(np.mean(err * err))),
        "ate_mean_se3_m": float(np.mean(err)),
        "ate_max_se3_m": float(np.max(err)),
        "ate_rmse_sim3_m": float(math.sqrt(np.mean(sim_err * sim_err))),
        "sim3_scale": float(scale),
        # Pipeline success is deliberately independent from the quality score.
        "pipeline_success": bool(len(est_m) >= 100 and coverage >= 0.70 and np.isfinite(err).all()),
        "metric_pass_ate_lt_1m": bool(np.sqrt(np.mean(err * err)) < 1.0),
    }
    text = json.dumps(metrics, indent=2) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
