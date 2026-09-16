#!/usr/bin/env python3
"""Clean-room implementation of IR-VIO's block noise removal stage.

The noise estimator follows the public MATLAB implementation referenced by
IR-VIO (Shin et al., IROS 2019).  It is intentionally independent from the
unavailable IR-VIO image-enhancement network: pass the raw and enhanced images
explicitly, or use ``--clahe-proxy`` only for a labelled component smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


NOISE_KERNEL = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
    dtype=np.float64,
)


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is empty")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"expected a grayscale image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def estimate_noise_level(image: np.ndarray) -> float:
    """Estimate image noise using the filter and masks cited by IR-VIO.

    The public reference implementation notes that its paper omitted the
    factor 1/6.  IR-VIO's ratio test is unchanged by this common factor, but we
    retain it here so absolute values match the cited implementation.
    """

    gray = _as_gray_u8(image)
    height, width = gray.shape
    if height < 3 or width < 3:
        raise ValueError("noise estimation needs an image of at least 3x3 pixels")

    values = gray.astype(np.float64)
    grad_x = cv2.Sobel(values, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(values, cv2.CV_64F, 0, 1, ksize=3)
    gradient_squared = grad_x * grad_x + grad_y * grad_y

    # MATLAB's int32(H*W*0.10) is a one-based order-statistic index.
    rank = max(1, int(round(height * width * 0.10)))
    threshold = float(np.partition(gradient_squared.ravel(), rank - 1)[rank - 1])
    homogeneous = gradient_squared <= threshold
    unsaturated = (values >= 15.0) & (values <= 235.0)
    reliable = homogeneous & unsaturated

    laplacian = cv2.filter2D(
        values, cv2.CV_64F, NOISE_KERNEL, borderType=cv2.BORDER_CONSTANT
    )
    reliable_count = int(np.count_nonzero(reliable))
    scale = math.sqrt(math.pi / 2.0) / 6.0

    if reliable_count >= height * width * 0.0001:
        return scale * float(np.abs(laplacian[reliable]).sum()) / reliable_count

    # Same fallback as the cited MATLAB code when no reliable region exists.
    return scale * float(np.abs(laplacian).sum()) / ((width - 2) * (height - 2))


def block_noise_ratios(
    raw: np.ndarray, enhanced: np.ndarray, rows: int = 6, cols: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return enhanced/raw noise ratios and both per-block noise maps."""

    raw = _as_gray_u8(raw)
    enhanced = _as_gray_u8(enhanced)
    if raw.shape != enhanced.shape:
        raise ValueError(f"image sizes differ: {raw.shape} versus {enhanced.shape}")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    height, width = raw.shape
    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, cols + 1, dtype=int)
    raw_noise = np.zeros((rows, cols), dtype=np.float64)
    enhanced_noise = np.zeros((rows, cols), dtype=np.float64)

    for row in range(rows):
        for col in range(cols):
            ys = slice(y_edges[row], y_edges[row + 1])
            xs = slice(x_edges[col], x_edges[col + 1])
            raw_noise[row, col] = estimate_noise_level(raw[ys, xs])
            enhanced_noise[row, col] = estimate_noise_level(enhanced[ys, xs])

    ratios = np.ones_like(raw_noise)
    nonzero = raw_noise > np.finfo(np.float64).eps
    ratios[nonzero] = enhanced_noise[nonzero] / raw_noise[nonzero]
    ratios[~nonzero & (enhanced_noise > np.finfo(np.float64).eps)] = np.inf
    return ratios, raw_noise, enhanced_noise


def select_features(
    points: np.ndarray,
    ratios: np.ndarray,
    image_shape: Tuple[int, int],
    threshold: float = 10.8,
    min_keep: int = 8,
) -> Tuple[np.ndarray, List[dict]]:
    """Apply the paper's descending-ratio removal with constraint protection.

    Points in blocks whose ratio exceeds ``threshold`` are considered in
    descending block-ratio order.  Removal stops when only ``min_keep`` points
    remain.  Stable input order breaks ties, since the paper does not specify a
    within-block ordering rule.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros(0, dtype=bool), []
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if min_keep < 0:
        raise ValueError("min_keep must be non-negative")

    height, width = image_shape
    rows, cols = ratios.shape
    block_rows = np.clip((points[:, 1] * rows / height).astype(int), 0, rows - 1)
    block_cols = np.clip((points[:, 0] * cols / width).astype(int), 0, cols - 1)
    point_ratios = ratios[block_rows, block_cols]
    candidates = [
        i for i in range(len(points)) if float(point_ratios[i]) > threshold
    ]
    candidates.sort(key=lambda i: (-float(point_ratios[i]), i))

    keep = np.ones(len(points), dtype=bool)
    removal_budget = max(0, len(points) - min_keep)
    removed = []
    for index in candidates[:removal_budget]:
        keep[index] = False
        removed.append(
            {
                "index": int(index),
                "x": float(points[index, 0]),
                "y": float(points[index, 1]),
                "block_row": int(block_rows[index]),
                "block_col": int(block_cols[index]),
                "noise_ratio": float(point_ratios[index]),
            }
        )
    return keep, removed


def _read_points(path: Path) -> np.ndarray:
    points = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("feature CSV has no header")
        x_name = "x" if "x" in reader.fieldnames else "u"
        y_name = "y" if "y" in reader.fieldnames else "v"
        if x_name not in reader.fieldnames or y_name not in reader.fieldnames:
            raise ValueError("feature CSV must contain x,y or u,v columns")
        for row in reader:
            points.append((float(row[x_name]), float(row[y_name])))
    return np.asarray(points, dtype=np.float64).reshape((-1, 2))


def _detect_points(image: np.ndarray, maximum: int) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(
        image, maxCorners=maximum, qualityLevel=0.01, minDistance=20
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float64)
    return corners.reshape((-1, 2)).astype(np.float64)


def _json_number(value: float):
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return float(value)


def _draw_result(
    raw: np.ndarray,
    enhanced: np.ndarray,
    points: np.ndarray,
    keep: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    left = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    height, width = raw.shape
    for row in range(1, rows):
        y = int(round(row * height / rows))
        cv2.line(right, (0, y), (width - 1, y), (90, 90, 90), 1)
    for col in range(1, cols):
        x = int(round(col * width / cols))
        cv2.line(right, (x, 0), (x, height - 1), (90, 90, 90), 1)
    for point, retained in zip(points, keep):
        center = (int(round(point[0])), int(round(point[1])))
        color = (40, 220, 40) if retained else (20, 20, 240)
        cv2.circle(right, center, 3, color, -1)
    cv2.putText(left, "raw", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(
        right,
        "enhanced: green=kept, red=removed",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return np.hstack((left, right))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("enhanced", nargs="?", type=Path)
    parser.add_argument(
        "--clahe-proxy",
        action="store_true",
        help="generate a CLAHE image for component testing; this is not IR-VIO enhancement",
    )
    parser.add_argument("--features", type=Path)
    parser.add_argument("--detect-max", type=int, default=150)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=10.8)
    parser.add_argument("--min-keep", type=int, default=8)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--visualization", type=Path)
    args = parser.parse_args(argv)

    raw = _as_gray_u8(cv2.imread(str(args.raw), cv2.IMREAD_GRAYSCALE))
    enhancement_source = "external"
    if args.enhanced:
        enhanced = _as_gray_u8(cv2.imread(str(args.enhanced), cv2.IMREAD_GRAYSCALE))
    elif args.clahe_proxy:
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(raw)
        enhancement_source = "CLAHE proxy (not the IR-VIO network)"
    else:
        parser.error("provide an enhanced image or use --clahe-proxy")

    ratios, raw_noise, enhanced_noise = block_noise_ratios(
        raw, enhanced, args.rows, args.cols
    )
    points = _read_points(args.features) if args.features else _detect_points(
        enhanced, args.detect_max
    )
    keep, removed = select_features(
        points, ratios, raw.shape, args.threshold, args.min_keep
    )

    report = {
        "status": "component_test" if args.clahe_proxy else "external_enhancement",
        "enhancement_source": enhancement_source,
        "raw_image": str(args.raw),
        "enhanced_image": str(args.enhanced) if args.enhanced else None,
        "grid": {"rows": args.rows, "cols": args.cols},
        "noise_ratio_threshold": args.threshold,
        "minimum_retained_features": args.min_keep,
        "feature_count_before": int(len(points)),
        "feature_count_removed": int(np.count_nonzero(~keep)),
        "feature_count_after": int(np.count_nonzero(keep)),
        "blocks_over_threshold": int(np.count_nonzero(ratios > args.threshold)),
        "maximum_noise_ratio": _json_number(float(np.max(ratios))),
        "raw_noise": [[_json_number(v) for v in row] for row in raw_noise],
        "enhanced_noise": [[_json_number(v) for v in row] for row in enhanced_noise],
        "noise_ratio": [[_json_number(v) for v in row] for row in ratios],
        "removed_features": removed,
        "reproduction_note": (
            "The BNR equations and published parameters are reproduced. "
            "The IR-VIO grayscale enhancement network/checkpoint is unavailable."
        ),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8")
    if args.visualization:
        args.visualization.parent.mkdir(parents=True, exist_ok=True)
        rendered = _draw_result(raw, enhanced, points, keep, args.rows, args.cols)
        if not cv2.imwrite(str(args.visualization), rendered):
            raise RuntimeError(f"failed to write {args.visualization}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
