#!/usr/bin/env python3
"""Summarize dual feature-tracker evidence without treating new points as TSR.

The paper's TSR formula was not found in the available IR-VIO text.  This
script therefore reports explicitly named engineering proxies.  The primary
proxy is the number of tracks surviving LK (or LK+fundamental-matrix filter)
divided by the number of LK attempts, aggregated over synchronized frames.
BNR survival is reported separately and is never silently counted as tracking
success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


FIELDS = (
    "timestamp", "raw_flow_attempts", "raw_flow_success", "raw_flow_after_f",
    "enhanced_flow_attempts", "enhanced_flow_success", "enhanced_flow_after_f",
    "enhanced_before_bnr", "enhanced_removed_bnr", "enhanced_after_bnr", "published",
)


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if not raw or raw.get("timestamp", "").startswith("#"):
                continue
            # The C++ tracker names this first column ``stamp``; accept the
            # more descriptive ``timestamp`` used by older shell headers too.
            if "timestamp" not in raw and "stamp" in raw:
                raw["timestamp"] = raw["stamp"]
            try:
                row = {field: float(raw[field]) for field in FIELDS}
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def frame_ratios(rows: list[dict[str, float]], numerator: str, denominator: str):
    values = []
    for row in rows:
        if row[denominator] > 0:
            values.append(row[numerator] / row[denominator])
    return values


def summary(path: Path, rows: list[dict[str, float]]) -> dict:
    published = [row for row in rows if row["published"] > 0.5]
    attempted = [row for row in rows if row["enhanced_flow_attempts"] > 0]
    published_attempted = [
        row for row in published if row["enhanced_flow_attempts"] > 0
    ]
    sums = {field: sum(row[field] for row in rows) for field in FIELDS[1:]}
    published_sums = {
        field: sum(row[field] for row in published)
        for field in FIELDS[1:]
    }
    published_attempted_sums = {
        field: sum(row[field] for row in published_attempted)
        for field in FIELDS[1:]
    }
    enhanced_attempts = sums["enhanced_flow_attempts"]
    raw_attempts = sums["raw_flow_attempts"]

    # The paper does not expose a mathematical TSR definition in the
    # available text.  These are therefore labelled proxies, not IR-VIO TSR.
    enhanced_lk_frame = frame_ratios(rows, "enhanced_flow_success", "enhanced_flow_attempts")
    enhanced_post_f_frame = frame_ratios(rows, "enhanced_flow_after_f", "enhanced_flow_attempts")
    raw_lk_frame = frame_ratios(rows, "raw_flow_success", "raw_flow_attempts")
    raw_post_f_frame = frame_ratios(rows, "raw_flow_after_f", "raw_flow_attempts")
    bnr_frame = frame_ratios(rows, "enhanced_after_bnr", "enhanced_before_bnr")

    def stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "std": None}
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "status": "tracking_stats_complete",
        "source": str(path.resolve()),
        "rows": len(rows),
        "published_rows": len(published),
        "published_rows_with_enhanced_lk_attempts": len(published_attempted),
        "rows_with_enhanced_lk_attempts": len(attempted),
        "timestamp_start": rows[0]["timestamp"] if rows else None,
        "timestamp_end": rows[-1]["timestamp"] if rows else None,
        "raw_flow_attempts": int(sums["raw_flow_attempts"]),
        "raw_flow_success": int(sums["raw_flow_success"]),
        "raw_flow_after_f": int(sums["raw_flow_after_f"]),
        "enhanced_flow_attempts": int(enhanced_attempts),
        "enhanced_flow_success": int(sums["enhanced_flow_success"]),
        "enhanced_flow_after_f": int(sums["enhanced_flow_after_f"]),
        "enhanced_before_bnr": int(sums["enhanced_before_bnr"]),
        "enhanced_removed_bnr": int(sums["enhanced_removed_bnr"]),
        "enhanced_after_bnr": int(sums["enhanced_after_bnr"]),
        "published_enhanced_flow_attempts": int(published_sums["enhanced_flow_attempts"]),
        "published_enhanced_flow_after_f": int(published_sums["enhanced_flow_after_f"]),
        "published_enhanced_after_bnr": int(published_sums["enhanced_after_bnr"]),
        "raw_lk_success_rate_attempt_based": ratio(sums["raw_flow_success"], raw_attempts),
        "raw_post_f_track_rate_attempt_based": ratio(sums["raw_flow_after_f"], raw_attempts),
        "enhanced_lk_success_rate_attempt_based": ratio(sums["enhanced_flow_success"], enhanced_attempts),
        "enhanced_post_f_track_rate_attempt_based": ratio(sums["enhanced_flow_after_f"], enhanced_attempts),
        "enhanced_bnr_survival_rate": ratio(sums["enhanced_after_bnr"], sums["enhanced_before_bnr"]),
        "tsr_proxy_post_f_attempt_based": ratio(sums["enhanced_flow_after_f"], enhanced_attempts),
        # The tracker adds newly detected points after LK, so a BNR-after
        # count cannot be divided by LK attempts without pretending new
        # points were tracked successfully.  Keep this value unavailable and
        # report BNR survival against enhanced_before_bnr instead.
        "tsr_proxy_post_f_then_bnr_attempt_based": None,
        "tsr_proxy_post_f_published_attempt_based": ratio(
            published_sums["enhanced_flow_after_f"], published_sums["enhanced_flow_attempts"]
        ),
        "tsr_proxy_post_f_then_bnr_published_attempt_based": None,
        "frame_level": {
            "raw_lk_success_rate": stats(raw_lk_frame),
            "raw_post_f_track_rate": stats(raw_post_f_frame),
            "enhanced_lk_success_rate": stats(enhanced_lk_frame),
            "enhanced_post_f_track_rate": stats(enhanced_post_f_frame),
            "enhanced_bnr_survival_rate": stats(bnr_frame),
        },
        "definition": {
            "denominator": "LK optical-flow attempts on the previous frame's tracked points",
            "post_f_numerator": "points surviving LK and the configured fundamental-matrix rejection",
            "bnr_numerator": "post-flow enhanced points remaining after BNR",
            "aggregation": "micro aggregate over synchronized CSV rows; published-frame variants use only rows actually sent to VIO; frame-level mean/median/std also reported",
            "paper_equivalence": False,
            "new_detection_caveat": "The tracker may add new points after LK, so BNR-after counts are not used as a TSR numerator.",
            "warning": "The available IR-VIO paper text reports a TSR improvement but does not define TSR mathematically. Do not call these values the paper TSR.",
        },
    }


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=workspace / "results" / "tracking_stats.json")
    args = parser.parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"no valid rows in {args.input}")
    report = summary(args.input, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "source", "rows", "published_rows", "enhanced_flow_attempts",
        "enhanced_post_f_track_rate_attempt_based", "enhanced_bnr_survival_rate",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
