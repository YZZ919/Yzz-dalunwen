#!/usr/bin/env python3
"""Compare fixed-configuration repeat runs with canonical dual-branch results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    results_dir = workspace / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-prefix", default="dual_branch_proxy_repeatability")
    parser.add_argument("--replay-speed", type=float)
    args = parser.parse_args()

    canonical = json.loads(
        (results_dir / "dual_branch_proxy_summary.json").read_text(encoding="utf-8")
    )
    canonical_by_sequence = {row["sequence"]: row for row in canonical["results"]}
    rows = []
    for sequence, original in canonical_by_sequence.items():
        repeat_path = results_dir / sequence / f"{args.run_label}_ate.json"
        if not repeat_path.is_file():
            continue
        repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
        original_ate = float(original["ate_rmse_se3_m"])
        repeat_ate = float(repeat["ate_rmse_se3_m"])
        rows.append(
            {
                "sequence": sequence,
                "canonical_ate_rmse_se3_m": original_ate,
                "repeat_ate_rmse_se3_m": repeat_ate,
                "absolute_ate_difference_m": repeat_ate - original_ate,
                "repeat_vs_canonical_difference_percent": (
                    (repeat_ate - original_ate) / original_ate * 100.0
                ),
                "canonical_matched_poses": int(original["matched_poses"]),
                "repeat_matched_poses": int(repeat["matched_poses"]),
                "canonical_time_coverage": float(original["time_coverage"]),
                "repeat_time_coverage": float(repeat["time_coverage"]),
                "repeat_pipeline_success": bool(repeat["pipeline_success"]),
            }
        )

    if not rows:
        raise SystemExit("no repeatability result found")
    output_csv = results_dir / f"{args.output_prefix}.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    abs_diffs = [abs(row["absolute_ate_difference_m"]) for row in rows]
    rel_diffs = [abs(row["repeat_vs_canonical_difference_percent"]) for row in rows]
    document = {
        "status": "clean_room_proxy_repeatability",
        "run_label": args.run_label,
        "replay_speed": args.replay_speed,
        "sequences_compared": len(rows),
        "successful_repeats": sum(row["repeat_pipeline_success"] for row in rows),
        "max_absolute_ate_difference_m": max(abs_diffs),
        "mean_absolute_ate_difference_m": sum(abs_diffs) / len(abs_diffs),
        "max_absolute_ate_difference_percent": max(rel_diffs),
        "results": rows,
        "warning": (
            "Repeatability is for the clean-room dual-branch proxy and does not validate "
            "the unavailable official IR-VIO implementation."
        ),
    }
    output_json = results_dir / f"{args.output_prefix}.json"
    output_json.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
