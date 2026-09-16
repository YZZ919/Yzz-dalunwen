#!/usr/bin/env python3
"""Compare clean-room trained ATE with the IR-VIO values reported in Table I.

The paper reports rounded per-sequence ATE values.  This script keeps those
published values separate from the locally measured trajectories and records
both absolute and relative gaps; it does not imply that the unpublished author
implementation was rerun here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAPER_IRVIO_ATE_M = {
    "MH_01_easy": 0.16,
    "MH_02_easy": 0.12,
    "MH_03_medium": 0.11,
    "MH_04_difficult": 0.20,
    "MH_05_difficult": 0.21,
    "V1_01_easy": 0.08,
    "V1_02_medium": 0.08,
    "V1_03_difficult": 0.10,
    "V2_01_easy": 0.06,
    "V2_02_medium": 0.10,
    "V2_03_difficult": 0.17,
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-json", type=Path,
        default=workspace / "results" / "gray_dce_cleanroom_comparison_11seq.json",
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        default=workspace / "results" / "gray_dce_vs_paper_irvio_11seq",
    )
    args = parser.parse_args()

    comparison = read_json(args.comparison_json)
    if not isinstance(comparison, dict):
        raise FileNotFoundError(f"comparison report not found: {args.comparison_json}")
    measured = {}
    for row in comparison.get("test_rows", []):
        if row.get("status") == "complete":
            measured.setdefault(row["sequence"], {})[row["profile"]] = row.get("ate_rmse_se3_m")

    rows = []
    for sequence, paper_ate in PAPER_IRVIO_ATE_M.items():
        row = {
            "sequence": sequence,
            "paper_irvio_ate_rmse_se3_m": paper_ate,
        }
        for profile in ("official", "paper"):
            value = measured.get(sequence, {}).get(profile)
            row[f"{profile}_ate_rmse_se3_m"] = value
            row[f"{profile}_gap_m"] = value - paper_ate if value is not None else None
            row[f"{profile}_gap_pct"] = (
                100.0 * (value - paper_ate) / paper_ate
                if value is not None and paper_ate else None
            )
            row[f"{profile}_ratio"] = value / paper_ate if value is not None and paper_ate else None
        rows.append(row)

    paper_mean_reported = 0.12
    paper_mean_from_rounded_rows = sum(PAPER_IRVIO_ATE_M.values()) / len(PAPER_IRVIO_ATE_M)
    summary = {
        "status": "cleanroom_gray_dce_vs_published_irvio_table1",
        "paper_table": "IR-VIO Table I, rounded values transcribed in references/paper_protocol.md",
        "paper_mean_reported_m": paper_mean_reported,
        "paper_mean_from_rounded_rows_m": paper_mean_from_rounded_rows,
        "cleanroom_means_m": {
            profile: comparison.get("test_summary", {}).get(profile, {}).get("mean_ate_rmse_se3_m")
            for profile in ("official", "paper")
        },
        "cleanroom_gap_vs_reported_mean_m": {},
        "cleanroom_gap_vs_reported_mean_pct": {},
        "cleanroom_ratio_vs_reported_mean": {},
        "rows": rows,
        "warning": (
            "Published IR-VIO values are rounded Table I results; clean-room trajectories "
            "were not produced by the unpublished author code or checkpoint."
        ),
    }
    for profile, value in summary["cleanroom_means_m"].items():
        if value is None:
            summary["cleanroom_gap_vs_reported_mean_m"][profile] = None
            summary["cleanroom_gap_vs_reported_mean_pct"][profile] = None
            summary["cleanroom_ratio_vs_reported_mean"][profile] = None
        else:
            summary["cleanroom_gap_vs_reported_mean_m"][profile] = value - paper_mean_reported
            summary["cleanroom_gap_vs_reported_mean_pct"][profile] = (
                100.0 * (value - paper_mean_reported) / paper_mean_reported
            )
            summary["cleanroom_ratio_vs_reported_mean"][profile] = value / paper_mean_reported

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "sequence", "paper_irvio_ate_rmse_se3_m",
        "official_ate_rmse_se3_m", "official_gap_m", "official_gap_pct", "official_ratio",
        "paper_ate_rmse_se3_m", "paper_gap_m", "paper_gap_pct", "paper_ratio",
    ]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "output": str(prefix),
        "paper_mean_reported_m": paper_mean_reported,
        "cleanroom_means_m": summary["cleanroom_means_m"],
        "gap_pct": summary["cleanroom_gap_vs_reported_mean_pct"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
