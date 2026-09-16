#!/usr/bin/env python3
"""Aggregate per-sequence tracker CSVs without inventing a TSR definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_tracking_stats import read_rows, summary


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=workspace / "results")
    parser.add_argument("--label", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sequence_reports = []
    all_rows = []
    missing = []
    for sequence in args.sequences:
        path = args.results_root / sequence / f"{args.label}_tracking_stats.csv"
        if not path.exists():
            missing.append(sequence)
            continue
        rows = read_rows(path)
        if not rows:
            missing.append(sequence)
            continue
        sequence_reports.append({
            "sequence": sequence,
            "report": summary(path, rows),
        })
        all_rows.extend(rows)

    micro = summary(Path("<multiple sequence tracker CSVs>"), all_rows) if all_rows else None
    report = {
        "status": "tracking_stats_suite_complete" if not missing else "tracking_stats_suite_partial",
        "label": args.label,
        "requested_sequences": list(args.sequences),
        "complete_sequences": len(sequence_reports),
        "missing_sequences": missing,
        "sequence_reports": sequence_reports,
        "micro_aggregate": micro,
        "warning": (
            "The paper's TSR formula is not defined in the available text. "
            "post-F attempt-based rates and BNR survival are explicitly named "
            "engineering proxies, not paper TSR."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "label": args.label,
        "complete_sequences": len(sequence_reports),
        "missing_sequences": missing,
        "micro_enhanced_post_f_track_rate_attempt_based": (
            micro.get("enhanced_post_f_track_rate_attempt_based") if micro else None
        ),
        "micro_enhanced_bnr_survival_rate": (
            micro.get("enhanced_bnr_survival_rate") if micro else None
        ),
    }, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
