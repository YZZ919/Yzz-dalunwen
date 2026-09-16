#!/usr/bin/env python3
"""Build a same-protocol comparison table without overwriting historical runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
    "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _summary_row(label: str, name: str, path: Path, checkpoint: str | None = None):
    payload = _load_json(path)
    if payload is None:
        return {
            "label": label, "name": name, "status": "missing",
            "completed": 0, "requested": len(SEQUENCES), "mean_ate_rmse_se3_m": None,
            "success_rate": None, "source": str(path.resolve()), "checkpoint": checkpoint,
        }
    return {
        "label": label,
        "name": name,
        "status": payload.get("status", "unknown"),
        "completed": payload.get("completed", payload.get("successful", 0)),
        "requested": len(payload.get("sequences_requested", SEQUENCES)),
        "mean_ate_rmse_se3_m": payload.get("mean_ate_rmse_se3_m"),
        "success_rate": payload.get("success_rate"),
        "source": str(path.resolve()),
        "checkpoint": checkpoint,
    }


def _partial_rows(results_root: Path, label: str):
    rows = []
    for sequence in SEQUENCES:
        path = results_root / sequence / f"{label}_ate.json"
        payload = _load_json(path)
        rows.append({
            "sequence": sequence,
            "status": "complete" if payload else "missing",
            "ate_rmse_se3_m": payload.get("ate_rmse_se3_m") if payload else None,
            "source": str(path.resolve()),
        })
    complete = [row["ate_rmse_se3_m"] for row in rows if row["ate_rmse_se3_m"] is not None]
    return rows, {
        "completed": len(complete),
        "requested": len(SEQUENCES),
        "mean_ate_rmse_se3_m": statistics.mean(complete) if complete else None,
        "success_rate": len(complete) / len(SEQUENCES),
    }


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=workspace / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        _summary_row(
            "dual_gray_dce_trained_official",
            "历史 official clean-room",
            args.results_root / "dual_gray_dce_trained_official_11seq_summary.json",
            "checkpoints/gray_dce_cleanroom/gray_official_mirror/best_val.pth",
        ),
        _summary_row(
            "dual_gray_dce_trained_paper",
            "历史 paper clean-room",
            args.results_root / "dual_gray_dce_trained_paper_11seq_summary.json",
            "checkpoints/gray_dce_cleanroom/gray_paper_mirror/best_val.pth",
        ),
        _summary_row(
            "dual_gray_dce_reference_official_11seq_tracking",
            "新 reference/official（low1 镜像）",
            args.results_root / "dual_gray_dce_reference_official_11seq_tracking_summary.json",
            "checkpoints/gray_dce_cleanroom_v2/gray_reference_official_mirror_resize256/best_val.pth",
        ),
        _summary_row(
            "dual_gray_dce_reference_paper_11seq_tracking",
            "新 reference/paper（low1 镜像，部分回放）",
            args.results_root / "dual_gray_dce_reference_paper_11seq_tracking_summary.json",
            "checkpoints/gray_dce_cleanroom_v2/gray_reference_paper_mirror_resize256/best_val.pth",
        ),
    ]
    paper_partial_rows, paper_partial = _partial_rows(
        args.results_root, "dual_gray_dce_reference_paper_11seq_tracking"
    )
    for row in rows:
        if row["label"] == "dual_gray_dce_reference_paper_11seq_tracking" and row["status"] == "missing":
            row.update(paper_partial)
            row["status"] = "partial" if paper_partial["completed"] else "missing"

    paper_table = [0.16, 0.12, 0.11, 0.20, 0.21, 0.08, 0.08, 0.10, 0.06, 0.10, 0.17]
    report = {
        "status": "reference_comparison_complete",
        "sequence_order": SEQUENCES,
        "runs": rows,
        "paper_table_i": {
            "reported_avg_m": 0.12,
            "rounded_row_values_m": paper_table,
            "rounded_row_arithmetic_mean_m": statistics.mean(paper_table),
            "note": "Table values are rounded; reported Avg and row arithmetic mean are retained as separate readings.",
        },
        "paper_partial_rows": paper_partial_rows,
        "warning": "EuRoC has been used for development/regression and these are not blind-test results; clean-room checkpoints are not the author checkpoint.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "label", "name", "status", "completed", "requested",
            "success_rate", "mean_ate_rmse_se3_m", "checkpoint", "source",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "runs": [{"label": row["label"], "status": row["status"], "completed": row["completed"], "mean_ate_rmse_se3_m": row["mean_ate_rmse_se3_m"]} for row in rows],
        "paper_table_row_mean_m": report["paper_table_i"]["rounded_row_arithmetic_mean_m"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
