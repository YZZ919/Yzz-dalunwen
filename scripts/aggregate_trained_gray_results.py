#!/usr/bin/env python3
"""Aggregate ATE reports produced by a trained clean-room gray DCE run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


DEFAULT_SEQUENCES = ["V1_01_easy", "V1_03_difficult", "V2_03_difficult"]


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_first_json(root: Path, names: tuple[str, ...]) -> dict:
    """Read the first available metrics file from a sequence result root.

    ASL runs use ``*_asl_metrics.json`` while the older bag/CSV runs use the
    shorter ``metrics.json`` names.  Keeping both fallbacks here prevents the
    trained-checkpoint comparison from silently omitting the MH baseline.
    """

    for name in names:
        value = read_json(root / name)
        if isinstance(value, dict):
            return value
    return {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=workspace / "results")
    parser.add_argument("--label", default="dual_gray_dce_trained_official")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    rows = []
    for sequence in args.sequences:
        sequence_root = args.results_root / sequence
        report = read_json(sequence_root / f"{args.label}_ate.json")
        if report is None:
            rows.append({"sequence": sequence, "status": "missing"})
            continue
        baseline = read_first_json(
            sequence_root, ("vins_mono_asl_metrics.json", "metrics.json")
        )
        tw = read_first_json(
            sequence_root, ("irvio_weighting_asl_metrics.json", "irvio_weighting_metrics.json")
        )
        enhancement = read_first_json(
            sequence_root,
            (
                f"enhanced_{args.label}.json",
                f"enhanced_trained_{args.label.replace('dual_gray_dce_trained_', '', 1)}.json",
            ),
        )
        bnr_rows = read_csv_rows(sequence_root / f"{args.label}_bnr_frames.csv")
        bnr_removed = [row for row in bnr_rows if float(row.get("enhanced_features_removed", 0) or 0) > 0]
        ate = report.get("ate_rmse_se3_m")
        baseline_ate = baseline.get("ate_rmse_se3_m")
        tw_ate = tw.get("ate_rmse_se3_m")
        rows.append({
            "sequence": sequence,
            "status": "complete",
            "pipeline_success": bool(report.get("pipeline_success", False)),
            "matched_poses": report.get("matched_poses"),
            "time_coverage": report.get("time_coverage"),
            "ate_rmse_se3_m": ate,
            "ate_mean_se3_m": report.get("ate_mean_se3_m"),
            "baseline_ate_rmse_se3_m": baseline_ate,
            "tw_ate_rmse_se3_m": tw_ate,
            "enhancement_frames": enhancement.get("frames"),
            "enhancement_processed_frames": enhancement.get("processed_frames"),
            "enhancement_mean_inference_seconds": enhancement.get("mean_inference_seconds"),
            "mean_input_intensity": enhancement.get("mean_input_intensity"),
            "mean_output_intensity": enhancement.get("mean_output_intensity"),
            "bnr_frames": len(bnr_rows),
            "frames_with_bnr_removal": len(bnr_removed),
            "enhanced_features_before": sum(
                int(float(row.get("enhanced_features_before", 0) or 0)) for row in bnr_rows
            ),
            "enhanced_features_removed": sum(
                int(float(row.get("enhanced_features_removed", 0) or 0)) for row in bnr_rows
            ),
            "reduction_vs_baseline_pct": (
                100.0 * (baseline_ate - ate) / baseline_ate
                if ate is not None and baseline_ate not in (None, 0) else None
            ),
            "reduction_vs_tw_pct": (
                100.0 * (tw_ate - ate) / tw_ate
                if ate is not None and tw_ate not in (None, 0) else None
            ),
        })

    completed = [row for row in rows if row.get("status") == "complete"]
    successful = [
        row for row in completed
        if row.get("pipeline_success")
        and row.get("matched_poses", 0) >= 100
        and row.get("time_coverage", 0.0) >= 0.70
        and row.get("ate_rmse_se3_m") is not None
    ]
    ates = [float(row["ate_rmse_se3_m"]) for row in successful]
    summary = {
        "status": "trained_cleanroom_gray_dce_euroc_summary",
        "label": args.label,
        "sequences_requested": list(args.sequences),
        "completed": len(completed),
        "successful": len(successful),
        "success_rate": len(successful) / len(args.sequences) if args.sequences else 0.0,
        "mean_ate_rmse_se3_m": statistics.mean(ates) if ates else None,
        "median_ate_rmse_se3_m": statistics.median(ates) if ates else None,
        "rows": rows,
        "warning": "This is a clean-room trained checkpoint, not the unpublished author IR-VIO model.",
    }
    prefix = args.output_prefix or (args.results_root / f"{args.label}_summary")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_suffix(".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "sequence", "status", "pipeline_success", "matched_poses", "time_coverage",
        "ate_rmse_se3_m", "ate_mean_se3_m", "baseline_ate_rmse_se3_m",
        "tw_ate_rmse_se3_m", "enhancement_frames", "enhancement_processed_frames",
        "enhancement_mean_inference_seconds", "mean_input_intensity", "mean_output_intensity",
        "bnr_frames", "frames_with_bnr_removal", "enhanced_features_before",
        "enhanced_features_removed", "reduction_vs_baseline_pct", "reduction_vs_tw_pct",
    ]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: summary[k] for k in (
        "label", "completed", "successful", "success_rate", "mean_ate_rmse_se3_m"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
