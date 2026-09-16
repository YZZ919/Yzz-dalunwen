#!/usr/bin/env python3
"""Aggregate complete clean-room dual-branch proxy runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reduction(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    results_dir = workspace / "results"
    base_rows = read_csv(results_dir / "summary.csv")
    references = {}
    order = []
    for row in base_rows:
        sequence = row["sequence"]
        if sequence not in references:
            references[sequence] = {}
            order.append(sequence)
        references[sequence][row["method"]] = float(row["ate_rmse_se3_m"])

    rows = []
    for sequence in order:
        sequence_dir = results_dir / sequence
        metrics_path = sequence_dir / "dual_gray_dce_full_ate.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        enhancement = json.loads(
            (sequence_dir / "enhanced_openvino_full.json").read_text(encoding="utf-8")
        )
        bnr_rows = read_csv(sequence_dir / "dual_gray_dce_full_bnr_frames.csv")
        weight_rows = read_csv(sequence_dir / "dual_gray_dce_full_adaptive_weights.csv")
        baseline = references[sequence]["vins_mono_baseline"]
        weighting = references[sequence]["clean_room_two_layer_weighting"]
        dual = float(metrics["ate_rmse_se3_m"])
        removed = sum(int(float(row["enhanced_features_removed"])) for row in bnr_rows)
        before = sum(int(float(row["enhanced_features_before"])) for row in bnr_rows)
        rows.append(
            {
                "sequence": sequence,
                "method": "clean_room_dual_branch_proxy",
                "enhancement": "folded_grayscale_dce_openvino",
                "estimate_poses": int(metrics["estimate_poses"]),
                "matched_poses": int(metrics["matched_poses"]),
                "time_coverage": float(metrics["time_coverage"]),
                "ate_rmse_se3_m": dual,
                "pipeline_success": bool(metrics["pipeline_success"]),
                "baseline_ate_rmse_se3_m": baseline,
                "dual_vs_baseline_reduction_percent": reduction(baseline, dual),
                "weighting_ate_rmse_se3_m": weighting,
                "dual_vs_weighting_reduction_percent": reduction(weighting, dual),
                "bnr_frames": len(bnr_rows),
                "frames_with_bnr_removal": sum(
                    int(float(row["enhanced_features_removed"])) > 0 for row in bnr_rows
                ),
                "enhanced_features_before": before,
                "enhanced_features_removed": removed,
                "source0_weight_rows": sum(row["feature_source"] == "0" for row in weight_rows),
                "source1_weight_rows": sum(row["feature_source"] == "1" for row in weight_rows),
                "enhancement_frames": int(enhancement["frames"]),
                "enhancement_processed_frames": int(enhancement.get("processed_frames", 0)),
                "enhancement_skipped_existing": int(enhancement.get("skipped_existing", 0)),
                "enhancement_statistics_scope": enhancement.get(
                    "statistics_scope", "all_selected_frames"
                ),
                "enhancement_timing_scope": enhancement.get(
                    "timing_scope", "processed_frames_this_run"
                ),
                "enhancement_mean_inference_seconds": float(enhancement["mean_inference_seconds"]),
                "mean_input_intensity": float(enhancement["mean_input_intensity"]),
                "mean_output_intensity": float(enhancement["mean_output_intensity"]),
            }
        )

    if not rows:
        raise SystemExit("no complete dual-branch result found")
    csv_path = results_dir / "dual_branch_proxy_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(row["pipeline_success"] for row in rows)
    mean_dual = mean(row["ate_rmse_se3_m"] for row in rows)
    mean_baseline = mean(row["baseline_ate_rmse_se3_m"] for row in rows)
    mean_weighting = mean(row["weighting_ate_rmse_se3_m"] for row in rows)
    document = {
        "status": "clean_room_proxy_not_author_irvio",
        "success_definition": ">=100 matched finite poses, >=70% camera-time coverage, finite SE(3)-aligned ATE",
        "completed_sequences": len(rows),
        "successful_sequences": successful,
        "pipeline_success_rate": successful / len(rows),
        "precision_wins_vs_baseline": sum(
            row["ate_rmse_se3_m"] < row["baseline_ate_rmse_se3_m"] for row in rows
        ),
        "precision_wins_vs_weighting": sum(
            row["ate_rmse_se3_m"] < row["weighting_ate_rmse_se3_m"] for row in rows
        ),
        "mean_ate_rmse_se3_m": mean_dual,
        "mean_baseline_ate_rmse_se3_m": mean_baseline,
        "mean_weighting_ate_rmse_se3_m": mean_weighting,
        "mean_dual_vs_baseline_reduction_percent": reduction(mean_baseline, mean_dual),
        "mean_dual_vs_weighting_reduction_percent": reduction(mean_weighting, mean_dual),
        "results": rows,
        "warning": (
            "The IR-VIO authors' enhancement checkpoint and full source are unavailable; "
            "do not report this as the official IR-VIO result."
        ),
    }
    (results_dir / "dual_branch_proxy_summary.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
