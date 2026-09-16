#!/usr/bin/env python3
"""Aggregate EuRoC per-sequence metrics into CSV and JSON summaries."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean

SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
    "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]
BASE_METRICS = ["vins_mono_asl_metrics.json", "metrics.json"]
WEIGHT_METRICS = ["irvio_weighting_asl_metrics.json", "irvio_weighting_metrics.json"]
WEIGHT_LOGS = ["irvio_weighting_asl_weights.csv", "irvio_weighting_weights.csv"]


def first_file(directory, names):
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_weight_stats(path):
    if path is None:
        return {}
    columns = {
        "alpha": "alpha_weight",
        "beta": "mean_beta_weight",
        "hybrid_weight": "mean_hybrid_weight",
    }
    values = {field: [] for field in columns}
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            try:
                valid = float(row["covariance_radius"]) > 0 and int(row["constraint_count"]) > 0
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                continue
            for field, column in columns.items():
                try:
                    value = float(row[column])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values[field].append(value)
    if not values["hybrid_weight"]:
        return {}
    return {
        "weight_rows": rows,
        "valid_weight_rows": len(values["hybrid_weight"]),
        "alpha_min": min(values["alpha"]),
        "alpha_mean": fmean(values["alpha"]),
        "alpha_max": max(values["alpha"]),
        "beta_mean": fmean(values["beta"]),
        "hybrid_mean": fmean(values["hybrid_weight"]),
    }


def source_label(sequence, baseline_path):
    if baseline_path.name == "vins_mono_asl_metrics.json":
        return "EuRoC ASL ZIP from Hugging Face mirror"
    if sequence == "MH_01_easy":
        return "EuRoC ROS1 bag from Hugging Face mirror; sparse Leica position"
    return "EuRoC ASL ZIP from Hugging Face mirror"


def format_value(value, digits=10):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--paper-sequences-total", type=int, default=11)
    args = parser.parse_args()
    root = Path(args.results_root).resolve()
    summary_rows, comparisons, records = [], [], []

    for sequence in SEQUENCES:
        directory = root / sequence
        baseline_path = first_file(directory, BASE_METRICS)
        weighting_path = first_file(directory, WEIGHT_METRICS)
        if baseline_path is None or weighting_path is None:
            records.append({"name": sequence, "status": "not_run"})
            continue
        baseline, weighting = load_json(baseline_path), load_json(weighting_path)
        source = source_label(sequence, baseline_path)
        for method, metric in (("vins_mono_baseline", baseline),
                               ("clean_room_two_layer_weighting", weighting)):
            summary_rows.append({
                "sequence": sequence, "method": method, "data_source": source,
                "replay": "complete", "estimate_poses": metric["estimate_poses"],
                "matched_poses": metric["matched_poses"],
                "time_coverage": metric["time_coverage"],
                "ate_rmse_se3_m": metric["ate_rmse_se3_m"],
                "ate_rmse_sim3_m": metric["ate_rmse_sim3_m"],
                "pipeline_success": metric["pipeline_success"],
                "metric_pass_ate_lt_1m": metric["metric_pass_ate_lt_1m"],
            })
        base_ate = float(baseline["ate_rmse_se3_m"])
        weight_ate = float(weighting["ate_rmse_se3_m"])
        reduction = 100.0 * (base_ate - weight_ate) / base_ate
        comparison = {
            "sequence": sequence,
            "baseline_ate_rmse_se3_m": base_ate,
            "weighting_ate_rmse_se3_m": weight_ate,
            "relative_ate_reduction_percent": reduction,
            "baseline_success": bool(baseline["pipeline_success"]),
            "weighting_success": bool(weighting["pipeline_success"]),
            **collect_weight_stats(first_file(directory, WEIGHT_LOGS)),
        }
        comparisons.append(comparison)
        records.append({
            "name": sequence,
            "status": "success" if comparison["baseline_success"] and comparison["weighting_success"] else "failed",
            "baseline_ate_rmse_se3_m": base_ate,
            "weighting_ate_rmse_se3_m": weight_ate,
            "relative_reduction_percent": reduction,
        })

    summary_fields = [
        "sequence", "method", "data_source", "replay", "estimate_poses",
        "matched_poses", "time_coverage", "ate_rmse_se3_m", "ate_rmse_sim3_m",
        "pipeline_success", "metric_pass_ate_lt_1m",
    ]
    comparison_fields = [
        "sequence", "baseline_ate_rmse_se3_m", "weighting_ate_rmse_se3_m",
        "relative_ate_reduction_percent", "baseline_success", "weighting_success",
        "weight_rows", "valid_weight_rows", "alpha_min", "alpha_mean", "alpha_max",
        "beta_mean", "hybrid_mean",
    ]
    write_csv(root / "summary.csv", summary_rows, summary_fields)
    write_csv(root / "weighting_comparison.csv", comparisons, comparison_fields)

    attempted = len(comparisons)
    base_ok = sum(row["baseline_success"] for row in comparisons)
    weight_ok = sum(row["weighting_success"] for row in comparisons)
    comparable = [row for row in comparisons if row["baseline_success"] and row["weighting_success"]]
    improved = sum(row["relative_ate_reduction_percent"] > 0 for row in comparable)
    payload = {
        "dataset": "EuRoC MAV", "attempted_sequences": attempted,
        "baseline_successful_sequences": base_ok,
        "baseline_attempted_success_rate": base_ok / attempted if attempted else None,
        "weighting_successful_sequences": weight_ok,
        "weighting_attempted_success_rate": weight_ok / attempted if attempted else None,
        "paper_sequences_total": args.paper_sequences_total,
        "completed_fraction_of_paper_set": attempted / args.paper_sequences_total,
        "full_set_success_rate": weight_ok / attempted if attempted == args.paper_sequences_total else None,
        "comparable_sequences": len(comparable),
        "weighting_improved_sequences": improved,
        "weighting_precision_win_rate": improved / len(comparable) if comparable else None,
        "definition": "success = >=100 matched finite poses, >=70% camera-time coverage, finite SE(3)-aligned ATE",
        "implementation_status": "clean-room equations (2)-(9); not official or full IR-VIO",
        "sequences": records,
    }
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"attempted": attempted, "baseline_successes": base_ok,
                      "weighting_successes": weight_ok, "improved": improved,
                      "comparable": len(comparable)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
