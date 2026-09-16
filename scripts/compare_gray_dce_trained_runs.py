#!/usr/bin/env python3
"""Create a reproducible comparison report for the two gray-DCE profiles.

The report intentionally keeps the training and EuRoC-test evidence separate:
the best checkpoint is read from SICE validation history, while ATE rows are
read from the fixed, preselected sequence results.  It does not choose a
profile or epoch from EuRoC performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_PROFILES = ("official", "paper")
DEFAULT_SEQUENCES = ("V1_01_easy", "V1_03_difficult", "V2_03_difficult")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_first_json(root: Path, names: tuple[str, ...]) -> dict:
    for name in names:
        value = read_json(root / name)
        if isinstance(value, dict):
            return value
    return {}


def read_history(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def training_summary(checkpoint_root: Path, profile: str) -> dict:
    run_root = checkpoint_root / f"gray_{profile}_mirror"
    config = read_json(run_root / "training_config.json") or {}
    history = read_history(run_root / "history.csv")
    rows = [row for row in history if as_float(row.get("val_loss")) is not None]
    best = min(rows, key=lambda row: as_float(row["val_loss"])) if rows else {}
    last = max(rows, key=lambda row: int(row.get("epoch", 0))) if rows else {}
    checkpoint = run_root / "best_val.pth"
    return {
        "profile": profile,
        "checkpoint_dir": str(run_root),
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_exists": checkpoint.is_file(),
        "history_epochs": len(rows),
        "best_epoch": int(best["epoch"]) if best else None,
        "best_val_loss": as_float(best.get("val_loss")),
        "best_val_spa": as_float(best.get("val_spa")),
        "best_val_exp": as_float(best.get("val_exp")),
        "best_val_tv": as_float(best.get("val_tv")),
        "final_epoch": int(last["epoch"]) if last else None,
        "final_train_loss": as_float(last.get("train_loss")),
        "final_val_loss": as_float(last.get("val_loss")),
        "training_config": config,
    }


def test_rows(results_root: Path, profile: str, sequences: tuple[str, ...]) -> list[dict]:
    label = f"dual_gray_dce_trained_{profile}"
    rows = []
    for sequence in sequences:
        root = results_root / sequence
        report = read_json(root / f"{label}_ate.json")
        baseline = read_first_json(root, ("vins_mono_asl_metrics.json", "metrics.json"))
        tw = read_first_json(
            root, ("irvio_weighting_asl_metrics.json", "irvio_weighting_metrics.json")
        )
        if report is None:
            rows.append({"profile": profile, "sequence": sequence, "status": "missing"})
            continue
        ate = as_float(report.get("ate_rmse_se3_m"))
        baseline_ate = as_float(baseline.get("ate_rmse_se3_m"))
        tw_ate = as_float(tw.get("ate_rmse_se3_m"))
        rows.append({
            "profile": profile,
            "sequence": sequence,
            "status": "complete",
            "pipeline_success": bool(report.get("pipeline_success", False)),
            "matched_poses": report.get("matched_poses"),
            "time_coverage": report.get("time_coverage"),
            "ate_rmse_se3_m": ate,
            "baseline_ate_rmse_se3_m": baseline_ate,
            "tw_ate_rmse_se3_m": tw_ate,
            "reduction_vs_baseline_pct": (
                100.0 * (baseline_ate - ate) / baseline_ate
                if ate is not None and baseline_ate not in (None, 0.0) else None
            ),
            "reduction_vs_tw_pct": (
                100.0 * (tw_ate - ate) / tw_ate
                if ate is not None and tw_ate not in (None, 0.0) else None
            ),
        })
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    complete = [row for row in rows if row.get("status") == "complete"]
    successful = [
        row for row in complete
        if row.get("pipeline_success")
        and (row.get("matched_poses") or 0) >= 100
        and (row.get("time_coverage") or 0.0) >= 0.70
        and row.get("ate_rmse_se3_m") is not None
    ]
    ates = [float(row["ate_rmse_se3_m"]) for row in successful]
    reductions_baseline = [
        float(row["reduction_vs_baseline_pct"])
        for row in successful if row.get("reduction_vs_baseline_pct") is not None
    ]
    reductions_tw = [
        float(row["reduction_vs_tw_pct"])
        for row in successful if row.get("reduction_vs_tw_pct") is not None
    ]
    baseline_ates = [
        float(row["baseline_ate_rmse_se3_m"])
        for row in successful if row.get("baseline_ate_rmse_se3_m") is not None
    ]
    tw_ates = [
        float(row["tw_ate_rmse_se3_m"])
        for row in successful if row.get("tw_ate_rmse_se3_m") is not None
    ]
    mean_ate = statistics.mean(ates) if ates else None
    mean_baseline = statistics.mean(baseline_ates) if baseline_ates else None
    mean_tw = statistics.mean(tw_ates) if tw_ates else None
    return {
        "requested": len(rows),
        "completed": len(complete),
        "successful": len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "mean_ate_rmse_se3_m": mean_ate,
        "median_ate_rmse_se3_m": statistics.median(ates) if ates else None,
        "mean_baseline_ate_rmse_se3_m": mean_baseline,
        "mean_tw_ate_rmse_se3_m": mean_tw,
        "reduction_of_mean_vs_baseline_pct": (
            100.0 * (mean_baseline - mean_ate) / mean_baseline
            if mean_ate is not None and mean_baseline not in (None, 0.0) else None
        ),
        "reduction_of_mean_vs_tw_pct": (
            100.0 * (mean_tw - mean_ate) / mean_tw
            if mean_ate is not None and mean_tw not in (None, 0.0) else None
        ),
        "mean_reduction_vs_baseline_pct": (
            statistics.mean(reductions_baseline) if reductions_baseline else None
        ),
        "mean_reduction_vs_tw_pct": statistics.mean(reductions_tw) if reductions_tw else None,
    }


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=workspace / "results")
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=workspace / "checkpoints" / "gray_dce_cleanroom",
    )
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument(
        "--output-prefix", type=Path,
        default=workspace / "results" / "gray_dce_cleanroom_comparison_3seq",
    )
    args = parser.parse_args()

    profiles = tuple(args.profiles)
    sequences = tuple(args.sequences)
    training = [training_summary(args.checkpoint_root, profile) for profile in profiles]
    rows_by_profile = {
        profile: test_rows(args.results_root, profile, sequences) for profile in profiles
    }
    test_rows_all = [row for profile in profiles for row in rows_by_profile[profile]]
    summaries = {profile: summarize_rows(rows_by_profile[profile]) for profile in profiles}

    profile_diff = {}
    if "official" in rows_by_profile and "paper" in rows_by_profile:
        official = {row["sequence"]: row for row in rows_by_profile["official"]}
        paper = {row["sequence"]: row for row in rows_by_profile["paper"]}
        for sequence in sequences:
            off_ate = as_float(official.get(sequence, {}).get("ate_rmse_se3_m"))
            paper_ate = as_float(paper.get(sequence, {}).get("ate_rmse_se3_m"))
            profile_diff[sequence] = {
                "official_ate_rmse_se3_m": off_ate,
                "paper_ate_rmse_se3_m": paper_ate,
                "paper_minus_official_m": (
                    paper_ate - off_ate
                    if off_ate is not None and paper_ate is not None else None
                ),
                "winner_lower_ate": (
                    "official" if off_ate < paper_ate else "paper" if paper_ate < off_ate else "tie"
                ) if off_ate is not None and paper_ate is not None else None,
            }

    report = {
        "status": "gray_dce_cleanroom_training_and_euroc_comparison",
        "profiles": list(profiles),
        "sequences": list(sequences),
        "training": training,
        "test_summary": summaries,
        "test_rows": test_rows_all,
        "profile_diff": profile_diff,
        "selection_rule": "best checkpoint selected by SICE validation loss only; EuRoC is test-only",
        "warning": (
            "Both profiles are clean-room grayscale DCE checkpoints trained from the stated "
            "architecture and mirror-derived data; neither is the unpublished author IR-VIO model."
        ),
    }
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "profile", "sequence", "status", "pipeline_success", "matched_poses", "time_coverage",
        "ate_rmse_se3_m", "baseline_ate_rmse_se3_m", "tw_ate_rmse_se3_m",
        "reduction_vs_baseline_pct", "reduction_vs_tw_pct",
    ]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(test_rows_all)
    print(json.dumps({"output": str(prefix), "summaries": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
