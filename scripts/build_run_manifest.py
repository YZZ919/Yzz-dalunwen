#!/usr/bin/env python3
"""Write a machine-readable manifest of the reproduction artifacts and statuses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
    "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]


def _exists(workspace: Path, relative: str) -> bool:
    return (workspace / relative).exists()


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=workspace / "results" / "reproduction_run_manifest_20260911.json")
    args = parser.parse_args()

    summary = workspace / "results" / "reference_comparison_20260911.json"
    all_low_history = workspace / "checkpoints" / "gray_dce_cleanroom_v2" / "gray_reference_official_all_low_resize256" / "history.csv"
    paper_ate = {
        sequence: _exists(workspace, f"results/{sequence}/dual_gray_dce_reference_paper_11seq_tracking_ate.json")
        for sequence in SEQUENCES
    }
    brightness = {
        sequence: _exists(workspace, f"results/{sequence}/dual_gray_dce_reference_official_11seq_tracking_brightness.json")
        for sequence in SEQUENCES
    }
    last_all_low_epoch = None
    if all_low_history.exists():
        for line in all_low_history.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split(",", 1)
            if fields and fields[0].isdigit():
                last_all_low_epoch = int(fields[0])
    manifest = {
        "status": "reproduction_manifest_complete",
        "date": "2026-09-11",
        "workspace": str(workspace.resolve()),
        "workspace_windows": r"D:\大论文实验\IR-VIO的复现",
        "platform": {
            "host": "Windows 11 + WSL2 Ubuntu 20.04.6",
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "torch": "2.4.1+cu124",
            "ros": "Noetic",
            "environment_log": str((workspace / "logs/environment.txt").resolve()),
        },
        "fixed_sources": {
            "irvio_pdf": r"C:\Users\Administrator\Desktop\大论文\很相关论文\IR-VIO_Illumination-Robust_Visual-Inertial_Odometry_Based_on_Adaptive_Weighting_Algorithm_With_Two-Layer_Confidence_Maximization.pdf",
            "zerodce_commit": "e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd",
            "zerodce_snapshot": "references/third_party/Zero-DCE",
            "sice_source": "https://huggingface.co/datasets/okhater/SICE",
            "sice_source_status": "mirror-derived; not verified equivalent to official Part1 2422/600 split",
        },
        "datasets": {
            "sice_gray_low1": {"train_images": 259, "val_images": 47, "selection": "low1"},
            "sice_gray_all_low": {"train_images": 1021, "val_images": 176, "selection": "all_low", "status": "mirror-derived"},
            "euroc": {"sequences": SEQUENCES, "status": "development/regression; not untouched blind test"},
        },
        "runs": {
            "historical_official": "results/dual_gray_dce_trained_official_11seq_summary.json",
            "historical_paper": "results/dual_gray_dce_trained_paper_11seq_summary.json",
            "reference_official_11seq": "results/dual_gray_dce_reference_official_11seq_tracking_summary.json",
            "reference_paper_11seq": "results/dual_gray_dce_reference_paper_11seq_tracking_summary.json",
            "reference_comparison": str(summary.resolve()),
            "all_low_training_history": str(all_low_history.resolve()),
        },
        "partial_status": {
            "reference_paper_ate_by_sequence": paper_ate,
            "reference_official_brightness_by_sequence": brightness,
            "all_low_training_last_completed_epoch": last_all_low_epoch,
            "all_low_training_best_checkpoint": _exists(workspace, "checkpoints/gray_dce_cleanroom_v2/gray_reference_official_all_low_resize256/best_val.pth"),
        },
        "audit_artifacts": [
            "references/implementation_audit_20260911.md",
            "references/sice_training_data.md",
            "results/loss_scale_audit_20260911.json",
            "results/loss_scale_audit_strict_fp32_20260911.json",
            "results/reproducibility_sanity_20260911.json",
        ],
        "warning": "Clean-room enhancement checkpoints are not the IR-VIO author checkpoint; paper TSR remains undefined in the available text.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "paper_ate_completed": sum(paper_ate.values()),
        "brightness_completed": sum(brightness.values()),
        "all_low_training_last_completed_epoch": manifest["partial_status"]["all_low_training_last_completed_epoch"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
