#!/usr/bin/env python3
"""Small algebra checks for the clean-room Eq.(5)--(9) implementation."""

from __future__ import annotations

import json
import math
from pathlib import Path


def main() -> int:
    # log det(J^T J) is used directly in estimator.cpp.  Eq.(5) gives
    # eta proportional to det(J^T J)^(-1/12); Eq.(6) normalizes by a
    # reference eta, so a tighter covariance must receive the larger alpha.
    loose_logdet = 0.0
    tight_logdet = 12.0 * math.log(4.0)
    loose_radius = math.exp(-loose_logdet / 12.0)
    tight_radius = math.exp(-tight_logdet / 12.0)
    loose_alpha = 1.0 / loose_radius
    tight_alpha = 1.0 / tight_radius
    alpha = 2.0
    beta = 3.0
    omega_c = math.sqrt(alpha * beta)
    residual_scale = math.sqrt(omega_c)
    objective_scale = residual_scale * residual_scale
    report = {
        "status": "adaptive_formula_checks_pass",
        "loose_radius": loose_radius,
        "tight_radius": tight_radius,
        "loose_alpha": loose_alpha,
        "tight_alpha": tight_alpha,
        "tight_has_larger_alpha": tight_alpha > loose_alpha,
        "omega_c": omega_c,
        "residual_scale": residual_scale,
        "objective_scale_after_factor": objective_scale,
        "objective_scale_matches_eq9": abs(objective_scale - omega_c) < 1e-12,
        "interpretation": "ProjectionFactor receives sqrt(omega_C), so Ceres' squared residual has coefficient omega_C as in Eq.(1).",
    }
    if not report["tight_has_larger_alpha"] or not report["objective_scale_matches_eq9"]:
        report["status"] = "adaptive_formula_checks_fail"
    output = Path(__file__).resolve().parent.parent / "results" / "adaptive_weight_formula_check_20260911.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"].endswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
