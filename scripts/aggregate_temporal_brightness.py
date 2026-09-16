#!/usr/bin/env python3
"""Aggregate adjacent-frame brightness diagnostics for a fixed checkpoint label.

The output deliberately calls these global frame-mean diagnostics rather than
TSR: the available IR-VIO text does not define the numerator/denominator of
TSR, and this measurement does not inspect feature tracks.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=workspace / "results")
    parser.add_argument("--label", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    missing = []
    for sequence in args.sequences:
        path = args.results_root / sequence / f"{args.label}_brightness.json"
        if not path.exists():
            missing.append(sequence)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("raw_adjacent_delta_abs", {})
        enhanced = payload.get("enhanced_adjacent_delta_abs", {})
        raw_mean = raw.get("mean")
        enhanced_mean = enhanced.get("mean")
        rows.append({
            "sequence": sequence,
            "frames_compared": payload.get("frames_compared"),
            "raw_mean_intensity": payload.get("raw_mean_intensity"),
            "enhanced_mean_intensity": payload.get("enhanced_mean_intensity"),
            "raw_adjacent_delta_abs_mean": raw_mean,
            "enhanced_adjacent_delta_abs_mean": enhanced_mean,
            "raw_adjacent_delta_abs_p95": raw.get("p95"),
            "enhanced_adjacent_delta_abs_p95": enhanced.get("p95"),
            "enhanced_over_raw_delta_mean_ratio": (
                float(enhanced_mean) / float(raw_mean)
                if raw_mean is not None and enhanced_mean is not None and raw_mean != 0
                else None
            ),
            "source": str(path.resolve()),
        })

    def mean(key):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return statistics.mean(values) if values else None

    report = {
        "status": "temporal_brightness_aggregate_complete" if not missing else "temporal_brightness_aggregate_partial",
        "label": args.label,
        "requested_sequences": list(args.sequences),
        "complete_sequences": len(rows),
        "missing_sequences": missing,
        "mean_raw_intensity": mean("raw_mean_intensity"),
        "mean_enhanced_intensity": mean("enhanced_mean_intensity"),
        "mean_raw_adjacent_delta_abs": mean("raw_adjacent_delta_abs_mean"),
        "mean_enhanced_adjacent_delta_abs": mean("enhanced_adjacent_delta_abs_mean"),
        "mean_raw_adjacent_delta_abs_p95": mean("raw_adjacent_delta_abs_p95"),
        "mean_enhanced_adjacent_delta_abs_p95": mean("enhanced_adjacent_delta_abs_p95"),
        "mean_enhanced_over_raw_delta_mean_ratio": mean("enhanced_over_raw_delta_mean_ratio"),
        "rows": rows,
        "warning": "Global frame-mean brightness continuity is not the paper's TSR definition.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "label": args.label,
        "complete_sequences": len(rows),
        "missing_sequences": missing,
        "mean_raw_adjacent_delta_abs": report["mean_raw_adjacent_delta_abs"],
        "mean_enhanced_adjacent_delta_abs": report["mean_enhanced_adjacent_delta_abs"],
    }, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
