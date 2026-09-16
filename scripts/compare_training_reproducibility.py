#!/usr/bin/env python3
"""Compare two training checkpoints produced with the same deterministic config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch  # pylint: disable=import-outside-toplevel

    first = torch.load(str(args.first), map_location="cpu", weights_only=False)
    second = torch.load(str(args.second), map_location="cpu", weights_only=False)
    max_abs = 0.0
    different_values = 0
    for name in first["model"]:
        difference = (first["model"][name] - second["model"][name]).abs()
        max_abs = max(max_abs, float(difference.max()))
        different_values += int(torch.count_nonzero(difference))
    report = {
        "status": "reproducibility_compare_complete",
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "epochs_equal": first.get("epoch") == second.get("epoch"),
        "val_loss_equal": first.get("val_loss") == second.get("val_loss"),
        "max_model_abs_difference": max_abs,
        "different_model_values": different_values,
        "bitwise_model_equal": different_values == 0,
        "note": "Deterministic CUDA run pair; checkpoint payload hashes can still differ because RNG/config payloads are serialized separately.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["bitwise_model_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
