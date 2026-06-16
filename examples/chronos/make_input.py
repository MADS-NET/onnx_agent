#!/usr/bin/env python3
"""
make_input.py — Convert chicken.csv to a chronos-2-onnx input JSON file.

Reads the ``value`` column from chicken.csv (180 monthly observations),
pads the context to the nearest multiple of the model's patch size (16),
and writes an input JSON file conforming to the project's tensor<->JSON
mapping ({"shape": [...], "data": [...], "dtype": "..."} per tensor).

Usage:
  python3 examples/chronos/make_input.py [--csv PATH] [--out PATH] \
          [--prediction-months N]

Defaults:
  --csv   chicken.csv          (in examples/chronos/, auto-detected)
  --out   examples/chronos/chronos_input.json
  --prediction-months 12       (rounded up to nearest patch = 16 months out)

Requires: Python 3.8+, stdlib only (no numpy/pandas).

Model tensor schema (from --inspect on model_fixed.onnx):
  context            float32  [batch, context_length]   — time series values
  group_ids          int64    [batch]                   — series group id (0)
  attention_mask     float32  [batch, context_length]   — 1=real, 0=padding
  future_covariates  float32  [batch, future_length]    — zeros (no covariates)
  num_output_patches int64    []                        — number of output patches

Output tensor:
  quantile_preds     float32  [batch, 21, future_length]
    21 quantile levels: 0.01, 0.06, 0.11, ..., 0.94, 0.99
    Index 10 (q=0.50) is the median forecast.
"""

import argparse
import csv
import json
import math
import pathlib
import sys

# Model constants (from config.json of kashif/chronos-2-onnx)
PATCH_SIZE = 16  # output_patch_size / context tokenization granularity


def read_csv_values(csv_path: pathlib.Path) -> list[float]:
    """Read the 'value' column from a CSV with header 'time,value,idx'."""
    values: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["value"]))
    return values


def make_input(
    values: list[float],
    prediction_months: int = 12,
) -> dict:
    """
    Build the chronos-2-onnx input JSON dict from a flat list of float values.

    The context is left-padded with zeros to the nearest multiple of
    PATCH_SIZE so the model's Reshape(1, -1, 16) operation succeeds.
    The attention_mask is 0 for padding positions and 1 for real data.

    :param values:            Historical time series (univariate).
    :param prediction_months: Desired forecast horizon; rounded up to the
                              nearest multiple of PATCH_SIZE.
    :return: JSON-serialisable dict mapping tensor names to tensor objects.
    """
    n = len(values)
    patch_size = PATCH_SIZE

    # Pad context to nearest multiple of patch_size
    padded_len = math.ceil(n / patch_size) * patch_size
    pad_len = padded_len - n

    padded = [0.0] * pad_len + values
    mask = [0.0] * pad_len + [1.0] * n

    # Number of output patches (ceil of desired prediction / patch_size)
    num_patches = math.ceil(prediction_months / patch_size)  # 1 for 12 months
    future_length = num_patches * patch_size

    return {
        "context": {
            "shape": [1, padded_len],
            "data": padded,
            "dtype": "float32",
        },
        "group_ids": {
            "shape": [1],
            "data": [0],
            "dtype": "int64",
        },
        "attention_mask": {
            "shape": [1, padded_len],
            "data": mask,
            "dtype": "float32",
        },
        "future_covariates": {
            "shape": [1, future_length],
            "data": [0.0] * future_length,
            "dtype": "float32",
        },
        "num_output_patches": {
            "shape": [],
            "data": [num_patches],
            "dtype": "int64",
        },
    }


def _find_csv(explicit: pathlib.Path | None) -> pathlib.Path:
    """Locate chicken.csv: explicit path, or search upward from script dir."""
    if explicit is not None:
        if not explicit.exists():
            sys.exit(f"Error: CSV file not found: {explicit}")
        return explicit

    # Search from script directory upward for chicken.csv
    candidate = pathlib.Path(__file__).resolve()
    for _ in range(6):
        candidate = candidate.parent
        p = candidate / "chicken.csv"
        if p.exists():
            return p
    sys.exit(
        "Error: could not find chicken.csv. "
        "Pass --csv <path> to specify its location."
    )


def main() -> None:
    script_dir = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=pathlib.Path, default=None,
                        help="Path to chicken.csv (default: auto-detected)")
    parser.add_argument("--out", type=pathlib.Path,
                        default=script_dir / "chronos_input.json",
                        help="Output JSON file path")
    parser.add_argument("--prediction-months", type=int, default=12,
                        help="Forecast horizon in months (default: 12)")
    args = parser.parse_args()

    csv_path = _find_csv(args.csv)
    print(f"Reading: {csv_path}")
    values = read_csv_values(csv_path)
    print(f"  {len(values)} data points: {values[0]:.2f} .. {values[-1]:.2f}")

    prediction_months = args.prediction_months
    num_patches = math.ceil(prediction_months / PATCH_SIZE)
    future_length = num_patches * PATCH_SIZE

    print(f"Building input tensors:")
    print(f"  prediction_months = {prediction_months}")
    print(f"  num_output_patches = {num_patches}  (patch_size={PATCH_SIZE})")
    print(f"  future_length = {future_length}")

    n = len(values)
    padded_len = math.ceil(n / PATCH_SIZE) * PATCH_SIZE
    print(f"  context_length (padded) = {padded_len}  ({padded_len - n} zeros prepended)")

    result = make_input(values, prediction_months)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"Written: {out_path}")
    print()
    print("Tensor summary:")
    for name, t in result.items():
        shape = t["shape"]
        dtype = t["dtype"]
        n_elems = 1
        for d in shape:
            n_elems *= d
        print(f"  {name:<22s} dtype={dtype:<8s} shape={shape}  ({n_elems} elements)")


if __name__ == "__main__":
    main()
