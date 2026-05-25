#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate run_summary.json files into tabular reports")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    rows: list[dict[str, object]] = []

    for summary_path in sorted((workspace / "artifacts" / "runs").glob("*/run_summary.json")):
        payload = json.loads(summary_path.read_text())
        row = {
            "run_name": payload.get("run_name"),
            "dataset_name": payload.get("dataset_name"),
            "split_name": payload.get("split_name"),
            "seed": payload.get("seed"),
            "status": payload.get("status"),
            "test_loss": payload.get("metrics", {}).get("test_loss"),
            "test_mae": payload.get("metrics", {}).get("test_mae"),
            "test_rmse": payload.get("metrics", {}).get("test_rmse"),
            "summary_path": str(summary_path),
        }
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "run_name", "dataset_name", "split_name", "seed", "status", "test_loss", "test_mae", "test_rmse", "summary_path"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    grouped_metrics: dict[str, dict[str, object]] = {}
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = f"{row['dataset_name']}|{row['split_name']}"
        buckets[key].append(row)
    for key, bucket in buckets.items():
        rmse_values = [float(row["test_rmse"]) for row in bucket if row["test_rmse"] is not None]
        mae_values = [float(row["test_mae"]) for row in bucket if row["test_mae"] is not None]
        grouped_metrics[key] = {
            "num_runs": len(bucket),
            "mean_test_rmse": round(sum(rmse_values) / len(rmse_values), 6) if rmse_values else None,
            "mean_test_mae": round(sum(mae_values) / len(mae_values), 6) if mae_values else None,
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(grouped_metrics, indent=2))
    print(json.dumps({"num_runs": len(rows), "output_csv": str(output_csv.resolve()), "output_json": str(output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
