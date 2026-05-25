#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_DATASETS = ("davis", "bindingdb", "kiba")
EXPECTED_SPLITS = ("random", "unseen_target", "unseen_drug", "all_unseen", "similarity_aware_unseen_target")
EXPECTED_SEEDS = (42, 43, 44)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate DeepDTAGen run summaries into dedicated reports.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def expected_run_names() -> set[str]:
    return {
        f"deepdtagen_ep15_{dataset}_{split}_seed{seed}"
        for dataset in EXPECTED_DATASETS
        for split in EXPECTED_SPLITS
        for seed in EXPECTED_SEEDS
    }


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    expected = expected_run_names()
    rows: list[dict[str, object]] = []

    for summary_path in sorted((workspace / "artifacts" / "runs").glob("deepdtagen_ep15_*/run_summary.json")):
        payload = json.loads(summary_path.read_text())
        run_name = payload.get("run_name")
        if run_name not in expected:
            continue
        metrics = payload.get("metrics", {})
        row = {
            "run_name": run_name,
            "dataset_name": payload.get("dataset_name"),
            "split_name": payload.get("split_name"),
            "seed": payload.get("seed"),
            "status": payload.get("status"),
            "test_rmse": metrics.get("test_rmse"),
            "test_mse": metrics.get("test_mse"),
            "test_mae": metrics.get("test_mae"),
            "test_ci": metrics.get("test_ci"),
            "test_pearson": metrics.get("test_pearson"),
            "test_spearman": metrics.get("test_spearman"),
            "test_rm2": metrics.get("test_rm2"),
            "summary_path": str(summary_path),
        }
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "dataset_name",
        "split_name",
        "seed",
        "status",
        "test_rmse",
        "test_mse",
        "test_mae",
        "test_ci",
        "test_pearson",
        "test_spearman",
        "test_rm2",
        "summary_path",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    grouped: dict[str, dict[str, object]] = {}
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = f"{row['dataset_name']}|{row['split_name']}"
        buckets[key].append(row)

    for key, bucket in buckets.items():
        def _mean(metric_name: str):
            values = [float(item[metric_name]) for item in bucket if item[metric_name] is not None]
            return round(sum(values) / len(values), 6) if values else None

        grouped[key] = {
            "num_runs": len(bucket),
            "mean_test_rmse": _mean("test_rmse"),
            "mean_test_mae": _mean("test_mae"),
            "mean_test_ci": _mean("test_ci"),
            "mean_test_pearson": _mean("test_pearson"),
            "mean_test_spearman": _mean("test_spearman"),
            "mean_test_rm2": _mean("test_rm2"),
        }

    missing = sorted(expected - {str(row["run_name"]) for row in rows})
    payload = {
        "num_runs": len(rows),
        "num_expected": len(expected),
        "num_missing": len(missing),
        "missing_run_names": missing,
        "grouped_metrics": grouped,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"num_runs": len(rows), "num_expected": len(expected), "num_missing": len(missing), "output_csv": str(output_csv.resolve()), "output_json": str(output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

