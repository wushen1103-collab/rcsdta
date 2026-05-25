#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate selective and post-hoc metric artifacts")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _iter_metric_artifacts(run_dir: Path) -> list[tuple[str, Path]]:
    metric_specs = [
        ("selective_eval", run_dir / "selective_eval"),
        ("posthoc_selector", run_dir / "posthoc_selector"),
    ]
    artifacts: list[tuple[str, Path]] = []
    for evaluation_kind, artifact_dir in metric_specs:
        if not artifact_dir.exists():
            continue
        for metrics_path in sorted(artifact_dir.glob("*_metrics.json")):
            artifacts.append((evaluation_kind, metrics_path))
    return artifacts


def _iter_run_summary_paths(workspace: Path) -> list[Path]:
    summary_paths: list[Path] = []
    summary_paths.extend(sorted((workspace / "artifacts" / "runs").glob("*/run_summary.json")))
    for external_root in sorted((workspace / "artifacts" / "external_runs").glob("*")):
        external_runs_dir = external_root / "runs"
        if not external_runs_dir.exists():
            continue
        summary_paths.extend(sorted(external_runs_dir.glob("*/run_summary.json")))
    return summary_paths


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    rows: list[dict[str, object]] = []
    metric_fields: set[str] = set()

    for summary_path in _iter_run_summary_paths(workspace):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        base_row = {
            "run_name": summary.get("run_name"),
            "dataset_name": summary.get("dataset_name"),
            "split_name": summary.get("split_name"),
            "seed": summary.get("seed"),
            "model_type": summary.get("model_type", "baseline"),
            "summary_path": str(summary_path),
        }
        for evaluation_kind, metrics_path in _iter_metric_artifacts(run_dir):
            payload = json.loads(metrics_path.read_text())
            for confidence_source, metrics in payload.items():
                row = dict(base_row)
                row["evaluation_kind"] = evaluation_kind
                row["confidence_source"] = confidence_source
                row["metrics_path"] = str(metrics_path)
                for key, value in metrics.items():
                    row[key] = value
                    metric_fields.add(key)
                rows.append(row)

    ordered_fields = [
        "run_name",
        "dataset_name",
        "split_name",
        "seed",
        "model_type",
        "evaluation_kind",
        "confidence_source",
        *sorted(metric_fields),
        "summary_path",
        "metrics_path",
    ]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    grouped_metrics: dict[str, dict[str, object]] = {}
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = "|".join(
            [
                str(row["dataset_name"]),
                str(row["split_name"]),
                str(row["model_type"]),
                str(row["evaluation_kind"]),
                str(row["confidence_source"]),
            ]
        )
        buckets[key].append(row)

    for key, bucket in buckets.items():
        grouped_payload: dict[str, object] = {"num_runs": len(bucket)}
        for metric_name in sorted(metric_fields):
            values: list[float] = []
            for row in bucket:
                value = row.get(metric_name)
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                grouped_payload[f"mean_{metric_name}"] = round(sum(values) / len(values), 6)
        grouped_metrics[key] = grouped_payload

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(grouped_metrics, indent=2))
    print(
        json.dumps(
            {
                "num_rows": len(rows),
                "output_csv": str(output_csv.resolve()),
                "output_json": str(output_json.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


