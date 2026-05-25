#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompletedRun:
    dataset_name: str
    split_name: str
    run_tag: str
    csv_path: Path
    mse: float
    ci: float
    rm2: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate KANPM external baseline CSV outputs into summary tables")
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "--csv-dir",
        default="artifacts/external_runs/kanpm_ep15_seed42/csv",
        help="Directory containing Test-*.csv files",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/summary/kanpm_ep15_seed42_results.csv",
    )
    parser.add_argument(
        "--output-json",
        default="reports/summary/kanpm_ep15_seed42_grouped.json",
    )
    return parser


def load_expected_runs(workspace: Path) -> list[tuple[str, str]]:
    root = workspace / "external" / "KANPM-DTA" / "datasets"
    runs: list[tuple[str, str]] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for split_dir in sorted(dataset_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            if all((split_dir / part).exists() for part in ("train.csv", "valid.csv", "test.csv")):
                runs.append((dataset_dir.name, split_dir.name))
    return runs


def parse_completed_runs(csv_dir: Path, expected_runs: list[tuple[str, str]]) -> dict[tuple[str, str], CompletedRun]:
    completed: dict[tuple[str, str], CompletedRun] = {}
    ordered_expected = sorted(expected_runs, key=lambda item: (-len(f"{item[0]}-{item[1]}-"), item[0], item[1]))
    for csv_path in sorted(csv_dir.glob("Test-*.csv")):
        stem = csv_path.stem.removeprefix("Test-")
        matched: tuple[str, str] | None = None
        run_tag = ""
        for dataset_name, split_name in ordered_expected:
            prefix = f"{dataset_name}-{split_name}-"
            if stem.startswith(prefix):
                matched = (dataset_name, split_name)
                run_tag = stem[len(prefix):]
                break
        if matched is None:
            continue
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
        if row is None:
            continue
        record = CompletedRun(
            dataset_name=matched[0],
            split_name=matched[1],
            run_tag=run_tag,
            csv_path=csv_path,
            mse=float(row["mse"]),
            ci=float(row["ci"]),
            rm2=float(row["rm2"]),
        )
        previous = completed.get(matched)
        if previous is None or csv_path.stat().st_mtime > previous.csv_path.stat().st_mtime:
            completed[matched] = record
    return completed


def write_outputs(
    *,
    output_csv: Path,
    output_json: Path,
    expected_runs: list[tuple[str, str]],
    completed: dict[tuple[str, str], CompletedRun],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    grouped: dict[str, object] = {
        "total_expected": len(expected_runs),
        "num_completed": len(completed),
        "num_missing": len(expected_runs) - len(completed),
        "completed": {},
        "missing": [],
    }

    for dataset_name, split_name in expected_runs:
        key = (dataset_name, split_name)
        record = completed.get(key)
        if record is None:
            grouped["missing"].append({"dataset_name": dataset_name, "split_name": split_name})
            continue
        row = {
            "dataset_name": dataset_name,
            "split_name": split_name,
            "run_tag": record.run_tag,
            "mse": record.mse,
            "ci": record.ci,
            "rm2": record.rm2,
            "csv_path": str(record.csv_path),
        }
        rows.append(row)
        grouped["completed"][f"{dataset_name}|{split_name}"] = row

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_name", "split_name", "run_tag", "mse", "ci", "rm2", "csv_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_json.write_text(json.dumps(grouped, indent=2))


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    csv_dir = (workspace / args.csv_dir).resolve()
    expected_runs = load_expected_runs(workspace)
    completed = parse_completed_runs(csv_dir, expected_runs)
    write_outputs(
        output_csv=(workspace / args.output_csv).resolve(),
        output_json=(workspace / args.output_json).resolve(),
        expected_runs=expected_runs,
        completed=completed,
    )
    print(
        json.dumps(
            {
                "workspace": str(workspace),
                "csv_dir": str(csv_dir),
                "total_expected": len(expected_runs),
                "num_completed": len(completed),
                "num_missing": len(expected_runs) - len(completed),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

