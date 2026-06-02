#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect_run_summaries(workspace: Path) -> pd.DataFrame:
    roots = [
        workspace / "artifacts" / "external_runs" / "adambind_formal" / "runs",
        workspace / "artifacts" / "external_runs" / "adambind_viable11_refcsv" / "runs",
    ]
    rows: list[dict[str, object]] = []
    for root in roots:
        for summary_path in sorted(root.glob("*/run_summary.json")):
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            row = {
                "model_type": "adambind",
                "run_group": root.parent.name,
                "run_name": summary_path.parent.name,
                "dataset": payload.get("dataset_name"),
                "split": payload.get("split_name"),
                "seed": payload.get("seed"),
                "gnn": payload.get("gnn"),
                "nums": payload.get("nums"),
                "min_total_interactions": payload.get("min_total_interactions"),
                "status": payload.get("status"),
                "result_file": payload.get("result_file"),
                "run_summary_path": str(summary_path),
            }
            row.update({key: metrics.get(key) for key in ("mse", "ci", "r2", "spearman", "pearson")})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate AdaMBind external baseline results.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-csv", default="reports/summary/adambind_external_results.csv")
    parser.add_argument("--output-json", default="reports/summary/adambind_external_grouped.json")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    frame = collect_run_summaries(workspace)
    if frame.empty:
        raise FileNotFoundError("No completed AdaMBind run_summary.json files found.")

    output_csv = workspace / args.output_csv
    output_json = workspace / args.output_json
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["dataset", "split", "seed"]).to_csv(output_csv, index=False)

    grouped = (
        frame.groupby(["dataset", "split"], dropna=False)
        .agg(
            n=("run_name", "count"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            ci_mean=("ci", "mean"),
            ci_std=("ci", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
        )
        .reset_index()
    )
    payload = {
        "num_runs": int(len(frame)),
        "known_incompatible": ["bindingdb/unseen_target/seed42"],
        "note": (
            "BindingDB AdaMBind runs use the viable-target variant with "
            "min_total_interactions=11 because the upstream AdaMBind code groups by target_sequence."
        ),
        "grouped": grouped.to_dict(orient="records"),
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"num_runs": len(frame), "output_csv": str(output_csv), "output_json": str(output_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

