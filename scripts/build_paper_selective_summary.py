#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - optional dependency
    wilcoxon = None


EXPECTED_DATASETS = {"bindingdb", "davis", "kiba"}
EXPECTED_SPLITS = {"random", "unseen_drug", "unseen_target", "all_unseen", "similarity_aware_unseen_target"}
EXPECTED_SEEDS = {42, 43, 44}
EXPECTED_FORMAL_RUNS_BY_MODEL = {
    "adambind": 17,
}
EXCLUDE_RUN_PATTERNS = (r"(^|_)smoke(_|$)", r"fastdev")
PAIRWISE_BASELINES = ("mc_dropout", "target_familiarity", "conformal_mc_dropout", "deep_ensemble", "aleatoric")
PAIRWISE_METRICS = ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse")
SELECTIVE_EVALUATION_KINDS = {"selective_eval", "posthoc_selector"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper-ready selective DTA summary tables")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--grouped-json", required=True)
    parser.add_argument("--pairwise-csv", required=True)
    parser.add_argument("--ranking-csv", required=True)
    parser.add_argument("--full-rmse-csv", required=True)
    parser.add_argument("--coverage-audit-csv", required=True)
    return parser


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _should_exclude_run(run_name: str) -> bool:
    lowered = str(run_name).lower()
    return any(re.search(pattern, lowered) for pattern in EXCLUDE_RUN_PATTERNS)


def _summary_model_type(summary: dict[str, object], default: str = "baseline") -> str:
    return str(summary.get("model_type", summary.get("model_family", default))).lower()


def _load_external_rows(workspace: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    external_root = workspace / "artifacts" / "external_runs"
    if not external_root.exists():
        return pd.DataFrame()

    for runs_dir in sorted(external_root.glob("*/runs")):
        for summary_path in sorted(runs_dir.glob("*/run_summary.json")):
            run_dir = summary_path.parent
            summary = json.loads(summary_path.read_text())
            base_row = {
                "run_name": summary.get("run_name", run_dir.name),
                "dataset_name": str(summary.get("dataset_name", "")).lower(),
                "split_name": str(summary.get("split_name", "")),
                "seed": summary.get("seed"),
                "model_type": _summary_model_type(summary),
                "summary_path": str(summary_path),
            }
            for evaluation_kind in SELECTIVE_EVALUATION_KINDS:
                eval_dir = run_dir / evaluation_kind
                if not eval_dir.exists():
                    continue
                for metrics_path in sorted(eval_dir.glob("*_metrics.json")):
                    payload = json.loads(metrics_path.read_text())
                    for confidence_source, metrics in payload.items():
                        row = dict(base_row)
                        row["evaluation_kind"] = evaluation_kind
                        row["confidence_source"] = confidence_source
                        row["metrics_path"] = str(metrics_path)
                        row.update(metrics)
                        rows.append(row)
    return pd.DataFrame(rows)


def _load_pmmr_rows(workspace: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pmmr_root = workspace / "reports" / "deployment_upgrade_experiments" / "pmmr_training"
    if not pmmr_root.exists():
        return pd.DataFrame()

    for summary_path in sorted(pmmr_root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        dataset_name = str(summary.get("dataset_name", summary_path.parent.parent.name)).lower()
        split_name = str(summary.get("split_name", ""))
        seed = summary.get("seed")
        test_metrics = summary.get("test_metrics", {}) or {}
        run_name = f"pmmr_{dataset_name}_{split_name}_seed{seed}"
        rows.append(
            {
                "run_name": run_name,
                "dataset_name": dataset_name,
                "split_name": split_name,
                "seed": seed,
                "model_type": "pmmr",
                "evaluation_kind": "external_baseline",
                "confidence_source": "external_baseline",
                "summary_path": str(summary_path),
                "full_rmse": test_metrics.get("rmse"),
                "full_mae": test_metrics.get("mae"),
                "num_examples": summary.get("num_test_rows"),
            }
        )
    return pd.DataFrame(rows)


def _load_adambind_rows(workspace: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    roots = [
        workspace / "artifacts" / "external_runs" / "adambind_formal" / "runs",
        workspace / "artifacts" / "external_runs" / "adambind_viable11_refcsv" / "runs",
    ]
    for root in roots:
        if not root.exists():
            continue
        for summary_path in sorted(root.glob("*/run_summary.json")):
            summary = json.loads(summary_path.read_text())
            metrics = summary.get("metrics", {}) or {}
            mse = metrics.get("mse")
            full_rmse = None if mse is None else float(mse) ** 0.5
            rows.append(
                {
                    "run_name": str(summary_path.parent.name),
                    "dataset_name": str(summary.get("dataset_name", "")).lower(),
                    "split_name": str(summary.get("split_name", "")),
                    "seed": summary.get("seed"),
                    "model_type": "adambind",
                    "evaluation_kind": "external_baseline",
                    "confidence_source": "external_baseline",
                    "summary_path": str(summary_path),
                    "full_rmse": full_rmse,
                    "full_mae": None,
                }
            )
    return pd.DataFrame(rows)


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result["dataset_name"] = result["dataset_name"].astype(str).str.lower()
    result["split_name"] = result["split_name"].astype(str)
    result["model_type"] = result["model_type"].astype(str).str.lower()
    result["run_name"] = result["run_name"].astype(str)
    result["seed"] = pd.to_numeric(result["seed"], errors="coerce").astype("Int64")
    metric_columns = [
        "aurc",
        "coverage_50_rmse",
        "coverage_70_rmse",
        "coverage_90_rmse",
        "coverage_100_rmse",
        "full_rmse",
        "full_mae",
    ]
    for column in metric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "num_examples" in result.columns:
        result["num_examples"] = pd.to_numeric(result["num_examples"], errors="coerce").astype("Int64")
    return result


def _filter_paper_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result = result[result["dataset_name"].isin(EXPECTED_DATASETS)]
    result = result[result["split_name"].isin(EXPECTED_SPLITS)]
    result = result[result["seed"].isin(EXPECTED_SEEDS)]
    result = result[~result["run_name"].map(_should_exclude_run)]
    result = result.drop_duplicates(
        subset=["run_name", "model_type", "evaluation_kind", "confidence_source"],
        keep="first",
    )
    return result.sort_values(
        ["dataset_name", "split_name", "model_type", "run_name", "evaluation_kind", "confidence_source"]
    ).reset_index(drop=True)


def _build_grouped_payload(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    metric_columns = [
        column
        for column in frame.columns
        if column.startswith("coverage_") or column in {"aurc", "full_rmse", "full_mae", "num_examples"}
    ]
    for keys, bucket in frame.groupby(
        ["dataset_name", "split_name", "model_type", "evaluation_kind", "confidence_source"],
        dropna=False,
    ):
        key = "|".join(str(value) for value in keys)
        payload: dict[str, object] = {"num_runs": int(bucket["run_name"].nunique())}
        for metric_name in metric_columns:
            if metric_name not in bucket.columns:
                continue
            values = bucket[metric_name].dropna()
            if not values.empty:
                payload[f"mean_{metric_name}"] = round(float(values.mean()), 6)
        grouped[key] = payload
    return grouped


def _build_pairwise_stats(frame: pd.DataFrame) -> pd.DataFrame:
    non_oracle = frame[frame["confidence_source"] != "oracle"].copy()
    if non_oracle.empty:
        return pd.DataFrame()
    per_run = (
        non_oracle.groupby(["run_name", "dataset_name", "split_name", "model_type", "confidence_source"], as_index=False)
        .agg({metric: "mean" for metric in PAIRWISE_METRICS if metric in non_oracle.columns})
    )
    stats_rows: list[dict[str, object]] = []
    for metric_name in PAIRWISE_METRICS:
        if metric_name not in per_run.columns:
            continue
        pivot = per_run.pivot_table(
            index=["run_name", "dataset_name", "split_name", "model_type"],
            columns="confidence_source",
            values=metric_name,
            aggfunc="mean",
        )
        if "posthoc_selector" not in pivot.columns:
            continue
        for baseline_confidence in PAIRWISE_BASELINES:
            if baseline_confidence not in pivot.columns:
                continue
            paired = pivot[["posthoc_selector", baseline_confidence]].dropna()
            if paired.empty:
                continue
            selector = paired["posthoc_selector"]
            baseline = paired[baseline_confidence]
            diff = baseline - selector
            stats_row = {
                "metric_name": metric_name,
                "baseline_confidence_source": baseline_confidence,
                "num_pairs": int(len(paired)),
                "selector_win_rate": float((selector < baseline).mean()),
                "mean_abs_gain": float(diff.mean()),
                "mean_rel_gain": float((diff / baseline.replace(0.0, pd.NA)).dropna().mean()),
            }
            if wilcoxon is not None and len(paired) > 0:
                try:
                    _, p_value = wilcoxon(diff, alternative="greater")
                    stats_row["wilcoxon_p_value"] = float(p_value)
                except ValueError:
                    stats_row["wilcoxon_p_value"] = None
            stats_rows.append(stats_row)
    if not stats_rows:
        return pd.DataFrame()
    return pd.DataFrame(stats_rows).sort_values(["metric_name", "baseline_confidence_source"]).reset_index(drop=True)


def _build_selector_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["confidence_source"] != "oracle"].copy()
    frame = frame[frame["evaluation_kind"].isin(SELECTIVE_EVALUATION_KINDS)].copy()
    metric_columns = [
        column
        for column in ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse", "full_rmse")
        if column in frame.columns
    ]
    if not metric_columns:
        return pd.DataFrame()
    ranking = frame.groupby(["model_type", "evaluation_kind", "confidence_source"], as_index=False).agg(
        num_runs=("run_name", "nunique"),
        **{metric: (metric, "mean") for metric in metric_columns},
    )
    return ranking.sort_values(["aurc", "coverage_50_rmse", "full_rmse"], na_position="last").reset_index(drop=True)


def _build_full_rmse_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("full_rmse", "full_mae") if column in frame.columns]
    if not columns:
        return pd.DataFrame()
    one_row_per_run = frame.sort_values(["run_name", "evaluation_kind", "confidence_source"]).drop_duplicates("run_name")
    ranking = one_row_per_run.groupby("model_type", as_index=False).agg(
        num_runs=("run_name", "nunique"),
        **{metric: (metric, "mean") for metric in columns},
    )
    return ranking.sort_values(["full_rmse", "full_mae"], na_position="last").reset_index(drop=True)


def _build_coverage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    default_expected_total = len(EXPECTED_DATASETS) * len(EXPECTED_SPLITS) * len(EXPECTED_SEEDS)
    audit_rows: list[dict[str, object]] = []
    for model_type, bucket in frame.groupby("model_type", dropna=False):
        expected_total = EXPECTED_FORMAL_RUNS_BY_MODEL.get(str(model_type), default_expected_total)
        formal_combos = bucket[["dataset_name", "split_name", "seed"]].drop_duplicates()
        selective_combos = (
            bucket[bucket["evaluation_kind"] == "selective_eval"][["dataset_name", "split_name", "seed"]]
            .drop_duplicates()
            .shape[0]
        )
        posthoc_combos = (
            bucket[bucket["evaluation_kind"] == "posthoc_selector"][["dataset_name", "split_name", "seed"]]
            .drop_duplicates()
            .shape[0]
        )
        formal_combo_count = int(formal_combos.shape[0])
        audit_rows.append(
            {
                "model_type": model_type,
                "formal_run_count": formal_combo_count,
                "expected_formal_runs": expected_total,
                "missing_formal_runs": int(max(expected_total - formal_combo_count, 0)),
                "selective_run_count": int(selective_combos),
                "posthoc_run_count": int(posthoc_combos),
            }
        )
    return pd.DataFrame(audit_rows).sort_values("model_type").reset_index(drop=True)


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    input_csv = Path(args.input_csv).resolve()
    frame = pd.read_csv(input_csv)
    external_rows = _load_external_rows(workspace)
    pmmr_rows = _load_pmmr_rows(workspace)
    adambind_rows = _load_adambind_rows(workspace)
    merged = pd.concat([frame, external_rows, pmmr_rows, adambind_rows], ignore_index=True, sort=False)
    merged = _coerce_types(merged)
    paper_frame = _filter_paper_rows(merged)

    output_csv = Path(args.output_csv).resolve()
    grouped_json = Path(args.grouped_json).resolve()
    pairwise_csv = Path(args.pairwise_csv).resolve()
    ranking_csv = Path(args.ranking_csv).resolve()
    full_rmse_csv = Path(args.full_rmse_csv).resolve()
    coverage_audit_csv = Path(args.coverage_audit_csv).resolve()
    for path in (output_csv, grouped_json, pairwise_csv, ranking_csv, full_rmse_csv, coverage_audit_csv):
        _ensure_parent(path)

    paper_frame.to_csv(output_csv, index=False)
    grouped_json.write_text(json.dumps(_build_grouped_payload(paper_frame), indent=2))
    _build_pairwise_stats(paper_frame).to_csv(pairwise_csv, index=False)
    _build_selector_ranking(paper_frame).to_csv(ranking_csv, index=False)
    _build_full_rmse_ranking(paper_frame).to_csv(full_rmse_csv, index=False)
    _build_coverage_audit(paper_frame).to_csv(coverage_audit_csv, index=False)

    print(
        json.dumps(
            {
                "num_rows": int(len(paper_frame)),
                "output_csv": str(output_csv),
                "grouped_json": str(grouped_json),
                "pairwise_csv": str(pairwise_csv),
                "ranking_csv": str(ranking_csv),
                "full_rmse_csv": str(full_rmse_csv),
                "coverage_audit_csv": str(coverage_audit_csv),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

