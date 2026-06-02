#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, wilcoxon

from selective_dta_b.eval.followup_experiments import (
    PredictionRecord,
    discover_prediction_records,
    ensure_error_columns,
    selective_metrics,
)
from selective_dta_b.eval.posthoc import fit_posthoc_error_regressor, predict_posthoc_error


PRIMARY_REGRESSOR = "ridge"
PRIMARY_FEATURE_SET = "enriched9"
PRIMARY_SELECTOR_NAME = "ridge_enriched9"
PRIMARY_RISK_LAMBDA = 1.0
PRIMARY_MODELS = ("adambind", "deepdtagen", "kanpm", "pmmr")
BASELINE_SOURCES = ("mc_dropout", "target_familiarity")
SELECTIVE_METRICS = ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse")
ABS_ERROR_THRESHOLDS = (0.75, 1.00, 1.25)
TARGET_VIOLATION_RATES = (0.10, 0.20)
CANDIDATE_COVERAGES = tuple(np.round(np.arange(0.05, 1.001, 0.05), 2))
VS_BUDGETS = (10, 50)
ACTIVE_FRACTION = 0.10


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _load(path: Path) -> pd.DataFrame:
    frame = ensure_error_columns(pd.read_csv(path))
    if "confidence_mc_dropout" not in frame and "prediction_std_mc_dropout" in frame:
        frame["confidence_mc_dropout"] = 1.0 / (1.0 + frame["prediction_std_mc_dropout"].clip(lower=0.0))
    return frame


def _score_primary(validation: pd.DataFrame, test: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["prediction_mean", "prediction_std_mc_dropout", "target_familiarity", "target_novelty", "abs_error"]
    missing = [column for column in required if column not in validation or column not in test]
    if missing:
        raise KeyError(f"Missing primary-selector columns: {missing}")
    regressor = fit_posthoc_error_regressor(
        validation,
        random_state=seed,
        regressor_type=PRIMARY_REGRESSOR,
        feature_set=PRIMARY_FEATURE_SET,
    )
    validation = validation.copy()
    test = test.copy()
    validation["predicted_abs_error_primary"] = predict_posthoc_error(regressor, validation)
    test["predicted_abs_error_primary"] = predict_posthoc_error(regressor, test)
    validation["confidence_primary_selector"] = 1.0 / (1.0 + validation["predicted_abs_error_primary"])
    test["confidence_primary_selector"] = 1.0 / (1.0 + test["predicted_abs_error_primary"])
    return validation, test


def _payload(record: PredictionRecord) -> dict[str, object]:
    return {
        "run_name": record.run_name,
        "dataset_name": record.dataset_name,
        "split_name": record.split_name,
        "seed": int(record.seed),
        "model_type": record.model_type,
        "primary_selector": PRIMARY_SELECTOR_NAME,
    }


def _unique_records(workspace: Path) -> list[PredictionRecord]:
    records = discover_prediction_records(workspace, paper_only=True, max_runs=None)
    selected: dict[str, PredictionRecord] = {}
    for record in records:
        if record.validation_path is not None and record.validation_path.exists() and record.test_path.exists():
            selected.setdefault(record.run_name, record)
    return list(selected.values())


def _holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda idx: pvalues[idx] if math.isfinite(pvalues[idx]) else float("inf"))
    output = [float("nan")] * len(pvalues)
    running = 0.0
    for rank, idx in enumerate(order):
        pvalue = pvalues[idx]
        if not math.isfinite(pvalue):
            continue
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalue))
        output[idx] = running
    return output


def _pairwise_rows(summary: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in SELECTIVE_METRICS:
        pivot = summary.pivot_table(index="run_name", columns="confidence_source", values=metric, aggfunc="first")
        for baseline in BASELINE_SOURCES:
            if "primary_selector" not in pivot or baseline not in pivot:
                continue
            paired = pivot[["primary_selector", baseline]].dropna()
            if paired.empty:
                continue
            delta = paired["primary_selector"] - paired[baseline]
            try:
                pvalue = float(wilcoxon(delta, alternative="less", zero_method="zsplit").pvalue)
            except ValueError:
                pvalue = float("nan")
            rows.append(
                {
                    "scope": scope,
                    "metric_name": metric,
                    "baseline_confidence_source": baseline,
                    "num_pairs": int(len(delta)),
                    "primary_mean": float(paired["primary_selector"].mean()),
                    "baseline_mean": float(paired[baseline].mean()),
                    "mean_delta_primary_minus_baseline": float(delta.mean()),
                    "primary_win_rate": float(np.mean(delta < 0)),
                    "wilcoxon_p_value": pvalue,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_p_value"] = _holm_adjust(result["wilcoxon_p_value"].tolist())
    return result


def _block_bootstrap(summary: pd.DataFrame, *, reps: int = 4000) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    for metric in SELECTIVE_METRICS:
        pivot = summary.pivot_table(
            index=["run_name", "dataset_name", "split_name", "model_type"],
            columns="confidence_source",
            values=metric,
            aggfunc="first",
        ).reset_index()
        for baseline in BASELINE_SOURCES:
            if "primary_selector" not in pivot or baseline not in pivot:
                continue
            paired = pivot.dropna(subset=["primary_selector", baseline]).copy()
            paired["delta"] = paired["primary_selector"] - paired[baseline]
            blocks = (
                paired.groupby(["dataset_name", "split_name", "model_type"], as_index=False)["delta"]
                .mean()["delta"]
                .to_numpy(dtype=float)
            )
            samples = rng.choice(blocks, size=(reps, len(blocks)), replace=True).mean(axis=1)
            rows.append(
                {
                    "metric_name": metric,
                    "baseline_confidence_source": baseline,
                    "num_runs": int(len(paired)),
                    "num_blocks": int(len(blocks)),
                    "block_mean_delta": float(blocks.mean()),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "block_win_rate": float(np.mean(blocks < 0)),
                }
            )
    return pd.DataFrame(rows)


def _cp_upper(failures: int, total: int, delta: float) -> float:
    if total <= 0:
        return float("nan")
    if failures >= total:
        return 1.0
    return float(beta.ppf(1.0 - delta, failures + 1, total - failures))


def _prefix(frame: pd.DataFrame, confidence_col: str, coverage: float) -> pd.DataFrame:
    valid = frame.loc[frame[confidence_col].notna() & frame["abs_error"].notna()].copy()
    k = max(1, int(math.ceil(len(valid) * coverage)))
    return valid.sort_values(confidence_col, ascending=False).head(k)


def _certified_select(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    confidence_col: str,
    threshold: float,
    target_violation_rate: float,
) -> dict[str, object]:
    accepted: list[tuple[float, float]] = []
    delta_each = 0.05 / len(CANDIDATE_COVERAGES)
    for coverage in CANDIDATE_COVERAGES:
        chosen = _prefix(calibration, confidence_col, coverage)
        failures = int((chosen["abs_error"] > threshold).sum())
        bound = _cp_upper(failures, len(chosen), delta_each)
        if bound <= target_violation_rate:
            accepted.append((coverage, bound))
    if not accepted:
        return {
            "selected_coverage": 0.0,
            "test_num_selected": 0,
            "test_violation_rate": float("nan"),
            "test_satisfies_target": False,
        }
    coverage, bound = max(accepted, key=lambda item: item[0])
    chosen = _prefix(test, confidence_col, coverage)
    observed = float(np.mean(chosen["abs_error"] > threshold))
    return {
        "selected_coverage": float(coverage),
        "calibration_upper_bound": float(bound),
        "test_num_selected": int(len(chosen)),
        "test_violation_rate": observed,
        "test_satisfies_target": bool(observed <= target_violation_rate),
    }


def _risk_limit_rows(records: list[PredictionRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        validation = _load(record.validation_path)
        test = _load(record.test_path)
        if len(validation) < 8:
            continue
        order = np.random.default_rng(42 + int(record.seed)).permutation(len(validation))
        midpoint = max(2, len(order) // 2)
        fit = validation.iloc[order[:midpoint]].copy()
        calibration = validation.iloc[order[midpoint:]].copy()
        if len(calibration) < 2:
            continue
        calibration, test = _score_primary(fit, calibration, seed=record.seed)[1], _score_primary(fit, test, seed=record.seed)[1]
        sources = (
            ("primary_selector", "confidence_primary_selector"),
            ("mc_dropout", "confidence_mc_dropout"),
            ("target_familiarity", "target_familiarity"),
        )
        for source, confidence_col in sources:
            if confidence_col not in calibration or confidence_col not in test:
                continue
            for threshold in ABS_ERROR_THRESHOLDS:
                for alpha in TARGET_VIOLATION_RATES:
                    rows.append(
                        {
                            **_payload(record),
                            "confidence_source": source,
                            "selector_fit_num_examples": int(len(fit)),
                            "independent_calibration_num_examples": int(len(calibration)),
                            "abs_error_threshold": threshold,
                            "target_violation_rate": alpha,
                            **_certified_select(
                                calibration,
                                test,
                                confidence_col=confidence_col,
                                threshold=threshold,
                                target_violation_rate=alpha,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _active_set(group: pd.DataFrame) -> set[object]:
    n_active = max(1, int(math.ceil(len(group) * ACTIVE_FRACTION)))
    return set(group.sort_values("target", ascending=False).head(n_active)["row_id"])


def _vs_rows(record: PredictionRecord, validation: pd.DataFrame, test: pd.DataFrame) -> list[dict[str, object]]:
    del validation
    data = test.copy()
    required_columns = {"row_id", "target_id", "target", "prediction_mean", "predicted_abs_error_primary", "abs_error"}
    if not required_columns.issubset(data.columns):
        return []
    data["score_prediction_only"] = data["prediction_mean"]
    data["score_primary_lcb"] = data["prediction_mean"] - PRIMARY_RISK_LAMBDA * data["predicted_abs_error_primary"]
    if "prediction_std_mc_dropout" in data:
        data["score_mc_lcb"] = data["prediction_mean"] - PRIMARY_RISK_LAMBDA * data["prediction_std_mc_dropout"]
    methods = [("prediction_only", "score_prediction_only"), ("primary_lcb", "score_primary_lcb")]
    if "score_mc_lcb" in data:
        methods.append(("mc_lcb", "score_mc_lcb"))
    rows: list[dict[str, object]] = []
    novelty = data.groupby("target_id")["target_novelty"].mean() if "target_novelty" in data else pd.Series(dtype=float)
    cutoff = float(novelty.quantile(0.75)) if len(novelty) else float("nan")
    for target_id, group in data.groupby("target_id"):
        group = group.loc[group["target"].notna()].copy()
        if len(group) < 10:
            continue
        active = _active_set(group)
        n_active = len(active)
        active_rate = n_active / float(len(group))
        pred_selected_by_budget: dict[int, set[object]] = {}
        for budget in VS_BUDGETS:
            n_select = min(budget, len(group))
            pred_selected_by_budget[budget] = set(
                group.sort_values("score_prediction_only", ascending=False).head(n_select)["row_id"]
            )
            for method, score_col in methods:
                selected = group.loc[group[score_col].notna()].sort_values(score_col, ascending=False).head(n_select)
                if selected.empty:
                    continue
                selected_set = set(selected["row_id"])
                hits = int(selected["row_id"].isin(active).sum())
                precision = hits / float(len(selected))
                enrichment = precision / active_rate
                baseline_set = pred_selected_by_budget[budget]
                changed = len(selected_set.symmetric_difference(baseline_set)) / float(2 * n_select)
                rows.append(
                    {
                        **_payload(record),
                        "target_id": target_id,
                        "decision_budget": budget,
                        "decision_protocol": method,
                        "fixed_lambda": PRIMARY_RISK_LAMBDA if method != "prediction_only" else 0.0,
                        "num_candidates": int(len(group)),
                        "num_selected": int(len(selected)),
                        "num_actives": int(n_active),
                        "selected_hits": hits,
                        "hit_recovery": float(hits / n_active),
                        "false_positive_risk": float((len(selected) - hits) / len(selected)),
                        "precision_at_budget": precision,
                        "enrichment_factor": float(enrichment),
                        "mean_selected_abs_error": float(selected["abs_error"].mean()),
                        "risk_adjusted_enrichment": float(enrichment / (1.0 + selected["abs_error"].mean())),
                        "recommendation_change_rate_vs_prediction": float(changed),
                        "novel_target_subgroup": bool(
                            len(novelty) and math.isfinite(cutoff) and float(novelty.get(target_id, 0.0)) >= cutoff
                        ),
                    }
                )
    return rows


def run_primary_experiments(workspace: Path, output_dir: Path) -> dict[str, object]:
    records = _unique_records(workspace)
    summary_rows: list[dict[str, object]] = []
    vs_rows: list[dict[str, object]] = []
    successful: list[PredictionRecord] = []
    skipped: list[dict[str, str]] = []
    for record in records:
        try:
            validation, test = _score_primary(_load(record.validation_path), _load(record.test_path), seed=record.seed)
        except Exception as exc:
            skipped.append({"run_name": record.run_name, "reason": str(exc)})
            continue
        successful.append(record)
        for source, column in (
            ("primary_selector", "confidence_primary_selector"),
            ("mc_dropout", "confidence_mc_dropout"),
            ("target_familiarity", "target_familiarity"),
        ):
            if column in test and test[column].notna().any():
                summary_rows.append({**_payload(record), "confidence_source": source, **selective_metrics(test, confidence_col=column)})
        vs_rows.extend(_vs_rows(record, validation, test))
    summary = pd.DataFrame(summary_rows)
    _write_frame(summary, output_dir / "primary_selective_runs.csv")
    pairwise = _pairwise_rows(summary, scope="full_primary_matrix")
    _write_frame(pairwise, output_dir / "primary_pairwise_stats.csv")
    named = summary.loc[summary["model_type"].isin(PRIMARY_MODELS)].copy()
    named_pairwise = [_pairwise_rows(named, scope="all_named_strong")]
    for model, frame in named.groupby("model_type"):
        named_pairwise.append(_pairwise_rows(frame, scope=f"named_{model}"))
    named_pairwise_frame = pd.concat(named_pairwise, ignore_index=True) if named_pairwise else pd.DataFrame()
    _write_frame(named_pairwise_frame, output_dir / "primary_named_strong_pairwise_stats.csv")
    block = _block_bootstrap(summary)
    _write_frame(block, output_dir / "primary_block_bootstrap_summary.csv")
    risk_detail = _risk_limit_rows(successful)
    _write_frame(risk_detail, output_dir / "primary_certified_risk_detail.csv")
    risk_summary = (
        risk_detail.groupby(["abs_error_threshold", "target_violation_rate", "confidence_source"], as_index=False)
        .agg(
            num_runs=("run_name", "count"),
            nonempty_selection_rate=("test_num_selected", lambda values: float(np.mean(values > 0))),
            mean_selected_coverage=("selected_coverage", "mean"),
            mean_test_violation_rate=("test_violation_rate", "mean"),
            test_target_satisfaction_rate=("test_satisfies_target", "mean"),
        )
    )
    _write_frame(risk_summary, output_dir / "primary_certified_risk_summary.csv")
    vs_target = pd.DataFrame(vs_rows)
    _write_frame(vs_target, output_dir / "primary_vs_target_rows.csv")
    if vs_target.empty:
        vs_run = pd.DataFrame()
    else:
        vs_run = (
            vs_target.groupby(
                ["run_name", "dataset_name", "split_name", "seed", "model_type", "decision_budget", "decision_protocol"],
                as_index=False,
            )
            .agg(
                num_targets=("target_id", "nunique"),
                mean_hit_recovery=("hit_recovery", "mean"),
                mean_false_positive_risk=("false_positive_risk", "mean"),
                mean_risk_adjusted_enrichment=("risk_adjusted_enrichment", "mean"),
                mean_selected_abs_error=("mean_selected_abs_error", "mean"),
                mean_recommendation_change_rate=("recommendation_change_rate_vs_prediction", "mean"),
            )
        )
    _write_frame(vs_run, output_dir / "primary_vs_run_summary.csv")
    if vs_target.empty:
        vs_novel = pd.DataFrame()
    else:
        vs_novel = (
            vs_target.groupby(
                [
                    "run_name",
                    "dataset_name",
                    "split_name",
                    "seed",
                    "model_type",
                    "decision_budget",
                    "decision_protocol",
                    "novel_target_subgroup",
                ],
                as_index=False,
            )
            .agg(
                num_targets=("target_id", "nunique"),
                mean_hit_recovery=("hit_recovery", "mean"),
                mean_false_positive_risk=("false_positive_risk", "mean"),
                mean_risk_adjusted_enrichment=("risk_adjusted_enrichment", "mean"),
                mean_selected_abs_error=("mean_selected_abs_error", "mean"),
                mean_recommendation_change_rate=("recommendation_change_rate_vs_prediction", "mean"),
            )
        )
    _write_frame(vs_novel, output_dir / "primary_vs_novel_target_summary.csv")
    status = {
        "primary_selector": {
            "name": PRIMARY_SELECTOR_NAME,
            "regressor": PRIMARY_REGRESSOR,
            "feature_set": PRIMARY_FEATURE_SET,
            "ridge_alpha": 1.0,
        },
        "discovered_records": len(records),
        "successful_records": len(successful),
        "skipped_records": skipped,
        "selective_summary_rows": int(len(summary)),
        "named_pairwise_rows": int(len(named_pairwise_frame)),
        "risk_detail_rows": int(len(risk_detail)),
        "vs_target_rows": int(len(vs_target)),
        "vs_novel_target_summary_rows": int(len(vs_novel)),
        "vs_records_with_target_level_candidates": int(vs_target["run_name"].nunique()) if not vs_target.empty else 0,
        "vs_policy": "fixed_lcb_score=prediction_mean-1.0*predicted_abs_error; no test-driven lambda tuning",
    }
    (output_dir / "primary_status.json").write_text(json.dumps(status, indent=2))
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the locked primary-selector submission experiments.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else workspace / "reports" / "primary_submission_experiments"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    status = run_primary_experiments(workspace, output_dir)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
