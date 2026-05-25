#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from selective_dta_b.eval.chembl_temporal_backtest import (
    _prediction_frame,
    _predict_ensemble,
    _score_posthoc,
    add_target_familiarity,
)
from selective_dta_b.eval.followup_experiments import selective_metrics


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _split_validation_by_year(val_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(pd.to_numeric(val_pred["document_year"], errors="coerce").dropna().unique())
    if len(years) >= 2:
        fit = val_pred.loc[val_pred["document_year"] <= years[0]].copy()
        tune = val_pred.loc[val_pred["document_year"] > years[0]].copy()
        if min(len(fit), len(tune)) >= 20:
            return fit, tune
    ordered = val_pred.sort_values(["document_year", "row_id"]).reset_index(drop=True)
    midpoint = max(20, len(ordered) // 2)
    midpoint = min(midpoint, len(ordered) - 20)
    return ordered.iloc[:midpoint].copy(), ordered.iloc[midpoint:].copy()


def _add_drift_confidence(frame: pd.DataFrame, *, gamma: float) -> pd.DataFrame:
    out = frame.copy()
    risk = pd.to_numeric(out["predicted_abs_error_posthoc"], errors="coerce").to_numpy(dtype=float)
    novelty = pd.to_numeric(out.get("target_novelty", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    adjusted = np.maximum(risk + float(gamma) * novelty, 1e-8)
    out["predicted_abs_error_drift_aware"] = adjusted
    out["confidence_drift_aware"] = 1.0 / (1.0 + adjusted)
    return out


def _tune_drift_gamma(tune_scored: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    best_gamma = 0.0
    best_aurc = math.inf
    for gamma in (0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        scored = _add_drift_confidence(tune_scored, gamma=gamma)
        metrics = selective_metrics(scored, confidence_col="confidence_drift_aware")
        aurc = float(metrics["aurc"])
        rows.append({"gamma": gamma, **metrics})
        if aurc < best_aurc:
            best_aurc = aurc
            best_gamma = gamma
    return best_gamma, pd.DataFrame(rows)


def _pairwise(summary: pd.DataFrame, *, primary: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse"):
        pivot = summary.pivot_table(index="backbone_name", columns="confidence_source", values=metric, aggfunc="first")
        if primary not in pivot:
            continue
        for baseline in ("posthoc_selector", "mc_dropout", "target_familiarity"):
            if baseline not in pivot or baseline == primary:
                continue
            matched = pivot[[primary, baseline]].dropna()
            if matched.empty:
                continue
            delta = matched[primary] - matched[baseline]
            rows.append(
                {
                    "metric_name": metric,
                    "baseline_confidence_source": baseline,
                    "num_backbones": int(len(delta)),
                    "primary_mean": float(matched[primary].mean()),
                    "baseline_mean": float(matched[baseline].mean()),
                    "mean_delta_primary_minus_baseline": float(delta.mean()),
                    "primary_win_rate": float((delta < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _clopper_pearson_upper(events: int, total: int, *, delta: float) -> float:
    if total <= 0:
        return 1.0
    if events >= total:
        return 1.0
    return float(beta.ppf(1.0 - delta, events + 1, total - events))


def _conservative_event_risk_rows(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    confidence_col: str,
    source_name: str,
    meta: dict[str, object],
    delta: float = 0.10,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ranked_val = validation.dropna(subset=[confidence_col, "abs_error"]).sort_values(confidence_col, ascending=False).reset_index(drop=True)
    ranked_test = test.dropna(subset=[confidence_col, "abs_error"]).sort_values(confidence_col, ascending=False).reset_index(drop=True)
    if ranked_val.empty or ranked_test.empty:
        return rows
    for abs_error_threshold in (1.0, 1.25, 1.5, 2.0):
        events = (ranked_val["abs_error"].to_numpy(dtype=float) > abs_error_threshold).astype(int)
        cumulative_events = np.cumsum(events)
        for target_event_rate in (0.10, 0.20, 0.30, 0.40):
            selected_k = 0
            selected_upper = 1.0
            for k in range(1, len(ranked_val) + 1):
                upper = _clopper_pearson_upper(int(cumulative_events[k - 1]), k, delta=delta)
                if upper <= target_event_rate:
                    selected_k = k
                    selected_upper = upper
            test_k = int(math.floor(len(ranked_test) * selected_k / len(ranked_val)))
            test_k = min(test_k, len(ranked_test))
            if test_k > 0:
                selected = ranked_test.iloc[:test_k]
                event_rate = float((selected["abs_error"] > abs_error_threshold).mean())
                achieved_mae = float(selected["abs_error"].mean())
                achieved_rmse = math.sqrt(float(selected["squared_error"].mean()))
            else:
                event_rate = math.nan
                achieved_mae = math.nan
                achieved_rmse = math.nan
            rows.append(
                {
                    **meta,
                    "confidence_source": source_name,
                    "abs_error_threshold": abs_error_threshold,
                    "target_event_rate": target_event_rate,
                    "calibration_delta": delta,
                    "validation_selected_coverage": float(selected_k / len(ranked_val)),
                    "validation_cp_upper_bound": selected_upper,
                    "test_coverage": float(test_k / len(ranked_test)),
                    "test_event_rate": event_rate,
                    "test_mae": achieved_mae,
                    "test_rmse": achieved_rmse,
                    "test_satisfies_target": bool(test_k > 0 and event_rate <= target_event_rate),
                    "selection_rule": "max_coverage_with_clopper_pearson_upper_bound",
                }
            )
    return rows


def run_chembl36_kbs_pr_audits(
    *,
    chembl_pairs: Path,
    output_dir: Path,
    ensemble_size: int,
    random_state: int,
) -> dict[str, object]:
    frame = pd.read_csv(chembl_pairs)
    frame = add_target_familiarity(frame)
    train = frame.loc[frame["split"] == "train"].copy()
    val = frame.loc[frame["split"] == "val"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    summary_rows: list[dict[str, object]] = []
    gamma_rows: list[pd.DataFrame] = []
    risk_rows: list[dict[str, object]] = []

    for backbone_name in ("SimBoost", "DeepDTA", "GraphDTA", "KANPM"):
        val_mean, val_std, test_mean, test_std = _predict_ensemble(
            backbone_name,
            train,
            val,
            test,
            ensemble_size=ensemble_size,
            random_state=random_state,
        )
        val_pred = _prediction_frame(val, val_mean, val_std, backbone_name=backbone_name)
        test_pred = _prediction_frame(test, test_mean, test_std, backbone_name=backbone_name)
        fit_val, tune_val = _split_validation_by_year(val_pred)
        tune_scored = _score_posthoc(fit_val, tune_val, random_state=random_state)
        best_gamma, grid = _tune_drift_gamma(tune_scored)
        gamma_rows.append(grid.assign(backbone_name=backbone_name, selected_gamma=best_gamma))

        posthoc_test = _score_posthoc(val_pred, test_pred, random_state=random_state)
        drift_test = _add_drift_confidence(posthoc_test, gamma=best_gamma)
        drift_tune = _add_drift_confidence(tune_scored, gamma=best_gamma)
        meta = {
            "run_name": f"chembl36_kbs_pr_{backbone_name.lower()}_seed{random_state}",
            "dataset_name": "chembl",
            "split_name": "publication_year_temporal",
            "seed": int(random_state),
            "backbone_name": backbone_name,
            "num_train": int(len(train)),
            "num_val": int(len(val)),
            "num_test": int(len(test)),
            "calibration_design": "validation-year split: fit earliest validation year, tune latest validation year, test 2022",
            "selected_gamma": float(best_gamma),
        }
        confidence_specs = (
            ("posthoc_selector", posthoc_test, "confidence_posthoc", tune_scored, "confidence_posthoc"),
            ("drift_aware_selector", drift_test, "confidence_drift_aware", drift_tune, "confidence_drift_aware"),
            ("mc_dropout", posthoc_test, "confidence_mc_dropout", tune_val, "confidence_mc_dropout"),
            ("target_familiarity", posthoc_test, "target_familiarity", tune_val, "target_familiarity"),
        )
        for source_name, test_frame, test_col, val_frame, val_col in confidence_specs:
            summary_rows.append({**meta, "confidence_source": source_name, **selective_metrics(test_frame, confidence_col=test_col)})
            risk_rows.extend(
                _conservative_event_risk_rows(
                    val_frame,
                    test_frame,
                    confidence_col=val_col if val_col in val_frame.columns else test_col,
                    source_name=source_name,
                    meta=meta,
                )
            )

    summary = pd.DataFrame(summary_rows)
    pairwise = _pairwise(summary, primary="drift_aware_selector")
    gamma_grid = pd.concat(gamma_rows, ignore_index=True) if gamma_rows else pd.DataFrame()
    conservative_risk = pd.DataFrame(risk_rows)
    _write_frame(summary, output_dir / "chembl36_drift_aware_summary.csv")
    _write_frame(pairwise, output_dir / "chembl36_drift_aware_pairwise_stats.csv")
    _write_frame(gamma_grid, output_dir / "chembl36_drift_gamma_grid.csv")
    _write_frame(conservative_risk, output_dir / "chembl36_conservative_event_risk.csv")
    return {
        "chembl_pairs": str(chembl_pairs),
        "summary_rows": int(len(summary)),
        "pairwise_rows": int(len(pairwise)),
        "gamma_grid_rows": int(len(gamma_grid)),
        "conservative_risk_rows": int(len(conservative_risk)),
    }


def _best_case_examples(vs_target_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    key_cols = ["run_name", "dataset_name", "split_name", "seed", "model_type", "target_id", "decision_budget"]
    pivot = vs_target_rows.pivot_table(
        index=key_cols,
        columns="decision_protocol",
        values=[
            "hit_recovery",
            "false_positive_risk",
            "risk_adjusted_enrichment",
            "mean_selected_abs_error",
            "recommendation_change_rate_vs_prediction",
            "num_candidates",
            "num_actives",
            "novel_target_subgroup",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{protocol}" for metric, protocol in pivot.columns]
    pivot = pivot.reset_index()
    required = [
        "mean_selected_abs_error__prediction_only",
        "mean_selected_abs_error__primary_lcb",
        "hit_recovery__prediction_only",
        "hit_recovery__primary_lcb",
        "recommendation_change_rate_vs_prediction__primary_lcb",
    ]
    pivot = pivot.dropna(subset=[col for col in required if col in pivot.columns])
    pivot["delta_abs_error_primary_minus_prediction"] = pivot["mean_selected_abs_error__primary_lcb"] - pivot["mean_selected_abs_error__prediction_only"]
    pivot["delta_hit_primary_minus_prediction"] = pivot["hit_recovery__primary_lcb"] - pivot["hit_recovery__prediction_only"]
    pivot["delta_enrichment_primary_minus_prediction"] = pivot["risk_adjusted_enrichment__primary_lcb"] - pivot["risk_adjusted_enrichment__prediction_only"]
    pivot["delta_fp_primary_minus_prediction"] = pivot["false_positive_risk__primary_lcb"] - pivot["false_positive_risk__prediction_only"]
    changed = pivot.loc[pivot["recommendation_change_rate_vs_prediction__primary_lcb"] > 0.05].copy()
    beneficial = changed.loc[
        (changed["delta_abs_error_primary_minus_prediction"] < -0.05)
        & (changed["delta_hit_primary_minus_prediction"] >= 0)
    ].sort_values(["delta_abs_error_primary_minus_prediction", "delta_fp_primary_minus_prediction"])
    boundary = changed.loc[
        (changed["delta_hit_primary_minus_prediction"] < 0)
        | (changed["delta_enrichment_primary_minus_prediction"] < -0.05)
    ].sort_values(["delta_hit_primary_minus_prediction", "delta_enrichment_primary_minus_prediction"])
    if not beneficial.empty:
        rows.append(beneficial.head(3).assign(case_type="risk_reduction_without_hit_loss"))
    if not boundary.empty:
        rows.append(boundary.head(3).assign(case_type="decision_boundary_or_yield_loss"))
    if not rows:
        return pd.DataFrame()
    keep = [
        "case_type",
        *key_cols,
        "num_candidates__prediction_only",
        "num_actives__prediction_only",
        "novel_target_subgroup__prediction_only",
        "hit_recovery__prediction_only",
        "hit_recovery__primary_lcb",
        "false_positive_risk__prediction_only",
        "false_positive_risk__primary_lcb",
        "risk_adjusted_enrichment__prediction_only",
        "risk_adjusted_enrichment__primary_lcb",
        "mean_selected_abs_error__prediction_only",
        "mean_selected_abs_error__primary_lcb",
        "recommendation_change_rate_vs_prediction__primary_lcb",
        "delta_abs_error_primary_minus_prediction",
        "delta_hit_primary_minus_prediction",
        "delta_enrichment_primary_minus_prediction",
        "delta_fp_primary_minus_prediction",
    ]
    return pd.concat(rows, ignore_index=True)[[col for col in keep if col in pd.concat(rows, ignore_index=True).columns]]


def run_vs_decision_trace(*, vs_target_rows: Path, output_dir: Path) -> dict[str, object]:
    frame = pd.read_csv(vs_target_rows)
    trace = _best_case_examples(frame)
    _write_frame(trace, output_dir / "vs_decision_trace_examples.csv")
    return {"vs_trace_rows": int(len(trace))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KBS/PR-oriented supplementary audits from saved RCSDTA data")
    parser.add_argument("--chembl-pairs", type=Path, required=True)
    parser.add_argument("--vs-target-rows", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = run_chembl36_kbs_pr_audits(
        chembl_pairs=args.chembl_pairs,
        output_dir=args.output_dir,
        ensemble_size=args.ensemble_size,
        random_state=args.random_state,
    )
    if args.vs_target_rows:
        status.update(run_vs_decision_trace(vs_target_rows=args.vs_target_rows, output_dir=args.output_dir))
    (args.output_dir / "kbs_pr_audit_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
