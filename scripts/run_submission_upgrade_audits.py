from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, wilcoxon

from selective_dta_b.eval.followup_experiments import discover_prediction_records, ensure_error_columns
from selective_dta_b.eval.posthoc import fit_posthoc_error_regressor, predict_posthoc_error


METRICS = ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse")
BASELINES = ("mc_dropout", "target_familiarity")
NAMED_STRONG_BACKBONES = ("adambind", "deepdtagen", "kanpm", "pmmr")
CONFIDENCE_COLUMNS = {
    "posthoc_selector": "confidence_posthoc_independent",
    "mc_dropout": "confidence_mc_dropout",
    "target_familiarity": "target_familiarity",
}
EXCESSIVE_ERROR_THRESHOLDS = (0.75, 1.00, 1.25)
TARGET_VIOLATION_RATES = (0.10, 0.20)
CANDIDATE_COVERAGES = tuple(np.round(np.arange(0.05, 1.001, 0.05), 2))


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _holm_adjust(values: list[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda idx: values[idx] if math.isfinite(values[idx]) else float("inf"))
    adjusted = [float("nan")] * n
    running = 0.0
    for rank, idx in enumerate(order):
        pvalue = values[idx]
        if not math.isfinite(pvalue):
            continue
        corrected = min(1.0, (n - rank) * pvalue)
        running = max(running, corrected)
        adjusted[idx] = running
    return adjusted


def _paired_stats(
    summary: pd.DataFrame,
    *,
    subgroup_name: str,
    group_value: str,
    index_columns: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        pivot = summary.pivot_table(
            index=index_columns,
            columns="confidence_source",
            values=metric,
            aggfunc="first",
        )
        for baseline in BASELINES:
            if "posthoc_selector" not in pivot or baseline not in pivot:
                continue
            paired = pivot[["posthoc_selector", baseline]].dropna()
            if paired.empty:
                continue
            delta = (paired["posthoc_selector"] - paired[baseline]).to_numpy(dtype=float)
            try:
                pvalue = float(wilcoxon(delta, alternative="less", zero_method="zsplit").pvalue)
            except ValueError:
                pvalue = float("nan")
            rows.append(
                {
                    "subgroup": subgroup_name,
                    "group_value": group_value,
                    "metric_name": metric,
                    "baseline_confidence_source": baseline,
                    "num_pairs": int(len(delta)),
                    "posthoc_mean": float(paired["posthoc_selector"].mean()),
                    "baseline_mean": float(paired[baseline].mean()),
                    "mean_delta_posthoc_minus_baseline": float(delta.mean()),
                    "posthoc_win_rate": float(np.mean(delta < 0)),
                    "wilcoxon_p_value": pvalue,
                }
            )
    return rows


def named_strong_backbone_audit(workspace: Path, output_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(workspace / "reports" / "summary" / "selective_paper_runs.csv")
    named = summary.loc[summary["model_type"].isin(NAMED_STRONG_BACKBONES)].copy()
    rows: list[dict[str, object]] = []
    rows.extend(
        _paired_stats(
            named,
            subgroup_name="named_strong_backbones",
            group_value="all_named_strong",
            index_columns=["run_name"],
        )
    )
    for model_type, frame in named.groupby("model_type"):
        rows.extend(
            _paired_stats(
                frame,
                subgroup_name="named_backbone",
                group_value=str(model_type),
                index_columns=["run_name"],
            )
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p_value"] = _holm_adjust(out["wilcoxon_p_value"].tolist())
    _write(out, output_dir / "named_strong_backbone_pairwise_stats.csv")
    return out


def _bootstrap_block_ci(deltas: pd.DataFrame, *, reps: int = 4000) -> tuple[float, float]:
    blocks = deltas["block_delta"].to_numpy(dtype=float)
    if not len(blocks):
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260525)
    samples = rng.choice(blocks, size=(reps, len(blocks)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _delta_frame(summary: pd.DataFrame, *, metric: str, baseline: str) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["run_name", "dataset_name", "split_name", "seed", "model_type"],
        columns="confidence_source",
        values=metric,
        aggfunc="first",
    ).reset_index()
    if "posthoc_selector" not in pivot or baseline not in pivot:
        return pd.DataFrame()
    out = pivot.dropna(subset=["posthoc_selector", baseline]).copy()
    out["delta"] = out["posthoc_selector"] - out[baseline]
    return out


def hierarchical_robustness_audit(workspace: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(workspace / "reports" / "summary" / "selective_paper_runs.csv")
    overview: list[dict[str, object]] = []
    leave_one_out: list[dict[str, object]] = []
    for metric in METRICS:
        for baseline in BASELINES:
            deltas = _delta_frame(summary, metric=metric, baseline=baseline)
            if deltas.empty:
                continue
            deltas["block"] = (
                deltas["dataset_name"].astype(str)
                + " / "
                + deltas["split_name"].astype(str)
                + " / "
                + deltas["model_type"].astype(str)
            )
            block = deltas.groupby("block", as_index=False)["delta"].mean().rename(columns={"delta": "block_delta"})
            ci_low, ci_high = _bootstrap_block_ci(block)
            overview.append(
                {
                    "metric_name": metric,
                    "baseline_confidence_source": baseline,
                    "num_runs": int(len(deltas)),
                    "num_dataset_split_model_blocks": int(len(block)),
                    "run_weighted_mean_delta": float(deltas["delta"].mean()),
                    "block_weighted_mean_delta": float(block["block_delta"].mean()),
                    "block_bootstrap_ci95_low": ci_low,
                    "block_bootstrap_ci95_high": ci_high,
                    "block_win_rate": float(np.mean(block["block_delta"] < 0)),
                }
            )
            for unit_column in ("dataset_name", "model_type"):
                for held_out in sorted(deltas[unit_column].astype(str).unique()):
                    retained = deltas.loc[deltas[unit_column].astype(str) != held_out].copy()
                    retained["block"] = (
                        retained["dataset_name"].astype(str)
                        + " / "
                        + retained["split_name"].astype(str)
                        + " / "
                        + retained["model_type"].astype(str)
                    )
                    retained_block = retained.groupby("block", as_index=False)["delta"].mean()
                    leave_one_out.append(
                        {
                            "metric_name": metric,
                            "baseline_confidence_source": baseline,
                            "leave_out_dimension": unit_column,
                            "left_out_value": held_out,
                            "num_retained_runs": int(len(retained)),
                            "num_retained_blocks": int(len(retained_block)),
                            "block_weighted_mean_delta": float(retained_block["delta"].mean()),
                            "block_win_rate": float(np.mean(retained_block["delta"] < 0)),
                        }
                    )
    overview_frame = pd.DataFrame(overview)
    leave_frame = pd.DataFrame(leave_one_out)
    _write(overview_frame, output_dir / "hierarchical_block_bootstrap_summary.csv")
    _write(leave_frame, output_dir / "leave_one_group_out_robustness.csv")
    return overview_frame, leave_frame


def _clopper_pearson_upper(k: int, n: int, *, delta: float) -> float:
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def _rank_prefix(frame: pd.DataFrame, confidence_col: str, coverage: float) -> pd.DataFrame:
    valid = frame.loc[frame[confidence_col].notna() & frame["abs_error"].notna()].copy()
    n_select = max(1, int(math.ceil(len(valid) * coverage)))
    return valid.sort_values(confidence_col, ascending=False).head(n_select)


def _risk_limited_selection(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    confidence_col: str,
    abs_error_threshold: float,
    max_violation_rate: float,
    delta: float,
) -> dict[str, object]:
    per_candidate_delta = delta / len(CANDIDATE_COVERAGES)
    accepted: list[tuple[float, float, int, int]] = []
    for coverage in CANDIDATE_COVERAGES:
        selected = _rank_prefix(calibration, confidence_col, coverage)
        failures = int((selected["abs_error"] > abs_error_threshold).sum())
        upper = _clopper_pearson_upper(failures, len(selected), delta=per_candidate_delta)
        if upper <= max_violation_rate:
            accepted.append((coverage, upper, failures, len(selected)))
    if not accepted:
        return {
            "selected_coverage": 0.0,
            "calibration_num_selected": 0,
            "calibration_violation_rate": float("nan"),
            "simultaneous_upper_bound": float("nan"),
            "test_num_selected": 0,
            "test_violation_rate": float("nan"),
            "test_satisfies_target": False,
        }
    coverage, upper, cal_failures, cal_n = max(accepted, key=lambda item: item[0])
    test_selected = _rank_prefix(test, confidence_col, coverage)
    test_rate = float(np.mean(test_selected["abs_error"] > abs_error_threshold))
    return {
        "selected_coverage": float(coverage),
        "calibration_num_selected": int(cal_n),
        "calibration_violation_rate": float(cal_failures / cal_n),
        "simultaneous_upper_bound": float(upper),
        "test_num_selected": int(len(test_selected)),
        "test_violation_rate": test_rate,
        "test_satisfies_target": bool(test_rate <= max_violation_rate),
    }


def risk_limited_error_event_audit(workspace: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = discover_prediction_records(workspace, paper_only=True, max_runs=None)
    rows: list[dict[str, object]] = []
    for record in records:
        if record.validation_path is None or not record.validation_path.exists() or not record.test_path.exists():
            continue
        validation = ensure_error_columns(pd.read_csv(record.validation_path))
        test = ensure_error_columns(pd.read_csv(record.test_path))
        if len(validation) < 8:
            continue
        rng = np.random.default_rng(70201 + int(record.seed))
        shuffled = rng.permutation(len(validation))
        midpoint = max(2, len(shuffled) // 2)
        selector_fit = validation.iloc[shuffled[:midpoint]].copy()
        calibration = validation.iloc[shuffled[midpoint:]].copy()
        if len(calibration) < 2:
            continue
        regressor = fit_posthoc_error_regressor(
            selector_fit,
            random_state=int(record.seed),
            regressor_type="gbr",
            feature_set="enriched9",
        )
        calibration["confidence_posthoc_independent"] = 1.0 / (
            1.0 + predict_posthoc_error(regressor, calibration)
        )
        test["confidence_posthoc_independent"] = 1.0 / (
            1.0 + predict_posthoc_error(regressor, test)
        )
        for source_name, confidence_col in CONFIDENCE_COLUMNS.items():
            if confidence_col not in calibration or confidence_col not in test:
                continue
            for threshold in EXCESSIVE_ERROR_THRESHOLDS:
                for max_violation_rate in TARGET_VIOLATION_RATES:
                    result = _risk_limited_selection(
                        calibration,
                        test,
                        confidence_col=confidence_col,
                        abs_error_threshold=threshold,
                        max_violation_rate=max_violation_rate,
                        delta=0.05,
                    )
                    rows.append(
                        {
                            "run_name": record.run_name,
                            "dataset_name": record.dataset_name,
                            "split_name": record.split_name,
                            "seed": int(record.seed),
                            "model_type": record.model_type,
                            "confidence_source": source_name,
                            "selector_fit_num_examples": int(len(selector_fit)),
                            "independent_calibration_num_examples": int(len(calibration)),
                            "abs_error_threshold": threshold,
                            "target_violation_rate": max_violation_rate,
                            "familywise_delta": 0.05,
                            **result,
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["abs_error_threshold", "target_violation_rate", "confidence_source"], as_index=False)
        .agg(
            num_runs=("run_name", "count"),
            nonempty_selection_rate=("test_num_selected", lambda values: float(np.mean(values > 0))),
            mean_selected_coverage=("selected_coverage", "mean"),
            mean_test_violation_rate=("test_violation_rate", "mean"),
            test_target_satisfaction_rate=("test_satisfies_target", "mean"),
        )
    )
    _write(detail, output_dir / "simultaneous_risk_limit_error_event_detail.csv")
    _write(summary, output_dir / "simultaneous_risk_limit_error_event_summary.csv")
    return detail, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build submission-upgrade audits from frozen DTA predictions.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else workspace / "reports" / "submission_upgrade_audits"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    named = named_strong_backbone_audit(workspace, output_dir)
    hierarchical, leave_one = hierarchical_robustness_audit(workspace, output_dir)
    risk_detail, risk_summary = risk_limited_error_event_audit(workspace, output_dir)
    status = {
        "output_dir": str(output_dir),
        "named_strong_backbone_rows": int(len(named)),
        "hierarchical_summary_rows": int(len(hierarchical)),
        "leave_one_group_out_rows": int(len(leave_one)),
        "risk_limit_detail_rows": int(len(risk_detail)),
        "risk_limit_summary_rows": int(len(risk_summary)),
        "risk_limit_definition": "An independent half of each validation set fits the selector and the remaining half selects coverage using simultaneous Clopper-Pearson upper bounds for Pr(|error| > tau); this is not a guarantee on mean RMSE or under distribution shift.",
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
