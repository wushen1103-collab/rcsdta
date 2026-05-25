from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd


def prepare_regression_frame(
    frame: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    target_col: str = "target",
) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["residual"] = prepared[prediction_col] - prepared[target_col]
    prepared["abs_error"] = prepared["residual"].abs()
    prepared["squared_error"] = prepared["residual"] ** 2
    return prepared


def build_risk_coverage_curve(
    frame: pd.DataFrame,
    *,
    confidence_col: str,
    error_col: str = "squared_error",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    ranked = frame.sort_values(confidence_col, ascending=not higher_is_better).reset_index(drop=True).copy()
    sample_count = len(ranked)
    if sample_count == 0:
        raise ValueError("Cannot build risk-coverage curve from an empty frame")
    ranked["coverage"] = [(index + 1) / sample_count for index in range(sample_count)]
    ranked["risk"] = ranked[error_col].expanding().mean()
    return ranked[["coverage", "risk"]]


def _coverage_key(coverage: float) -> str:
    return str(int(round(coverage * 100)))


def summarize_selective_regression(
    frame: pd.DataFrame,
    *,
    confidence_col: str,
    coverage_levels: tuple[float, ...] = (0.5, 0.7, 0.9, 1.0),
    higher_is_better: bool = True,
) -> dict[str, float | int]:
    if "squared_error" not in frame.columns or "abs_error" not in frame.columns:
        prepared = prepare_regression_frame(frame)
    else:
        prepared = frame.copy()

    curve = build_risk_coverage_curve(
        prepared,
        confidence_col=confidence_col,
        error_col="squared_error",
        higher_is_better=higher_is_better,
    )
    aurc = 0.0
    previous_coverage = 0.0
    previous_risk = 0.0
    for row in curve.itertuples(index=False):
        width = float(row.coverage) - previous_coverage
        aurc += width * (previous_risk + float(row.risk)) / 2.0
        previous_coverage = float(row.coverage)
        previous_risk = float(row.risk)
    full_mse = float(prepared["squared_error"].mean())
    summary: dict[str, float | int] = {
        "num_examples": len(prepared),
        "full_rmse": math.sqrt(full_mse),
        "full_mae": float(prepared["abs_error"].mean()),
        "aurc": aurc,
    }

    ranked = prepared.sort_values(confidence_col, ascending=not higher_is_better).reset_index(drop=True)
    for coverage in coverage_levels:
        top_k = max(1, int(math.ceil(len(ranked) * coverage)))
        subset = ranked.iloc[:top_k]
        key = _coverage_key(coverage)
        summary[f"coverage_{key}_rmse"] = math.sqrt(float(subset["squared_error"].mean()))
        summary[f"coverage_{key}_mae"] = float(subset["abs_error"].mean())
    return summary


def summarize_predictive_intervals(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    target_col: str,
    std_col: str,
    interval_levels: tuple[float, ...] = (0.8, 0.9),
) -> dict[str, float]:
    summary: dict[str, float] = {}
    normal = NormalDist()
    for level in interval_levels:
        z_score = normal.inv_cdf(0.5 + level / 2.0)
        half_width = frame[std_col].astype(float) * z_score
        lower = frame[prediction_col].astype(float) - half_width
        upper = frame[prediction_col].astype(float) + half_width
        covered = ((frame[target_col].astype(float) >= lower) & (frame[target_col].astype(float) <= upper)).mean()
        interval_key = str(int(round(level * 100)))
        summary[f"interval_{interval_key}_coverage"] = float(covered)
        summary[f"interval_{interval_key}_mean_width"] = float((2.0 * half_width).mean())
    return summary

