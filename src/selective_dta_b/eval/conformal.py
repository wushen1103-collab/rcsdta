from __future__ import annotations

import math

import numpy as np
import pandas as pd


def fit_scaled_split_conformal(
    validation_frame: pd.DataFrame,
    *,
    scale_col: str,
    error_col: str = "abs_error",
    interval_levels: tuple[float, ...] = (0.8, 0.9),
    min_scale: float = 1e-6,
) -> dict[str, float]:
    if error_col not in validation_frame.columns:
        raise KeyError(f"Missing error column: {error_col}")
    if scale_col not in validation_frame.columns:
        raise KeyError(f"Missing scale column: {scale_col}")

    scale = validation_frame[scale_col].astype(float).clip(lower=min_scale)
    scores = (validation_frame[error_col].astype(float) / scale).to_numpy(dtype=float)
    if len(scores) == 0:
        raise ValueError("Cannot fit split conformal on an empty validation frame")
    scores = np.sort(scores)

    calibration: dict[str, float] = {}
    for level in interval_levels:
        interval_key = str(int(round(level * 100)))
        rank = min(len(scores), max(1, math.ceil((len(scores) + 1) * level)))
        calibration[f"conformal_{interval_key}_qhat"] = float(scores[rank - 1])
    return calibration


def attach_conformal_confidence(
    frame: pd.DataFrame,
    *,
    calibration: dict[str, float],
    scale_col: str,
    interval_level: float = 0.9,
    prefix: str = "conformal",
    min_scale: float = 1e-6,
) -> pd.DataFrame:
    interval_key = str(int(round(interval_level * 100)))
    qhat_key = f"conformal_{interval_key}_qhat"
    if qhat_key not in calibration:
        raise KeyError(f"Missing conformal calibration key: {qhat_key}")
    if scale_col not in frame.columns:
        raise KeyError(f"Missing scale column: {scale_col}")

    prepared = frame.copy()
    scale = prepared[scale_col].astype(float).clip(lower=min_scale)
    width = 2.0 * calibration[qhat_key] * scale
    prepared[f"conformal_interval_width_{prefix}_{interval_key}"] = width
    prepared[f"confidence_{prefix}_{interval_key}"] = 1.0 / (1.0 + width)
    return prepared


def summarize_scaled_conformal_intervals(
    frame: pd.DataFrame,
    *,
    calibration: dict[str, float],
    scale_col: str,
    prediction_col: str,
    target_col: str,
    min_scale: float = 1e-6,
) -> dict[str, float]:
    if scale_col not in frame.columns:
        raise KeyError(f"Missing scale column: {scale_col}")
    scale = frame[scale_col].astype(float).clip(lower=min_scale)

    summary: dict[str, float] = {}
    for key, value in calibration.items():
        if not key.startswith("conformal_") or not key.endswith("_qhat"):
            continue
        interval_key = key.removeprefix("conformal_").removesuffix("_qhat")
        half_width = scale * float(value)
        lower = frame[prediction_col].astype(float) - half_width
        upper = frame[prediction_col].astype(float) + half_width
        covered = ((frame[target_col].astype(float) >= lower) & (frame[target_col].astype(float) <= upper)).mean()
        summary[f"interval_{interval_key}_coverage"] = float(covered)
        summary[f"interval_{interval_key}_mean_width"] = float((2.0 * half_width).mean())
    return summary


__all__ = [
    "attach_conformal_confidence",
    "fit_scaled_split_conformal",
    "summarize_scaled_conformal_intervals",
]

