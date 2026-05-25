from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from selective_dta_b.eval.selective import prepare_regression_frame


def _prepare_seed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"row_id", "target", "prediction_mean"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns for seed ensemble: {missing}")
    return frame.sort_values("row_id").reset_index(drop=True).copy()


def build_seed_ensemble_frame(
    frames: Iterable[pd.DataFrame],
    *,
    prediction_col: str = "prediction_mean",
) -> pd.DataFrame:
    prepared_frames = [_prepare_seed_frame(frame) for frame in frames]
    if len(prepared_frames) < 2:
        raise ValueError("Seed ensemble requires at least two prediction frames")

    base = prepared_frames[0].copy()
    prediction_columns: list[str] = []
    for index, frame in enumerate(prepared_frames):
        if not base["row_id"].equals(frame["row_id"]):
            raise ValueError("Seed ensemble frames must share the same row_id ordering")
        if not np.allclose(base["target"].to_numpy(dtype=float), frame["target"].to_numpy(dtype=float)):
            raise ValueError("Seed ensemble frames must share the same targets")
        seed_prediction_col = f"prediction_seed_{index}"
        base[seed_prediction_col] = frame[prediction_col].to_numpy(dtype=float)
        prediction_columns.append(seed_prediction_col)

    base["prediction_mean"] = base.loc[:, prediction_columns].mean(axis=1)
    base["prediction_std_ensemble"] = base.loc[:, prediction_columns].std(axis=1, ddof=0)
    return prepare_regression_frame(base, prediction_col="prediction_mean", target_col="target")


__all__ = ["build_seed_ensemble_frame"]

