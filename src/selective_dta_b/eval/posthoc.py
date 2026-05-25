from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


POSTHOC_FEATURE_COLUMNS = (
    "prediction_mean",
    "prediction_std_mc_dropout",
    "target_familiarity",
    "target_novelty",
)
POSTHOC_FEATURE_SETS = {
    "base4": POSTHOC_FEATURE_COLUMNS,
    "enriched9": POSTHOC_FEATURE_COLUMNS
    + (
        "mc_x_novelty",
        "mc_x_familiarity",
        "mean_x_novelty",
        "mc_sq",
        "novelty_sq",
    ),
}


@dataclass
class PosthocErrorRegressor:
    model: object
    feature_columns: tuple[str, ...] = POSTHOC_FEATURE_COLUMNS
    regressor_type: str = "gbr"
    feature_set: str = "base4"


def build_posthoc_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_set: str = "base4",
) -> pd.DataFrame:
    if feature_set not in POSTHOC_FEATURE_SETS:
        raise KeyError(f"Unsupported feature set: {feature_set}")
    prepared = frame.copy()
    if feature_set == "enriched9":
        prepared["mc_x_novelty"] = prepared["prediction_std_mc_dropout"] * prepared["target_novelty"]
        prepared["mc_x_familiarity"] = prepared["prediction_std_mc_dropout"] * prepared["target_familiarity"]
        prepared["mean_x_novelty"] = prepared["prediction_mean"] * prepared["target_novelty"]
        prepared["mc_sq"] = prepared["prediction_std_mc_dropout"] ** 2
        prepared["novelty_sq"] = prepared["target_novelty"] ** 2
    return prepared


def _validate_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    require_target: bool,
) -> pd.DataFrame:
    required_columns = list(POSTHOC_FEATURE_SETS[feature_set])
    if require_target:
        required_columns.append("abs_error")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return frame.loc[:, required_columns].copy()


def _build_regressor(
    regressor_type: str,
    random_state: int,
    *,
    sample_count: int,
) -> object:
    if regressor_type == "gbr":
        return GradientBoostingRegressor(random_state=random_state)
    if regressor_type == "knn":
        neighbor_count = max(1, min(64, sample_count))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=neighbor_count, weights="distance")),
            ]
        )
    if regressor_type == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    raise KeyError(f"Unsupported regressor type: {regressor_type}")


def fit_posthoc_error_regressor(
    validation_frame: pd.DataFrame,
    *,
    random_state: int = 42,
    regressor_type: str = "gbr",
    feature_set: str = "base4",
) -> PosthocErrorRegressor:
    validation_features = build_posthoc_feature_frame(validation_frame, feature_set=feature_set)
    prepared = _validate_feature_frame(
        validation_features,
        feature_set=feature_set,
        require_target=True,
    )
    feature_columns = POSTHOC_FEATURE_SETS[feature_set]
    model = _build_regressor(
        regressor_type,
        random_state=random_state,
        sample_count=len(prepared),
    )
    model.fit(prepared.loc[:, feature_columns], prepared["abs_error"])
    return PosthocErrorRegressor(
        model=model,
        feature_columns=feature_columns,
        regressor_type=regressor_type,
        feature_set=feature_set,
    )


def predict_posthoc_error(
    regressor: PosthocErrorRegressor,
    frame: pd.DataFrame,
) -> np.ndarray:
    prediction_features = build_posthoc_feature_frame(frame, feature_set=regressor.feature_set)
    prepared = _validate_feature_frame(
        prediction_features,
        feature_set=regressor.feature_set,
        require_target=False,
    )
    predictions = regressor.model.predict(prepared.loc[:, regressor.feature_columns])
    return np.clip(np.asarray(predictions, dtype=float), a_min=0.0, a_max=None)


__all__ = [
    "POSTHOC_FEATURE_COLUMNS",
    "POSTHOC_FEATURE_SETS",
    "PosthocErrorRegressor",
    "build_posthoc_feature_frame",
    "fit_posthoc_error_regressor",
    "predict_posthoc_error",
]

