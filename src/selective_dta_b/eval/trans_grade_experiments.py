from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from selective_dta_b.eval.followup_experiments import (
    FEATURE_SETS,
    PredictionRecord,
    add_enriched_features,
    discover_prediction_records,
    ensure_error_columns,
    selective_metrics,
)
from selective_dta_b.eval.posthoc import fit_posthoc_error_regressor, predict_posthoc_error


MODERN_BACKBONE_ADAPTERS: dict[str, dict[str, object]] = {
    "gfl_2025_invariant_proxy": {
        "family": "GFLearn-like generalization/invariant feature learning",
        "component_models": ("simboost", "graphdta", "moltrans", "kanpm", "deepdtagen"),
        "posthoc_claim": "cached-output adapter; not a raw reimplementation",
    },
    "mlc_dta_2025_structural_proxy": {
        "family": "MLC-DTA-like equivariant/3D/contrastive representation",
        "component_models": ("graphdta", "moltrans", "kanpm", "deepdtagen"),
        "posthoc_claim": "cached-output adapter; not a raw reimplementation",
    },
    "balm_esm_plm_2025_proxy": {
        "family": "BALM/ESM/PLM-like sequence/structure multimodal backbone",
        "component_models": ("kanpm", "deepdtagen", "pmmr", "moltrans"),
        "posthoc_claim": "cached-output adapter; not a raw reimplementation",
    },
}

RISK_TARGETS: tuple[tuple[str, float], ...] = (
    ("rmse", 0.50),
    ("rmse", 0.75),
    ("rmse", 1.00),
    ("rmse", 1.25),
    ("rmse", 1.50),
    ("mae", 0.35),
    ("mae", 0.50),
    ("mae", 0.75),
    ("mae", 1.00),
    ("mae", 1.25),
)

MODERN_SOURCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("posthoc_selector", "confidence_posthoc"),
    ("mc_dropout", "confidence_mc_dropout"),
    ("target_familiarity", "target_familiarity"),
    ("oracle", "confidence_oracle"),
)

BASE_SOURCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("posthoc_selector", "confidence_posthoc"),
    ("mc_dropout", "confidence_mc_dropout"),
    ("target_familiarity", "target_familiarity"),
)


@dataclass(frozen=True)
class AdapterBuildResult:
    adapter_name: str
    dataset_name: str
    split_name: str
    seed: int
    validation: pd.DataFrame
    test: pd.DataFrame
    component_models: tuple[str, ...]
    component_weights: dict[str, float]


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _payload(record: PredictionRecord) -> dict[str, object]:
    return {
        "run_name": record.run_name,
        "dataset_name": record.dataset_name,
        "split_name": record.split_name,
        "seed": int(record.seed),
        "model_type": record.model_type,
    }


def _read_prediction_frame(path: Path | None, *, compact: bool = True) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    if not compact:
        return _prepare_prediction_frame(pd.read_csv(path))
    wanted = [
        "row_id",
        "dataset_name",
        "drug_id",
        "drug_smiles",
        "target_id",
        "target_sequence",
        "target",
        "affinity_model_target",
        "split",
        "prediction_mean",
        "prediction_std_mc_dropout",
        "target_familiarity",
        "target_novelty",
        "predicted_abs_error_posthoc",
        "confidence_posthoc",
        "confidence_mc_dropout",
        "abs_error",
        "squared_error",
    ]
    columns = pd.read_csv(path, nrows=0).columns
    usecols = [column for column in wanted if column in columns]
    return _prepare_prediction_frame(pd.read_csv(path, usecols=usecols))


def _prepare_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = ensure_error_columns(frame)
    if "prediction_std_mc_dropout" not in out:
        out["prediction_std_mc_dropout"] = 0.0
    if "confidence_mc_dropout" not in out:
        out["confidence_mc_dropout"] = 1.0 / (1.0 + out["prediction_std_mc_dropout"].fillna(0.0))
    if "confidence_oracle" not in out and "abs_error" in out:
        out["confidence_oracle"] = 1.0 / (1.0 + out["abs_error"].fillna(out["abs_error"].max()))
    return out


def _run_name_priority(run_name: str) -> tuple[int, int, str]:
    text = run_name.lower()
    score = 0
    if "smoke" in text:
        score += 100
    if "ensfix" in text:
        score += 20
    for idx, pattern in enumerate(("ep15_required", "ep15_core", "ep15_saw", "ep15", "fullpaper")):
        if pattern in text:
            score -= 20 - idx
    return score, len(run_name), run_name


def choose_canonical_records(records: list[PredictionRecord]) -> dict[tuple[str, str, str, int], PredictionRecord]:
    grouped: dict[tuple[str, str, str, int], list[PredictionRecord]] = {}
    for record in records:
        grouped.setdefault((record.model_type, record.dataset_name, record.split_name, int(record.seed)), []).append(record)
    return {key: sorted(items, key=lambda item: _run_name_priority(item.run_name))[0] for key, items in grouped.items()}


def _finite_or_nan(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    return float(valid.iloc[0]) if len(valid) else float("nan")


def _frame_for_component(record: PredictionRecord, *, split: str) -> pd.DataFrame | None:
    path = record.validation_path if split == "validation" else record.test_path
    frame = _read_prediction_frame(path)
    if frame is None or frame.empty:
        return None
    keep = [
        "row_id",
        "dataset_name",
        "drug_id",
        "drug_smiles",
        "target_id",
        "target_sequence",
        "target",
        "affinity_model_target",
        "split",
        "prediction_mean",
        "prediction_std_mc_dropout",
        "target_familiarity",
        "target_novelty",
    ]
    out = frame[[column for column in keep if column in frame.columns]].copy()
    if "target" not in out and "affinity_model_target" in out:
        out["target"] = out["affinity_model_target"]
    return out.drop_duplicates("row_id")


def _component_weights(validation_frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    raw: dict[str, float] = {}
    for model, frame in validation_frames.items():
        prepared = ensure_error_columns(frame)
        mae = float(prepared["abs_error"].mean()) if len(prepared) else float("nan")
        if math.isfinite(mae):
            raw[model] = 1.0 / max(mae, 1e-6)
    total = sum(raw.values())
    if total <= 0:
        return {model: 1.0 / len(validation_frames) for model in validation_frames}
    return {model: value / total for model, value in raw.items()}


def _weighted_nanmean(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(matrix)
    weighted = np.where(valid, matrix * weights.reshape(1, -1), 0.0)
    denom = np.where(valid, weights.reshape(1, -1), 0.0).sum(axis=1)
    out = weighted.sum(axis=1) / np.maximum(denom, 1e-12)
    out[denom <= 0] = np.nan
    return out


def _assemble_adapter_frame(
    component_frames: dict[str, pd.DataFrame],
    *,
    component_weights: dict[str, float],
    min_components: int = 2,
) -> pd.DataFrame:
    models = tuple(component_frames)
    merged: pd.DataFrame | None = None
    meta_columns = ["dataset_name", "drug_id", "drug_smiles", "target_id", "target_sequence", "target", "affinity_model_target", "split"]
    familiarity_columns: list[str] = []
    novelty_columns: list[str] = []
    prediction_columns: list[str] = []
    std_columns: list[str] = []

    for model, frame in component_frames.items():
        renamed = frame.copy()
        renamed = renamed.rename(
            columns={
                "prediction_mean": f"prediction_mean__{model}",
                "prediction_std_mc_dropout": f"prediction_std_mc_dropout__{model}",
                "target_familiarity": f"target_familiarity__{model}",
                "target_novelty": f"target_novelty__{model}",
            }
        )
        keep = ["row_id", *[column for column in meta_columns if column in renamed.columns]]
        keep.extend(
            [
                f"prediction_mean__{model}",
                f"prediction_std_mc_dropout__{model}",
                f"target_familiarity__{model}",
                f"target_novelty__{model}",
            ]
        )
        renamed = renamed[[column for column in keep if column in renamed.columns]]
        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed, on="row_id", how="outer", suffixes=("", f"__meta_{model}"))
        prediction_columns.append(f"prediction_mean__{model}")
        std_columns.append(f"prediction_std_mc_dropout__{model}")
        familiarity_columns.append(f"target_familiarity__{model}")
        novelty_columns.append(f"target_novelty__{model}")

    if merged is None or merged.empty:
        return pd.DataFrame()

    for column in meta_columns:
        candidates = [c for c in merged.columns if c == column or c.startswith(f"{column}__meta_")]
        if not candidates:
            continue
        base = merged[candidates[0]]
        for candidate in candidates[1:]:
            base = base.combine_first(merged[candidate])
        merged[column] = base

    prediction_matrix = merged[prediction_columns].to_numpy(dtype=float)
    counts = np.isfinite(prediction_matrix).sum(axis=1)
    merged = merged.loc[counts >= min_components].copy()
    prediction_matrix = merged[prediction_columns].to_numpy(dtype=float)
    if merged.empty:
        return merged

    weights = np.asarray([component_weights.get(model, 1.0) for model in models], dtype=float)
    merged["prediction_mean"] = _weighted_nanmean(prediction_matrix, weights)
    between_std = np.nanstd(prediction_matrix, axis=1)
    std_matrix = merged[[column for column in std_columns if column in merged.columns]].to_numpy(dtype=float)
    within_std = np.nanmean(std_matrix, axis=1) if std_matrix.size else np.zeros(len(merged))
    within_std = np.nan_to_num(within_std, nan=0.0)
    merged["prediction_std_mc_dropout"] = np.sqrt(np.nan_to_num(between_std, nan=0.0) ** 2 + within_std**2)

    fam_cols = [column for column in familiarity_columns if column in merged.columns]
    nov_cols = [column for column in novelty_columns if column in merged.columns]
    if fam_cols:
        merged["target_familiarity"] = np.nanmean(merged[fam_cols].to_numpy(dtype=float), axis=1)
    if nov_cols:
        merged["target_novelty"] = np.nanmean(merged[nov_cols].to_numpy(dtype=float), axis=1)
    if "target_familiarity" not in merged:
        merged["target_familiarity"] = 0.0
    if "target_novelty" not in merged:
        merged["target_novelty"] = 1.0 - merged["target_familiarity"]
    merged["target_familiarity"] = merged["target_familiarity"].fillna(merged["target_familiarity"].median()).fillna(0.0)
    merged["target_novelty"] = merged["target_novelty"].fillna(merged["target_novelty"].median()).fillna(1.0 - merged["target_familiarity"])

    if "target" not in merged and "affinity_model_target" in merged:
        merged["target"] = merged["affinity_model_target"]
    merged["prediction"] = merged["prediction_mean"]
    merged = ensure_error_columns(merged)
    merged["confidence_mc_dropout"] = 1.0 / (1.0 + merged["prediction_std_mc_dropout"].fillna(0.0))
    merged["confidence_oracle"] = 1.0 / (1.0 + merged["abs_error"].fillna(merged["abs_error"].max()))
    merged["component_count"] = np.isfinite(prediction_matrix).sum(axis=1)
    return merged


def _build_one_adapter(
    canonical: dict[tuple[str, str, str, int], PredictionRecord],
    *,
    adapter_name: str,
    dataset_name: str,
    split_name: str,
    seed: int,
    min_components: int = 2,
) -> AdapterBuildResult | None:
    spec = MODERN_BACKBONE_ADAPTERS[adapter_name]
    requested = tuple(spec["component_models"])  # type: ignore[index]
    records = {
        model: canonical[(model, dataset_name, split_name, seed)]
        for model in requested
        if (model, dataset_name, split_name, seed) in canonical
    }
    if len(records) < min_components:
        return None

    validation_frames = {
        model: frame
        for model, record in records.items()
        if (frame := _frame_for_component(record, split="validation")) is not None
    }
    test_frames = {
        model: frame
        for model, record in records.items()
        if (frame := _frame_for_component(record, split="test")) is not None
    }
    common_models = tuple(model for model in requested if model in validation_frames and model in test_frames)
    if len(common_models) < min_components:
        return None
    validation_frames = {model: validation_frames[model] for model in common_models}
    test_frames = {model: test_frames[model] for model in common_models}
    weights = _component_weights(validation_frames)
    validation = _assemble_adapter_frame(validation_frames, component_weights=weights, min_components=min_components)
    test = _assemble_adapter_frame(test_frames, component_weights=weights, min_components=min_components)
    if validation.empty or test.empty:
        return None
    return AdapterBuildResult(
        adapter_name=adapter_name,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=int(seed),
        validation=validation,
        test=test,
        component_models=common_models,
        component_weights=weights,
    )


def _score_posthoc(validation: pd.DataFrame, test: pd.DataFrame, *, random_state: int) -> pd.DataFrame:
    train = add_enriched_features(ensure_error_columns(validation)).dropna(subset=list(FEATURE_SETS["enriched9"]) + ["abs_error"])
    score = add_enriched_features(ensure_error_columns(test)).copy()
    valid = score.dropna(subset=list(FEATURE_SETS["enriched9"])).index
    if len(train) < 10 or len(valid) == 0:
        score["predicted_abs_error_posthoc"] = np.nan
        score["confidence_posthoc"] = np.nan
        return score
    regressor = fit_posthoc_error_regressor(train, random_state=random_state, regressor_type="ridge", feature_set="enriched9")
    predicted = np.full(len(score), np.nan, dtype=float)
    predicted[score.index.get_indexer(valid)] = predict_posthoc_error(regressor, score.loc[valid])
    score["predicted_abs_error_posthoc"] = predicted
    score["confidence_posthoc"] = 1.0 / (1.0 + score["predicted_abs_error_posthoc"])
    return score


def run_modern_backbone_adapters(
    workspace: str | Path,
    *,
    output_dir: str | Path,
    paper_only: bool = True,
    max_runs: int | None = None,
    min_components: int = 2,
    write_predictions: bool = False,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = discover_prediction_records(root, paper_only=paper_only, max_runs=max_runs)
    canonical = choose_canonical_records(records)

    combos = sorted({(dataset, split, seed) for _, dataset, split, seed in canonical})
    summary_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for adapter_name, spec in MODERN_BACKBONE_ADAPTERS.items():
        for dataset_name, split_name, seed in combos:
            result = _build_one_adapter(
                canonical,
                adapter_name=adapter_name,
                dataset_name=dataset_name,
                split_name=split_name,
                seed=int(seed),
                min_components=min_components,
            )
            if result is None:
                continue
            scored = _score_posthoc(result.validation, result.test, random_state=int(seed))
            run_name = f"{adapter_name}_{dataset_name}_{split_name}_seed{seed}"
            meta = {
                "run_name": run_name,
                "adapter_name": adapter_name,
                "modern_baseline_family": spec["family"],
                "adapter_claim": spec["posthoc_claim"],
                "dataset_name": dataset_name,
                "split_name": split_name,
                "seed": int(seed),
                "model_type": adapter_name,
                "component_models": ",".join(result.component_models),
                "component_weights": json.dumps(result.component_weights, sort_keys=True),
                "num_validation_examples": int(len(result.validation)),
                "num_test_examples": int(len(scored)),
                "mean_component_count": float(scored["component_count"].mean()) if "component_count" in scored else float("nan"),
            }
            manifest_rows.append(meta)
            if write_predictions:
                pred_path = out / "modern_adapter_predictions" / f"{run_name}_test_predictions.csv"
                _write_frame(scored, pred_path)
            for source_name, confidence_col in MODERN_SOURCE_COLUMNS:
                if confidence_col not in scored or scored[confidence_col].notna().sum() == 0:
                    continue
                summary_rows.append({**meta, "confidence_source": source_name, **selective_metrics(scored, confidence_col=confidence_col)})

    summary = pd.DataFrame(summary_rows)
    manifest = pd.DataFrame(manifest_rows)
    pairwise = summarize_pairwise_advantage(summary, treatment_source="posthoc_selector")
    _write_frame(summary, out / "modern_baseline_posthoc_summary.csv")
    _write_frame(manifest, out / "modern_baseline_adapter_manifest.csv")
    _write_frame(pairwise, out / "modern_baseline_pairwise_stats.csv")
    return {
        "modern_adapter_runs": int(len(manifest)),
        "modern_adapter_metric_rows": int(len(summary)),
        "modern_adapter_pairwise_rows": int(len(pairwise)),
    }


def summarize_pairwise_advantage(
    metrics_frame: pd.DataFrame,
    *,
    treatment_source: str,
    baseline_sources: tuple[str, ...] = ("mc_dropout", "target_familiarity"),
    metric_names: tuple[str, ...] = ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse"),
) -> pd.DataFrame:
    if metrics_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    index_columns = [column for column in ("run_name", "adapter_name", "dataset_name", "split_name", "seed", "model_type") if column in metrics_frame.columns]
    for metric_name in metric_names:
        if metric_name not in metrics_frame:
            continue
        for baseline_source in baseline_sources:
            pair = metrics_frame.loc[
                metrics_frame["confidence_source"].isin((treatment_source, baseline_source)),
                [*index_columns, "confidence_source", metric_name],
            ].dropna()
            if pair.empty:
                continue
            pivot = pair.pivot_table(index=index_columns, columns="confidence_source", values=metric_name, aggfunc="first").dropna()
            if treatment_source not in pivot.columns or baseline_source not in pivot.columns or pivot.empty:
                continue
            delta = pivot[treatment_source] - pivot[baseline_source]
            wins = delta < 0
            rows.append(
                {
                    "metric_name": metric_name,
                    "baseline_confidence_source": baseline_source,
                    "num_pairs": int(len(pivot)),
                    "treatment_mean": float(pivot[treatment_source].mean()),
                    "baseline_mean": float(pivot[baseline_source].mean()),
                    "mean_delta_treatment_minus_baseline": float(delta.mean()),
                    "treatment_win_rate": float(wins.mean()),
                    "binom_p_value": float(binomtest(int(wins.sum()), len(wins), p=0.5, alternative="greater").pvalue),
                }
            )
    return pd.DataFrame(rows)


def _risk_array(frame: pd.DataFrame, metric_name: str) -> np.ndarray:
    prepared = ensure_error_columns(frame)
    if metric_name == "rmse":
        return prepared["squared_error"].to_numpy(dtype=float)
    if metric_name == "mae":
        return prepared["abs_error"].to_numpy(dtype=float)
    raise KeyError(metric_name)


def _risk_value(values: np.ndarray, metric_name: str) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    if metric_name == "rmse":
        return float(math.sqrt(float(np.mean(values))))
    if metric_name == "mae":
        return float(np.mean(values))
    raise KeyError(metric_name)


def _cumulative_risk(values: np.ndarray, metric_name: str) -> np.ndarray:
    valid = np.where(np.isfinite(values), values, np.nan)
    counts = np.cumsum(np.isfinite(valid))
    sums = np.nancumsum(valid)
    mean = sums / np.maximum(counts, 1)
    if metric_name == "rmse":
        return np.sqrt(mean)
    return mean


def _choose_coverage(validation: pd.DataFrame, *, confidence_col: str, metric_name: str, threshold: float) -> dict[str, object]:
    ranked = validation.loc[validation[confidence_col].notna()].sort_values(confidence_col, ascending=False)
    if ranked.empty:
        return {"coverage": float("nan"), "risk": float("nan"), "attainable": False, "selection_rule": "no_validation_confidence"}
    risks = _cumulative_risk(_risk_array(ranked, metric_name), metric_name)
    ok = np.where(np.isfinite(risks) & (risks <= threshold))[0]
    if len(ok):
        idx = int(ok[-1])
        attainable = True
        rule = "max_validation_coverage_under_threshold"
    else:
        finite = np.where(np.isfinite(risks))[0]
        if len(finite) == 0:
            return {"coverage": float("nan"), "risk": float("nan"), "attainable": False, "selection_rule": "no_finite_validation_risk"}
        idx = int(finite[np.nanargmin(risks[finite])])
        attainable = False
        rule = "best_effort_min_validation_risk"
    return {
        "coverage": float((idx + 1) / len(ranked)),
        "risk": float(risks[idx]),
        "attainable": bool(attainable),
        "selection_rule": rule,
    }


def _bootstrap_ci(
    values: np.ndarray,
    *,
    metric_name: str,
    reps: int,
    sample_cap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) > sample_cap:
        values = rng.choice(values, size=sample_cap, replace=False)
    boot = np.empty(reps, dtype=float)
    for idx in range(reps):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[idx] = _risk_value(sample, metric_name)
    return float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def _source_columns(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return [(name, column) for name, column in BASE_SOURCE_COLUMNS if column in frame and frame[column].notna().sum() > 0]


def run_formal_risk_control(
    workspace: str | Path,
    *,
    output_dir: str | Path,
    paper_only: bool = True,
    max_runs: int | None = None,
    bootstrap_reps: int = 200,
    bootstrap_sample_cap: int = 5000,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve()
    records = discover_prediction_records(root, paper_only=paper_only, max_runs=max_runs)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260520)

    for record in records:
        validation = _read_prediction_frame(record.validation_path)
        test = _read_prediction_frame(record.test_path)
        if validation is None or test is None or validation.empty or test.empty:
            continue
        meta = _payload(record)
        for source_name, confidence_col in _source_columns(test):
            if confidence_col not in validation:
                continue
            for metric_name, threshold in RISK_TARGETS:
                chosen = _choose_coverage(validation, confidence_col=confidence_col, metric_name=metric_name, threshold=threshold)
                coverage = float(chosen["coverage"])
                if not math.isfinite(coverage) or coverage <= 0:
                    continue
                ranked_test = test.loc[test[confidence_col].notna()].sort_values(confidence_col, ascending=False)
                k = max(1, min(len(ranked_test), int(math.ceil(len(ranked_test) * coverage))))
                selected = ranked_test.iloc[:k]
                risk_values = _risk_array(selected, metric_name)
                achieved = _risk_value(risk_values, metric_name)
                ci_low, ci_high = _bootstrap_ci(
                    risk_values,
                    metric_name=metric_name,
                    reps=bootstrap_reps,
                    sample_cap=bootstrap_sample_cap,
                    rng=rng,
                )
                rows.append(
                    {
                        **meta,
                        "confidence_source": source_name,
                        "risk_metric": metric_name,
                        "target_risk_threshold": float(threshold),
                        "validation_selected_coverage": coverage,
                        "validation_achieved_risk": float(chosen["risk"]),
                        "target_attainable_on_validation": bool(chosen["attainable"]),
                        "selection_rule": chosen["selection_rule"],
                        "test_num_examples": int(len(ranked_test)),
                        "test_num_selected": int(k),
                        "test_coverage": float(k / len(ranked_test)) if len(ranked_test) else float("nan"),
                        "test_achieved_risk": achieved,
                        "test_risk_ci95_low": ci_low,
                        "test_risk_ci95_high": ci_high,
                        "violates_target": bool(math.isfinite(achieved) and achieved > threshold),
                        "bootstrap_reps": int(bootstrap_reps),
                        "bootstrap_sample_cap": int(bootstrap_sample_cap),
                    }
                )

    run_frame = pd.DataFrame(rows)
    if run_frame.empty:
        summary = pd.DataFrame()
    else:
        group_cols = ["confidence_source", "risk_metric", "target_risk_threshold"]
        summary = (
            run_frame.groupby(group_cols, dropna=False)
            .agg(
                num_runs=("run_name", "nunique"),
                mean_test_achieved_risk=("test_achieved_risk", "mean"),
                median_test_achieved_risk=("test_achieved_risk", "median"),
                mean_test_coverage=("test_coverage", "mean"),
                median_test_coverage=("test_coverage", "median"),
                violation_rate=("violates_target", "mean"),
                validation_attainable_rate=("target_attainable_on_validation", "mean"),
                mean_ci95_low=("test_risk_ci95_low", "mean"),
                mean_ci95_high=("test_risk_ci95_high", "mean"),
            )
            .reset_index()
        )
        by_model = (
            run_frame.groupby(["dataset_name", "split_name", "model_type", *group_cols], dropna=False)
            .agg(
                num_runs=("run_name", "nunique"),
                mean_test_achieved_risk=("test_achieved_risk", "mean"),
                mean_test_coverage=("test_coverage", "mean"),
                violation_rate=("violates_target", "mean"),
                validation_attainable_rate=("target_attainable_on_validation", "mean"),
            )
            .reset_index()
        )
        _write_frame(by_model, out / "formal_risk_control_by_model.csv")

    _write_frame(run_frame, out / "formal_risk_control_run_rows.csv")
    _write_frame(summary, out / "formal_risk_control_summary.csv")
    return {"formal_risk_control_rows": int(len(run_frame)), "formal_risk_control_summary_rows": int(len(summary))}


def _active_set(group: pd.DataFrame, *, active_fraction: float) -> set[object]:
    n_active = max(1, int(math.ceil(len(group) * active_fraction)))
    return set(group.sort_values("target", ascending=False).head(n_active)["row_id"])


def _decision_scores(frame: pd.DataFrame) -> dict[str, str]:
    data = frame
    scores = {"prediction_only": "prediction_mean"}
    if "confidence_posthoc" in data:
        data["score_posthoc_weighted"] = data["prediction_mean"] * data["confidence_posthoc"]
        scores["posthoc_weighted"] = "score_posthoc_weighted"
    if "predicted_abs_error_posthoc" in data:
        data["score_posthoc_lower_bound"] = data["prediction_mean"] - data["predicted_abs_error_posthoc"]
        scores["posthoc_lower_bound"] = "score_posthoc_lower_bound"
        data["score_posthoc_risk_gate_mae1"] = np.where(data["predicted_abs_error_posthoc"] <= 1.0, data["prediction_mean"], -np.inf)
        scores["posthoc_risk_gate_mae1"] = "score_posthoc_risk_gate_mae1"
    if "confidence_mc_dropout" in data:
        data["score_mc_weighted"] = data["prediction_mean"] * data["confidence_mc_dropout"]
        scores["mc_weighted"] = "score_mc_weighted"
    if "target_familiarity" in data:
        data["score_target_familiarity_weighted"] = data["prediction_mean"] * data["target_familiarity"]
        scores["target_familiarity_weighted"] = "score_target_familiarity_weighted"
    return scores


def _vs_target_rows(
    frame: pd.DataFrame,
    *,
    budgets: tuple[int, ...],
    active_fraction: float,
    min_compounds_per_target: int,
) -> list[dict[str, object]]:
    data = ensure_error_columns(frame).copy()
    rows: list[dict[str, object]] = []
    scores = _decision_scores(data)
    if "target_id" not in data:
        return rows

    target_novelty = (
        data.groupby("target_id")["target_novelty"].mean().rename("mean_target_novelty").reset_index()
        if "target_novelty" in data
        else pd.DataFrame({"target_id": [], "mean_target_novelty": []})
    )
    novelty_cutoff = float(target_novelty["mean_target_novelty"].quantile(0.75)) if not target_novelty.empty else float("nan")
    novelty_lookup = dict(zip(target_novelty["target_id"], target_novelty["mean_target_novelty"]))

    for target_id, group in data.groupby("target_id"):
        group = group.loc[group["target"].notna()].copy()
        if len(group) < min_compounds_per_target:
            continue
        active = _active_set(group, active_fraction=active_fraction)
        n_active = len(active)
        active_rate = n_active / float(len(group))
        mean_novelty = float(novelty_lookup.get(target_id, float("nan")))
        novel_flag = bool(math.isfinite(mean_novelty) and math.isfinite(novelty_cutoff) and mean_novelty >= novelty_cutoff)
        for budget in budgets:
            max_select = min(int(budget), len(group))
            if max_select <= 0:
                continue
            for method, score_col in scores.items():
                ranked = group.loc[group[score_col].notna()].sort_values(score_col, ascending=False)
                if method == "posthoc_risk_gate_mae1":
                    ranked = ranked.loc[np.isfinite(ranked[score_col])]
                selected = ranked.head(max_select)
                if selected.empty:
                    continue
                hits = int(selected["row_id"].isin(active).sum())
                precision = hits / float(len(selected))
                enrichment = precision / active_rate if active_rate > 0 else float("nan")
                mean_abs_error = float(selected["abs_error"].mean())
                rows.append(
                    {
                        "target_id": target_id,
                        "decision_budget": int(budget),
                        "decision_protocol": method,
                        "num_compounds": int(len(group)),
                        "num_selected": int(len(selected)),
                        "num_actives": int(n_active),
                        "selected_hits": int(hits),
                        "hit_recovery": float(hits / n_active) if n_active else float("nan"),
                        "precision_at_budget": precision,
                        "false_positive_risk": float((len(selected) - hits) / len(selected)),
                        "active_rate": float(active_rate),
                        "enrichment_factor": float(enrichment),
                        "mean_selected_abs_error": mean_abs_error,
                        "risk_adjusted_enrichment": float(enrichment / (1.0 + mean_abs_error)) if math.isfinite(enrichment) else float("nan"),
                        "mean_target_novelty": mean_novelty,
                        "novel_target_subgroup": novel_flag,
                    }
                )
    return rows


def run_vs_decision_budget(
    workspace: str | Path,
    *,
    output_dir: str | Path,
    paper_only: bool = True,
    max_runs: int | None = None,
    budgets: tuple[int, ...] = (10, 50),
    active_fraction: float = 0.1,
    min_compounds_per_target: int = 10,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve()
    records = discover_prediction_records(root, paper_only=paper_only, max_runs=max_runs)
    rows: list[dict[str, object]] = []
    for record in records:
        test = _read_prediction_frame(record.test_path)
        if test is None or test.empty or "target_id" not in test:
            continue
        payload = _payload(record)
        rows.extend({**payload, **row} for row in _vs_target_rows(test, budgets=budgets, active_fraction=active_fraction, min_compounds_per_target=min_compounds_per_target))

    target_frame = pd.DataFrame(rows)
    if target_frame.empty:
        run_summary = pd.DataFrame()
        subgroup_summary = pd.DataFrame()
    else:
        group_cols = ["run_name", "dataset_name", "split_name", "seed", "model_type", "decision_budget", "decision_protocol"]
        run_summary = (
            target_frame.groupby(group_cols, dropna=False)
            .agg(
                num_targets=("target_id", "nunique"),
                mean_hit_recovery=("hit_recovery", "mean"),
                mean_false_positive_risk=("false_positive_risk", "mean"),
                mean_risk_adjusted_enrichment=("risk_adjusted_enrichment", "mean"),
                mean_enrichment_factor=("enrichment_factor", "mean"),
                mean_precision_at_budget=("precision_at_budget", "mean"),
                mean_selected_abs_error=("mean_selected_abs_error", "mean"),
            )
            .reset_index()
        )
        subgroup_summary = (
            target_frame.groupby(["dataset_name", "split_name", "model_type", "decision_budget", "decision_protocol", "novel_target_subgroup"], dropna=False)
            .agg(
                num_target_decisions=("target_id", "count"),
                mean_hit_recovery=("hit_recovery", "mean"),
                mean_false_positive_risk=("false_positive_risk", "mean"),
                mean_risk_adjusted_enrichment=("risk_adjusted_enrichment", "mean"),
                mean_enrichment_factor=("enrichment_factor", "mean"),
            )
            .reset_index()
        )
    _write_frame(target_frame, out / "vs_decision_budget_target_rows.csv")
    _write_frame(run_summary, out / "vs_decision_budget_run_summary.csv")
    _write_frame(subgroup_summary, out / "vs_decision_budget_novel_target_summary.csv")
    return {
        "vs_decision_budget_target_rows": int(len(target_frame)),
        "vs_decision_budget_run_rows": int(len(run_summary)),
        "vs_decision_budget_subgroup_rows": int(len(subgroup_summary)),
    }


def run_failure_mode_analysis(
    workspace: str | Path,
    *,
    output_dir: str | Path,
    paper_only: bool = True,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve()
    summary_path = root / "reports" / "summary" / ("selective_paper_runs.csv" if paper_only else "selective_all_runs.csv")
    if not summary_path.exists():
        summary_path = root / "reports" / "summary" / "selective_all_runs.csv"
    base = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    metric_names = ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse")
    rows: list[dict[str, object]] = []
    if not base.empty:
        index_columns = ["run_name", "dataset_name", "split_name", "seed", "model_type"]
        for metric_name in metric_names:
            if metric_name not in base:
                continue
            for comparator in ("mc_dropout", "target_familiarity", "aleatoric", "conformal_mc_dropout"):
                pair = base.loc[
                    base["confidence_source"].isin(("posthoc_selector", comparator)),
                    [*index_columns, "confidence_source", metric_name],
                ].dropna()
                if pair.empty:
                    continue
                pivot = pair.pivot_table(index=index_columns, columns="confidence_source", values=metric_name, aggfunc="first").reset_index().dropna()
                if "posthoc_selector" not in pivot or comparator not in pivot:
                    continue
                pivot["metric_name"] = metric_name
                pivot["comparator"] = comparator
                pivot["delta_posthoc_minus_comparator"] = pivot["posthoc_selector"] - pivot[comparator]
                pivot["posthoc_loses"] = pivot["delta_posthoc_minus_comparator"] > 0
                rows.extend(pivot.to_dict("records"))
    case_frame = pd.DataFrame(rows)
    loss_frame = case_frame.loc[case_frame["posthoc_loses"]].copy() if not case_frame.empty else pd.DataFrame()
    grouped = (
        case_frame.groupby(["dataset_name", "split_name", "model_type", "metric_name", "comparator"], dropna=False)
        .agg(
            num_pairs=("run_name", "count"),
            num_posthoc_losses=("posthoc_loses", "sum"),
            posthoc_loss_rate=("posthoc_loses", "mean"),
            mean_delta_posthoc_minus_comparator=("delta_posthoc_minus_comparator", "mean"),
            worst_delta_posthoc_minus_comparator=("delta_posthoc_minus_comparator", "max"),
        )
        .reset_index()
        if not case_frame.empty
        else pd.DataFrame()
    )
    novelty = _failure_subgroup_decomposition(root / "reports" / "followup_experiments" / "novelty_bin_summary.csv", subgroup_columns=("novelty_type", "novelty_bin"), metric_names=metric_names)
    cliffs = _failure_subgroup_decomposition(root / "reports" / "followup_experiments" / "activity_cliff_summary.csv", subgroup_columns=("activity_cliff_flag",), metric_names=metric_names)

    _write_frame(case_frame, out / "failure_mode_case_rows.csv")
    _write_frame(loss_frame, out / "failure_mode_losses.csv")
    _write_frame(grouped, out / "failure_mode_grouped.csv")
    _write_frame(novelty, out / "failure_mode_novelty_decomposition.csv")
    _write_frame(cliffs, out / "failure_mode_activity_cliff_decomposition.csv")
    return {
        "failure_mode_pairs": int(len(case_frame)),
        "failure_mode_losses": int(len(loss_frame)),
        "failure_mode_grouped_rows": int(len(grouped)),
        "failure_mode_novelty_rows": int(len(novelty)),
        "failure_mode_activity_cliff_rows": int(len(cliffs)),
    }


def _failure_subgroup_decomposition(path: Path, *, subgroup_columns: tuple[str, ...], metric_names: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    index_columns = ["run_name", "dataset_name", "split_name", "seed", "model_type", *subgroup_columns]
    rows: list[dict[str, object]] = []
    for metric_name in metric_names:
        if metric_name not in frame:
            continue
        for comparator in ("mc_dropout", "target_familiarity"):
            pair = frame.loc[
                frame["confidence_source"].isin(("posthoc_selector", comparator)),
                [*index_columns, "confidence_source", metric_name],
            ].dropna()
            if pair.empty:
                continue
            pivot = pair.pivot_table(index=index_columns, columns="confidence_source", values=metric_name, aggfunc="first").reset_index().dropna()
            if "posthoc_selector" not in pivot or comparator not in pivot:
                continue
            pivot["metric_name"] = metric_name
            pivot["comparator"] = comparator
            pivot["delta_posthoc_minus_comparator"] = pivot["posthoc_selector"] - pivot[comparator]
            pivot["posthoc_loses"] = pivot["delta_posthoc_minus_comparator"] > 0
            rows.extend(pivot.to_dict("records"))
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    group_cols = ["dataset_name", "split_name", "model_type", *subgroup_columns, "metric_name", "comparator"]
    return (
        detail.groupby(group_cols, dropna=False)
        .agg(
            num_pairs=("run_name", "count"),
            num_posthoc_losses=("posthoc_loses", "sum"),
            posthoc_loss_rate=("posthoc_loses", "mean"),
            mean_delta_posthoc_minus_comparator=("delta_posthoc_minus_comparator", "mean"),
        )
        .reset_index()
    )


def run_chembl_release_backtest_audit(
    workspace: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve()
    candidate_files: list[Path] = []
    for base in (root / "data", root / "artifacts"):
        if base.exists():
            candidate_files.extend(
                path
                for path in base.rglob("*")
                if path.is_file()
                and "chembl" in str(path).lower()
                and path.suffix.lower() in {".csv", ".tsv", ".tab", ".txt", ".parquet"}
                and ".venv" not in str(path)
            )
    audit_rows: list[dict[str, object]] = []
    release_like = ("release", "version", "chembl", "assay", "document", "pub", "year", "date")
    for path in candidate_files:
        try:
            if path.suffix.lower() == ".parquet":
                columns = pd.read_parquet(path, engine="pyarrow").columns.tolist()
            else:
                columns = pd.read_csv(path, sep=None, engine="python", nrows=0).columns.tolist()
        except Exception as exc:
            audit_rows.append({"path": str(path), "readable": False, "error": str(exc), "num_columns": 0, "release_like_columns": ""})
            continue
        matches = [column for column in columns if any(token in column.lower() for token in release_like)]
        audit_rows.append({"path": str(path), "readable": True, "error": "", "num_columns": len(columns), "release_like_columns": ",".join(matches)})
    audit_columns = ["path", "readable", "error", "num_columns", "release_like_columns"]
    audit = pd.DataFrame(audit_rows, columns=audit_columns)

    chembl_predictions = [
        path
        for path in root.glob("artifacts/runs/*/posthoc_selector/*_test_predictions.csv")
        if "chembl" in str(path).lower()
    ]
    summary_rows: list[dict[str, object]] = []
    for test_path in chembl_predictions:
        run_name = test_path.name[: -len("_test_predictions.csv")]
        frame = _read_prediction_frame(test_path)
        if frame is None or frame.empty:
            continue
        meta = {
            "run_name": run_name,
            "dataset_name": "chembl",
            "split_name": "release_temporal",
            "seed": 42,
            "model_type": "unknown",
        }
        for source_name, confidence_col in _source_columns(frame):
            summary_rows.append({**meta, "confidence_source": source_name, **selective_metrics(frame, confidence_col=confidence_col)})
    summary_columns = [
        "run_name",
        "dataset_name",
        "split_name",
        "seed",
        "model_type",
        "confidence_source",
        "num_examples",
        "full_rmse",
        "full_mae",
        "aurc",
        "coverage_50_rmse",
        "coverage_50_mae",
        "coverage_70_rmse",
        "coverage_70_mae",
        "coverage_90_rmse",
        "coverage_90_mae",
        "coverage_95_rmse",
        "coverage_95_mae",
    ]
    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    blueprint = pd.DataFrame(
        [
            {
                "dataset_name": "chembl",
                "split_protocol": "release_or_assay_publication_date_train_old_test_new",
                "required_models": "SimBoost,DeepDTA,GraphDTA,KANPM,posthoc_selector",
                "required_columns": "row_id,drug_smiles,target_id,target_sequence,affinity_model_target,chembl_release_or_assay_date",
                "status": "ready_to_run_when_chembl_release_file_and_predictions_are_present",
            }
        ]
    )
    status = {
        "chembl_candidate_files": int(len(candidate_files)),
        "chembl_candidate_readable_files": int(audit["readable"].sum()) if not audit.empty and "readable" in audit else 0,
        "chembl_prediction_files": int(len(chembl_predictions)),
        "chembl_backtest_rows": int(len(summary)),
        "status": "completed_from_existing_chembl_predictions" if len(summary) else "blocked_missing_chembl_release_predictions",
        "note": "No BindingDB proxy rows are relabeled as ChEMBL; this audit is intentionally strict.",
    }
    _write_frame(audit, out / "chembl_release_source_audit.csv")
    _write_frame(summary, out / "chembl_release_backtest_summary.csv")
    _write_frame(blueprint, out / "chembl_release_backtest_blueprint.csv")
    (out / "chembl_release_backtest_status.json").write_text(json.dumps(status, indent=2))
    return status


def run_trans_grade_experiments(
    workspace: str | Path,
    *,
    output_dir: str | Path | None = None,
    paper_only: bool = True,
    max_runs: int | None = None,
    sections: tuple[str, ...] = ("modern", "risk", "vs", "failure", "chembl"),
    bootstrap_reps: int = 200,
    bootstrap_sample_cap: int = 5000,
    write_modern_predictions: bool = False,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "reports" / "trans_grade_experiments"
    out.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "workspace": str(root),
        "output_dir": str(out),
        "paper_only": bool(paper_only),
        "max_runs": max_runs,
        "sections": list(sections),
    }
    if "modern" in sections:
        status.update(
            run_modern_backbone_adapters(
                root,
                output_dir=out,
                paper_only=paper_only,
                max_runs=max_runs,
                write_predictions=write_modern_predictions,
            )
        )
    if "risk" in sections:
        status.update(
            run_formal_risk_control(
                root,
                output_dir=out,
                paper_only=paper_only,
                max_runs=max_runs,
                bootstrap_reps=bootstrap_reps,
                bootstrap_sample_cap=bootstrap_sample_cap,
            )
        )
    if "vs" in sections:
        status.update(run_vs_decision_budget(root, output_dir=out, paper_only=paper_only, max_runs=max_runs))
    if "failure" in sections:
        status.update(run_failure_mode_analysis(root, output_dir=out, paper_only=paper_only))
    if "chembl" in sections:
        try:
            from selective_dta_b.eval.chembl_temporal_backtest import run_chembl_publication_year_backtest

            status["chembl_release_backtest"] = run_chembl_publication_year_backtest(root, output_dir=out)
        except Exception as exc:
            fallback_status = run_chembl_release_backtest_audit(root, output_dir=out)
            fallback_status["fallback_reason"] = str(exc)
            status["chembl_release_backtest"] = fallback_status

    (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


__all__ = [
    "MODERN_BACKBONE_ADAPTERS",
    "RISK_TARGETS",
    "choose_canonical_records",
    "run_chembl_release_backtest_audit",
    "run_failure_mode_analysis",
    "run_formal_risk_control",
    "run_modern_backbone_adapters",
    "run_trans_grade_experiments",
    "run_vs_decision_budget",
    "summarize_pairwise_advantage",
]
