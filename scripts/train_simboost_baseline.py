#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import HistGradientBoostingRegressor

from selective_dta_b.data.loading import load_split_frame
from selective_dta_b.eval.novelty import attach_target_novelty
from selective_dta_b.eval.posthoc import fit_posthoc_error_regressor, predict_posthoc_error
from selective_dta_b.eval.selective import (
    build_risk_coverage_curve,
    prepare_regression_frame,
    summarize_predictive_intervals,
    summarize_selective_regression,
)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class SimBoostFeaturizer:
    fingerprint_bits: int = 256
    target_column: str = "affinity_model_target"
    global_mean: float = 0.0
    drug_mean: dict[str, float] | None = None
    drug_count: dict[str, int] | None = None
    target_mean: dict[str, float] | None = None
    target_count: dict[str, int] | None = None

    def fit(self, train_frame: pd.DataFrame) -> "SimBoostFeaturizer":
        self.global_mean = float(train_frame[self.target_column].mean())
        drug_stats = train_frame.groupby("drug_id")[self.target_column].agg(["mean", "count"])
        target_stats = train_frame.groupby("target_id")[self.target_column].agg(["mean", "count"])
        self.drug_mean = drug_stats["mean"].astype(float).to_dict()
        self.drug_count = drug_stats["count"].astype(int).to_dict()
        self.target_mean = target_stats["mean"].astype(float).to_dict()
        self.target_count = target_stats["count"].astype(int).to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        drug_fp_cache: dict[str, np.ndarray] = {}
        protein_cache: dict[str, np.ndarray] = {}
        rows: list[np.ndarray] = []
        for row in frame.itertuples(index=False):
            smiles = str(getattr(row, "drug_smiles"))
            sequence = str(getattr(row, "target_sequence"))
            if smiles not in drug_fp_cache:
                drug_fp_cache[smiles] = _morgan_fingerprint(smiles, self.fingerprint_bits)
            if sequence not in protein_cache:
                protein_cache[sequence] = _protein_composition(sequence)
            rows.append(
                np.concatenate(
                    [
                        drug_fp_cache[smiles],
                        protein_cache[sequence],
                        self._network_features(row),
                    ]
                )
            )
        return np.asarray(rows, dtype=np.float32)

    def _network_features(self, row: object) -> np.ndarray:
        drug_id = str(getattr(row, "drug_id"))
        target_id = str(getattr(row, "target_id"))
        drug_mean = (self.drug_mean or {}).get(drug_id, self.global_mean)
        target_mean = (self.target_mean or {}).get(target_id, self.global_mean)
        drug_count = (self.drug_count or {}).get(drug_id, 0)
        target_count = (self.target_count or {}).get(target_id, 0)
        target_familiarity = float(getattr(row, "target_familiarity", 0.0))
        target_novelty = float(getattr(row, "target_novelty", 1.0 - target_familiarity))
        return np.asarray(
            [
                self.global_mean,
                float(drug_mean),
                float(target_mean),
                math.log1p(float(drug_count)),
                math.log1p(float(target_count)),
                0.5 * (float(drug_mean) + float(target_mean)),
                target_familiarity,
                target_novelty,
            ],
            dtype=np.float32,
        )


def _morgan_fingerprint(smiles: str, fingerprint_bits: int) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    vector = np.zeros((fingerprint_bits,), dtype=np.float32)
    if molecule is None:
        return vector
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=fingerprint_bits)
    DataStructs.ConvertToNumpyArray(fingerprint, vector)
    return vector


def _protein_composition(sequence: str) -> np.ndarray:
    sequence = str(sequence)
    length = max(len(sequence), 1)
    counts = np.asarray([sequence.count(amino_acid) / length for amino_acid in AMINO_ACIDS], dtype=np.float32)
    extras = np.asarray(
        [
            math.log1p(length),
            sum(sequence.count(amino_acid) for amino_acid in "FWY") / length,
            sum(sequence.count(amino_acid) for amino_acid in "KRH") / length,
            sum(sequence.count(amino_acid) for amino_acid in "DE") / length,
        ],
        dtype=np.float32,
    )
    return np.concatenate([counts, extras])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a SimBoost-style classical DTA baseline")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--target-column", default="affinity_model_target")
    parser.add_argument("--fingerprint-bits", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--posthoc-regressor-type", choices=["gbr", "knn", "ridge"], default="knn")
    parser.add_argument("--posthoc-feature-set", choices=["base4", "enriched9"], default="enriched9")
    return parser


def _fit_models(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    ensemble_size: int,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    l2_regularization: float,
) -> list[HistGradientBoostingRegressor]:
    models: list[HistGradientBoostingRegressor] = []
    sample_count = len(y_train)
    for ensemble_index in range(max(1, ensemble_size)):
        rng = np.random.default_rng(seed + 997 * ensemble_index)
        if ensemble_size > 1 and sample_count > 8:
            indices = rng.integers(0, sample_count, size=sample_count)
        else:
            indices = np.arange(sample_count)
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            early_stopping=False,
            random_state=seed + ensemble_index,
        )
        model.fit(x_train[indices], y_train[indices])
        models.append(model)
    return models


def _predict_ensemble(models: list[HistGradientBoostingRegressor], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.vstack([model.predict(features) for model in models])
    prediction_mean = predictions.mean(axis=0)
    prediction_std = predictions.std(axis=0)
    if len(models) == 1:
        prediction_std = np.zeros_like(prediction_mean)
    return prediction_mean.astype(float), prediction_std.astype(float)


def _prediction_frame(split_frame: pd.DataFrame, prediction_mean: np.ndarray, prediction_std: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": split_frame["row_id"].astype(str).to_numpy(),
            "target": split_frame["affinity_model_target"].astype(float).to_numpy(),
            "prediction_mean": prediction_mean,
            "prediction_std": prediction_std,
            "prediction_std_mc_dropout": prediction_std,
        }
    )


def _prepare_prediction_frame(split_frame: pd.DataFrame, prediction_frame: pd.DataFrame) -> pd.DataFrame:
    frame = split_frame.merge(prediction_frame, on="row_id", how="inner")
    frame["prediction"] = frame["prediction_mean"]
    frame["confidence_mc_dropout"] = 1.0 / (1.0 + frame["prediction_std_mc_dropout"])
    prepared = prepare_regression_frame(frame, prediction_col="prediction_mean", target_col="target")
    prepared["confidence_oracle"] = 1.0 / (1.0 + prepared["abs_error"])
    return prepared


def _write_selective_outputs(run_dir: Path, run_name: str, test_frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    output_dir = run_dir / "selective_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_payload: dict[str, dict[str, float | int]] = {}
    curve_rows: list[pd.DataFrame] = []
    confidence_specs = [
        ("mc_dropout", "confidence_mc_dropout"),
        ("target_familiarity", "target_familiarity"),
        ("oracle", "confidence_oracle"),
    ]
    for label, confidence_column in confidence_specs:
        metric_payload[label] = summarize_selective_regression(test_frame, confidence_col=confidence_column)
        if label == "mc_dropout":
            metric_payload[label].update(
                summarize_predictive_intervals(
                    test_frame,
                    prediction_col="prediction_mean",
                    target_col="target",
                    std_col="prediction_std",
                )
            )
        curve = build_risk_coverage_curve(test_frame, confidence_col=confidence_column)
        curve["confidence_source"] = label
        curve_rows.append(curve)
    test_frame.to_csv(output_dir / f"{run_name}_test_predictions.csv", index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(output_dir / f"{run_name}_risk_coverage.csv", index=False)
    (output_dir / f"{run_name}_selective_metrics.json").write_text(json.dumps(metric_payload, indent=2))
    return metric_payload


def _write_posthoc_outputs(
    *,
    run_dir: Path,
    run_name: str,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    regressor_type: str,
    feature_set: str,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    output_dir = run_dir / "posthoc_selector"
    output_dir.mkdir(parents=True, exist_ok=True)
    regressor = fit_posthoc_error_regressor(
        validation_frame,
        random_state=seed,
        regressor_type=regressor_type,
        feature_set=feature_set,
    )
    validation_frame = validation_frame.copy()
    test_frame = test_frame.copy()
    validation_frame["predicted_abs_error_posthoc"] = predict_posthoc_error(regressor, validation_frame)
    test_frame["predicted_abs_error_posthoc"] = predict_posthoc_error(regressor, test_frame)
    validation_frame["confidence_posthoc"] = 1.0 / (1.0 + validation_frame["predicted_abs_error_posthoc"])
    test_frame["confidence_posthoc"] = 1.0 / (1.0 + test_frame["predicted_abs_error_posthoc"])

    metric_payload: dict[str, dict[str, float | int]] = {}
    curve_rows: list[pd.DataFrame] = []
    confidence_specs = [
        ("posthoc_selector", "confidence_posthoc"),
        ("mc_dropout", "confidence_mc_dropout"),
        ("target_familiarity", "target_familiarity"),
        ("oracle", "confidence_oracle"),
    ]
    for label, confidence_column in confidence_specs:
        metric_payload[label] = summarize_selective_regression(test_frame, confidence_col=confidence_column)
        curve = build_risk_coverage_curve(test_frame, confidence_col=confidence_column)
        curve["confidence_source"] = label
        curve_rows.append(curve)

    validation_frame.to_csv(output_dir / f"{run_name}_validation_predictions.csv", index=False)
    test_frame.to_csv(output_dir / f"{run_name}_test_predictions.csv", index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(output_dir / f"{run_name}_posthoc_risk_coverage.csv", index=False)
    (output_dir / f"{run_name}_posthoc_metrics.json").write_text(json.dumps(metric_payload, indent=2))
    return metric_payload


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    run_name = args.run_name or f"simboost_ep15_{args.dataset_name}_{args.split_name}_seed{args.seed}"
    run_dir = workspace / "artifacts" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    split_frame = load_split_frame(
        workspace=workspace,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        seed=args.seed,
    )
    enriched_frame = attach_target_novelty(split_frame)
    train_frame = enriched_frame.loc[enriched_frame["split"] == "train"].reset_index(drop=True)
    validation_split = enriched_frame.loc[enriched_frame["split"] == "val"].reset_index(drop=True)
    test_split = enriched_frame.loc[enriched_frame["split"] == "test"].reset_index(drop=True)

    featurizer = SimBoostFeaturizer(
        fingerprint_bits=args.fingerprint_bits,
        target_column=args.target_column,
    ).fit(train_frame)
    x_train = featurizer.transform(train_frame)
    y_train = train_frame[args.target_column].astype(float).to_numpy()
    models = _fit_models(
        x_train=x_train,
        y_train=y_train,
        seed=args.seed,
        ensemble_size=args.ensemble_size,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
    )

    validation_mean, validation_std = _predict_ensemble(models, featurizer.transform(validation_split))
    test_mean, test_std = _predict_ensemble(models, featurizer.transform(test_split))
    validation_predictions = _prediction_frame(validation_split, validation_mean, validation_std)
    test_predictions = _prediction_frame(test_split, test_mean, test_std)
    validation_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    validation_frame = _prepare_prediction_frame(validation_split, validation_predictions)
    test_frame = _prepare_prediction_frame(test_split, test_predictions)
    _write_selective_outputs(run_dir, run_name, test_frame)
    _write_posthoc_outputs(
        run_dir=run_dir,
        run_name=run_name,
        validation_frame=validation_frame,
        test_frame=test_frame,
        regressor_type=args.posthoc_regressor_type,
        feature_set=args.posthoc_feature_set,
        seed=args.seed,
    )

    test_mse = float(test_frame["squared_error"].mean())
    summary = {
        "run_name": run_name,
        "dataset_name": args.dataset_name,
        "split_name": args.split_name,
        "seed": args.seed,
        "split_seed": args.seed,
        "model_type": "simboost",
        "status": "finished",
        "run_dir": str(run_dir),
        "metrics": {
            "test_loss": test_mse,
            "test_mae": float(test_frame["abs_error"].mean()),
            "test_rmse": math.sqrt(test_mse),
        },
        "config": {
            "fingerprint_bits": args.fingerprint_bits,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "ensemble_size": args.ensemble_size,
            "posthoc_regressor_type": args.posthoc_regressor_type,
            "posthoc_feature_set": args.posthoc_feature_set,
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"run_name": run_name, "run_dir": str(run_dir), "status": "finished"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
