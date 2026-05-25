from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from selective_dta_b.eval.followup_experiments import (
    PredictionRecord,
    discover_prediction_records,
    ensure_error_columns,
    selective_metrics,
)


TEMPORAL_DATE_PATTERNS = ("date", "year", "time", "document", "publication")
OOD_FEATURE_CANDIDATES = (
    "prediction_mean",
    "prediction_std_mc_dropout",
    "target_familiarity",
    "target_novelty",
)


@dataclass(frozen=True)
class TemporalMetadataCandidate:
    dataset_name: str
    source_path: Path
    join_key: str
    date_column: str


def _payload(record: PredictionRecord) -> dict[str, object]:
    return {
        "run_name": record.run_name,
        "dataset_name": record.dataset_name,
        "split_name": record.split_name,
        "seed": record.seed,
        "model_type": record.model_type,
    }


def _read_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return ensure_error_columns(pd.read_csv(path))


def _available_ood_features(frame: pd.DataFrame) -> list[str]:
    return [column for column in OOD_FEATURE_CANDIDATES if column in frame.columns and frame[column].notna().any()]


def summarize_ood_distance_baselines(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_neighbors: int = 15,
) -> list[dict[str, object]]:
    validation = ensure_error_columns(validation)
    test = ensure_error_columns(test)
    feature_columns = _available_ood_features(validation)
    feature_columns = [column for column in feature_columns if column in test.columns]
    if not feature_columns:
        return []

    val = validation.dropna(subset=feature_columns).copy()
    tst = test.dropna(subset=feature_columns).copy()
    if len(val) < 3 or len(tst) == 0:
        return []

    scaler = StandardScaler()
    val_scaled = scaler.fit_transform(val.loc[:, feature_columns])
    tst_scaled = scaler.transform(tst.loc[:, feature_columns])

    neighbors = NearestNeighbors(n_neighbors=min(n_neighbors, len(val_scaled)), metric="euclidean")
    neighbors.fit(val_scaled)
    knn_distances, _ = neighbors.kneighbors(tst_scaled)
    knn_distance = knn_distances.mean(axis=1)
    knn_frame = tst.copy()
    knn_frame["confidence_ood_knn"] = 1.0 / (1.0 + knn_distance)

    center = val_scaled.mean(axis=0)
    covariance = np.cov(val_scaled, rowvar=False)
    if np.ndim(covariance) == 0:
        covariance = np.array([[float(covariance)]], dtype=float)
    inverse_covariance = np.linalg.pinv(np.atleast_2d(covariance))
    centered_test = tst_scaled - center
    mahal_distance = np.sqrt(np.sum((centered_test @ inverse_covariance) * centered_test, axis=1))
    mahal_frame = tst.copy()
    mahal_frame["confidence_ood_mahalanobis"] = 1.0 / (1.0 + mahal_distance)

    rows = []
    for method_name, frame, confidence_col in (
        ("ood_knn_distance", knn_frame, "confidence_ood_knn"),
        ("ood_mahalanobis", mahal_frame, "confidence_ood_mahalanobis"),
    ):
        rows.append(
            {
                "confidence_source": method_name,
                "num_reference_examples": int(len(val)),
                "num_features": int(len(feature_columns)),
                "feature_columns": ",".join(feature_columns),
                **selective_metrics(frame, confidence_col=confidence_col),
            }
        )
    return rows


def _murcko_scaffold(smiles: object) -> str:
    if pd.isna(smiles):
        return ""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)


def summarize_scaffold_split(
    standardized: pd.DataFrame,
    split_frame: pd.DataFrame,
    *,
    evaluation_split: str = "test",
) -> dict[str, object]:
    merged = standardized.merge(split_frame[["row_id", "split"]], on="row_id", how="inner")
    merged = merged.loc[merged["split"].isin(("train", evaluation_split))].copy()
    if merged.empty:
        return {"evaluation_split": evaluation_split, "num_rows": 0}

    merged["drug_smiles"] = merged["drug_smiles"].fillna("")
    if "scaffold" not in merged.columns:
        merged["scaffold"] = merged["drug_smiles"].map(_murcko_scaffold)
    train = merged.loc[merged["split"] == "train"].copy()
    test = merged.loc[merged["split"] == evaluation_split].copy()
    if train.empty or test.empty:
        return {"evaluation_split": evaluation_split, "num_rows": int(len(merged))}

    train_drugs = set(train["drug_smiles"])
    test_drugs = set(test["drug_smiles"])
    train_scaffolds = {value for value in train["scaffold"] if value}
    test_scaffolds = {value for value in test["scaffold"] if value}
    overlapping_scaffolds = train_scaffolds & test_scaffolds

    row_seen_scaffold = float(test["scaffold"].isin(train_scaffolds).mean()) if len(test) else float("nan")
    row_exact_drug_overlap = float(test["drug_smiles"].isin(train_drugs).mean()) if len(test) else float("nan")
    return {
        "evaluation_split": evaluation_split,
        "num_rows": int(len(merged)),
        "num_train_rows": int(len(train)),
        "num_eval_rows": int(len(test)),
        "num_train_unique_drugs": int(train["drug_smiles"].nunique()),
        "num_eval_unique_drugs": int(test["drug_smiles"].nunique()),
        "num_train_unique_scaffolds": int(len(train_scaffolds)),
        "num_eval_unique_scaffolds": int(len(test_scaffolds)),
        "num_overlapping_scaffolds": int(len(overlapping_scaffolds)),
        "eval_scaffold_overlap_rate": float(len(overlapping_scaffolds) / len(test_scaffolds)) if test_scaffolds else float("nan"),
        "eval_drug_overlap_rate": float(len(train_drugs & test_drugs) / len(test_drugs)) if test_drugs else float("nan"),
        "eval_row_seen_scaffold_rate": row_seen_scaffold,
        "eval_row_exact_drug_overlap_rate": row_exact_drug_overlap,
    }


def summarize_fewshot_support_calibration(
    frame: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    shrinkage_alpha: float = 0.5,
) -> list[dict[str, object]]:
    data = ensure_error_columns(frame)
    if "target_id" not in data.columns:
        return []

    if "predicted_abs_error_posthoc" in data.columns:
        base_error_col = "predicted_abs_error_posthoc"
        support_source = "posthoc_abs_error"
    elif "prediction_std_mc_dropout" in data.columns:
        base_error_col = "prediction_std_mc_dropout"
        support_source = "mc_dropout_std"
    else:
        return []

    rows: list[dict[str, object]] = []
    for k in k_values:
        query_frames: list[pd.DataFrame] = []
        support_examples = 0
        supported_targets = 0
        for _, group in data.groupby("target_id", dropna=False):
            ordered = group.sort_values("row_id").reset_index(drop=True)
            if len(ordered) <= k:
                continue
            support = ordered.iloc[:k]
            query = ordered.iloc[k:].copy()
            support_error = float(support["abs_error"].mean())
            query["predicted_abs_error_support"] = (
                (1.0 - shrinkage_alpha) * query[base_error_col].astype(float)
                + shrinkage_alpha * support_error
            )
            query["confidence_support_fewshot"] = 1.0 / (1.0 + query["predicted_abs_error_support"])
            query_frames.append(query)
            support_examples += int(len(support))
            supported_targets += 1
        if not query_frames:
            continue
        query_data = pd.concat(query_frames, ignore_index=True)
        rows.append(
            {
                "confidence_source": "fewshot_support_calibration",
                "support_error_source": support_source,
                "support_k": int(k),
                "support_targets": int(supported_targets),
                "support_examples": int(support_examples),
                "query_examples": int(len(query_data)),
                **selective_metrics(query_data, confidence_col="confidence_support_fewshot"),
            }
        )
    return rows


def build_true_temporal_split_from_metadata(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    join_key: str = "row_id",
    date_column: str = "year",
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
) -> pd.DataFrame:
    if join_key not in frame.columns:
        raise KeyError(f"join key not found in frame: {join_key}")
    if join_key not in metadata.columns:
        raise KeyError(f"join key not found in metadata: {join_key}")
    if date_column not in metadata.columns:
        raise KeyError(f"date column not found in metadata: {date_column}")
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("invalid split fractions")

    merged = frame.merge(metadata[[join_key, date_column]].drop_duplicates(join_key), on=join_key, how="left")
    merged = merged.loc[merged[date_column].notna()].copy()
    if merged.empty:
        raise ValueError("no temporal metadata rows matched")

    merged[date_column] = pd.to_datetime(merged[date_column], errors="coerce")
    if merged[date_column].isna().all():
        merged[date_column] = pd.to_numeric(merged[date_column], errors="coerce")
    merged = merged.loc[merged[date_column].notna()].copy()
    if merged.empty:
        raise ValueError("temporal metadata could not be parsed")

    merged = merged.sort_values([date_column, join_key]).reset_index(drop=True)
    n = len(merged)
    train_end = max(1, int(math.floor(n * train_fraction)))
    val_end = min(max(train_end + 1, int(math.floor(n * (train_fraction + val_fraction)))), n - 1)
    merged["split"] = "test"
    merged.loc[: train_end - 1, "split"] = "train"
    merged.loc[train_end : val_end - 1, "split"] = "val"
    merged["true_temporal_rank"] = np.arange(n, dtype=int)
    return merged


def summarize_temporal_metadata_support(workspace: Path) -> tuple[list[dict[str, object]], list[TemporalMetadataCandidate]]:
    rows: list[dict[str, object]] = []
    candidates: list[TemporalMetadataCandidate] = []
    data_root = workspace / "data"
    for dataset_dir in sorted((data_root / "processed").glob("*")):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        search_roots = [
            data_root / "external_temporal",
            data_root / "raw" / dataset_name,
            data_root / dataset_name,
        ]
        dataset_candidates: list[TemporalMetadataCandidate] = []
        for root in search_roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if path.suffix.lower() not in {".csv", ".tsv", ".json"}:
                    continue
                try:
                    if path.suffix.lower() == ".json":
                        raw = json.loads(path.read_text())
                        if not isinstance(raw, list):
                            continue
                        frame = pd.DataFrame(raw)
                    else:
                        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", nrows=5)
                except Exception:
                    continue
                lowered = {column.lower(): column for column in frame.columns}
                date_column = next((lowered[name] for name in lowered if any(pattern in name for pattern in TEMPORAL_DATE_PATTERNS)), None)
                if date_column is None:
                    continue
                join_key = "row_id" if "row_id" in frame.columns else ("drug_id" if "drug_id" in frame.columns else "")
                if not join_key:
                    continue
                dataset_candidates.append(
                    TemporalMetadataCandidate(
                        dataset_name=dataset_name,
                        source_path=path,
                        join_key=join_key,
                        date_column=date_column,
                    )
                )
        candidates.extend(dataset_candidates)
        rows.append(
            {
                "dataset_name": dataset_name,
                "has_true_temporal_metadata": bool(dataset_candidates),
                "num_candidate_sources": int(len(dataset_candidates)),
                "candidate_sources": ";".join(str(item.source_path) for item in dataset_candidates),
                "candidate_join_keys": ";".join(item.join_key for item in dataset_candidates),
                "candidate_date_columns": ";".join(item.date_column for item in dataset_candidates),
            }
        )
    return rows, candidates


def select_virtual_screening_case_studies(
    summary_frame: pd.DataFrame,
    *,
    max_cases_per_dataset: int = 3,
    min_compounds: int = 10,
    min_actives: int = 2,
) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame()
    frame = summary_frame.copy()
    frame = frame.loc[
        (frame["num_compounds"] >= min_compounds)
        & (frame["num_actives"] >= min_actives)
        & frame["enrichment_factor"].notna()
    ].copy()
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["dataset_name", "enrichment_factor", "selected_hits", "num_compounds"],
        ascending=[True, False, False, False],
    )
    return frame.groupby("dataset_name", group_keys=False).head(max_cases_per_dataset).reset_index(drop=True)


def export_virtual_screening_case_compounds(
    records: list[PredictionRecord],
    cases: pd.DataFrame,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()
    record_map = {record.run_name: record for record in records}
    rows: list[dict[str, object]] = []
    for case in cases.to_dict("records"):
        record = record_map.get(str(case["run_name"]))
        if record is None:
            continue
        frame = _read_frame(record.test_path)
        if frame is None or "target_id" not in frame.columns:
            continue
        target_frame = frame.loc[frame["target_id"] == case["target_id"]].copy()
        if target_frame.empty:
            continue
        method = str(case["method"])
        score_column = "prediction_mean"
        if method == "posthoc_weighted" and "confidence_posthoc" in target_frame.columns:
            target_frame["score_posthoc_weighted"] = target_frame["prediction_mean"] * target_frame["confidence_posthoc"]
            score_column = "score_posthoc_weighted"
        elif method == "posthoc_lower_bound" and "predicted_abs_error_posthoc" in target_frame.columns:
            target_frame["score_posthoc_lower_bound"] = target_frame["prediction_mean"] - target_frame["predicted_abs_error_posthoc"]
            score_column = "score_posthoc_lower_bound"
        elif method == "mc_weighted" and "confidence_mc_dropout" in target_frame.columns:
            target_frame["score_mc_weighted"] = target_frame["prediction_mean"] * target_frame["confidence_mc_dropout"]
            score_column = "score_mc_weighted"

        n_active = max(1, int(math.ceil(len(target_frame) * 0.1)))
        active_rows = set(target_frame.sort_values("target", ascending=False).head(n_active)["row_id"])
        ranked = target_frame.sort_values(score_column, ascending=False).head(top_k).copy()
        ranked["case_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked["score_used"] = ranked[score_column]
        ranked["is_top_decile_active"] = ranked["row_id"].isin(active_rows)
        keep_columns = [
            "row_id",
            "drug_id",
            "drug_smiles",
            "target_id",
            "prediction_mean",
            "predicted_abs_error_posthoc",
            "confidence_posthoc",
            "confidence_mc_dropout",
            "target",
            "score_used",
            "case_rank",
            "is_top_decile_active",
        ]
        for row in ranked[[column for column in keep_columns if column in ranked.columns]].to_dict("records"):
            rows.append(
                {
                    "run_name": case["run_name"],
                    "dataset_name": case["dataset_name"],
                    "split_name": case["split_name"],
                    "seed": case["seed"],
                    "model_type": case["model_type"],
                    "method": method,
                    **row,
                }
            )
    return pd.DataFrame(rows)


def estimate_runtime_overhead(
    run_dir: Path,
    *,
    training_log_path: Path | None = None,
) -> dict[str, object]:
    run_summary_path = run_dir / "run_summary.json"
    summary = json.loads(run_summary_path.read_text()) if run_summary_path.exists() else {}

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_sizes = [path.stat().st_size for path in checkpoint_dir.glob("*.ckpt")] + [path.stat().st_size for path in checkpoint_dir.glob("*.pt")]
    checkpoint_size_mb = float(max(checkpoint_sizes) / (1024 * 1024)) if checkpoint_sizes else float("nan")

    parameter_size_mb = float("nan")
    if training_log_path is not None and training_log_path.exists():
        text = training_log_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Total estimated model params size \(MB\)\s+([0-9.]+)", text)
        if match:
            parameter_size_mb = float(match.group(1))

    training_times = []
    if training_log_path is not None and training_log_path.exists():
        training_times.append(training_log_path.stat().st_mtime)
    for path in checkpoint_dir.glob("*"):
        training_times.append(path.stat().st_mtime)
    for path in (run_dir / "logs").rglob("*"):
        if path.is_file():
            training_times.append(path.stat().st_mtime)
    train_wall_minutes = (
        float((max(training_times) - min(training_times)) / 60.0)
        if len(training_times) >= 2
        else float("nan")
    )

    def _dir_minutes(path: Path) -> float:
        if not path.exists():
            return float("nan")
        mtimes = [item.stat().st_mtime for item in path.rglob("*") if item.is_file()]
        if len(mtimes) < 2:
            return float("nan")
        return float((max(mtimes) - min(mtimes)) / 60.0)

    def _dir_size_mb(path: Path) -> float:
        if not path.exists():
            return float("nan")
        return float(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024 * 1024))

    selective_dir = run_dir / "selective_eval"
    posthoc_dir = run_dir / "posthoc_selector"
    test_predictions_path = next(iter(posthoc_dir.glob("*_test_predictions.csv")), None)
    num_examples = 0
    if test_predictions_path is not None:
        num_examples = max(0, sum(1 for _ in test_predictions_path.open(encoding="utf-8")) - 1)

    return {
        "run_name": summary.get("run_name", run_dir.name),
        "dataset_name": summary.get("dataset_name", ""),
        "split_name": summary.get("split_name", ""),
        "seed": summary.get("seed"),
        "model_type": summary.get("model_type", ""),
        "accelerator": summary.get("accelerator", ""),
        "precision": summary.get("precision", ""),
        "num_examples": int(num_examples),
        "parameter_size_mb_estimate": parameter_size_mb,
        "checkpoint_size_mb": checkpoint_size_mb,
        "train_wall_minutes_estimate": train_wall_minutes,
        "posthoc_wall_minutes_estimate": _dir_minutes(posthoc_dir),
        "selective_wall_minutes_estimate": _dir_minutes(selective_dir),
        "posthoc_artifact_size_mb": _dir_size_mb(posthoc_dir),
        "selective_artifact_size_mb": _dir_size_mb(selective_dir),
    }


def run_gap_closing_experiments(
    workspace: str | Path,
    *,
    output_dir: str | Path,
    paper_only: bool = True,
    max_runs: int | None = None,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    records = discover_prediction_records(root, paper_only=paper_only, max_runs=max_runs)
    ood_rows: list[dict[str, object]] = []
    fewshot_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for record in records:
        validation = _read_frame(record.validation_path)
        test = _read_frame(record.test_path)
        if test is None:
            continue
        payload = _payload(record)
        if validation is not None:
            ood_rows.extend({**payload, **row} for row in summarize_ood_distance_baselines(validation, test))
        if record.split_name in {"unseen_target", "similarity_aware_unseen_target"}:
            fewshot_rows.extend({**payload, **row} for row in summarize_fewshot_support_calibration(test))
        training_log_path = root / "logs" / "training" / f"{record.run_name}.log"
        runtime_rows.append(estimate_runtime_overhead(record.test_path.parents[1], training_log_path=training_log_path))

    scaffold_rows: list[dict[str, object]] = []
    for dataset_dir in sorted((root / "data" / "processed").glob("*")):
        standardized_path = dataset_dir / "standardized_pairs.csv"
        splits_dir = dataset_dir / "splits"
        if not standardized_path.exists() or not splits_dir.exists():
            continue
        standardized = pd.read_csv(standardized_path, usecols=["row_id", "drug_smiles"])
        standardized["scaffold"] = standardized["drug_smiles"].fillna("").map(_murcko_scaffold)
        for split_path in sorted(splits_dir.glob("*_seed*.csv")):
            split_columns = pd.read_csv(split_path, nrows=0).columns.tolist()
            if "row_id" not in split_columns or "split" not in split_columns:
                continue
            split_frame = pd.read_csv(split_path, usecols=["row_id", "split"])
            metadata = re.match(r"(?P<split_name>.+)_seed(?P<seed>\d+)\.csv$", split_path.name)
            if metadata is None:
                continue
            row = summarize_scaffold_split(standardized, split_frame, evaluation_split="test")
            scaffold_rows.append(
                {
                    "dataset_name": dataset_dir.name,
                    "split_name": metadata.group("split_name"),
                    "seed": int(metadata.group("seed")),
                    **row,
                }
            )

    temporal_rows, temporal_candidates = summarize_temporal_metadata_support(root)
    if temporal_candidates:
        temporal_blueprint = {
            "status": "temporal_metadata_candidates_found",
            "candidates": [
                {
                    "dataset_name": candidate.dataset_name,
                    "source_path": str(candidate.source_path),
                    "join_key": candidate.join_key,
                    "date_column": candidate.date_column,
                }
                for candidate in temporal_candidates
            ],
        }
    else:
        temporal_blueprint = {
            "status": "missing_true_temporal_metadata",
            "recommended_sources": ["BindingDB entry metadata", "ChEMBL document/year metadata"],
            "supported_join_keys": ["row_id", "drug_id"],
        }

    vs_summary_path = root / "reports" / "followup_experiments" / "virtual_screening_target_summary.csv"
    case_studies = pd.DataFrame()
    top_compounds = pd.DataFrame()
    if vs_summary_path.exists():
        case_studies = select_virtual_screening_case_studies(pd.read_csv(vs_summary_path))
        if not case_studies.empty:
            top_compounds = export_virtual_screening_case_compounds(records, case_studies)

    pd.DataFrame(ood_rows).to_csv(out / "ood_uncertainty_summary.csv", index=False)
    pd.DataFrame(scaffold_rows).to_csv(out / "scaffold_leakage_audit.csv", index=False)
    pd.DataFrame(fewshot_rows).to_csv(out / "fewshot_support_summary.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(out / "runtime_overhead_summary.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(out / "true_temporal_support_audit.csv", index=False)
    (out / "true_temporal_blueprint.json").write_text(json.dumps(temporal_blueprint, indent=2))
    case_studies.to_csv(out / "virtual_screening_case_studies.csv", index=False)
    top_compounds.to_csv(out / "virtual_screening_top_compounds.csv", index=False)

    status = {
        "num_records": len(records),
        "ood_rows": len(ood_rows),
        "scaffold_rows": len(scaffold_rows),
        "fewshot_rows": len(fewshot_rows),
        "runtime_rows": len(runtime_rows),
        "temporal_rows": len(temporal_rows),
        "virtual_screening_case_rows": int(len(case_studies)),
        "virtual_screening_top_compound_rows": int(len(top_compounds)),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


__all__ = [
    "build_true_temporal_split_from_metadata",
    "estimate_runtime_overhead",
    "export_virtual_screening_case_compounds",
    "run_gap_closing_experiments",
    "select_virtual_screening_case_studies",
    "summarize_fewshot_support_calibration",
    "summarize_ood_distance_baselines",
    "summarize_scaffold_split",
    "summarize_temporal_metadata_support",
]

