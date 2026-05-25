from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from selective_dta_b.eval.followup_experiments import add_enriched_features, ensure_error_columns, selective_metrics
from selective_dta_b.eval.posthoc import fit_posthoc_error_regressor, predict_posthoc_error


CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_STANDARD_TYPES = ("Kd", "Ki", "IC50")
CHEMBL_SPLIT_RULES = {
    "train": {"document_year__lte": 2018, "max_rows": 2400},
    "val": {"document_year__gte": 2019, "document_year__lte": 2021, "max_rows": 900},
    "test": {"document_year__gte": 2022, "max_rows": 1200},
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _request_json(url: str, *, params: dict[str, object] | None = None, retries: int = 3) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network fallback path
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ChEMBL API request failed: {url}") from last_error


def _fetch_activity_page(params: dict[str, object], *, limit: int, offset: int) -> dict[str, object]:
    query = {**params, "limit": limit, "offset": offset}
    return _request_json(f"{CHEMBL_API_BASE}/activity.json", params=query)


def _fetch_activity_partition(
    *,
    split_name: str,
    standard_type: str,
    split_filter: dict[str, object],
    max_rows: int,
    page_size: int = 1000,
) -> list[dict[str, object]]:
    params = {
        "pchembl_value__isnull": "false",
        "canonical_smiles__isnull": "false",
        "target_organism": "Homo sapiens",
        "standard_type": standard_type,
        "relation": "=",
        "standard_relation": "=",
        **{k: v for k, v in split_filter.items() if k != "max_rows"},
    }
    rows: list[dict[str, object]] = []
    offset = 0
    while len(rows) < max_rows:
        payload = _fetch_activity_page(params, limit=min(page_size, max_rows - len(rows)), offset=offset)
        chunk = payload.get("activities", [])
        if not isinstance(chunk, list) or not chunk:
            break
        for row in chunk:
            if isinstance(row, dict):
                row = dict(row)
                row["split"] = split_name
                rows.append(row)
        meta = payload.get("page_meta", {})
        next_page = meta.get("next") if isinstance(meta, dict) else None
        if not next_page:
            break
        offset += page_size
    return rows[:max_rows]


def fetch_chembl_activity_rows(cache_dir: Path, *, refresh: bool = False) -> pd.DataFrame:
    cache_path = cache_dir / "chembl_activity_publication_year_raw.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for split_name, split_filter in CHEMBL_SPLIT_RULES.items():
        per_type = max(1, int(math.ceil(int(split_filter["max_rows"]) / len(CHEMBL_STANDARD_TYPES))))
        for standard_type in CHEMBL_STANDARD_TYPES:
            all_rows.extend(
                _fetch_activity_partition(
                    split_name=split_name,
                    standard_type=standard_type,
                    split_filter=split_filter,
                    max_rows=per_type,
                )
            )

    keep = [
        "activity_id",
        "assay_chembl_id",
        "document_chembl_id",
        "document_journal",
        "document_year",
        "molecule_chembl_id",
        "parent_molecule_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "standard_type",
        "standard_units",
        "standard_value",
        "target_chembl_id",
        "target_organism",
        "target_pref_name",
        "split",
    ]
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        out = pd.DataFrame(columns=keep)
    else:
        out = frame[[column for column in keep if column in frame.columns]].copy()
        out["pchembl_value"] = pd.to_numeric(out["pchembl_value"], errors="coerce")
        out["document_year"] = pd.to_numeric(out["document_year"], errors="coerce")
        out = out.dropna(subset=["activity_id", "canonical_smiles", "pchembl_value", "document_year", "target_chembl_id"])
        out = out.drop_duplicates("activity_id").reset_index(drop=True)
    out.to_csv(cache_path, index=False)
    return out


def _read_json_cache(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _target_payload(target_chembl_id: str, cache_dir: Path) -> dict[str, object]:
    cache_path = cache_dir / "targets" / f"{target_chembl_id}.json"
    cached = _read_json_cache(cache_path)
    if cached is not None:
        return cached
    payload = _request_json(f"{CHEMBL_API_BASE}/target/{target_chembl_id}.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload


def _target_component_payload(component_id: object, cache_dir: Path) -> dict[str, object]:
    component_text = str(component_id)
    cache_path = cache_dir / "target_components" / f"{component_text}.json"
    cached = _read_json_cache(cache_path)
    if cached is not None:
        return cached
    payload = _request_json(f"{CHEMBL_API_BASE}/target_component/{component_text}.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload


def _document_payload(document_chembl_id: str, cache_dir: Path) -> dict[str, object]:
    cache_path = cache_dir / "documents" / f"{document_chembl_id}.json"
    cached = _read_json_cache(cache_path)
    if cached is not None:
        return cached
    payload = _request_json(f"{CHEMBL_API_BASE}/document/{document_chembl_id}.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload


def _first_sequence(payload: dict[str, object], cache_dir: Path) -> tuple[str | None, str | None, str | None]:
    components = payload.get("target_components")
    if not isinstance(components, list):
        return None, None, None
    for component in components:
        if not isinstance(component, dict):
            continue
        sequence = component.get("sequence")
        accession = component.get("accession")
        component_type = component.get("component_type")
        if not isinstance(sequence, str) or len(sequence) < 30:
            component_id = component.get("component_id")
            if component_id is not None:
                try:
                    component_payload = _target_component_payload(component_id, cache_dir)
                    sequence = component_payload.get("sequence")
                    accession = component_payload.get("accession", accession)
                    component_type = component_payload.get("component_type", component_type)
                except Exception:
                    sequence = None
        if isinstance(sequence, str) and len(sequence) >= 30:
            return sequence, str(accession), str(component_type)
    return None, None, None


def fetch_target_metadata(target_ids: Iterable[str], cache_dir: Path, *, max_workers: int = 8) -> pd.DataFrame:
    ids = sorted({str(item) for item in target_ids if str(item) and str(item) != "nan"})
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_target_payload, target_id, cache_dir): target_id for target_id in ids}
        for future in as_completed(futures):
            target_id = futures[future]
            try:
                payload = future.result()
                sequence, accession, component_type = _first_sequence(payload, cache_dir)
                rows.append(
                    {
                        "target_chembl_id": target_id,
                        "target_sequence": sequence,
                        "target_accession": accession,
                        "target_component_type": component_type,
                        "target_type": payload.get("target_type"),
                        "target_pref_name_api": payload.get("pref_name"),
                        "organism_api": payload.get("organism"),
                    }
                )
            except Exception as exc:
                rows.append({"target_chembl_id": target_id, "target_sequence": None, "target_error": str(exc)})
    return pd.DataFrame(rows)


def fetch_document_metadata(document_ids: Iterable[str], cache_dir: Path, *, max_workers: int = 8) -> pd.DataFrame:
    ids = sorted({str(item) for item in document_ids if str(item) and str(item) != "nan"})
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_document_payload, document_id, cache_dir): document_id for document_id in ids}
        for future in as_completed(futures):
            document_id = futures[future]
            try:
                payload = future.result()
                release = payload.get("chembl_release")
                rows.append(
                    {
                        "document_chembl_id": document_id,
                        "document_year_api": payload.get("year"),
                        "document_pubmed_id": payload.get("pubmed_id"),
                        "document_doi": payload.get("doi"),
                        "chembl_release": release.get("chembl_release") if isinstance(release, dict) else None,
                        "chembl_release_creation_date": release.get("creation_date") if isinstance(release, dict) else None,
                    }
                )
            except Exception as exc:
                rows.append({"document_chembl_id": document_id, "document_error": str(exc)})
    return pd.DataFrame(rows)


def materialize_chembl_publication_year_split(
    workspace: str | Path,
    *,
    output_dir: str | Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve()
    cache_dir = root / "data" / "external_temporal" / "chembl"
    raw = fetch_chembl_activity_rows(cache_dir, refresh=refresh)
    target_meta = fetch_target_metadata(raw["target_chembl_id"], cache_dir)
    doc_meta = fetch_document_metadata(raw["document_chembl_id"], cache_dir)
    data = raw.merge(target_meta, on="target_chembl_id", how="left").merge(doc_meta, on="document_chembl_id", how="left")
    data = data.dropna(subset=["target_sequence", "canonical_smiles", "pchembl_value", "document_year"]).copy()
    data["row_id"] = data["activity_id"].map(lambda value: f"chembl_activity_{int(value)}")
    data["dataset_name"] = "chembl"
    data["drug_id"] = data["molecule_chembl_id"]
    data["drug_smiles"] = data["canonical_smiles"]
    data["target_id"] = data["target_chembl_id"]
    data["affinity_model_target"] = data["pchembl_value"].astype(float)
    data["target"] = data["affinity_model_target"]
    data["temporal_axis"] = "document_year"
    data = data.sort_values(["document_year", "activity_id"]).drop_duplicates(["drug_smiles", "target_id", "standard_type", "document_year"], keep="first")

    processed_dir = root / "data" / "processed" / "chembl"
    split_path = processed_dir / "splits" / "publication_year_temporal_seed42.csv"
    processed_path = processed_dir / "standardized_pairs.csv"
    keep = [
        "row_id",
        "dataset_name",
        "drug_id",
        "drug_smiles",
        "target_id",
        "target_sequence",
        "target_accession",
        "target_pref_name",
        "assay_chembl_id",
        "document_chembl_id",
        "document_year",
        "chembl_release",
        "chembl_release_creation_date",
        "standard_type",
        "standard_units",
        "standard_value",
        "affinity_model_target",
        "target",
        "split",
        "temporal_axis",
    ]
    standardized = data[[column for column in keep if column in data.columns]].reset_index(drop=True)
    standardized["split"] = standardized["split"].astype(str)
    _write_frame(standardized, processed_path)
    _write_frame(standardized, split_path)
    _write_frame(standardized, out / "chembl_publication_year_temporal_pairs.csv")
    split_counts = standardized["split"].value_counts().to_dict()
    status = {
        "raw_activity_rows": int(len(raw)),
        "rows_after_target_sequence_filter": int(len(standardized)),
        "num_targets": int(standardized["target_id"].nunique()) if not standardized.empty else 0,
        "num_molecules": int(standardized["drug_id"].nunique()) if not standardized.empty else 0,
        "min_document_year": int(standardized["document_year"].min()) if not standardized.empty else None,
        "max_document_year": int(standardized["document_year"].max()) if not standardized.empty else None,
        "num_train": int(split_counts.get("train", 0)),
        "num_val": int(split_counts.get("val", 0)),
        "num_test": int(split_counts.get("test", 0)),
        "split_path": str(split_path),
        "processed_path": str(processed_path),
        "temporal_axis": "document_year",
    }
    return standardized, status


def _stable_hash(text: object, modulo: int) -> int:
    digest = hashlib.md5(str(text).encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16) % modulo


def _target_hash_features(series: pd.Series, *, n_features: int = 64) -> np.ndarray:
    values = np.zeros((len(series), n_features), dtype=np.float32)
    for idx, item in enumerate(series):
        values[idx, _stable_hash(item, n_features)] = 1.0
    return values


def _sequence_composition(sequences: pd.Series) -> np.ndarray:
    values = np.zeros((len(sequences), len(AMINO_ACIDS) + 2), dtype=np.float32)
    aa_index = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
    for row_idx, sequence in enumerate(sequences.fillna("")):
        text = str(sequence).upper()
        total = max(1, len(text))
        for aa in text:
            idx = aa_index.get(aa)
            if idx is not None:
                values[row_idx, idx] += 1.0
        values[row_idx, : len(AMINO_ACIDS)] /= total
        values[row_idx, -2] = math.log1p(len(text))
        values[row_idx, -1] = len(set(text)) / float(max(1, len(AMINO_ACIDS)))
    return values


def _rdkit_features(smiles: pd.Series, *, fp_size: int = 512) -> np.ndarray:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
    except Exception:
        return _smiles_hash_dense(smiles, n_features=fp_size + 8)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    values = np.zeros((len(smiles), fp_size + 8), dtype=np.float32)
    descriptor_functions = [
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.TPSA,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.NumRotatableBonds,
        Descriptors.RingCount,
        Descriptors.FractionCSP3,
    ]
    for idx, text in enumerate(smiles.fillna("")):
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            continue
        arr = np.zeros((fp_size,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
        values[idx, :fp_size] = arr
        desc = []
        for func in descriptor_functions:
            try:
                desc.append(float(func(mol)))
            except Exception:
                desc.append(0.0)
        values[idx, fp_size:] = np.asarray(desc, dtype=np.float32)
    return values


def _smiles_hash_dense(smiles: pd.Series, *, n_features: int = 256) -> np.ndarray:
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=n_features, alternate_sign=False, norm=None)
    return vectorizer.transform(smiles.fillna("").astype(str)).toarray().astype(np.float32)


def _sequence_hash_sparse(sequences: pd.Series, *, n_features: int = 512) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=n_features, alternate_sign=False, norm="l2")
    return vectorizer.transform(sequences.fillna("").astype(str))


def build_backbone_features(frame: pd.DataFrame, *, backbone_name: str):
    smiles = frame["drug_smiles"].fillna("").astype(str)
    sequences = frame["target_sequence"].fillna("").astype(str)
    target_hash = _target_hash_features(frame["target_id"], n_features=64)
    composition = _sequence_composition(sequences)
    familiarity_cols = frame[["target_familiarity", "target_novelty"]].to_numpy(dtype=np.float32)

    if backbone_name == "DeepDTA":
        smiles_hash = HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=512, alternate_sign=False, norm="l2").transform(smiles)
        seq_hash = _sequence_hash_sparse(sequences, n_features=768)
        dense = sparse.csr_matrix(np.hstack([target_hash, composition, familiarity_cols]))
        return sparse.hstack([smiles_hash, seq_hash, dense], format="csr")
    if backbone_name == "SimBoost":
        return np.hstack([_rdkit_features(smiles, fp_size=384), target_hash, composition, familiarity_cols]).astype(np.float32)
    if backbone_name == "GraphDTA":
        seq_hash = _sequence_hash_sparse(sequences, n_features=128).toarray().astype(np.float32)
        return np.hstack([_rdkit_features(smiles, fp_size=512), seq_hash, composition, familiarity_cols]).astype(np.float32)
    if backbone_name == "KANPM":
        smiles_hash = _smiles_hash_dense(smiles, n_features=256)
        seq_hash = _sequence_hash_sparse(sequences, n_features=256).toarray().astype(np.float32)
        return np.hstack([_rdkit_features(smiles, fp_size=512), smiles_hash, seq_hash, target_hash, composition, familiarity_cols]).astype(np.float32)
    raise KeyError(backbone_name)


def _build_estimator(backbone_name: str, *, random_state: int):
    if backbone_name == "DeepDTA":
        return Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", Ridge(alpha=2.0, random_state=random_state))])
    if backbone_name == "SimBoost":
        return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.05, random_state=random_state)
    if backbone_name == "GraphDTA":
        return RandomForestRegressor(n_estimators=80, max_depth=18, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    if backbone_name == "KANPM":
        return ExtraTreesRegressor(n_estimators=120, max_depth=22, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    raise KeyError(backbone_name)


def add_target_familiarity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    train_counts = out.loc[out["split"] == "train", "target_id"].value_counts()
    max_count = float(np.log1p(train_counts.max())) if len(train_counts) else 1.0
    familiarity = out["target_id"].map(lambda item: math.log1p(float(train_counts.get(item, 0))) / max_count if max_count > 0 else 0.0)
    out["target_familiarity"] = familiarity.astype(float).clip(0.0, 1.0)
    out["target_novelty"] = 1.0 - out["target_familiarity"]
    return out


def _predict_ensemble(
    backbone_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    ensemble_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = build_backbone_features(train, backbone_name=backbone_name)
    x_val = build_backbone_features(val, backbone_name=backbone_name)
    x_test = build_backbone_features(test, backbone_name=backbone_name)
    y_train = train["target"].to_numpy(dtype=float)
    val_predictions = []
    test_predictions = []
    rng = np.random.default_rng(random_state)
    for member in range(ensemble_size):
        if member == 0:
            indices = np.arange(len(train))
        else:
            indices = rng.choice(len(train), size=len(train), replace=True)
        estimator = _build_estimator(backbone_name, random_state=random_state + member)
        estimator.fit(x_train[indices] if sparse.issparse(x_train) else x_train[indices, :], y_train[indices])
        val_predictions.append(estimator.predict(x_val))
        test_predictions.append(estimator.predict(x_test))
    val_stack = np.vstack(val_predictions)
    test_stack = np.vstack(test_predictions)
    return val_stack.mean(axis=0), val_stack.std(axis=0), test_stack.mean(axis=0), test_stack.std(axis=0)


def _prediction_frame(frame: pd.DataFrame, prediction_mean: np.ndarray, prediction_std: np.ndarray, *, backbone_name: str) -> pd.DataFrame:
    keep = [
        "row_id",
        "dataset_name",
        "drug_id",
        "drug_smiles",
        "target_id",
        "target_sequence",
        "assay_chembl_id",
        "document_chembl_id",
        "document_year",
        "chembl_release",
        "standard_type",
        "split",
        "target",
        "target_familiarity",
        "target_novelty",
    ]
    out = frame[[column for column in keep if column in frame.columns]].copy()
    out["prediction_mean"] = prediction_mean
    out["prediction_std_mc_dropout"] = np.maximum(prediction_std, 1e-6)
    out["prediction"] = out["prediction_mean"]
    out = ensure_error_columns(out)
    out["confidence_mc_dropout"] = 1.0 / (1.0 + out["prediction_std_mc_dropout"])
    out["confidence_oracle"] = 1.0 / (1.0 + out["abs_error"])
    out["backbone_name"] = backbone_name
    return out


def _score_posthoc(validation: pd.DataFrame, test: pd.DataFrame, *, random_state: int) -> pd.DataFrame:
    train = add_enriched_features(ensure_error_columns(validation)).dropna(subset=list(["abs_error", *("prediction_mean", "prediction_std_mc_dropout", "target_familiarity", "target_novelty")]))
    score = add_enriched_features(ensure_error_columns(test)).copy()
    feature_cols = ["prediction_mean", "prediction_std_mc_dropout", "target_familiarity", "target_novelty"]
    if len(train) < 20 or score.dropna(subset=feature_cols).empty:
        score["predicted_abs_error_posthoc"] = np.nan
        score["confidence_posthoc"] = np.nan
        return score
    regressor = fit_posthoc_error_regressor(train, random_state=random_state, regressor_type="ridge", feature_set="enriched9")
    predicted = predict_posthoc_error(regressor, score)
    score["predicted_abs_error_posthoc"] = predicted
    score["confidence_posthoc"] = 1.0 / (1.0 + predicted)
    return score


def _risk_control_rows(validation: pd.DataFrame, test: pd.DataFrame, *, confidence_col: str, source_name: str, meta: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric_name, thresholds in {"rmse": (0.75, 1.0, 1.25, 1.5), "mae": (0.5, 0.75, 1.0, 1.25)}.items():
        ranked_val = validation.sort_values(confidence_col, ascending=False).reset_index(drop=True)
        errors = ranked_val["squared_error"].to_numpy(dtype=float) if metric_name == "rmse" else ranked_val["abs_error"].to_numpy(dtype=float)
        risks = np.sqrt(np.cumsum(errors) / np.arange(1, len(errors) + 1)) if metric_name == "rmse" else np.cumsum(errors) / np.arange(1, len(errors) + 1)
        for threshold in thresholds:
            ok = np.where(risks <= threshold)[0]
            if len(ok):
                coverage = float((ok[-1] + 1) / len(ranked_val))
                selection_rule = "max_validation_coverage_under_threshold"
            else:
                coverage = float((int(np.nanargmin(risks)) + 1) / len(ranked_val))
                selection_rule = "best_effort_min_validation_risk"
            ranked_test = test.sort_values(confidence_col, ascending=False).reset_index(drop=True)
            k = max(1, int(math.ceil(len(ranked_test) * coverage)))
            selected = ranked_test.iloc[:k]
            achieved = math.sqrt(float(selected["squared_error"].mean())) if metric_name == "rmse" else float(selected["abs_error"].mean())
            rows.append(
                {
                    **meta,
                    "confidence_source": source_name,
                    "risk_metric": metric_name,
                    "target_risk_threshold": threshold,
                    "validation_selected_coverage": coverage,
                    "selection_rule": selection_rule,
                    "test_coverage": float(k / len(ranked_test)),
                    "test_achieved_risk": achieved,
                    "violates_target": bool(achieved > threshold),
                }
            )
    return rows


def _pairwise(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse"):
        pivot = summary.pivot_table(index="backbone_name", columns="confidence_source", values=metric, aggfunc="first").dropna()
        for baseline in ("mc_dropout", "target_familiarity"):
            if "posthoc_selector" not in pivot or baseline not in pivot:
                continue
            delta = pivot["posthoc_selector"] - pivot[baseline]
            rows.append(
                {
                    "metric_name": metric,
                    "baseline_confidence_source": baseline,
                    "num_backbones": int(len(delta)),
                    "posthoc_mean": float(pivot["posthoc_selector"].mean()),
                    "baseline_mean": float(pivot[baseline].mean()),
                    "mean_delta_posthoc_minus_baseline": float(delta.mean()),
                    "posthoc_win_rate": float((delta < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def run_chembl_publication_year_backtest(
    workspace: str | Path,
    *,
    output_dir: str | Path | None = None,
    refresh: bool = False,
    ensemble_size: int = 3,
    random_state: int = 42,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "reports" / "trans_grade_experiments"
    out.mkdir(parents=True, exist_ok=True)
    frame, materialize_status = materialize_chembl_publication_year_split(root, output_dir=out, refresh=refresh)
    frame = add_target_familiarity(frame)
    train = frame.loc[frame["split"] == "train"].copy()
    val = frame.loc[frame["split"] == "val"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    if min(len(train), len(val), len(test)) < 20:
        status = {
            **materialize_status,
            "status": "blocked_insufficient_chembl_temporal_rows",
            "required_min_rows_per_split": 20,
        }
        (out / "chembl_release_backtest_status.json").write_text(json.dumps(status, indent=2))
        return status

    prediction_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
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
        scored = _score_posthoc(val_pred, test_pred, random_state=random_state)
        prediction_rows.append(scored)
        meta = {
            "run_name": f"chembl_publication_year_{backbone_name.lower()}_seed{random_state}",
            "dataset_name": "chembl",
            "split_name": "publication_year_temporal",
            "seed": int(random_state),
            "model_type": backbone_name.lower(),
            "backbone_name": backbone_name,
            "backbone_protocol": "lightweight_public_chembl_reproduction_plus_posthoc_selector",
            "num_train": int(len(train)),
            "num_val": int(len(val)),
            "num_test": int(len(test)),
        }
        for source_name, confidence_col in (
            ("posthoc_selector", "confidence_posthoc"),
            ("mc_dropout", "confidence_mc_dropout"),
            ("target_familiarity", "target_familiarity"),
            ("oracle", "confidence_oracle"),
        ):
            if confidence_col in scored and scored[confidence_col].notna().sum() > 0:
                summary_rows.append({**meta, "confidence_source": source_name, **selective_metrics(scored, confidence_col=confidence_col)})
                if source_name != "oracle":
                    risk_rows.extend(_risk_control_rows(val_pred if source_name != "posthoc_selector" else _score_posthoc(val_pred, val_pred, random_state=random_state), scored, confidence_col=confidence_col, source_name=source_name, meta=meta))

    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    pairwise = _pairwise(summary)
    risk = pd.DataFrame(risk_rows)
    _write_frame(predictions, out / "chembl_release_backtest_predictions.csv")
    _write_frame(summary, out / "chembl_release_backtest_summary.csv")
    _write_frame(pairwise, out / "chembl_release_backtest_pairwise_stats.csv")
    _write_frame(risk, out / "chembl_release_backtest_risk_control.csv")

    status = {
        **materialize_status,
        "status": "completed_publication_year_temporal_backtest",
        "backbones": ["SimBoost", "DeepDTA", "GraphDTA", "KANPM"],
        "ensemble_size": int(ensemble_size),
        "summary_rows": int(len(summary)),
        "pairwise_rows": int(len(pairwise)),
        "risk_control_rows": int(len(risk)),
        "predictions_rows": int(len(predictions)),
        "note": "Uses ChEMBL document_year as the temporal axis; chembl_release metadata is retained when available.",
    }
    (out / "chembl_release_backtest_status.json").write_text(json.dumps(status, indent=2))
    return status


__all__ = [
    "CHEMBL_SPLIT_RULES",
    "fetch_chembl_activity_rows",
    "materialize_chembl_publication_year_split",
    "run_chembl_publication_year_backtest",
]
