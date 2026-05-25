from __future__ import annotations

import json
import math
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from selective_dta_b.eval.followup_experiments import (
    FEATURE_SETS,
    PredictionRecord,
    add_enriched_features,
    discover_prediction_records,
    ensure_error_columns,
    selective_metrics,
)


BINDINGDB_DEFAULT_SOURCE_URL = (
    "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_PubChem_202604_tsv.zip"
)

BINDINGDB_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "pubchem_cid": ("PubChem CID",),
    "ligand_smiles": ("Ligand SMILES", "Ligand Canonical SMILES"),
    "target_id": (
        "UniProt (SwissProt) Primary ID of Target Chain",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain",
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
    ),
    "target_sequence": ("BindingDB Target Chain Sequence", "BindingDB Target Chain Sequence 1"),
    "kd_nm": ("Kd (nM)",),
    "publication_date": ("Publication Date", "Article Publication Date", "Date of publication"),
    "publication_year": ("Publication Year",),
    "curation_date": ("Curation/Data Entry Date", "Curation Date", "Date in BindingDB"),
    "pmid": ("PMID", "PubMed ID"),
    "doi": ("Article DOI", "DOI"),
}

RECENT_BASELINES = (
    {
        "baseline_name": "PMMR",
        "repo_dir": "PMMR",
        "repo_url": "https://github.com/NENUBioCompute/PMMR",
        "category": "recent_multimodal_backbone",
        "input_mode": "csv_plus_precomputed_compound_and_protein_features",
        "integration_priority": 1,
        "primary_blocker": "needs compound/protein feature generation before formal runs",
        "required_files": ("README.md", "main.py", "data.py"),
    },
    {
        "baseline_name": "UAMRL",
        "repo_dir": "UAMRL",
        "repo_url": "https://github.com/Astraea2xu/UAMRL",
        "category": "recent_uncertainty_aware_backbone",
        "input_mode": "drug_sdf_plus_target_pdb",
        "integration_priority": 2,
        "primary_blocker": "current benchmark lacks per-pair sdf/pdb assets",
        "required_files": ("README.md", "code/training.py"),
    },
    {
        "baseline_name": "EviDTI",
        "repo_dir": "EviDTI",
        "repo_url": "https://github.com/zhaoyanpeng208/EviDTI",
        "category": "recent_evidential_backbone",
        "input_mode": "repo_not_cloned_yet",
        "integration_priority": 3,
        "primary_blocker": "repo still needs clone and protocol audit",
        "required_files": ("README.md",),
    },
)


def _normalized_key(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _match_alias(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lookup = {_normalized_key(column): column for column in columns}
    for alias in aliases:
        matched = lookup.get(_normalized_key(alias))
        if matched is not None:
            return matched
    return None


def _bindingdb_candidate_usecols() -> set[str]:
    return {
        _normalized_key(alias)
        for aliases in BINDINGDB_COLUMN_ALIASES.values()
        for alias in aliases
    }


def _download_to_path(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=context, timeout=180) as response:
        destination.write_bytes(response.read())
    return destination


def read_bindingdb_temporal_source(
    source_path: str | Path | None = None,
    *,
    source_url: str | None = None,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    if source_path is None:
        if source_url is None:
            raise ValueError("either source_path or source_url must be provided")
        if cache_path is None:
            raise ValueError("cache_path is required when downloading from source_url")
        source = _download_to_path(source_url, Path(cache_path).resolve())
    else:
        source = Path(source_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"BindingDB temporal source not found: {source}")

    usecols = _bindingdb_candidate_usecols()
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith((".tsv", ".txt", ".csv"))]
            if not members:
                raise ValueError(f"no tabular member found inside: {source}")
            with archive.open(members[0]) as handle:
                return pd.read_csv(
                    handle,
                    sep="\t",
                    usecols=lambda column: _normalized_key(column) in usecols,
                    low_memory=False,
                )
    separator = "\t" if source.suffix.lower() in {".tsv", ".tab", ".txt"} else ","
    return pd.read_csv(
        source,
        sep=separator,
        usecols=lambda column: _normalized_key(column) in usecols,
        low_memory=False,
    )


def normalize_bindingdb_temporal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    mapping: dict[str, str] = {}
    for canonical, aliases in BINDINGDB_COLUMN_ALIASES.items():
        matched = _match_alias(list(renamed.columns), aliases)
        if matched is not None:
            mapping[matched] = canonical
    renamed = renamed.rename(columns=mapping)

    if "publication_date" not in renamed.columns and "publication_year" in renamed.columns:
        years = pd.to_numeric(renamed["publication_year"], errors="coerce")
        renamed["publication_date"] = years.map(lambda value: f"{int(value)}-01-01" if pd.notna(value) else np.nan)

    for column in ("publication_date", "curation_date"):
        if column in renamed.columns:
            renamed[column] = pd.to_datetime(renamed[column], errors="coerce")
    for column in ("pubchem_cid", "kd_nm"):
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    for column in ("ligand_smiles", "target_id", "target_sequence", "pmid", "doi"):
        if column in renamed.columns:
            renamed[column] = renamed[column].astype(str).str.strip()
    return renamed


def _first_nonempty(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    for value in valid:
        text = str(value).strip()
        if text and text.lower() != "nan":
            return value
    return np.nan


def _aggregate_bindingdb_matches(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    match_strategy: str,
) -> pd.DataFrame:
    rows = frame.copy()
    rows = rows.loc[rows["publication_date"].notna() | rows["curation_date"].notna()].copy()
    if rows.empty:
        return pd.DataFrame()
    agg_spec: dict[str, tuple[str, object]] = {
        "publication_date": ("publication_date", "min"),
        "curation_date": ("curation_date", "min"),
        "source_pubchem_cid": ("pubchem_cid", "min"),
        "source_target_id": ("target_id", _first_nonempty),
        "source_kd_nm": ("kd_nm", "min"),
        "num_candidate_records": ("publication_date", "size"),
    }
    if "pmid" in rows.columns:
        agg_spec["pmid"] = ("pmid", _first_nonempty)
    if "doi" in rows.columns:
        agg_spec["doi"] = ("doi", _first_nonempty)
    if "ligand_smiles" not in group_columns:
        agg_spec["ligand_smiles"] = ("ligand_smiles", _first_nonempty)
    if "target_sequence" not in group_columns:
        agg_spec["target_sequence"] = ("target_sequence", _first_nonempty)
    aggregated = rows.groupby(group_columns, dropna=False).agg(**agg_spec).reset_index()
    aggregated["match_strategy"] = match_strategy
    return aggregated


def _apply_temporal_match_stage(
    *,
    unmatched: pd.DataFrame,
    normalized_with_time: pd.DataFrame,
    source_group_columns: list[str],
    left_on: list[str],
    right_on: list[str],
    match_strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if unmatched.empty:
        return unmatched.iloc[0:0].copy(), unmatched
    source = normalized_with_time.copy()
    for column in source_group_columns:
        source = source.loc[source[column].notna()]
    if source.empty:
        return unmatched.iloc[0:0].copy(), unmatched
    aggregated = _aggregate_bindingdb_matches(
        source,
        group_columns=source_group_columns,
        match_strategy=match_strategy,
    )
    merged = unmatched.merge(
        aggregated,
        left_on=left_on,
        right_on=right_on,
        how="left",
    )
    matched = merged.loc[_has_temporal_signal(merged)].copy()
    still_unmatched = unmatched.loc[~unmatched["row_id"].isin(matched["row_id"])].copy()
    return matched, still_unmatched


def _has_temporal_signal(frame: pd.DataFrame) -> pd.Series:
    publication = pd.to_datetime(frame["publication_date"], errors="coerce")
    curation = pd.to_datetime(frame["curation_date"], errors="coerce")
    return publication.notna() | curation.notna()


def match_bindingdb_temporal_metadata(
    standardized: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    data = standardized.copy()
    required = {"row_id", "drug_id", "drug_smiles", "target_id", "target_sequence"}
    if not required.issubset(data.columns):
        missing = sorted(required - set(data.columns))
        raise KeyError(f"standardized BindingDB frame is missing columns: {missing}")
    if "affinity" not in data.columns:
        raise KeyError("standardized BindingDB frame is missing raw 'affinity' column")

    if "affinity_measure" in data.columns:
        data = data.loc[data["affinity_measure"].astype(str).str.lower() == "kd"].copy()

    data["drug_id_key"] = pd.to_numeric(data["drug_id"], errors="coerce")
    data["target_id_key"] = data["target_id"].astype(str).str.strip()
    data["drug_smiles_key"] = data["drug_smiles"].astype(str).str.strip()
    data["target_sequence_key"] = data["target_sequence"].astype(str).str.strip()
    data["affinity_key"] = pd.to_numeric(data["affinity"], errors="coerce").round(6)
    data = data.loc[data["drug_id_key"].notna() & data["affinity_key"].notna()].copy()

    normalized = normalize_bindingdb_temporal_frame(metadata)
    normalized["target_id"] = normalized["target_id"].astype(str).str.strip()
    normalized["ligand_smiles"] = normalized.get("ligand_smiles", pd.Series(dtype=str)).astype(str).str.strip()
    normalized["target_sequence"] = normalized.get("target_sequence", pd.Series(dtype=str)).astype(str).str.strip()
    normalized["kd_key"] = pd.to_numeric(normalized["kd_nm"], errors="coerce").round(6)
    normalized_with_time = normalized.loc[_has_temporal_signal(normalized)].copy()

    matched_buckets: list[pd.DataFrame] = []
    unmatched = data.copy()

    exact_match_configs = [
        (
            ["pubchem_cid", "target_id", "kd_key"],
            ["drug_id_key", "target_id_key", "affinity_key"],
            ["pubchem_cid", "target_id", "kd_key"],
            "pubchem_target_kd",
        ),
        (
            ["pubchem_cid", "target_sequence", "kd_key"],
            ["drug_id_key", "target_sequence_key", "affinity_key"],
            ["pubchem_cid", "target_sequence", "kd_key"],
            "pubchem_sequence_kd",
        ),
        (
            ["ligand_smiles", "target_id", "kd_key"],
            ["drug_smiles_key", "target_id_key", "affinity_key"],
            ["ligand_smiles", "target_id", "kd_key"],
            "smiles_target_kd",
        ),
        (
            ["ligand_smiles", "target_sequence", "kd_key"],
            ["drug_smiles_key", "target_sequence_key", "affinity_key"],
            ["ligand_smiles", "target_sequence", "kd_key"],
            "smiles_sequence_kd",
        ),
    ]
    for group_columns, left_on, right_on, match_strategy in exact_match_configs:
        stage_matched, unmatched = _apply_temporal_match_stage(
            unmatched=unmatched,
            normalized_with_time=normalized_with_time,
            source_group_columns=group_columns,
            left_on=left_on,
            right_on=right_on,
            match_strategy=match_strategy,
        )
        if not stage_matched.empty:
            matched_buckets.append(stage_matched)

    pair_match_configs = [
        (
            ["pubchem_cid", "target_id"],
            ["drug_id_key", "target_id_key"],
            ["pubchem_cid", "target_id"],
            "pubchem_target_pair_first_seen",
        ),
        (
            ["pubchem_cid", "target_sequence"],
            ["drug_id_key", "target_sequence_key"],
            ["pubchem_cid", "target_sequence"],
            "pubchem_sequence_pair_first_seen",
        ),
        (
            ["ligand_smiles", "target_id"],
            ["drug_smiles_key", "target_id_key"],
            ["ligand_smiles", "target_id"],
            "smiles_target_pair_first_seen",
        ),
        (
            ["ligand_smiles", "target_sequence"],
            ["drug_smiles_key", "target_sequence_key"],
            ["ligand_smiles", "target_sequence"],
            "smiles_sequence_pair_first_seen",
        ),
    ]
    for group_columns, left_on, right_on, match_strategy in pair_match_configs:
        stage_matched, unmatched = _apply_temporal_match_stage(
            unmatched=unmatched,
            normalized_with_time=normalized_with_time,
            source_group_columns=group_columns,
            left_on=left_on,
            right_on=right_on,
            match_strategy=match_strategy,
        )
        if not stage_matched.empty:
            matched_buckets.append(stage_matched)

    matched = pd.concat(matched_buckets, ignore_index=True) if matched_buckets else pd.DataFrame()

    if matched.empty:
        return pd.DataFrame(
            columns=[
                "row_id",
                "publication_date",
                "curation_date",
                "pmid",
                "doi",
                "match_strategy",
                "num_candidate_records",
                "source_pubchem_cid",
                "source_target_id",
                "source_kd_nm",
            ]
        )

    keep = [
        "row_id",
        "publication_date",
        "curation_date",
        "match_strategy",
        "num_candidate_records",
        "source_pubchem_cid",
        "source_target_id",
        "source_kd_nm",
    ]
    for optional_column in ("pmid", "doi"):
        if optional_column in matched.columns:
            keep.append(optional_column)
    out = matched[keep].sort_values("row_id").drop_duplicates("row_id", keep="first").reset_index(drop=True)
    return out


def _rotate_tied_rows(group: pd.DataFrame, seed: int) -> pd.DataFrame:
    ordered = group.sort_values("row_id").reset_index(drop=True)
    if len(ordered) <= 1:
        return ordered
    offset = seed % len(ordered)
    return pd.concat([ordered.iloc[offset:], ordered.iloc[:offset]], ignore_index=True)


def _apply_seeded_tie_break(frame: pd.DataFrame, *, date_column: str, seed: int) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for _, group in frame.sort_values([date_column, "row_id"]).groupby(date_column, sort=True):
        groups.append(_rotate_tied_rows(group, seed))
    return pd.concat(groups, ignore_index=True) if groups else frame.iloc[0:0].copy()


def build_seeded_true_temporal_split(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    seed: int,
    date_column: str = "publication_date",
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
) -> pd.DataFrame:
    if "row_id" not in frame.columns:
        raise KeyError("row_id not found in frame")
    if "row_id" not in metadata.columns:
        raise KeyError("row_id not found in metadata")
    if date_column not in metadata.columns:
        raise KeyError(f"{date_column} not found in metadata")
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("invalid split fractions")

    merged = frame.merge(metadata[["row_id", date_column]].drop_duplicates("row_id"), on="row_id", how="inner")
    merged[date_column] = pd.to_datetime(merged[date_column], errors="coerce")
    merged = merged.loc[merged[date_column].notna()].copy()
    if merged.empty:
        raise ValueError("no rows matched valid temporal metadata")

    merged = _apply_seeded_tie_break(merged, date_column=date_column, seed=seed).reset_index(drop=True)

    n = len(merged)
    train_end = max(1, int(math.floor(n * train_fraction)))
    val_end = min(max(train_end + 1, int(math.floor(n * (train_fraction + val_fraction)))), n - 1)
    merged["split"] = "test"
    merged.loc[: train_end - 1, "split"] = "train"
    merged.loc[train_end : val_end - 1, "split"] = "val"
    merged["true_temporal_rank"] = np.arange(n, dtype=int)
    merged["true_temporal_seed"] = int(seed)
    merged["true_temporal_note"] = "bindingdb_publication_date_temporal_split"
    return merged


def materialize_bindingdb_true_temporal_artifacts(
    workspace: str | Path,
    *,
    source_path: str | Path | None = None,
    source_url: str | None = None,
    seeds: tuple[int, ...] = (42, 43, 44),
    output_dir: str | Path | None = None,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "reports" / "deployment_upgrade_experiments"
    out.mkdir(parents=True, exist_ok=True)

    cache_path = root / "data" / "external_temporal" / "bindingdb" / "BindingDB_PubChem_202604_tsv.zip"
    raw = read_bindingdb_temporal_source(
        source_path=source_path,
        source_url=source_url,
        cache_path=cache_path,
    )
    standardized_path = root / "data" / "processed" / "bindingdb" / "standardized_pairs.csv"
    standardized = pd.read_csv(standardized_path)
    matched = match_bindingdb_temporal_metadata(standardized, raw)

    metadata_out = root / "data" / "external_temporal" / "bindingdb" / "bindingdb_true_temporal_metadata.csv"
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    publication_dates = pd.to_datetime(matched["publication_date"], errors="coerce")
    curation_dates = pd.to_datetime(matched["curation_date"], errors="coerce")
    publication_unique = int(publication_dates.nunique(dropna=True))
    curation_unique = int(curation_dates.nunique(dropna=True))
    if publication_unique <= 1 and curation_unique > publication_unique:
        matched["effective_temporal_date"] = curation_dates
        effective_date_source = "curation_date_fallback"
    else:
        matched["effective_temporal_date"] = publication_dates
        effective_date_source = "publication_date"

    matched.to_csv(metadata_out, index=False)
    matched.to_csv(out / "bindingdb_true_temporal_matches.csv", index=False)

    split_rows: list[dict[str, object]] = []
    for seed in seeds:
        split = build_seeded_true_temporal_split(
            standardized,
            matched,
            seed=seed,
            date_column="effective_temporal_date",
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        split_path = root / "data" / "processed" / "bindingdb" / "splits" / f"true_temporal_seed{seed}.csv"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(split_path, index=False)
        counts = split["split"].value_counts().to_dict()
        split_rows.append(
            {
                "dataset_name": "bindingdb",
                "split_name": "true_temporal",
                "seed": int(seed),
                "split_path": str(split_path),
                "num_rows": int(len(split)),
                "num_train": int(counts.get("train", 0)),
                "num_val": int(counts.get("val", 0)),
                "num_test": int(counts.get("test", 0)),
            }
        )
    split_summary = pd.DataFrame(split_rows)
    split_summary.to_csv(out / "bindingdb_true_temporal_split_summary.csv", index=False)

    coverage = float(len(matched) / len(standardized)) if len(standardized) else 0.0
    effective_dates = pd.to_datetime(matched["effective_temporal_date"], errors="coerce")
    status = {
        "dataset_name": "bindingdb",
        "num_standardized_rows": int(len(standardized)),
        "num_matched_rows": int(len(matched)),
        "match_coverage": coverage,
        "publication_unique_dates": publication_unique,
        "curation_unique_dates": curation_unique,
        "effective_date_source": effective_date_source,
        "min_publication_date": publication_dates.min().strftime("%Y-%m-%d") if publication_dates.notna().any() else None,
        "max_publication_date": publication_dates.max().strftime("%Y-%m-%d") if publication_dates.notna().any() else None,
        "min_effective_temporal_date": effective_dates.min().strftime("%Y-%m-%d") if effective_dates.notna().any() else None,
        "max_effective_temporal_date": effective_dates.max().strftime("%Y-%m-%d") if effective_dates.notna().any() else None,
        "metadata_path": str(metadata_out),
        "num_split_files": int(len(split_rows)),
    }
    (out / "bindingdb_true_temporal_status.json").write_text(json.dumps(status, indent=2))
    return status


def _ridge_model() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


def _compact_frame(record: PredictionRecord, frame: pd.DataFrame) -> pd.DataFrame:
    data = add_enriched_features(ensure_error_columns(frame))
    keep = [
        "row_id",
        "target",
        "prediction_mean",
        "prediction_std_mc_dropout",
        "target_familiarity",
        "target_novelty",
        "abs_error",
        "squared_error",
        "confidence_posthoc",
        "confidence_mc_dropout",
        *FEATURE_SETS["enriched9"],
    ]
    data = data[[column for column in dict.fromkeys(keep) if column in data.columns]].copy()
    data["run_name"] = record.run_name
    data["dataset_name"] = record.dataset_name
    data["split_name"] = record.split_name
    data["seed"] = record.seed
    data["model_type"] = record.model_type
    return data


def _transfer_training_examples(source: pd.DataFrame) -> int:
    cols = FEATURE_SETS["enriched9"]
    return int(len(source.dropna(subset=list(cols) + ["abs_error"])))


def score_cross_library_transfer(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame | None:
    cols = FEATURE_SETS["enriched9"]
    src = source.dropna(subset=list(cols) + ["abs_error"])
    tgt = target.dropna(subset=list(cols)).copy()
    if len(src) < 10 or len(tgt) == 0:
        return None
    model = _ridge_model()
    model.fit(src.loc[:, cols], src["abs_error"])
    predicted_error = np.clip(model.predict(tgt.loc[:, cols]), a_min=0.0, a_max=None)
    tgt["confidence_cross_library_transfer"] = 1.0 / (1.0 + predicted_error)
    return tgt


def build_cross_library_transfer_rows(
    validation_frames: list[pd.DataFrame],
    test_frames: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    if not validation_frames:
        return []
    validation = pd.concat(validation_frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for run_name, target in test_frames.items():
        meta = {key: target[key].iloc[0] for key in ("run_name", "dataset_name", "split_name", "seed", "model_type")}
        source = validation[
            (validation["model_type"] == meta["model_type"])
            & (validation["split_name"] == meta["split_name"])
            & (validation["dataset_name"] != meta["dataset_name"])
        ]
        scored = score_cross_library_transfer(source, target)
        if scored is None:
            continue
        rows.append(
            {
                **meta,
                "confidence_source": "cross_library_transfer",
                "transfer_setting": "same_model_other_datasets",
                "num_transfer_train_examples": _transfer_training_examples(source),
                **selective_metrics(scored, confidence_col="confidence_cross_library_transfer"),
            }
        )
    return rows


def summarize_transfer_pairwise_advantage(
    metrics_frame: pd.DataFrame,
    *,
    transfer_source: str = "cross_library_transfer",
    baseline_sources: tuple[str, ...] = ("mc_dropout", "target_familiarity"),
    metric_names: tuple[str, ...] = ("aurc", "coverage_50_rmse"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_name in metric_names:
        for baseline_source in baseline_sources:
            pair = metrics_frame.loc[
                metrics_frame["confidence_source"].isin((transfer_source, baseline_source)),
                ["run_name", "confidence_source", metric_name],
            ].dropna()
            if pair.empty:
                continue
            pivot = pair.pivot_table(index="run_name", columns="confidence_source", values=metric_name, aggfunc="first").dropna()
            if transfer_source not in pivot.columns or baseline_source not in pivot.columns or pivot.empty:
                continue
            delta = pivot[transfer_source] - pivot[baseline_source]
            wins = delta < 0
            p_value = float(binomtest(int(wins.sum()), len(wins), p=0.5, alternative="greater").pvalue)
            rows.append(
                {
                    "metric_name": metric_name,
                    "baseline_confidence_source": baseline_source,
                    "num_pairs": int(len(pivot)),
                    "transfer_mean": float(pivot[transfer_source].mean()),
                    "baseline_mean": float(pivot[baseline_source].mean()),
                    "mean_delta_transfer_minus_baseline": float(delta.mean()),
                    "transfer_win_rate": float(wins.mean()),
                    "binom_p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def _read_prediction_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return ensure_error_columns(pd.read_csv(path))


def summarize_cross_library_generalization(records: list[PredictionRecord]) -> pd.DataFrame:
    validation_frames: list[pd.DataFrame] = []
    test_frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for record in records:
        test = _read_prediction_frame(record.test_path)
        if test is None:
            continue
        validation = _read_prediction_frame(record.validation_path)
        meta = {
            "run_name": record.run_name,
            "dataset_name": record.dataset_name,
            "split_name": record.split_name,
            "seed": record.seed,
            "model_type": record.model_type,
        }
        if validation is not None:
            validation_frames.append(_compact_frame(record, validation))
        test_frames[record.run_name] = _compact_frame(record, test)

        if "confidence_mc_dropout" in test.columns:
            rows.append(
                {
                    **meta,
                    "confidence_source": "mc_dropout",
                    "transfer_setting": "target_run_baseline",
                    "num_transfer_train_examples": 0,
                    **selective_metrics(test, confidence_col="confidence_mc_dropout"),
                }
            )
        if "target_familiarity" in test.columns:
            rows.append(
                {
                    **meta,
                    "confidence_source": "target_familiarity",
                    "transfer_setting": "target_run_baseline",
                    "num_transfer_train_examples": 0,
                    **selective_metrics(test, confidence_col="target_familiarity"),
                }
            )
        if "confidence_posthoc" in test.columns:
            rows.append(
                {
                    **meta,
                    "confidence_source": "in_dataset_posthoc_selector",
                    "transfer_setting": "target_run_upper_bound",
                    "num_transfer_train_examples": 0,
                    **selective_metrics(test, confidence_col="confidence_posthoc"),
                }
            )

    rows.extend(build_cross_library_transfer_rows(validation_frames, test_frames))
    return pd.DataFrame(rows)


def audit_recent_baseline_repos(workspace: str | Path) -> pd.DataFrame:
    root = Path(workspace).resolve()
    rows: list[dict[str, object]] = []
    for item in RECENT_BASELINES:
        repo_path = root / "external" / item["repo_dir"]
        present = repo_path.exists()
        required_files = [str(repo_path / file_name) for file_name in item["required_files"]]
        missing_files = [path for path in required_files if not Path(path).exists()]
        rows.append(
            {
                **item,
                "repo_path": str(repo_path),
                "repo_present": bool(present),
                "all_required_files_present": bool(present and not missing_files),
                "missing_files": ";".join(missing_files),
            }
        )
    return pd.DataFrame(rows)


def run_deployment_upgrade_experiments(
    workspace: str | Path,
    *,
    output_dir: str | Path,
    bindingdb_source_path: str | Path | None = None,
    bindingdb_source_url: str | None = None,
    paper_only: bool = True,
    max_runs: int | None = None,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    status: dict[str, object] = {}
    if bindingdb_source_path is not None or bindingdb_source_url is not None:
        status["bindingdb_true_temporal"] = materialize_bindingdb_true_temporal_artifacts(
            root,
            source_path=bindingdb_source_path,
            source_url=bindingdb_source_url,
            output_dir=out,
        )

    records = discover_prediction_records(root, paper_only=paper_only, max_runs=max_runs)
    generalization = summarize_cross_library_generalization(records)
    generalization.to_csv(out / "cross_library_generalization_summary.csv", index=False)
    pairwise = summarize_transfer_pairwise_advantage(generalization) if not generalization.empty else pd.DataFrame()
    pairwise.to_csv(out / "cross_library_pairwise_stats.csv", index=False)

    baseline_audit = audit_recent_baseline_repos(root)
    baseline_audit.to_csv(out / "recent_baseline_audit.csv", index=False)

    status.update(
        {
            "num_prediction_records": int(len(records)),
            "cross_library_rows": int(len(generalization)),
            "cross_library_pairwise_rows": int(len(pairwise)),
            "recent_baseline_rows": int(len(baseline_audit)),
        }
    )
    (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


__all__ = [
    "BINDINGDB_DEFAULT_SOURCE_URL",
    "audit_recent_baseline_repos",
    "build_cross_library_transfer_rows",
    "build_seeded_true_temporal_split",
    "materialize_bindingdb_true_temporal_artifacts",
    "match_bindingdb_temporal_metadata",
    "normalize_bindingdb_temporal_frame",
    "read_bindingdb_temporal_source",
    "run_deployment_upgrade_experiments",
    "score_cross_library_transfer",
    "summarize_cross_library_generalization",
    "summarize_transfer_pairwise_advantage",
]

