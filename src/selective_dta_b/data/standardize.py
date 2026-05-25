from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


LOG_FLOOR = 1e-6


def _fallback_target_id(dataset_name: str, sequence: str) -> str:
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:12]
    return f"{dataset_name}_seq_{digest}"


def _canonical_affinity_measure(affinity_measure: str) -> str:
    measure = affinity_measure.strip().lower()
    if measure == "kd":
        return "Kd"
    if measure == "kiba":
        return "KIBA"
    raise ValueError(f"Unsupported affinity_measure: {affinity_measure}")


def _build_affinity_columns(raw_affinity: pd.Series, affinity_measure: str, affinity_unit: str) -> pd.DataFrame:
    affinity = raw_affinity.astype(float)
    clipped = affinity.clip(lower=LOG_FLOOR)
    clipped_for_log = affinity <= 0
    affinity_log10_raw = np.log10(clipped)
    measure = _canonical_affinity_measure(affinity_measure)

    if measure == "Kd":
        if affinity_unit != "nM":
            raise ValueError(f"Kd labels currently require nM units, got: {affinity_unit}")
        affinity_pkd = 9.0 - affinity_log10_raw
        affinity_model_target = affinity_pkd
        affinity_model_target_name = pd.Series("pKd", index=affinity.index)
    else:
        affinity_pkd = pd.Series(np.nan, index=affinity.index, dtype=float)
        affinity_model_target = affinity
        affinity_model_target_name = pd.Series("KIBA_score", index=affinity.index)

    return pd.DataFrame(
        {
            "affinity": affinity,
            "affinity_measure": measure,
            "affinity_unit": affinity_unit,
            "affinity_log10_raw": affinity_log10_raw,
            "affinity_pKd": affinity_pkd,
            "affinity_model_target": affinity_model_target,
            "affinity_model_target_name": affinity_model_target_name,
            "affinity_clipped_for_log": clipped_for_log,
        }
    )


def standardize_dta_frame(
    raw: pd.DataFrame,
    dataset_name: str,
    affinity_measure: str,
    affinity_unit: str,
) -> pd.DataFrame:
    frame = raw.copy()
    frame["Drug_ID"] = frame["Drug_ID"].astype(str)
    frame["Drug"] = frame["Drug"].astype(str)
    frame["Target"] = frame["Target"].astype(str)

    target_id = frame["Target_ID"].astype("string")
    missing_mask = target_id.isna() | (target_id.str.strip() == "")
    frame.loc[missing_mask, "Target_ID"] = frame.loc[missing_mask, "Target"].map(
        lambda seq: _fallback_target_id(dataset_name, seq)
    )
    frame["Target_ID"] = frame["Target_ID"].astype(str)

    affinity_frame = _build_affinity_columns(
        raw_affinity=frame["Y"],
        affinity_measure=affinity_measure,
        affinity_unit=affinity_unit,
    )

    standardized = pd.DataFrame(
        {
            "row_id": [f"{dataset_name}_{idx}" for idx in range(len(frame))],
            "dataset_name": dataset_name,
            "drug_id": frame["Drug_ID"],
            "drug_smiles": frame["Drug"],
            "target_id": frame["Target_ID"],
            "target_sequence": frame["Target"],
            "affinity": affinity_frame["affinity"],
            "affinity_measure": affinity_frame["affinity_measure"],
            "affinity_unit": affinity_frame["affinity_unit"],
            "affinity_log10_raw": affinity_frame["affinity_log10_raw"],
            "affinity_pKd": affinity_frame["affinity_pKd"],
            "affinity_model_target": affinity_frame["affinity_model_target"],
            "affinity_model_target_name": affinity_frame["affinity_model_target_name"],
            "affinity_clipped_for_log": affinity_frame["affinity_clipped_for_log"],
        }
    )
    return standardized
