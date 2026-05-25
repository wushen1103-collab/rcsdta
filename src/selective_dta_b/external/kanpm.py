from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_dta_b.data.loading import load_split_frame
from selective_dta_b.data.materialize import resolve_standardized_pairs_path


KANPM_DATASET_NAME_MAP = {
    "bindingdb": "bindingDB",
    "davis": "davis",
    "kiba": "kiba",
}

KANPM_RUNNING_SET_NAME_MAP = {
    "random": "warm",
    "unseen_target": "unseen-prot",
    "unseen_drug": "unseen-drug",
    "all_unseen": "unseen-pair",
    "similarity_aware_unseen_target": "similarity-aware-unseen-prot",
}

KANPM_INTERACTION_COLUMNS = [
    "drug_key",
    "compound_iso_smiles",
    "target_key",
    "target_sequence",
    "affinity",
]


def resolve_kanpm_dataset_name(dataset_name: str) -> str:
    try:
        return KANPM_DATASET_NAME_MAP[dataset_name]
    except KeyError as exc:
        raise KeyError(f"Unsupported KANPM dataset: {dataset_name}") from exc


def resolve_kanpm_running_set_name(split_name: str) -> str:
    try:
        return KANPM_RUNNING_SET_NAME_MAP[split_name]
    except KeyError as exc:
        raise KeyError(f"Unsupported KANPM split: {split_name}") from exc


def resolve_kanpm_seeded_running_set_name(split_name: str, seed: int) -> str:
    return f"{resolve_kanpm_running_set_name(split_name)}-seed{seed}"


def _prepare_interaction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "drug_id",
        "drug_smiles",
        "target_id",
        "target_sequence",
        "affinity_model_target",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns for KANPM export: {missing}")
    exported = frame.loc[
        :,
        [
            "drug_id",
            "drug_smiles",
            "target_id",
            "target_sequence",
            "affinity_model_target",
        ],
    ].copy()
    exported.columns = KANPM_INTERACTION_COLUMNS
    return exported.reset_index(drop=True)


def _prepare_drug_frame(standardized: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["drug_id", "drug_smiles"]
    missing = [column for column in required_columns if column not in standardized.columns]
    if missing:
        raise KeyError(f"Missing columns for KANPM drug export: {missing}")
    drugs = standardized.loc[:, ["drug_id", "drug_smiles"]].drop_duplicates().reset_index(drop=True)
    drugs.columns = ["drug_key", "compound_iso_smiles"]
    return drugs


def _prepare_protein_frame(standardized: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["target_id", "target_sequence"]
    missing = [column for column in required_columns if column not in standardized.columns]
    if missing:
        raise KeyError(f"Missing columns for KANPM protein export: {missing}")
    proteins = standardized.loc[:, ["target_id", "target_sequence"]].drop_duplicates().reset_index(drop=True)
    proteins.columns = ["target_key", "target_sequence"]
    return proteins


def materialize_kanpm_dataset(
    workspace: str | Path,
    dataset_name: str,
    split_name: str,
    seed: int,
    external_root: str | Path,
) -> dict[str, object]:
    workspace_path = Path(workspace)
    external_root_path = Path(external_root)
    kanpm_dataset_name = resolve_kanpm_dataset_name(dataset_name)
    base_running_set_name = resolve_kanpm_running_set_name(split_name)
    running_set_name = resolve_kanpm_seeded_running_set_name(split_name, seed)

    standardized_path = resolve_standardized_pairs_path(workspace_path, dataset_name)
    standardized = pd.read_csv(standardized_path)
    split_frame = load_split_frame(
        workspace=workspace_path,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=seed,
    )

    dataset_root = external_root_path / "datasets" / kanpm_dataset_name
    running_root = dataset_root / running_set_name
    running_root.mkdir(parents=True, exist_ok=True)

    data_df = _prepare_interaction_frame(standardized)
    drugs_df = _prepare_drug_frame(standardized)
    proteins_df = _prepare_protein_frame(standardized)

    split_lookup = {
        "train": "train.csv",
        "val": "valid.csv",
        "test": "test.csv",
    }
    split_outputs: dict[str, str] = {}
    for split_value, file_name in split_lookup.items():
        subset = split_frame.loc[split_frame["split"] == split_value].copy()
        subset_df = _prepare_interaction_frame(subset)
        output_path = running_root / file_name
        subset_df.to_csv(output_path, index=False)
        split_outputs[split_value] = str(output_path)

    data_path = dataset_root / "data.csv"
    drugs_path = dataset_root / f"{kanpm_dataset_name}_drugs.csv"
    proteins_path = dataset_root / f"{kanpm_dataset_name}_prots.csv"
    data_df.to_csv(data_path, index=False)
    drugs_df.to_csv(drugs_path, index=False)
    proteins_df.to_csv(proteins_path, index=False)

    return {
        "dataset_name": dataset_name,
        "kanpm_dataset_name": kanpm_dataset_name,
        "split_name": split_name,
        "base_running_set": base_running_set_name,
        "running_set": running_set_name,
        "seed": seed,
        "dataset_root": str(dataset_root),
        "data_path": str(data_path),
        "drugs_path": str(drugs_path),
        "proteins_path": str(proteins_path),
        "split_outputs": split_outputs,
        "rows": {
            "data": len(data_df),
            "drugs": len(drugs_df),
            "proteins": len(proteins_df),
            "train": int((split_frame["split"] == "train").sum()),
            "val": int((split_frame["split"] == "val").sum()),
            "test": int((split_frame["split"] == "test").sum()),
        },
    }

