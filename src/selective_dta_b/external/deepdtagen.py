from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_dta_b.data.loading import load_split_frame


DEEPDTAGEN_COLUMNS = [
    "compound_iso_smiles",
    "target_smiles",
    "target_sequence",
    "affinity",
]


def _resolve_standardized_pairs_path(workspace: Path, dataset_name: str) -> Path:
    return workspace / "data" / "processed" / dataset_name / "standardized_pairs.csv"


def _prepare_deepdtagen_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["drug_smiles", "target_sequence", "affinity_model_target"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns for DeepDTAGen export: {missing}")
    exported = pd.DataFrame(
        {
            "compound_iso_smiles": frame["drug_smiles"],
            "target_smiles": frame["drug_smiles"],
            "target_sequence": frame["target_sequence"],
            "affinity": frame["affinity_model_target"],
        }
    )
    return exported.reset_index(drop=True)


def materialize_deepdtagen_dataset(
    workspace: str | Path,
    dataset_name: str,
    split_name: str,
    seed: int,
    external_root: str | Path,
    merge_validation_into_train: bool = True,
) -> dict[str, object]:
    workspace_path = Path(workspace)
    external_root_path = Path(external_root)
    split_frame = load_split_frame(
        workspace=workspace_path,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=seed,
    )

    _ = pd.read_csv(_resolve_standardized_pairs_path(workspace_path, dataset_name))
    train_splits = ["train", "val"] if merge_validation_into_train else ["train"]
    train_frame = split_frame.loc[split_frame["split"].isin(train_splits)].copy()
    test_frame = split_frame.loc[split_frame["split"] == "test"].copy()

    output_root = external_root_path / "selective_data" / dataset_name / f"{split_name}_seed{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train.csv"
    test_path = output_root / "test.csv"

    _prepare_deepdtagen_frame(train_frame).to_csv(train_path, index=False)
    _prepare_deepdtagen_frame(test_frame).to_csv(test_path, index=False)

    return {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "seed": seed,
        "merge_validation_into_train": merge_validation_into_train,
        "output_root": str(output_root),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "rows": {
            "train": len(train_frame),
            "test": len(test_frame),
        },
    }

