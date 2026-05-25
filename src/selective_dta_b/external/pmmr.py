from __future__ import annotations

import pickle
import re
from argparse import Namespace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from selective_dta_b.data.loading import load_split_frame


PMMR_COLUMNS = [
    "row_id",
    "compound_iso_smiles",
    "target_id",
    "target_sequence",
    "label",
]

PMMR_DEFAULTS: dict[str, object] = {
    "objective": "regression",
    "batch_size": 128,
    "max_epochs": 200,
    "num_workers": 4,
    "learning_rate": 1e-3,
    "decoder_layers": 3,
    "linear_heads": 10,
    "linear_hidden_dim": 32,
    "decoder_heads": 4,
    "encoder_heads": 4,
    "gnn_layers": 3,
    "encoder_layers": 1,
    "decoder_nums": 1,
    "decoder_dim": 128,
    "compound_gnn_dim": 78,
    "pf_dim": 1024,
    "dropout": 0.2,
    "protein_dim": 128,
    "compound_structure_dim": 78,
    "compound_text_dim": 128,
    "compound_pretrained_dim": 384,
    "protein_pretrained_dim": 480,
}

DEFAULT_COMPOUND_MODEL = "DeepChem/ChemBERTa-77M-MLM"
DEFAULT_PROTEIN_MODEL = "facebook/esm2_t12_35M_UR50D"
PMMR_RUN_NAME_PATTERN = re.compile(r"^pmmr_([^_]+)_(.+)_seed(\d+)$")


def pmmr_num_workers(requested_workers: int, *, stage: str) -> int:
    workers = max(0, int(requested_workers))
    if stage == "train":
        return workers
    if stage in {"valid", "test", "eval", "evaluation"}:
        # PMMR batches can fan out to many open tensor handles during evaluation.
        return 0
    raise ValueError(f"unsupported PMMR stage: {stage}")


def configure_pmmr_runtime() -> str:
    import torch

    strategy = "file_system"
    torch.multiprocessing.set_sharing_strategy(strategy)
    return str(torch.multiprocessing.get_sharing_strategy())


def build_pmmr_run_name(dataset_name: str, split_name: str, seed: int) -> str:
    return f"pmmr_{str(dataset_name).lower()}_{split_name}_seed{int(seed)}"


def parse_pmmr_run_name(run_name: str) -> dict[str, object]:
    match = PMMR_RUN_NAME_PATTERN.fullmatch(str(run_name))
    if match is None:
        raise ValueError(f"invalid PMMR run name: {run_name}")
    return {
        "dataset_name": str(match.group(1)).lower(),
        "split_name": str(match.group(2)),
        "seed": int(match.group(3)),
    }


def resolve_pmmr_report_dir(workspace: str | Path, dataset_name: str, split_name: str, seed: int) -> Path:
    workspace_path = Path(workspace).resolve()
    return (
        workspace_path
        / "reports"
        / "deployment_upgrade_experiments"
        / "pmmr_training"
        / str(dataset_name).lower()
        / f"{split_name}_seed{int(seed)}"
    )


def resolve_pmmr_report_dir_from_run_name(workspace: str | Path, run_name: str) -> Path:
    parsed = parse_pmmr_run_name(run_name)
    return resolve_pmmr_report_dir(
        workspace=workspace,
        dataset_name=str(parsed["dataset_name"]),
        split_name=str(parsed["split_name"]),
        seed=int(parsed["seed"]),
    )


def _prepare_pmmr_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "row_id",
        "drug_smiles",
        "target_id",
        "target_sequence",
        "affinity_model_target",
    }
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise KeyError(f"missing PMMR export columns: {missing}")
    exported = pd.DataFrame(
        {
            "row_id": frame["row_id"],
            "compound_iso_smiles": frame["drug_smiles"],
            "target_id": frame["target_id"],
            "target_sequence": frame["target_sequence"],
            "label": frame["affinity_model_target"],
        }
    )
    return exported.reset_index(drop=True)


def load_pmmr_split_frames(
    workspace: str | Path,
    dataset_name: str,
    split_name: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    workspace_path = Path(workspace)
    split_frame = load_split_frame(
        workspace=workspace_path,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=seed,
    )
    return {
        "train": _prepare_pmmr_frame(split_frame.loc[split_frame["split"] == "train"].copy()),
        "valid": _prepare_pmmr_frame(split_frame.loc[split_frame["split"] == "val"].copy()),
        "test": _prepare_pmmr_frame(split_frame.loc[split_frame["split"] == "test"].copy()),
    }


def _write_split_csvs(output_root: Path, split_frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split_key, frame in split_frames.items():
        file_path = output_root / f"{split_key}.csv"
        frame.to_csv(file_path, index=False)
        paths[f"{split_key}_path"] = str(file_path)
    return paths


def materialize_pmmr_dataset(
    workspace: str | Path,
    dataset_name: str,
    split_name: str,
    seed: int,
    external_root: str | Path,
) -> dict[str, object]:
    external_root_path = Path(external_root)
    split_frames = load_pmmr_split_frames(
        workspace=workspace,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=seed,
    )

    output_root = external_root_path / "selective_data" / dataset_name / f"{split_name}_seed{seed}"
    paths = _write_split_csvs(output_root, split_frames)

    return {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "seed": int(seed),
        "output_root": str(output_root),
        **paths,
        "rows": {split_key: int(len(frame)) for split_key, frame in split_frames.items()},
    }


def _coerce_feature_matrix(value: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"expected a 2D feature matrix, got shape={array.shape}")
    return array


def _unique_targets(frame: pd.DataFrame) -> pd.DataFrame:
    targets = (
        frame.loc[:, ["target_id", "target_sequence"]]
        .drop_duplicates()
        .sort_values(["target_id", "target_sequence"])
        .reset_index(drop=True)
    )
    if targets.empty:
        return targets
    sequence_counts = targets.groupby("target_id")["target_sequence"].nunique()
    conflicting = sequence_counts.loc[sequence_counts > 1]
    if not conflicting.empty:
        ids = ", ".join(conflicting.index.astype(str).tolist()[:5])
        raise ValueError(f"target_id maps to multiple sequences in PMMR export: {ids}")
    return targets


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class ChemBERTaEncoder:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_COMPOUND_MODEL,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoTokenizer, RobertaModel

        self._torch = torch
        self.device = _resolve_device(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def __call__(self, smiles: str) -> np.ndarray:
        encoded = self.tokenizer(
            smiles,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.no_grad():
            outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state[0]
        if hidden.shape[0] > 2:
            hidden = hidden[1:-1]
        return hidden.detach().cpu().numpy().astype(np.float32)


class ESM2Encoder:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_PROTEIN_MODEL,
        device: str = "auto",
        max_length: int = 1024,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = _resolve_device(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def __call__(self, sequence: str) -> np.ndarray:
        encoded = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.no_grad():
            outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state[0]
        if hidden.shape[0] > 2:
            hidden = hidden[1:-1]
        return hidden.detach().cpu().numpy().astype(np.float32)


def materialize_pmmr_assets(
    workspace: str | Path,
    dataset_name: str,
    split_name: str,
    seed: int,
    external_root: str | Path,
    *,
    compound_encoder: Callable[[str], np.ndarray] | None = None,
    protein_encoder: Callable[[str], np.ndarray] | None = None,
    compound_model_name: str = DEFAULT_COMPOUND_MODEL,
    protein_model_name: str = DEFAULT_PROTEIN_MODEL,
    device: str = "auto",
) -> dict[str, object]:
    output_root = Path(external_root) / "selective_data" / dataset_name / f"{split_name}_seed{seed}"
    split_frames = load_pmmr_split_frames(
        workspace=workspace,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=seed,
    )
    paths = _write_split_csvs(output_root, split_frames)

    if compound_encoder is None:
        compound_encoder = ChemBERTaEncoder(model_name=compound_model_name, device=device)
    if protein_encoder is None:
        protein_encoder = ESM2Encoder(model_name=protein_model_name, device=device)

    compound_dir = output_root / "compound"
    compound_dir.mkdir(parents=True, exist_ok=True)
    compound_dict_path = compound_dir / "mol_dict.pkl"
    unique_smiles = sorted(
        {
            str(smiles)
            for frame in split_frames.values()
            for smiles in frame["compound_iso_smiles"].dropna().astype(str).tolist()
        }
    )
    compound_dict = {smiles: _coerce_feature_matrix(compound_encoder(smiles)) for smiles in unique_smiles}
    with open(compound_dict_path, "wb") as handle:
        pickle.dump(compound_dict, handle)

    protein_root = output_root / "protein"
    protein_split_dirs: dict[str, Path] = {}
    num_targets_by_split: dict[str, int] = {}
    for split_key, frame in split_frames.items():
        split_dir = protein_root / split_key
        split_dir.mkdir(parents=True, exist_ok=True)
        protein_split_dirs[split_key] = split_dir
        targets = _unique_targets(frame)
        num_targets_by_split[split_key] = int(len(targets))
        for row in targets.itertuples(index=False):
            feature_matrix = _coerce_feature_matrix(protein_encoder(str(row.target_sequence)))
            np.save(split_dir / f"{row.target_id}.npy", feature_matrix)

    return {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "seed": int(seed),
        "output_root": output_root,
        **paths,
        "compound_dict_path": compound_dict_path,
        "protein_split_dirs": protein_split_dirs,
        "rows": {split_key: int(len(frame)) for split_key, frame in split_frames.items()},
        "num_unique_smiles": int(len(unique_smiles)),
        "num_targets_by_split": num_targets_by_split,
    }


def build_pmmr_training_args(
    *,
    root_data_path: str | Path,
    dataset: str,
    seed: int,
    overrides: dict[str, object] | None = None,
) -> Namespace:
    payload = dict(PMMR_DEFAULTS)
    if overrides:
        payload.update(overrides)
    payload["root_data_path"] = str(root_data_path)
    payload["dataset"] = str(dataset)
    payload["seed"] = int(seed)
    return Namespace(**payload)


__all__ = [
    "ChemBERTaEncoder",
    "DEFAULT_COMPOUND_MODEL",
    "DEFAULT_PROTEIN_MODEL",
    "ESM2Encoder",
    "PMMR_COLUMNS",
    "PMMR_DEFAULTS",
    "PMMR_RUN_NAME_PATTERN",
    "build_pmmr_run_name",
    "build_pmmr_training_args",
    "configure_pmmr_runtime",
    "load_pmmr_split_frames",
    "materialize_pmmr_assets",
    "materialize_pmmr_dataset",
    "parse_pmmr_run_name",
    "pmmr_num_workers",
    "resolve_pmmr_report_dir",
    "resolve_pmmr_report_dir_from_run_name",
]

