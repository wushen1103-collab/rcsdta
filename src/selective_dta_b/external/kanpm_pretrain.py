from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch


KANPM_DRUG_COLUMNS = ["drug_key", "compound_iso_smiles"]
KANPM_PROTEIN_COLUMNS = ["target_key", "target_sequence"]


EmbeddingFn = Callable[[str, str], torch.Tensor | np.ndarray]
ContactMapFn = Callable[[str, str], torch.Tensor | np.ndarray]


def _load_records(csv_path: str | Path, id_column: str, value_column: str) -> list[tuple[str, str]]:
    frame = pd.read_csv(csv_path)
    missing = [column for column in [id_column, value_column] if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing KANPM columns in {csv_path}: {missing}")
    return [(str(record_id), str(value)) for record_id, value in zip(frame[id_column], frame[value_column])]


def load_kanpm_drug_records(csv_path: str | Path) -> list[tuple[str, str]]:
    return _load_records(csv_path, id_column="drug_key", value_column="compound_iso_smiles")


def load_kanpm_protein_records(csv_path: str | Path) -> list[tuple[str, str]]:
    return _load_records(csv_path, id_column="target_key", value_column="target_sequence")


def shard_records(
    records: Iterable[tuple[str, str]],
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[tuple[str, str]]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    records_list = list(records)
    if num_shards == 1:
        return records_list
    return [record for index, record in enumerate(records_list) if index % num_shards == shard_index]


def resolve_kanpm_pretrained_paths(external_root: str | Path, dataset_name: str) -> dict[str, Path]:
    pretrained_dir = Path(external_root) / "pretrained" / dataset_name
    return {
        "pretrained_dir": pretrained_dir,
        "chem": pretrained_dir / f"{dataset_name}_chem_pretrained.pkl",
        "esmc": pretrained_dir / f"{dataset_name}_esmc_pretrain.pkl",
        "esm2_contact_map": pretrained_dir / f"{dataset_name}_esm2_contact_map.pkl",
    }


def _mean_embedding(matrix: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.mean(dim=0)
    return matrix.mean(axis=0)


def _to_numpy(matrix: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def pad_square_matrix(matrix: torch.Tensor | np.ndarray, target_size: int | None) -> np.ndarray:
    array = _to_numpy(matrix)
    if target_size is None:
        return array
    if target_size < 1:
        raise ValueError(f"target_size must be >= 1, got {target_size}")
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {array.shape}")
    if array.shape[0] >= target_size:
        return array[:target_size, :target_size]
    padded = np.zeros((target_size, target_size), dtype=array.dtype)
    padded[: array.shape[0], : array.shape[1]] = array
    return padded


def build_embedding_payload(
    dataset_name: str,
    records: Iterable[tuple[str, str]],
    encode_fn: EmbeddingFn,
    max_length: int,
) -> dict[str, object]:
    vec_dict: dict[str, torch.Tensor | np.ndarray] = {}
    mat_dict: dict[str, torch.Tensor | np.ndarray] = {}
    length_dict: dict[str, int] = {}

    for record_id, value in records:
        truncated_value = value[:max_length]
        matrix = encode_fn(record_id, truncated_value)
        mat_dict[record_id] = matrix
        vec_dict[record_id] = _mean_embedding(matrix)
        length_dict[record_id] = len(truncated_value)

    return {
        "dataset": dataset_name,
        "vec_dict": vec_dict,
        "mat_dict": mat_dict,
        "length_dict": length_dict,
    }


def build_contact_map_payload(
    dataset_name: str,
    records: Iterable[tuple[str, str]],
    contact_map_fn: ContactMapFn,
    max_length: int,
    pad_square_size: int | None = None,
) -> dict[str, object]:
    contact_map: dict[str, np.ndarray] = {}
    length_dict: dict[str, int] = {}

    for record_id, value in records:
        truncated_value = value[:max_length]
        contact_map[record_id] = pad_square_matrix(
            contact_map_fn(record_id, truncated_value),
            target_size=pad_square_size,
        )
        length_dict[record_id] = len(truncated_value)

    return {
        "dataset": dataset_name,
        "contact_map": contact_map,
        "length_dict": length_dict,
    }


def merge_contact_map_payloads(
    dataset_name: str,
    payloads: Iterable[dict[str, object]],
) -> dict[str, object]:
    contact_map: dict[str, np.ndarray] = {}
    length_dict: dict[str, int] = {}
    for payload in payloads:
        payload_dataset = payload.get("dataset")
        if payload_dataset != dataset_name:
            raise ValueError(f"Expected dataset {dataset_name}, got {payload_dataset}")

        payload_contact_map = payload.get("contact_map", {})
        payload_length_dict = payload.get("length_dict", {})
        for record_id, matrix in payload_contact_map.items():
            if record_id in contact_map:
                raise ValueError(f"Duplicate contact-map record: {record_id}")
            contact_map[record_id] = np.asarray(matrix)
        for record_id, value in payload_length_dict.items():
            if record_id in length_dict and length_dict[record_id] != value:
                raise ValueError(f"Conflicting record length for {record_id}: {length_dict[record_id]} vs {value}")
            length_dict[record_id] = int(value)

    return {
        "dataset": dataset_name,
        "contact_map": contact_map,
        "length_dict": length_dict,
    }


def save_pickle(payload: dict[str, object], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle)
    return output

