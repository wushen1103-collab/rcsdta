from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd


DEFAULT_TEST_FRACTION = 0.2
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_KMER_SIZE = 3


def _allocate_counts(n_targets: int, test_fraction: float, val_fraction: float) -> tuple[int, int]:
    test_count = max(1, int(round(n_targets * test_fraction)))
    remaining = max(1, n_targets - test_count)
    val_count = max(1, int(round(n_targets * val_fraction))) if remaining > 1 else 0
    if test_count + val_count >= n_targets:
        val_count = max(0, n_targets - test_count - 1)
    return test_count, val_count


def make_random_split(standardized: pd.DataFrame, test_fraction: float, val_fraction: float, random_seed: int) -> pd.DataFrame:
    indices = list(standardized.index)
    random.Random(random_seed).shuffle(indices)

    test_count = max(1, int(round(len(indices) * test_fraction)))
    val_count = max(1, int(round(len(indices) * val_fraction)))
    if test_count + val_count >= len(indices):
        val_count = max(1, len(indices) - test_count - 1)

    test_idx = set(indices[:test_count])
    val_idx = set(indices[test_count:test_count + val_count])

    split_df = standardized.copy()
    split_df["split"] = "train"
    split_df.loc[split_df.index.isin(val_idx), "split"] = "val"
    split_df.loc[split_df.index.isin(test_idx), "split"] = "test"
    return split_df


def make_unseen_target_split(standardized: pd.DataFrame, test_fraction: float, val_fraction: float, random_seed: int) -> pd.DataFrame:
    target_ids = sorted(standardized["target_id"].unique())
    shuffled = target_ids[:]
    random.Random(random_seed).shuffle(shuffled)

    test_count, val_count = _allocate_counts(len(shuffled), test_fraction, val_fraction)
    test_targets = set(shuffled[:test_count])
    val_targets = set(shuffled[test_count:test_count + val_count])

    split_df = standardized.copy()
    split_df["split"] = "train"
    split_df.loc[split_df["target_id"].isin(val_targets), "split"] = "val"
    split_df.loc[split_df["target_id"].isin(test_targets), "split"] = "test"
    return split_df


def make_unseen_drug_split(standardized: pd.DataFrame, test_fraction: float, val_fraction: float, random_seed: int) -> pd.DataFrame:
    drug_ids = sorted(standardized["drug_id"].unique())
    shuffled = drug_ids[:]
    random.Random(random_seed).shuffle(shuffled)

    test_count, val_count = _allocate_counts(len(shuffled), test_fraction, val_fraction)
    test_drugs = set(shuffled[:test_count])
    val_drugs = set(shuffled[test_count:test_count + val_count])

    split_df = standardized.copy()
    split_df["split"] = "train"
    split_df.loc[split_df["drug_id"].isin(val_drugs), "split"] = "val"
    split_df.loc[split_df["drug_id"].isin(test_drugs), "split"] = "test"
    return split_df


def _kmers(sequence: str, k: int) -> set[str]:
    if len(sequence) < k:
        return {sequence}
    return {sequence[i:i + k] for i in range(len(sequence) - k + 1)}


def make_similarity_aware_unseen_target_split(
    standardized: pd.DataFrame,
    test_fraction: float,
    val_fraction: float,
    kmer_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_sequences = standardized[["target_id", "target_sequence"]].drop_duplicates().reset_index(drop=True)
    kmer_lookup = {row.target_id: _kmers(row.target_sequence, kmer_size) for row in target_sequences.itertuples(index=False)}
    similarity_scores: dict[str, list[float]] = defaultdict(list)

    for left, right in combinations(target_sequences["target_id"], 2):
        left_kmers = kmer_lookup[left]
        right_kmers = kmer_lookup[right]
        union = left_kmers | right_kmers
        similarity = 0.0 if not union else len(left_kmers & right_kmers) / len(union)
        similarity_scores[left].append(similarity)
        similarity_scores[right].append(similarity)

    target_scores = []
    for target_id in target_sequences["target_id"]:
        max_similarity = max(similarity_scores[target_id]) if similarity_scores[target_id] else 0.0
        target_scores.append({
            "target_id": target_id,
            "novelty_score": 1.0 - max_similarity,
        })

    target_scores_df = pd.DataFrame(target_scores).sort_values(["novelty_score", "target_id"], ascending=[False, True]).reset_index(drop=True)
    test_count, val_count = _allocate_counts(len(target_scores_df), test_fraction, val_fraction)
    test_targets = set(target_scores_df.loc[:test_count - 1, "target_id"])
    val_targets = set(target_scores_df.loc[test_count:test_count + val_count - 1, "target_id"])

    split_df = standardized.copy()
    split_df["split"] = "train"
    split_df.loc[split_df["target_id"].isin(val_targets), "split"] = "val"
    split_df.loc[split_df["target_id"].isin(test_targets), "split"] = "test"
    return split_df, target_scores_df


def _partition_entities(entity_ids: list[str], test_fraction: float, val_fraction: float, random_seed: int) -> dict[str, set[str]]:
    shuffled = entity_ids[:]
    random.Random(random_seed).shuffle(shuffled)
    test_count, val_count = _allocate_counts(len(shuffled), test_fraction, val_fraction)
    test_entities = set(shuffled[:test_count])
    val_entities = set(shuffled[test_count:test_count + val_count])
    train_entities = set(shuffled[test_count + val_count:])
    return {
        "train": train_entities,
        "val": val_entities,
        "test": test_entities,
    }


def make_all_unseen_split(
    standardized: pd.DataFrame,
    test_fraction: float,
    val_fraction: float,
    random_seed: int,
    max_attempts: int = 32,
) -> pd.DataFrame:
    drug_ids = sorted(standardized["drug_id"].unique())
    target_ids = sorted(standardized["target_id"].unique())

    for attempt in range(max_attempts):
        drug_groups = _partition_entities(
            entity_ids=drug_ids,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            random_seed=random_seed + attempt * 2,
        )
        target_groups = _partition_entities(
            entity_ids=target_ids,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            random_seed=random_seed + attempt * 2 + 1,
        )

        drug_split_lookup = {
            entity_id: split_name
            for split_name, members in drug_groups.items()
            for entity_id in members
        }
        target_split_lookup = {
            entity_id: split_name
            for split_name, members in target_groups.items()
            for entity_id in members
        }

        split_df = standardized.copy()
        split_df["drug_split"] = split_df["drug_id"].map(drug_split_lookup)
        split_df["target_split"] = split_df["target_id"].map(target_split_lookup)
        split_df = split_df.loc[split_df["drug_split"] == split_df["target_split"]].copy()
        split_df["split"] = split_df["drug_split"]
        split_df = split_df.drop(columns=["drug_split", "target_split"])

        if {"train", "val", "test"}.issubset(set(split_df["split"])):
            return split_df.reset_index(drop=True)

    raise ValueError("Unable to construct all_unseen split with non-empty train/val/test partitions")


def resolve_standardized_pairs_path(workspace: str | Path, dataset_name: str) -> Path:
    workspace_path = Path(workspace)
    return workspace_path / "data" / "processed" / dataset_name / "standardized_pairs.csv"


def resolve_split_output_path(workspace: str | Path, dataset_name: str, split_name: str, seed: int) -> Path:
    workspace_path = Path(workspace)
    return workspace_path / "data" / "processed" / dataset_name / "splits" / f"{split_name}_seed{seed}.csv"


def resolve_target_scores_output_path(workspace: str | Path, dataset_name: str, split_name: str, seed: int) -> Path:
    workspace_path = Path(workspace)
    return workspace_path / "data" / "processed" / dataset_name / "splits" / f"{split_name}_seed{seed}_target_novelty.csv"


def materialize_dataset_split(
    workspace: str | Path,
    dataset_name: str,
    split_name: str,
    seed: int,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    kmer_size: int = DEFAULT_KMER_SIZE,
    overwrite: bool = False,
) -> dict[str, object]:
    standardized_path = resolve_standardized_pairs_path(workspace, dataset_name)
    if not standardized_path.exists():
        raise FileNotFoundError(f"Standardized dataset not found: {standardized_path}")

    split_path = resolve_split_output_path(workspace, dataset_name, split_name, seed)
    target_scores_path = resolve_target_scores_output_path(workspace, dataset_name, split_name, seed)
    if split_path.exists() and not overwrite:
        payload: dict[str, object] = {
            "dataset_name": dataset_name,
            "split_name": split_name,
            "seed": seed,
            "split_path": str(split_path),
            "created": False,
        }
        if target_scores_path.exists():
            payload["target_scores_path"] = str(target_scores_path)
        return payload

    standardized = pd.read_csv(standardized_path)
    target_scores_df: pd.DataFrame | None = None
    if split_name == "random":
        split_df = make_random_split(
            standardized=standardized,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            random_seed=seed,
        )
    elif split_name == "unseen_target":
        split_df = make_unseen_target_split(
            standardized=standardized,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            random_seed=seed,
        )
    elif split_name == "unseen_drug":
        split_df = make_unseen_drug_split(
            standardized=standardized,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            random_seed=seed,
        )
    elif split_name == "all_unseen":
        split_df = make_all_unseen_split(
            standardized=standardized,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            random_seed=seed,
        )
    elif split_name == "similarity_aware_unseen_target":
        split_df, target_scores_df = make_similarity_aware_unseen_target_split(
            standardized=standardized,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            kmer_size=kmer_size,
        )
    else:
        raise ValueError(f"Unsupported split_name: {split_name}")

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_path, index=False)

    payload = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "seed": seed,
        "split_path": str(split_path),
        "created": True,
        "row_count": int(len(split_df)),
    }
    if target_scores_df is not None:
        target_scores_df.to_csv(target_scores_path, index=False)
        payload["target_scores_path"] = str(target_scores_path)
    return payload

