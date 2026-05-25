from __future__ import annotations

from pathlib import Path

import pandas as pd


def _kmers(sequence: str, k: int) -> set[str]:
    sequence = str(sequence)
    if len(sequence) < k:
        return {sequence}
    return {sequence[index:index + k] for index in range(len(sequence) - k + 1)}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def compute_target_novelty(
    frame: pd.DataFrame,
    *,
    split_column: str = "split",
    train_split_value: str = "train",
    target_id_column: str = "target_id",
    target_sequence_column: str = "target_sequence",
    kmer_size: int = 3,
) -> pd.DataFrame:
    target_frame = frame[[target_id_column, target_sequence_column, split_column]].drop_duplicates(subset=[target_id_column]).reset_index(drop=True)
    train_targets = target_frame.loc[target_frame[split_column] == train_split_value, [target_id_column, target_sequence_column]].reset_index(drop=True)
    train_kmers = {
        row[target_id_column]: _kmers(row[target_sequence_column], kmer_size)
        for _, row in train_targets.iterrows()
    }

    rows: list[dict[str, object]] = []
    for _, row in target_frame.iterrows():
        target_id = row[target_id_column]
        target_kmers = _kmers(row[target_sequence_column], kmer_size)
        max_train_similarity = 0.0
        for train_set in train_kmers.values():
            similarity = _jaccard_similarity(target_kmers, train_set)
            if similarity > max_train_similarity:
                max_train_similarity = similarity
        rows.append(
            {
                target_id_column: target_id,
                "max_train_similarity": max_train_similarity,
                "target_familiarity": max_train_similarity,
                "target_novelty": 1.0 - max_train_similarity,
            }
        )
    return pd.DataFrame(rows)


def attach_target_novelty(
    frame: pd.DataFrame,
    *,
    target_id_column: str = "target_id",
    kmer_size: int = 3,
) -> pd.DataFrame:
    novelty = compute_target_novelty(
        frame,
        target_id_column=target_id_column,
        kmer_size=kmer_size,
    )
    return frame.merge(novelty, on=target_id_column, how="left")

