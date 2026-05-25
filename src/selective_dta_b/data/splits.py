from __future__ import annotations

from pathlib import Path


SPLIT_METADATA = {
    "random": {
        "group_key": "row_id",
        "novelty_metric": "none",
    },
    "unseen_target": {
        "group_key": "target_id",
        "novelty_metric": "target_holdout",
    },
    "similarity_aware_unseen_target": {
        "group_key": "target_id",
        "novelty_metric": "protein_sequence_similarity",
    },
    "unseen_drug": {
        "group_key": "drug_id",
        "novelty_metric": "drug_holdout",
    },
    "all_unseen": {
        "group_key": "drug_id+target_id",
        "novelty_metric": "dual_holdout",
    },
}


def build_split_plan(dataset_name: str, split_type: str, random_seed: int) -> dict[str, object]:
    if split_type not in SPLIT_METADATA:
        raise ValueError(f"Unsupported split_type: {split_type}")

    metadata = SPLIT_METADATA[split_type]
    output_filename = f"{dataset_name}_{split_type}_seed{random_seed}.json"
    return {
        "dataset_name": dataset_name,
        "split_type": split_type,
        "random_seed": random_seed,
        "group_key": metadata["group_key"],
        "novelty_metric": metadata["novelty_metric"],
        "output_filename": output_filename,
        "default_output_path": str(Path("configs") / "splits" / output_filename),
    }

