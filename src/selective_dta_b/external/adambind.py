from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from selective_dta_b.data.loading import load_split_frame
from selective_dta_b.data.materialize import resolve_standardized_pairs_path


ADAMBIND_SPLIT_FILE_INDEX = {
    "train": 1,
    "val": 2,
    "test": 3,
}


def _resolve_split_targets(standardized: pd.DataFrame, split_frame: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    standardized_required_columns = {"target_sequence"}
    split_required_columns = {"target_sequence", "split"}
    missing_standardized = sorted(standardized_required_columns - set(standardized.columns))
    missing_split = sorted(split_required_columns - set(split_frame.columns))
    if missing_standardized:
        raise KeyError(f"Standardized dataset missing columns for AdaMBind export: {missing_standardized}")
    if missing_split:
        raise KeyError(f"Split frame missing columns for AdaMBind export: {missing_split}")

    full_counts = standardized.groupby("target_sequence").size().rename("full_count")
    split_counts = (
        split_frame.groupby(["target_sequence", "split"]).size().unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
        .astype(int)
    )

    full_targets = set(full_counts.index.tolist())
    split_targets = set(split_counts.index.tolist())

    missing_targets = sorted(full_targets - split_targets)
    unknown_targets = sorted(split_targets - full_targets)
    multi_split_targets = sorted(
        target
        for target, row in split_counts.iterrows()
        if int((row > 0).sum()) > 1
    )
    partial_targets = sorted(
        target
        for target in split_counts.index
        if int(split_counts.loc[target].sum()) != int(full_counts.loc[target])
    )

    incompatibilities = {
        "missing_targets": missing_targets,
        "unknown_targets": unknown_targets,
        "multi_split_targets": multi_split_targets,
        "partial_targets": partial_targets,
    }
    if any(incompatibilities.values()):
        return {}, incompatibilities

    target_lists: dict[str, list[str]] = {}
    for split_name in ["train", "val", "test"]:
        members = split_counts.index[split_counts[split_name] > 0].tolist()
        target_lists[split_name] = sorted(members)
    return target_lists, incompatibilities


def _apply_min_total_interactions_filter(
    count_frame: pd.DataFrame,
    target_lists: dict[str, list[str]],
    *,
    min_total_interactions: int | None,
) -> tuple[dict[str, list[str]], dict[str, object]]:
    if min_total_interactions is None:
        return target_lists, {
            "enabled": False,
            "min_total_interactions": None,
            "retained_counts": {split_name: len(targets) for split_name, targets in target_lists.items()},
            "dropped_counts": {split_name: 0 for split_name in target_lists},
        }

    total_counts = count_frame.groupby("target_sequence").size()
    filtered_target_lists: dict[str, list[str]] = {}
    retained_counts: dict[str, int] = {}
    dropped_counts: dict[str, int] = {}
    for split_name, targets in target_lists.items():
        filtered_targets = [
            target
            for target in targets
            if int(total_counts.get(target, 0)) >= min_total_interactions
        ]
        filtered_target_lists[split_name] = filtered_targets
        retained_counts[split_name] = len(filtered_targets)
        dropped_counts[split_name] = len(targets) - len(filtered_targets)

    return filtered_target_lists, {
        "enabled": True,
        "min_total_interactions": min_total_interactions,
        "retained_counts": retained_counts,
        "dropped_counts": dropped_counts,
    }


def stage_adambind_split(
    workspace: str | Path,
    dataset_name: str,
    split_name: str,
    seed: int,
    external_root: str | Path,
    *,
    activate: bool = False,
    min_total_interactions: int | None = None,
) -> dict[str, object]:
    workspace_path = Path(workspace).resolve()
    external_root_path = Path(external_root).resolve()

    standardized_path = resolve_standardized_pairs_path(workspace_path, dataset_name)
    split_frame = load_split_frame(
        workspace=workspace_path,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=seed,
    )
    standardized = pd.read_csv(standardized_path)
    external_dataset_path = external_root_path / "data" / f"{dataset_name}-full-data.csv"
    count_frame = pd.read_csv(external_dataset_path) if external_dataset_path.exists() else standardized

    target_lists, incompatibilities = _resolve_split_targets(standardized=standardized, split_frame=split_frame)
    if not target_lists:
        raise ValueError(
            "AdaMBind split export requires each target_sequence to map to exactly one split "
            "and to include all interactions for that target. "
            f"Incompatible dataset/split detected for {dataset_name}/{split_name}/seed{seed}: "
            f"{json.dumps(incompatibilities, ensure_ascii=False)}"
        )
    target_lists, viability_filter = _apply_min_total_interactions_filter(
        count_frame=count_frame,
        target_lists=target_lists,
        min_total_interactions=min_total_interactions,
    )
    empty_splits = [split_value for split_value, targets in target_lists.items() if not targets]
    if empty_splits:
        raise ValueError(
            "AdaMBind viable-target filtering removed every target from one or more partitions. "
            f"dataset={dataset_name} split={split_name} seed={seed} "
            f"min_total_interactions={min_total_interactions} empty_partitions={empty_splits}"
        )

    staged_root = external_root_path / "selective_splits" / dataset_name / f"{split_name}_seed{seed}"
    staged_root.mkdir(parents=True, exist_ok=True)
    data_root = external_root_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    staged_files: dict[str, str] = {}
    active_files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split_value, index in ADAMBIND_SPLIT_FILE_INDEX.items():
        output_path = staged_root / f"{dataset_name}_{index}.txt"
        lines = target_lists[split_value]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        staged_files[split_value] = str(output_path)
        counts[split_value] = len(lines)
        if activate:
            active_path = data_root / f"{dataset_name}_{index}.txt"
            shutil.copyfile(output_path, active_path)
            active_files[split_value] = str(active_path)

    manifest = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "seed": seed,
        "staged_root": str(staged_root),
        "standardized_path": str(standardized_path),
        "count_reference_path": str(external_dataset_path if external_dataset_path.exists() else standardized_path),
        "staged_files": staged_files,
        "counts": counts,
        "activate": activate,
        "active_files": active_files,
        "viability_filter": viability_filter,
        "compatibility": {
            "representable": True,
            "incompatibilities": incompatibilities,
        },
    }
    manifest_path = staged_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


__all__ = [
    "ADAMBIND_SPLIT_FILE_INDEX",
    "stage_adambind_split",
]

