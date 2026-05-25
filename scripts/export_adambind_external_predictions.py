#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from selective_dta_b.data.loading import load_split_frame


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_NUMS = 10
DEFAULT_GNN = "gat_gcn"


def _patch_torch_load_weights_only() -> None:
    original_torch_load = torch.load
    if getattr(original_torch_load, "_selective_dta_b_patched", False):
        return

    def patched_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    patched_torch_load._selective_dta_b_patched = True  # type: ignore[attr-defined]
    torch.load = patched_torch_load


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export AdaMBind validation/test predictions for external posthoc selector")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-mc-samples", type=int, default=1)
    parser.add_argument("--output-path", required=True)
    return parser


def _find_run_summary(workspace: Path, run_name: str) -> Path:
    matches = sorted(workspace.glob(f"artifacts/external_runs/*/runs/{run_name}/run_summary.json"))
    if not matches:
        raise FileNotFoundError(f"AdaMBind run_summary.json not found for {run_name!r}")
    return matches[0]


def _seed_torch(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _import_adambind_modules(external_root: Path):
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))
    from model.Trainer import Trainer  # type: ignore
    from model.gat_gcn import GAT_GCN  # type: ignore
    from model.gcn import GCNNet  # type: ignore
    from utils.TestbedDataset import TestbedDataset  # type: ignore

    return Trainer, {"gat_gcn": GAT_GCN, "gcn": GCNNet}, TestbedDataset


def _read_task_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"AdaMBind split task file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _full_data_with_row_ids(workspace: Path, dataset_name: str, full_data_csv: pd.DataFrame) -> pd.DataFrame:
    standardized_path = workspace / "data" / "processed" / dataset_name / "standardized_pairs.csv"
    standardized = pd.read_csv(standardized_path).copy()
    standardized["affinity_key"] = pd.to_numeric(standardized["affinity_model_target"], errors="coerce").round(6)
    standardized["duplicate_rank"] = standardized.groupby(
        ["drug_smiles", "target_sequence", "affinity_key"],
        dropna=False,
    ).cumcount()

    full = full_data_csv.copy().reset_index().rename(columns={"index": "full_index"})
    full["affinity_key"] = pd.to_numeric(full["affinity"], errors="coerce").round(6)
    full["duplicate_rank"] = full.groupby(
        ["compound_iso_smiles", "target_sequence", "affinity_key"],
        dropna=False,
    ).cumcount()

    merged = full.merge(
        standardized[
            [
                "row_id",
                "drug_smiles",
                "target_sequence",
                "affinity_key",
                "duplicate_rank",
                "affinity_model_target",
            ]
        ],
        left_on=["compound_iso_smiles", "target_sequence", "affinity_key", "duplicate_rank"],
        right_on=["drug_smiles", "target_sequence", "affinity_key", "duplicate_rank"],
        how="left",
    )
    return merged


def _build_task_payloads(
    *,
    full_data_csv: pd.DataFrame,
    encoded_dataset,
    nums: int,
) -> tuple[dict[str, list[object]], dict[str, list[int]]]:
    task_data: dict[str, list[object]] = {}
    query_indices: dict[str, list[int]] = {}
    for target_sequence, group_df in full_data_csv.groupby("target_sequence"):
        indices = group_df.index.tolist()
        random.shuffle(indices)
        encoded_samples = [encoded_dataset[index] for index in indices]
        task_data[target_sequence] = [encoded_samples[:nums], encoded_samples[nums:]]
        query_indices[target_sequence] = indices[nums:]
    return task_data, query_indices


def _checkpoint_path(checkpoints_dir: Path) -> Path:
    matches = sorted(checkpoints_dir.glob("*.pt"))
    if not matches:
        raise FileNotFoundError(f"No AdaMBind checkpoint found under {checkpoints_dir}")
    return matches[-1]


def main() -> int:
    args = build_parser().parse_args()
    del args.accelerator, args.batch_size, args.num_workers, args.num_mc_samples
    workspace = Path(args.workspace).resolve()
    summary_path = _find_run_summary(workspace, args.run_name)
    summary = json.loads(summary_path.read_text())
    dataset_name = str(summary["dataset_name"])
    split_name = str(summary["split_name"])
    seed = int(summary["seed"])
    nums = int(summary.get("nums", DEFAULT_NUMS))
    gnn = str(summary.get("gnn", DEFAULT_GNN))
    run_root = Path(str(summary["run_root"]))
    checkpoints_dir = Path(str(summary["checkpoints_dir"]))
    manifest_path = run_root / "adambind_run_root_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    external_root = (
        Path(args.external_root).resolve()
        if args.external_root
        else Path(str(manifest.get("shared_external_root", workspace / "external" / "AdaMBind"))).resolve()
    )

    _seed_torch(seed)
    _patch_torch_load_weights_only()
    Trainer, model_registry, TestbedDataset = _import_adambind_modules(external_root)

    dataset_stem = f"{dataset_name}-full-data"
    full_data_csv = pd.read_csv(run_root / "data" / f"{dataset_stem}.csv")
    mapped_full_data = _full_data_with_row_ids(workspace, dataset_name, full_data_csv)
    split_frame = load_split_frame(workspace=workspace, dataset_name=dataset_name, split_name=split_name, seed=seed)
    expected_row_ids = set(split_frame.loc[split_frame["split"] == args.split, "row_id"].astype(str).tolist())
    encoded_dataset = TestbedDataset(root=str(run_root / "data"), dataset=dataset_stem)
    task_data, query_indices = _build_task_payloads(
        full_data_csv=full_data_csv,
        encoded_dataset=encoded_dataset,
        nums=nums,
    )

    split_index = 2 if args.split == "val" else 3
    task_ids = _read_task_list(run_root / "data" / f"{dataset_name}_{split_index}.txt")
    ordered_query_indices: list[int] = []
    for task_id in task_ids:
        ordered_query_indices.extend(query_indices.get(task_id, []))

    if gnn not in model_registry:
        raise KeyError(f"Unsupported AdaMBind encoder: {gnn}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_registry[gnn]().to(device)
    model.load_state_dict(torch.load(_checkpoint_path(checkpoints_dir), map_location=device))
    trainer = Trainer(model)
    predict_args = SimpleNamespace(reg_lr=1e-4, update_step_test=3)
    y_pred, y_true = trainer.predict(model, predict_args, task_ids, task_data, ct=False)

    if len(y_pred) != len(ordered_query_indices):
        raise ValueError(
            f"AdaMBind prediction length mismatch for {args.run_name}: "
            f"predictions={len(y_pred)} query_rows={len(ordered_query_indices)}"
        )

    query_frame = mapped_full_data.loc[ordered_query_indices, ["row_id"]].copy().reset_index(drop=True)
    query_frame["prediction_mean"] = np.asarray(y_pred, dtype=float)
    query_frame["prediction_std"] = 0.0
    query_frame["prediction_std_mc_dropout"] = 0.0
    query_frame["prediction"] = query_frame["prediction_mean"]
    query_frame["target"] = np.asarray(y_true, dtype=float)
    query_frame["abs_error"] = np.abs(query_frame["prediction_mean"] - query_frame["target"])
    exported = (
        query_frame.loc[query_frame["row_id"].notna()].copy()
        .assign(row_id=lambda frame: frame["row_id"].astype(str))
        .loc[lambda frame: frame["row_id"].isin(expected_row_ids)]
        .drop_duplicates("row_id", keep="first")
        .reset_index(drop=True)
    )
    if exported.empty:
        raise ValueError(f"No AdaMBind query predictions aligned with {dataset_name}/{split_name}/{args.split}")

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "dataset_name": dataset_name,
                "split_name": split_name,
                "seed": seed,
                "split": args.split,
                "num_query_rows": int(len(query_frame)),
                "num_rows": int(len(exported)),
                "expected_split_rows": int(len(expected_row_ids)),
                "split_row_coverage": float(len(exported) / max(len(expected_row_ids), 1)),
                "output_path": str(output_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

