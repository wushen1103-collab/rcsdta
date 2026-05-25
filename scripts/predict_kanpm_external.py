#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from selective_dta_b.data.loading import load_split_frame
from selective_dta_b.eval.inference import resolve_device
from selective_dta_b.external.kanpm import (
    resolve_kanpm_dataset_name,
    resolve_kanpm_seeded_running_set_name,
)
from selective_dta_b.external.predictions import resolve_run_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate cached KANPM predictions for a split")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-mc-samples", type=int, default=16)
    parser.add_argument("--output-path", required=True)
    return parser


def _enable_dropout_only(module: nn.Module) -> None:
    module.eval()
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.train()


def _move_batch_to_device(batch, device: torch.device):
    moved = []
    for item in batch:
        if hasattr(item, "to"):
            moved.append(item.to(device))
        else:
            moved.append(item)
    return tuple(moved)


def _forward_prediction(model: nn.Module, batch) -> torch.Tensor:
    mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, drug_graph, protein_graph = batch
    return model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, drug_graph, protein_graph)


@contextmanager
def _kanpm_import_context(code_root: Path):
    module_names = ["hyperparameter", "MyDataset", "model", "kan", "gnn"]
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    sys.path.insert(0, str(code_root))
    try:
        yield
    finally:
        sys.path = [entry for entry in sys.path if entry != str(code_root)]
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _resolve_external_path(external_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    text = str(path)
    if text.startswith("KANPM-DTA/"):
        return external_root / text.removeprefix("KANPM-DTA/")
    if text.startswith("./KANPM-DTA/"):
        return external_root / text.removeprefix("./KANPM-DTA/")
    return external_root / text


def _load_model_state(model, checkpoint_path: Path, device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    target = model.module if hasattr(model, "module") else model
    try:
        target.load_state_dict(state_dict)
        return
    except RuntimeError:
        if isinstance(state_dict, dict):
            normalized = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    normalized[key.removeprefix("module.")] = value
                else:
                    normalized[f"module.{key}"] = value
            try:
                target.load_state_dict(normalized)
                return
            except RuntimeError:
                pass
        raise


def main() -> int:
    args = build_parser().parse_args()
    context = resolve_run_context(args.workspace, args.run_name)
    if context.model_type != "kanpm":
        raise ValueError(f"Run {args.run_name!r} is not a KANPM run")

    split_frame = load_split_frame(
        workspace=context.workspace,
        dataset_name=context.dataset_name,
        split_name=context.split_name,
        seed=context.split_seed,
    )
    split_value = "val" if args.split == "val" else "test"
    subset = split_frame.loc[split_frame["split"] == split_value].reset_index(drop=True)

    workspace = context.workspace
    code_root = workspace / "external" / "KANPM-DTA" / "code"
    device = resolve_device(args.accelerator)
    external_dataset = resolve_kanpm_dataset_name(context.dataset_name)
    running_set = resolve_kanpm_seeded_running_set_name(context.split_name, context.seed)
    external_root = workspace / "external" / "KANPM-DTA"
    data_root = workspace / "external" / "KANPM-DTA" / "datasets"

    env_updates = {
        "KANPM_DATA_ROOT": str(data_root),
        "KANPM_DATASET": external_dataset,
        "KANPM_RUNNING_SET": running_set,
        "KANPM_BATCH_SIZE": str(args.batch_size),
        "KANPM_NUM_WORKERS": str(args.num_workers),
        "KANPM_CUDA": "0",
    }
    previous_env = {key: os.environ.get(key) for key in env_updates}
    os.environ.update(env_updates)
    try:
        with _kanpm_import_context(code_root):
            HyperParameter = importlib.import_module("hyperparameter").HyperParameter
            MyDataset = importlib.import_module("MyDataset")
            Model = importlib.import_module("model").MODEL

            hp = HyperParameter()
            drug_df = pd.read_csv(hp.drugs_dir)
            prot_df = pd.read_csv(hp.prots_dir)
            mol2vec_dict = _load_pickle(_resolve_external_path(external_root, hp.mol2vec_dir))
            protvec_dict = _load_pickle(_resolve_external_path(external_root, hp.protvec_dir))
            contact_map = _load_pickle(_resolve_external_path(external_root, hp.contact_map))

            split_csv_name = "valid.csv" if args.split == "val" else "test.csv"
            split_csv_path = data_root / external_dataset / running_set / split_csv_name
            split_df = pd.read_csv(split_csv_path)
            dataset = MyDataset.CustomDataSet(split_df, hp)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                collate_fn=lambda batch: MyDataset.pred_my_collate_fn(
                    batch,
                    device,
                    hp,
                    drug_df,
                    prot_df,
                    mol2vec_dict,
                    protvec_dict,
                    contact_map,
                ),
            )

            base_model = Model(hp, device)
            model: nn.Module
            if device.type == "cuda":
                model = nn.DataParallel(base_model).to(device)
            else:
                model = base_model.to(device)
            checkpoint_path = Path(context.summary["paths"]["model"])
            _load_model_state(model, checkpoint_path, device)

            rows: list[dict[str, float | str]] = []
            with torch.inference_mode():
                row_offset = 0
                for batch in dataloader:
                    batch = _move_batch_to_device(batch, device)
                    model.eval()
                    prediction_mean = _forward_prediction(model, batch).detach().cpu().reshape(-1)
                    if args.num_mc_samples <= 1:
                        prediction_std = torch.zeros_like(prediction_mean)
                    else:
                        _enable_dropout_only(model)
                        samples = [_forward_prediction(model, batch).detach().cpu().reshape(-1) for _ in range(args.num_mc_samples)]
                        prediction_std = torch.stack(samples, dim=0).std(dim=0, unbiased=False)

                    for local_index in range(len(prediction_mean)):
                        split_row = subset.iloc[row_offset + local_index]
                        rows.append(
                            {
                                "row_id": str(split_row["row_id"]),
                                "target": float(split_row["affinity_model_target"]),
                                "prediction_mean": float(prediction_mean[local_index].item()),
                                "prediction_std": float(prediction_std[local_index].item()),
                                "prediction_std_mc_dropout": float(prediction_std[local_index].item()),
                            }
                        )
                    row_offset += len(prediction_mean)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(json.dumps({"run_name": args.run_name, "split": args.split, "output_path": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

