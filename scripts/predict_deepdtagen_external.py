#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from selective_dta_b.data.loading import load_split_frame
from selective_dta_b.eval.inference import resolve_device
from selective_dta_b.external.deepdtagen_runtime import (
    ensure_deepdtagen_state_dict_capacity,
    ensure_deepdtagen_sequence_capacity,
    legacy_torch_load_context,
    load_deepdtagen_modules,
    seq_cat,
    smile_to_graph,
)
from selective_dta_b.external.predictions import resolve_run_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate cached DeepDTAGen predictions for a split")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-mc-samples", type=int, default=16)
    parser.add_argument("--output-path", required=True)
    return parser


def _enable_dropout_only(module: nn.Module) -> None:
    module.eval()
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.train()


def _resolve_effective_batch_size(dataset_name: str, requested_batch_size: int) -> int:
    dataset_caps = {
        "bindingdb": 4,
        "kiba": 8,
    }
    return max(1, min(requested_batch_size, dataset_caps.get(dataset_name.lower(), requested_batch_size)))


def _build_export_frame(split_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound_iso_smiles": split_frame["drug_smiles"],
            "target_smiles": split_frame["drug_smiles"],
            "target_sequence": split_frame["target_sequence"],
            "affinity": split_frame["affinity_model_target"],
        }
    ).reset_index(drop=True)


def _build_dataset(frame: pd.DataFrame, *, data_root: Path, dataset_name: str, split_label: str, tokenizer, utils_module):
    compound_smiles = frame["compound_iso_smiles"].tolist()
    smile_graph = {smile: smile_to_graph(smile) for smile in sorted(set(compound_smiles))}
    tokenized = [torch.LongTensor(tokenizer.parse(smile)) for smile in frame["target_smiles"]]
    proteins = np.asarray([seq_cat(sequence) for sequence in frame["target_sequence"]])

    with legacy_torch_load_context():
        return utils_module.TestbedDataset(
            root=str(data_root),
            dataset=f"{dataset_name}_{split_label}",
            xd=np.asarray(compound_smiles),
            xdt=tokenized,
            xt=proteins,
            y=frame["affinity"].to_numpy(),
            smile_graph=smile_graph,
        )


def main() -> int:
    args = build_parser().parse_args()
    context = resolve_run_context(args.workspace, args.run_name)
    if context.model_type != "deepdtagen":
        raise ValueError(f"Run {args.run_name!r} is not a DeepDTAGen run")

    split_frame = load_split_frame(
        workspace=context.workspace,
        dataset_name=context.dataset_name,
        split_name=context.split_name,
        seed=context.split_seed,
    )
    subset = split_frame.loc[split_frame["split"] == args.split].reset_index(drop=True)
    export_frame = _build_export_frame(subset)
    device = resolve_device(args.accelerator)
    modules = load_deepdtagen_modules(context.workspace / "external" / "DeepDTAGen")
    utils_module = modules["utils"]
    model_module = modules["model"]

    tokenizer_path = context.run_dir / "data" / f"{context.dataset_name}_tokenizer.pkl"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"DeepDTAGen tokenizer missing: {tokenizer_path}")
    with tokenizer_path.open("rb") as handle:
        tokenizer = pickle.load(handle)

    cache_root = context.run_dir / "external_predictions" / "cache" / args.split
    cache_root.mkdir(parents=True, exist_ok=True)
    dataset = _build_dataset(
        export_frame,
        data_root=cache_root,
        dataset_name=context.dataset_name,
        split_label=args.split,
        tokenizer=tokenizer,
        utils_module=utils_module,
    )
    dataloader = utils_module.DataLoader(
        dataset,
        batch_size=_resolve_effective_batch_size(context.dataset_name, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = model_module.DeepDTAGen(tokenizer).to(device)
    checkpoint_path = Path(context.summary["paths"]["checkpoint"])
    with legacy_torch_load_context():
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    max_target_length = int(dataset[0].target_seq.shape[-1])
    model = ensure_deepdtagen_state_dict_capacity(
        model,
        model_module,
        state_dict=state_dict,
        sequence_length=max_target_length,
        device=device,
    )
    model.load_state_dict(state_dict)
    model = ensure_deepdtagen_sequence_capacity(model, model_module, sequence_length=max_target_length, device=device)

    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        row_offset = 0
        for batch in dataloader:
            batch = batch.to(device)
            model.eval()
            prediction_mean = model(batch)[0].detach().cpu().reshape(-1)
            if args.num_mc_samples <= 1:
                prediction_std = torch.zeros_like(prediction_mean)
            else:
                _enable_dropout_only(model)
                samples = [model(batch)[0].detach().cpu().reshape(-1) for _ in range(args.num_mc_samples)]
                prediction_std = torch.stack(samples, dim=0).std(dim=0, unbiased=False)
            targets = batch.y.detach().cpu().reshape(-1)

            for local_index in range(len(prediction_mean)):
                split_row = subset.iloc[row_offset + local_index]
                rows.append(
                    {
                        "row_id": str(split_row["row_id"]),
                        "target": float(targets[local_index].item()),
                        "prediction_mean": float(prediction_mean[local_index].item()),
                        "prediction_std": float(prediction_std[local_index].item()),
                        "prediction_std_mc_dropout": float(prediction_std[local_index].item()),
                    }
                )
            row_offset += len(prediction_mean)

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(json.dumps({"run_name": args.run_name, "split": args.split, "output_path": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

