#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from selective_dta_b.external.pmmr import (
    build_pmmr_training_args,
    configure_pmmr_runtime,
    parse_pmmr_run_name,
    pmmr_num_workers,
    resolve_pmmr_report_dir_from_run_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export cached PMMR validation/test predictions for external posthoc selector")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-mc-samples", type=int, default=1)
    parser.add_argument("--output-path", required=True)
    return parser


def _import_pmmr_modules(external_root: Path):
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))
    from data import CPIDataset  # type: ignore
    from models.core import PMMRNet  # type: ignore

    return CPIDataset, PMMRNet


def _make_loader(
    dataset_cls,
    *,
    csv_path: Path,
    compound_dir: Path,
    protein_dir: Path,
    batch_size: int,
    num_workers: int,
):
    dataset = dataset_cls(str(csv_path), str(compound_dir), str(protein_dir))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )


def _predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch).detach().cpu().numpy().reshape(-1)
            y_true = batch["LABEL"].detach().cpu().numpy().reshape(-1)
            predictions.append(outputs)
            labels.append(y_true)
    if not predictions:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(labels), np.concatenate(predictions)


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "PMMR"
    parsed = parse_pmmr_run_name(args.run_name)
    dataset_name = str(parsed["dataset_name"])
    split_name = str(parsed["split_name"])
    seed = int(parsed["seed"])
    report_dir = resolve_pmmr_report_dir_from_run_name(workspace, args.run_name)
    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"PMMR summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text())
    checkpoint_path = Path(str(summary["checkpoint_path"]))
    asset_root = Path(str(summary["asset_root"]))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PMMR checkpoint not found: {checkpoint_path}")
    if not asset_root.exists():
        raise FileNotFoundError(f"PMMR asset root not found: {asset_root}")

    if not torch.cuda.is_available():
        raise RuntimeError("PMMR prediction export requires CUDA")
    configure_pmmr_runtime()
    CPIDataset, PMMRNet = _import_pmmr_modules(external_root)

    split_key = "valid" if args.split == "val" else "test"
    protein_key = "valid" if args.split == "val" else "test"
    csv_path = asset_root / f"{split_key}.csv"
    protein_dir = asset_root / "protein" / protein_key
    if not csv_path.exists():
        raise FileNotFoundError(f"PMMR split csv not found: {csv_path}")
    if not protein_dir.exists():
        raise FileNotFoundError(f"PMMR protein dir not found: {protein_dir}")

    eval_workers = pmmr_num_workers(args.num_workers, stage="test")
    loader = _make_loader(
        CPIDataset,
        csv_path=csv_path,
        compound_dir=asset_root / "compound",
        protein_dir=protein_dir,
        batch_size=args.batch_size,
        num_workers=eval_workers,
    )
    model_args = build_pmmr_training_args(
        root_data_path=asset_root,
        dataset=f"{dataset_name}_{split_name}_seed{seed}",
        seed=seed,
        overrides={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_epochs": 1,
        },
    )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    model = PMMRNet(model_args).cuda()
    model.load_state_dict(torch.load(checkpoint_path))
    y_true, y_pred = _predict(model, loader)

    rows = pd.read_csv(csv_path).copy()
    rows["prediction_mean"] = y_pred
    rows["prediction_std"] = 0.0
    rows["prediction_std_mc_dropout"] = 0.0
    rows["prediction"] = y_pred
    rows["target"] = y_true
    rows["abs_error"] = np.abs(rows["prediction_mean"] - rows["target"])

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "split": args.split,
                "num_rows": int(len(rows)),
                "output_path": str(output_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

