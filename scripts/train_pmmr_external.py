#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from selective_dta_b.external.pmmr import (
    build_pmmr_training_args,
    configure_pmmr_runtime,
    materialize_pmmr_assets,
    pmmr_num_workers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PMMR on selective DTA exported splits")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--compound-model-name", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--protein-model-name", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--output-dir", default=None)
    return parser


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _import_pmmr_modules(external_root: Path):
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))
    from data import CPIDataset  # type: ignore
    from models.core import PMMRNet  # type: ignore
    from utils import ci, mae, mse, pearson, rmse, spearman  # type: ignore

    return CPIDataset, PMMRNet, ci, mae, mse, pearson, rmse, spearman


def _make_loader(
    dataset_cls,
    *,
    csv_path: Path,
    compound_dir: Path,
    protein_dir: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
):
    dataset = dataset_cls(str(csv_path), str(compound_dir), str(protein_dir))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )


def _train_one_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        labels = batch["LABEL"].view(-1, 1).float().cuda()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


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


def _safe_metric(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    try:
        value = fn(y_true, y_pred)
    except Exception:
        return None
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _regression_metrics(ci_fn, mae_fn, mse_fn, pearson_fn, rmse_fn, spearman_fn, y_true, y_pred) -> dict[str, float | None]:
    return {
        "rmse": _safe_metric(rmse_fn, y_true, y_pred),
        "mae": _safe_metric(mae_fn, y_true, y_pred),
        "mse": _safe_metric(mse_fn, y_true, y_pred),
        "pearson": _safe_metric(pearson_fn, y_true, y_pred),
        "spearman": _safe_metric(spearman_fn, y_true, y_pred),
        "ci": _safe_metric(ci_fn, y_true, y_pred),
    }


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "PMMR"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else workspace / "reports" / "deployment_upgrade_experiments" / "pmmr_training" / args.dataset_name / f"{args.split_name}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device != "cuda":
        raise ValueError("train_pmmr_external.py currently supports CUDA only because upstream PMMR hardcodes .cuda()")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PMMR training but no GPU is visible")
    sharing_strategy = configure_pmmr_runtime()

    if args.prepare_assets:
        asset_info = materialize_pmmr_assets(
            workspace=workspace,
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            seed=args.seed,
            external_root=external_root,
            device="cuda",
            compound_model_name=args.compound_model_name,
            protein_model_name=args.protein_model_name,
        )
    else:
        asset_root = external_root / "selective_data" / args.dataset_name / f"{args.split_name}_seed{args.seed}"
        asset_info = {
            "output_root": asset_root,
            "train_path": str(asset_root / "train.csv"),
            "valid_path": str(asset_root / "valid.csv"),
            "test_path": str(asset_root / "test.csv"),
            "compound_dict_path": asset_root / "compound" / "mol_dict.pkl",
            "protein_split_dirs": {
                "train": asset_root / "protein" / "train",
                "valid": asset_root / "protein" / "valid",
                "test": asset_root / "protein" / "test",
            },
        }

    asset_root = Path(asset_info["output_root"])
    required_paths = [
        Path(asset_info["train_path"]),
        Path(asset_info["valid_path"]),
        Path(asset_info["test_path"]),
        Path(asset_info["compound_dict_path"]),
        Path(asset_info["protein_split_dirs"]["train"]),
        Path(asset_info["protein_split_dirs"]["valid"]),
        Path(asset_info["protein_split_dirs"]["test"]),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing PMMR assets: {missing}")

    _set_seeds(args.seed)
    CPIDataset, PMMRNet, ci_fn, mae_fn, mse_fn, pearson_fn, rmse_fn, spearman_fn = _import_pmmr_modules(external_root)

    # Reset visibility after PMMR import before the first CUDA allocation.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    train_workers = pmmr_num_workers(args.num_workers, stage="train")
    eval_workers = pmmr_num_workers(args.num_workers, stage="valid")

    train_loader = _make_loader(
        CPIDataset,
        csv_path=Path(asset_info["train_path"]),
        compound_dir=asset_root / "compound",
        protein_dir=asset_root / "protein" / "train",
        batch_size=args.batch_size,
        num_workers=train_workers,
        shuffle=True,
    )
    valid_loader = _make_loader(
        CPIDataset,
        csv_path=Path(asset_info["valid_path"]),
        compound_dir=asset_root / "compound",
        protein_dir=asset_root / "protein" / "valid",
        batch_size=args.batch_size,
        num_workers=eval_workers,
        shuffle=False,
    )
    test_loader = _make_loader(
        CPIDataset,
        csv_path=Path(asset_info["test_path"]),
        compound_dir=asset_root / "compound",
        protein_dir=asset_root / "protein" / "test",
        batch_size=args.batch_size,
        num_workers=eval_workers,
        shuffle=False,
    )

    model_args = build_pmmr_training_args(
        root_data_path=asset_root,
        dataset=f"{args.dataset_name}_{args.split_name}_seed{args.seed}",
        seed=args.seed,
        overrides={
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
        },
    )
    model = PMMRNet(model_args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, eps=1e-8)
    loss_fn = nn.MSELoss()

    history: list[dict[str, float | int | None]] = []
    best_epoch = -1
    best_mae = float("inf")
    best_checkpoint = output_dir / "best_model.pth"

    for epoch in range(1, args.max_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn)
        valid_true, valid_pred = _predict(model, valid_loader)
        valid_metrics = _regression_metrics(
            ci_fn,
            mae_fn,
            mse_fn,
            pearson_fn,
            rmse_fn,
            spearman_fn,
            valid_true,
            valid_pred,
        )
        valid_mae = valid_metrics["mae"]
        if valid_mae is None:
            raise RuntimeError("validation MAE could not be computed")
        scheduler.step(valid_mae)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        history.append(row)
        if valid_mae < best_mae:
            best_mae = valid_mae
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint)

    if best_epoch < 0:
        raise RuntimeError("PMMR training finished without a valid checkpoint")

    model.load_state_dict(torch.load(best_checkpoint))
    valid_true, valid_pred = _predict(model, valid_loader)
    test_true, test_pred = _predict(model, test_loader)
    test_metrics = _regression_metrics(
        ci_fn,
        mae_fn,
        mse_fn,
        pearson_fn,
        rmse_fn,
        spearman_fn,
        test_true,
        test_pred,
    )

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    valid_rows = pd.read_csv(asset_info["valid_path"]).copy()
    valid_rows["prediction_mean"] = valid_pred
    valid_rows["prediction_std"] = 0.0
    valid_rows["prediction_std_mc_dropout"] = 0.0
    valid_rows["prediction"] = valid_pred
    valid_rows["target"] = valid_true
    valid_rows["abs_error"] = np.abs(valid_rows["prediction"] - valid_rows["target"])
    valid_rows.to_csv(output_dir / "validation_predictions.csv", index=False)

    test_rows = pd.read_csv(asset_info["test_path"]).copy()
    test_rows["prediction_mean"] = test_pred
    test_rows["prediction_std"] = 0.0
    test_rows["prediction_std_mc_dropout"] = 0.0
    test_rows["prediction"] = test_pred
    test_rows["target"] = test_true
    test_rows["abs_error"] = np.abs(test_rows["prediction"] - test_rows["target"])
    test_rows.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "dataset_name": args.dataset_name,
        "split_name": args.split_name,
        "seed": args.seed,
        "asset_root": str(asset_root),
        "best_epoch": best_epoch,
        "best_valid_mae": best_mae,
        "valid_metrics": _regression_metrics(
            ci_fn,
            mae_fn,
            mse_fn,
            pearson_fn,
            rmse_fn,
            spearman_fn,
            valid_true,
            valid_pred,
        ),
        "test_metrics": test_metrics,
        "num_train_rows": int(len(pd.read_csv(asset_info["train_path"]))),
        "num_valid_rows": int(len(valid_rows)),
        "num_test_rows": int(len(test_rows)),
        "train_num_workers": train_workers,
        "eval_num_workers": eval_workers,
        "torch_sharing_strategy": sharing_strategy,
        "checkpoint_path": str(best_checkpoint),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

