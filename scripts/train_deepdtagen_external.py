#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from selective_dta_b.external import (
    ensure_deepdtagen_sequence_capacity,
    load_deepdtagen_modules,
    prepare_deepdtagen_data,
    stage_deepdtagen_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DeepDTAGen baseline on a split-specific dataset export.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--default-root-dir", default=None)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-test-batches", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _resolve_device(accelerator: str) -> tuple[str, torch.device]:
    if accelerator == "auto":
        resolved = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        resolved = accelerator
    if resolved == "gpu" and torch.cuda.is_available():
        return resolved, torch.device("cuda")
    return "cpu", torch.device("cpu")


def _resolve_effective_batch_size(dataset_name: str, requested_batch_size: int, smoke: bool) -> int:
    dataset_caps = {
        "bindingdb": 4,
        "kiba": 8,
    }
    cap = dataset_caps.get(dataset_name.lower(), requested_batch_size)
    if smoke:
        cap = min(cap, 4)
    return max(1, min(requested_batch_size, cap))


def _max_batches(total_batches: int, limit: float) -> int:
    if total_batches <= 0:
        return 0
    if limit <= 0:
        return 1
    if limit <= 1.0:
        return max(1, math.ceil(total_batches * limit))
    return min(total_batches, int(limit))


def _clean_metric(value: float) -> float | None:
    numeric = float(value)
    if math.isfinite(numeric):
        return numeric
    return None


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return _clean_metric(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    ranks_x = np.argsort(np.argsort(x))
    ranks_y = np.argsort(np.argsort(y))
    return _safe_corrcoef(ranks_x.astype(float), ranks_y.astype(float))


def evaluate_model(model, loader, device, utils_module, max_batches: int) -> tuple[dict[str, float | None], np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    mse_losses: list[float] = []
    mae_losses: list[float] = []
    lm_losses: list[float] = []
    kl_losses: list[float] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = batch.to(device)
            prediction, _, lm_loss, kl_loss = model(batch)
            truth = batch.y.view(-1, 1).float()
            prediction_cpu = prediction.detach().cpu().numpy().reshape(-1)
            truth_cpu = truth.detach().cpu().numpy().reshape(-1)
            predictions.append(prediction_cpu)
            truths.append(truth_cpu)
            mse_losses.append(float(np.mean((prediction_cpu - truth_cpu) ** 2)))
            mae_losses.append(float(np.mean(np.abs(prediction_cpu - truth_cpu))))
            lm_losses.append(float(lm_loss.detach().cpu().item()))
            kl_losses.append(float(kl_loss.detach().cpu().item()))

    if not predictions:
        raise RuntimeError("DeepDTAGen evaluation received zero batches.")

    pred_array = np.concatenate(predictions)
    truth_array = np.concatenate(truths)
    metrics = {
        "test_rmse": _clean_metric(np.sqrt(np.mean((pred_array - truth_array) ** 2))),
        "test_mse": _clean_metric(np.mean((pred_array - truth_array) ** 2)),
        "test_mae": _clean_metric(np.mean(np.abs(pred_array - truth_array))),
        "test_ci": _clean_metric(utils_module.get_cindex(truth_array, pred_array)),
        "test_pearson": _safe_corrcoef(truth_array, pred_array),
        "test_spearman": _safe_spearman(truth_array, pred_array),
        "test_rm2": _clean_metric(utils_module.get_rm2(truth_array, pred_array)),
        "test_lm_loss": _clean_metric(np.mean(lm_losses)),
        "test_kl_loss": _clean_metric(np.mean(kl_losses)),
    }
    return metrics, truth_array, pred_array


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "DeepDTAGen"
    run_name = args.run_name or f"deepdtagen_{args.dataset_name}_{args.split_name}_seed{args.seed}"
    run_dir = Path(args.default_root_dir).resolve() if args.default_root_dir else workspace / "artifacts" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_accelerator, device = _resolve_device(args.accelerator)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    modules = load_deepdtagen_modules(external_root)
    staged = stage_deepdtagen_split(
        external_root=external_root,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        seed=args.seed,
        run_dir=run_dir,
    )
    tokenizer, train_data, test_data = prepare_deepdtagen_data(staged, args.dataset_name, modules)

    utils_module = modules["utils"]
    model_module = modules["model"]
    fettergrad_module = modules["fettergrad"]
    effective_batch_size = _resolve_effective_batch_size(args.dataset_name, args.batch_size, args.smoke)
    if effective_batch_size != args.batch_size:
        print(
            f"Adjusting DeepDTAGen batch size from {args.batch_size} to {effective_batch_size} "
            f"for dataset={args.dataset_name}",
            flush=True,
        )

    train_loader = utils_module.DataLoader(train_data, batch_size=effective_batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = utils_module.DataLoader(test_data, batch_size=effective_batch_size, shuffle=False, num_workers=args.num_workers)
    max_train_batches = _max_batches(len(train_loader), 0.1 if args.smoke else args.limit_train_batches)
    max_test_batches = _max_batches(len(test_loader), 0.1 if args.smoke else args.limit_test_batches)

    model = model_module.DeepDTAGen(tokenizer).to(device)
    max_target_length = int(
        max(
            train_data[0].target_seq.shape[-1],
            test_data[0].target_seq.shape[-1],
        )
    )
    model = ensure_deepdtagen_sequence_capacity(
        model,
        model_module,
        sequence_length=max_target_length,
        device=device,
    )
    optimizer = fettergrad_module.FetterGrad(torch.optim.Adam(model.parameters(), lr=args.learning_rate))
    mse_loss_fn = nn.MSELoss()

    history_rows: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(args.max_epochs):
        mse_epoch: list[float] = []
        total_epoch: list[float] = []
        lm_epoch: list[float] = []
        kl_epoch: list[float] = []
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= max_train_batches:
                break
            optimizer.zero_grad()
            batch = batch.to(device)
            prediction, _, lm_loss, kl_loss = model(batch)
            truth = batch.y.view(-1, 1).float()
            mse_loss = mse_loss_fn(prediction, truth)
            total_loss = mse_loss + lm_loss + (0.001 * kl_loss)
            optimizer.ft_backward([total_loss, mse_loss])
            optimizer.step()
            mse_epoch.append(float(mse_loss.detach().cpu().item()))
            total_epoch.append(float(total_loss.detach().cpu().item()))
            lm_epoch.append(float(lm_loss.detach().cpu().item()))
            kl_epoch.append(float(kl_loss.detach().cpu().item()))

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_total_loss": float(np.mean(total_epoch)) if total_epoch else float("nan"),
                "train_mse_loss": float(np.mean(mse_epoch)) if mse_epoch else float("nan"),
                "train_lm_loss": float(np.mean(lm_epoch)) if lm_epoch else float("nan"),
                "train_kl_loss": float(np.mean(kl_epoch)) if kl_epoch else float("nan"),
            }
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset_name": args.dataset_name,
            "split_name": args.split_name,
            "seed": args.seed,
            "max_epochs": args.max_epochs,
        },
        staged.checkpoint_path,
    )

    metrics, truth_array, pred_array = evaluate_model(model, test_loader, device, utils_module, max_test_batches)

    with staged.history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_total_loss", "train_mse_loss", "train_lm_loss", "train_kl_loss"])
        writer.writeheader()
        writer.writerows(history_rows)

    with staged.predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["y_true", "y_pred"])
        writer.writeheader()
        for truth, pred in zip(truth_array.tolist(), pred_array.tolist()):
            writer.writerow({"y_true": truth, "y_pred": pred})

    summary = {
        "run_name": run_name,
        "dataset_name": args.dataset_name,
        "split_name": args.split_name,
        "seed": args.seed,
        "model_type": "deepdtagen",
        "accelerator": resolved_accelerator,
        "max_epochs": args.max_epochs,
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "smoke": args.smoke,
        "status": "finished",
        "run_dir": str(run_dir),
        "metrics": metrics,
        "rows": {
            "train": staged.train_rows,
            "test": staged.test_rows,
        },
        "paths": {
            "checkpoint": str(staged.checkpoint_path),
            "history": str(staged.history_path),
            "predictions": str(staged.predictions_path),
        },
    }
    staged.summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

