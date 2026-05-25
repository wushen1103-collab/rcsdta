#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from selective_dta_b.data.loading import SelectiveDTADataModule
from selective_dta_b.models import MODEL_REGISTRY, build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a char-level DTA regressor")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--default-root-dir", default=None)
    parser.add_argument("--model-type", choices=sorted(MODEL_REGISTRY), default="baseline")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--fast-dev-run", type=int, default=0)
    parser.add_argument("--drug-max-length", type=int, default=128)
    parser.add_argument("--protein-max-length", type=int, default=512)
    parser.add_argument("--char-embed-dim", type=int, default=64)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--limit-test-batches", type=float, default=1.0)
    return parser


def _resolve_runtime(accelerator: str, precision: str) -> tuple[str, int, str]:
    if accelerator == "auto":
        resolved_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        resolved_accelerator = accelerator

    if precision == "auto":
        resolved_precision = "bf16-mixed" if resolved_accelerator == "gpu" else "32-true"
    else:
        resolved_precision = precision

    return resolved_accelerator, 1, resolved_precision


def _scalar_metrics(callback_metrics: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in callback_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics[key] = float(value.detach().cpu().item())
        elif isinstance(value, (float, int)):
            metrics[key] = float(value)
    return metrics


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    prefix_lookup = {
        "baseline": "char_baseline",
        "heteroscedastic": "char_heteroscedastic",
        "deepdta": "deepdta",
        "graphdta": "graphdta",
        "moltrans": "moltrans",
    }
    default_prefix = prefix_lookup.get(args.model_type, args.model_type)
    resolved_split_seed = args.split_seed if args.split_seed is not None else args.seed
    if args.run_name:
        run_name = args.run_name
    elif resolved_split_seed == args.seed:
        run_name = f"{default_prefix}_{args.dataset_name}_{args.split_name}_seed{args.seed}"
    else:
        run_name = (
            f"{default_prefix}_{args.dataset_name}_{args.split_name}_"
            f"split{resolved_split_seed}_seed{args.seed}"
        )
    run_dir = Path(args.default_root_dir).resolve() if args.default_root_dir else workspace / "artifacts" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    resolved_accelerator, resolved_devices, resolved_precision = _resolve_runtime(args.accelerator, args.precision)
    if resolved_accelerator == "gpu":
        torch.backends.cudnn.benchmark = True

    datamodule = SelectiveDTADataModule(
        workspace=workspace,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        seed=resolved_split_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(resolved_accelerator == "gpu"),
        include_drug_graph=(args.model_type == "graphdta"),
    )
    model = build_model(
        args.model_type,
        drug_max_length=args.drug_max_length,
        protein_max_length=args.protein_max_length,
        char_embed_dim=args.char_embed_dim,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logger = CSVLogger(save_dir=str(run_dir / "logs"), name="csv")
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = L.Trainer(
        accelerator=resolved_accelerator,
        devices=resolved_devices,
        precision=resolved_precision,
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        default_root_dir=str(run_dir),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path="best" if not args.fast_dev_run else None,
        verbose=False,
    )

    summary = {
        "run_name": run_name,
        "dataset_name": args.dataset_name,
        "split_name": args.split_name,
        "seed": args.seed,
        "split_seed": resolved_split_seed,
        "model_type": args.model_type,
        "accelerator": resolved_accelerator,
        "precision": resolved_precision,
        "status": "finished" if trainer.state.finished else str(trainer.state.status),
        "run_dir": str(run_dir),
        "metrics": _scalar_metrics(trainer.callback_metrics),
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

