#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selective_dta_b.data.loading import SelectiveDTADataModule, load_split_frame
from selective_dta_b.eval.inference import collect_model_predictions, resolve_checkpoint_path, resolve_device
from selective_dta_b.eval.novelty import attach_target_novelty
from selective_dta_b.eval.selective import (
    build_risk_coverage_curve,
    prepare_regression_frame,
    summarize_predictive_intervals,
    summarize_selective_regression,
)
from selective_dta_b.external.predictions import EXTERNAL_MODEL_TYPES, load_external_prediction_frame, resolve_run_context
from selective_dta_b.models import MODEL_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export predictions and selective-eval summaries for a trained run")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-mc-samples", type=int, default=16)
    parser.add_argument("--output-dir", default=None)
    return parser

def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    context = resolve_run_context(workspace, args.run_name)
    run_dir = context.run_dir
    summary = context.summary
    dataset_name = str(summary["dataset_name"])
    split_name = str(summary["split_name"])
    seed = int(summary["seed"])
    split_seed = int(summary.get("split_seed", seed))
    model_type = str(summary.get("model_type", "baseline"))
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "selective_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frame = load_split_frame(
        workspace=workspace,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=split_seed,
    )
    enriched_frame = attach_target_novelty(split_frame)
    test_frame = enriched_frame.loc[enriched_frame["split"] == "test"].reset_index(drop=True)

    checkpoint_path = None
    if model_type in EXTERNAL_MODEL_TYPES:
        _, predictions = load_external_prediction_frame(
            workspace=workspace,
            run_name=args.run_name,
            split_value="test",
            accelerator=args.accelerator,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_mc_samples=args.num_mc_samples,
        )
        prediction_frame = test_frame.merge(predictions, on="row_id", how="inner")
    else:
        device = resolve_device(args.accelerator)
        checkpoint_path = resolve_checkpoint_path(run_dir)
        model_class = MODEL_REGISTRY[model_type]
        model = model_class.load_from_checkpoint(str(checkpoint_path), map_location=device)
        datamodule = SelectiveDTADataModule(
            workspace=workspace,
            dataset_name=dataset_name,
            split_name=split_name,
            seed=split_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            include_drug_graph=(model_type == "graphdta"),
        )
        datamodule.setup("test")
        predictions = collect_model_predictions(
            model,
            dataloader=datamodule.test_dataloader(),
            device=device,
            num_mc_samples=args.num_mc_samples,
        )
        prediction_frame = test_frame.merge(predictions, on="row_id", how="inner")

    prediction_frame["prediction"] = prediction_frame["prediction_mean"]
    prediction_frame["confidence_mc_dropout"] = 1.0 / (1.0 + prediction_frame["prediction_std"])
    if "prediction_std_aleatoric" in prediction_frame.columns:
        prediction_frame["confidence_aleatoric"] = 1.0 / (1.0 + prediction_frame["prediction_std_aleatoric"])
    prediction_frame = prepare_regression_frame(
        prediction_frame,
        prediction_col="prediction_mean",
        target_col="target",
    )
    prediction_frame["confidence_oracle"] = 1.0 / (1.0 + prediction_frame["abs_error"])

    metric_payload: dict[str, dict[str, float | int]] = {}
    curve_rows: list[pd.DataFrame] = []
    confidence_specs = [
        ("mc_dropout", "confidence_mc_dropout"),
        ("target_familiarity", "target_familiarity"),
        ("oracle", "confidence_oracle"),
    ]
    if "confidence_aleatoric" in prediction_frame.columns:
        confidence_specs.insert(1, ("aleatoric", "confidence_aleatoric"))

    for label, confidence_column in confidence_specs:
        metric_payload[label] = summarize_selective_regression(
            prediction_frame,
            confidence_col=confidence_column,
        )
        if label == "mc_dropout":
            metric_payload[label].update(
                summarize_predictive_intervals(
                    prediction_frame,
                    prediction_col="prediction_mean",
                    target_col="target",
                    std_col="prediction_std",
                )
            )
        elif label == "aleatoric":
            metric_payload[label].update(
                summarize_predictive_intervals(
                    prediction_frame,
                    prediction_col="prediction_mean",
                    target_col="target",
                    std_col="prediction_std_aleatoric",
                )
            )
        curve = build_risk_coverage_curve(
            prediction_frame,
            confidence_col=confidence_column,
        )
        curve["confidence_source"] = label
        curve_rows.append(curve)

    predictions_path = output_dir / f"{args.run_name}_test_predictions.csv"
    curves_path = output_dir / f"{args.run_name}_risk_coverage.csv"
    metrics_path = output_dir / f"{args.run_name}_selective_metrics.json"
    prediction_frame.to_csv(predictions_path, index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(curves_path, index=False)
    metrics_path.write_text(json.dumps(metric_payload, indent=2))

    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "model_type": model_type,
                "checkpoint_path": str(checkpoint_path),
                "predictions_path": str(predictions_path),
                "curves_path": str(curves_path),
                "metrics_path": str(metrics_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



