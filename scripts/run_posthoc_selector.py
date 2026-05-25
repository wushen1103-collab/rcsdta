#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selective_dta_b.data.loading import SelectiveDTADataModule, load_split_frame
from selective_dta_b.eval.inference import collect_model_predictions, resolve_checkpoint_path, resolve_device
from selective_dta_b.eval.novelty import attach_target_novelty
from selective_dta_b.eval.posthoc import fit_posthoc_error_regressor, predict_posthoc_error
from selective_dta_b.eval.selective import build_risk_coverage_curve, prepare_regression_frame, summarize_selective_regression
from selective_dta_b.external.predictions import EXTERNAL_MODEL_TYPES, load_external_prediction_frame, resolve_run_context
from selective_dta_b.models import MODEL_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a post-hoc error selector on validation predictions")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--regressor-type", choices=["gbr", "knn", "ridge"], default="gbr")
    parser.add_argument("--feature-set", choices=["base4", "enriched9"], default="base4")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-mc-samples", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser


def _build_prediction_frame(
    *,
    split_frame: pd.DataFrame,
    dataloader,
    model,
    device,
    num_mc_samples: int,
) -> pd.DataFrame:
    predictions = collect_model_predictions(
        model,
        dataloader=dataloader,
        device=device,
        num_mc_samples=num_mc_samples,
    )
    frame = split_frame.merge(predictions, on="row_id", how="inner")
    frame["prediction"] = frame["prediction_mean"]
    frame["confidence_mc_dropout"] = 1.0 / (1.0 + frame["prediction_std_mc_dropout"])
    return prepare_regression_frame(
        frame,
        prediction_col="prediction_mean",
        target_col="target",
    )


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
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "posthoc_selector"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frame = load_split_frame(
        workspace=workspace,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=split_seed,
    )
    enriched_frame = attach_target_novelty(split_frame)
    validation_split = enriched_frame.loc[enriched_frame["split"] == "val"].reset_index(drop=True)
    test_split = enriched_frame.loc[enriched_frame["split"] == "test"].reset_index(drop=True)

    checkpoint_path = None
    if model_type in EXTERNAL_MODEL_TYPES:
        _, validation_predictions = load_external_prediction_frame(
            workspace=workspace,
            run_name=args.run_name,
            split_value="val",
            accelerator=args.accelerator,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_mc_samples=args.num_mc_samples,
        )
        _, test_predictions = load_external_prediction_frame(
            workspace=workspace,
            run_name=args.run_name,
            split_value="test",
            accelerator=args.accelerator,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_mc_samples=args.num_mc_samples,
        )
        validation_frame = prepare_regression_frame(
            validation_split.merge(validation_predictions, on="row_id", how="inner"),
            prediction_col="prediction_mean",
            target_col="target",
        )
        test_frame = prepare_regression_frame(
            test_split.merge(test_predictions, on="row_id", how="inner"),
            prediction_col="prediction_mean",
            target_col="target",
        )
        validation_frame["prediction"] = validation_frame["prediction_mean"]
        test_frame["prediction"] = test_frame["prediction_mean"]
        validation_frame["confidence_mc_dropout"] = 1.0 / (1.0 + validation_frame["prediction_std_mc_dropout"])
        test_frame["confidence_mc_dropout"] = 1.0 / (1.0 + test_frame["prediction_std_mc_dropout"])
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
        datamodule.setup("fit")
        validation_frame = _build_prediction_frame(
            split_frame=validation_split,
            dataloader=datamodule.val_dataloader(),
            model=model,
            device=device,
            num_mc_samples=args.num_mc_samples,
        )
        test_frame = _build_prediction_frame(
            split_frame=test_split,
            dataloader=datamodule.test_dataloader(),
            model=model,
            device=device,
            num_mc_samples=args.num_mc_samples,
        )

    regressor = fit_posthoc_error_regressor(
        validation_frame,
        random_state=args.random_state,
        regressor_type=args.regressor_type,
        feature_set=args.feature_set,
    )
    validation_frame["predicted_abs_error_posthoc"] = predict_posthoc_error(regressor, validation_frame)
    test_frame["predicted_abs_error_posthoc"] = predict_posthoc_error(regressor, test_frame)
    validation_frame["confidence_posthoc"] = 1.0 / (1.0 + validation_frame["predicted_abs_error_posthoc"])
    test_frame["confidence_posthoc"] = 1.0 / (1.0 + test_frame["predicted_abs_error_posthoc"])
    test_frame["confidence_oracle"] = 1.0 / (1.0 + test_frame["abs_error"])

    metric_payload: dict[str, dict[str, float | int]] = {}
    curve_rows: list[pd.DataFrame] = []
    confidence_specs = [
        ("posthoc_selector", "confidence_posthoc"),
        ("mc_dropout", "confidence_mc_dropout"),
        ("target_familiarity", "target_familiarity"),
        ("oracle", "confidence_oracle"),
    ]
    for label, confidence_column in confidence_specs:
        metric_payload[label] = summarize_selective_regression(
            test_frame,
            confidence_col=confidence_column,
        )
        curve = build_risk_coverage_curve(
            test_frame,
            confidence_col=confidence_column,
        )
        curve["confidence_source"] = label
        curve_rows.append(curve)

    validation_predictions_path = output_dir / f"{args.run_name}_validation_predictions.csv"
    test_predictions_path = output_dir / f"{args.run_name}_test_predictions.csv"
    metrics_path = output_dir / f"{args.run_name}_posthoc_metrics.json"
    curves_path = output_dir / f"{args.run_name}_posthoc_risk_coverage.csv"

    validation_frame.to_csv(validation_predictions_path, index=False)
    test_frame.to_csv(test_predictions_path, index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(curves_path, index=False)
    metrics_path.write_text(json.dumps(metric_payload, indent=2))

    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "model_type": model_type,
                "regressor_type": args.regressor_type,
                "feature_set": args.feature_set,
                "checkpoint_path": str(checkpoint_path),
                "validation_predictions_path": str(validation_predictions_path),
                "test_predictions_path": str(test_predictions_path),
                "metrics_path": str(metrics_path),
                "curves_path": str(curves_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



