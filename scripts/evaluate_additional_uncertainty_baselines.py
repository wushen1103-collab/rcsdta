#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from selective_dta_b.eval.conformal import (
    attach_conformal_confidence,
    fit_scaled_split_conformal,
    summarize_scaled_conformal_intervals,
)
from selective_dta_b.eval.ensemble import build_seed_ensemble_frame
from selective_dta_b.eval.selective import summarize_selective_regression


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate conformal and deep-ensemble uncertainty baselines")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--models", default="baseline,deepdta,graphdta,moltrans")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _iter_finished_runs(workspace: Path, models: set[str]) -> list[tuple[Path, dict[str, object]]]:
    rows: list[tuple[Path, dict[str, object]]] = []
    for summary_path in sorted((workspace / "artifacts" / "runs").glob("*/run_summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("model_type") not in models:
            continue
        if summary.get("status") != "finished":
            continue
        rows.append((summary_path.parent, summary))
    return rows


def _posthoc_paths(run_dir: Path, run_name: str) -> tuple[Path, Path]:
    posthoc_dir = run_dir / "posthoc_selector"
    return (
        posthoc_dir / f"{run_name}_validation_predictions.csv",
        posthoc_dir / f"{run_name}_test_predictions.csv",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def evaluate_per_run_conformal(
    workspace: Path,
    *,
    models: set[str],
    overwrite: bool,
) -> dict[str, int]:
    completed = 0
    skipped = 0
    for run_dir, summary in _iter_finished_runs(workspace, models):
        run_name = str(summary["run_name"])
        validation_path, test_path = _posthoc_paths(run_dir, run_name)
        output_path = run_dir / "selective_eval" / f"{run_name}_conformal_metrics.json"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        if not validation_path.exists() or not test_path.exists():
            skipped += 1
            continue

        validation_frame = pd.read_csv(validation_path)
        test_frame = pd.read_csv(test_path)
        if "prediction_std_mc_dropout" not in validation_frame.columns or "prediction_std_mc_dropout" not in test_frame.columns:
            skipped += 1
            continue

        calibration = fit_scaled_split_conformal(
            validation_frame,
            scale_col="prediction_std_mc_dropout",
            interval_levels=(0.8, 0.9),
        )
        enriched_test = attach_conformal_confidence(
            test_frame,
            calibration=calibration,
            scale_col="prediction_std_mc_dropout",
            interval_level=0.9,
            prefix="conformal_mc_dropout",
        )
        metrics = summarize_selective_regression(
            enriched_test,
            confidence_col="confidence_conformal_mc_dropout_90",
        )
        metrics.update(
            summarize_scaled_conformal_intervals(
                enriched_test,
                calibration=calibration,
                scale_col="prediction_std_mc_dropout",
                prediction_col="prediction_mean",
                target_col="target",
            )
        )
        metrics["num_calibration_examples"] = len(validation_frame)
        _write_json(output_path, {"conformal_mc_dropout": metrics})
        completed += 1
    return {"completed": completed, "skipped": skipped}


def _build_deepensemble_summary(
    *,
    run_name: str,
    dataset_name: str,
    split_name: str,
    backbone_model: str,
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "dataset_name": dataset_name,
        "split_name": split_name,
        "seed": "ensemble",
        "model_type": f"deepensemble_{backbone_model}",
        "backbone_model": backbone_model,
        "status": "finished",
    }


def evaluate_seed_ensembles(
    workspace: Path,
    *,
    models: set[str],
    overwrite: bool,
) -> dict[str, int]:
    grouped_runs: dict[tuple[str, str, str], list[tuple[Path, dict[str, object]]]] = defaultdict(list)
    for run_dir, summary in _iter_finished_runs(workspace, models):
        key = (str(summary["model_type"]), str(summary["dataset_name"]), str(summary["split_name"]))
        grouped_runs[key].append((run_dir, summary))

    completed = 0
    skipped = 0
    for (model_type, dataset_name, split_name), group in sorted(grouped_runs.items()):
        if len(group) < 2:
            skipped += 1
            continue
        run_name = f"deepensemble_{model_type}_{dataset_name}_{split_name}"
        ensemble_run_dir = workspace / "artifacts" / "runs" / run_name
        output_path = ensemble_run_dir / "selective_eval" / f"{run_name}_selective_metrics.json"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        validation_frames: list[pd.DataFrame] = []
        test_frames: list[pd.DataFrame] = []
        missing_files = False
        for run_dir, summary in sorted(group, key=lambda item: int(item[1]["seed"])):
            validation_path, test_path = _posthoc_paths(run_dir, str(summary["run_name"]))
            if not validation_path.exists() or not test_path.exists():
                missing_files = True
                break
            validation_frames.append(pd.read_csv(validation_path))
            test_frames.append(pd.read_csv(test_path))
        if missing_files:
            skipped += 1
            continue

        try:
            validation_ensemble = build_seed_ensemble_frame(validation_frames)
            test_ensemble = build_seed_ensemble_frame(test_frames)
        except ValueError:
            skipped += 1
            continue
        test_ensemble["confidence_deep_ensemble"] = 1.0 / (1.0 + test_ensemble["prediction_std_ensemble"])
        deep_ensemble_metrics = summarize_selective_regression(
            test_ensemble,
            confidence_col="confidence_deep_ensemble",
        )

        calibration = fit_scaled_split_conformal(
            validation_ensemble,
            scale_col="prediction_std_ensemble",
            interval_levels=(0.8, 0.9),
        )
        conformal_ensemble = attach_conformal_confidence(
            test_ensemble,
            calibration=calibration,
            scale_col="prediction_std_ensemble",
            interval_level=0.9,
            prefix="conformal_ensemble",
        )
        conformal_metrics = summarize_selective_regression(
            conformal_ensemble,
            confidence_col="confidence_conformal_ensemble_90",
        )
        conformal_metrics.update(
            summarize_scaled_conformal_intervals(
                conformal_ensemble,
                calibration=calibration,
                scale_col="prediction_std_ensemble",
                prediction_col="prediction_mean",
                target_col="target",
            )
        )
        conformal_metrics["num_calibration_examples"] = len(validation_ensemble)

        ensemble_run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(ensemble_run_dir / "run_summary.json", _build_deepensemble_summary(
            run_name=run_name,
            dataset_name=dataset_name,
            split_name=split_name,
            backbone_model=model_type,
        ))
        selective_dir = ensemble_run_dir / "selective_eval"
        selective_dir.mkdir(parents=True, exist_ok=True)
        test_ensemble.to_csv(selective_dir / f"{run_name}_test_predictions.csv", index=False)
        _write_json(
            output_path,
            {
                "deep_ensemble": deep_ensemble_metrics,
                "conformal_ensemble": conformal_metrics,
            },
        )
        completed += 1
    return {"completed": completed, "skipped": skipped}


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    models = {part.strip() for part in args.models.split(",") if part.strip()}

    per_run = evaluate_per_run_conformal(
        workspace,
        models=models,
        overwrite=args.overwrite,
    )
    ensemble = evaluate_seed_ensembles(
        workspace,
        models=models,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "workspace": str(workspace),
                "models": sorted(models),
                "per_run_conformal": per_run,
                "seed_ensemble": ensemble,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

