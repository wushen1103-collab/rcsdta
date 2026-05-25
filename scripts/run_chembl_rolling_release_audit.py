from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from selective_dta_b.eval.chembl_temporal_backtest import (
    _predict_ensemble,
    _prediction_frame,
    _score_posthoc,
    _write_frame,
    add_target_familiarity,
)
from selective_dta_b.eval.followup_experiments import selective_metrics
from selective_dta_b.eval.maximal_trans_experiments import _predict_chemberta_hybrid


BACKBONES = ("SimBoost", "DeepDTA", "GraphDTA", "KANPM", "ChemBERTaHybrid")
SOURCES = (
    ("posthoc_selector", "confidence_posthoc"),
    ("mc_dropout", "confidence_mc_dropout"),
    ("target_familiarity", "target_familiarity"),
)


def _paired_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, grouped in [("all_windows", summary), *list(summary.groupby("window_name"))]:
        scope_name = scope if isinstance(scope, str) else str(scope)
        for metric in ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse"):
            pivot = grouped.pivot_table(
                index=["window_name", "backbone_name", "seed"],
                columns="confidence_source",
                values=metric,
                aggfunc="first",
            )
            for baseline in ("mc_dropout", "target_familiarity"):
                paired = pivot[["posthoc_selector", baseline]].dropna()
                delta = paired["posthoc_selector"] - paired[baseline]
                rows.append(
                    {
                        "analysis_scope": scope_name,
                        "metric_name": metric,
                        "baseline_confidence_source": baseline,
                        "num_pairs": int(len(delta)),
                        "posthoc_mean": float(paired["posthoc_selector"].mean()),
                        "baseline_mean": float(paired[baseline].mean()),
                        "mean_delta_posthoc_minus_baseline": float(delta.mean()),
                        "posthoc_win_rate": float(np.mean(delta < 0)),
                    }
                )
    return pd.DataFrame(rows)


def _release_number(value: object) -> int:
    return int(str(value).rsplit("_", 1)[-1])


def _rolling_windows(frame: pd.DataFrame) -> list[dict[str, object]]:
    releases = sorted(frame["release_number"].dropna().astype(int).unique())
    windows: list[dict[str, object]] = []
    for idx in range(2, len(releases)):
        windows.append(
            {
                "window_name": f"release_window_{idx - 1}",
                "train_releases": releases[: idx - 1],
                "val_release": releases[idx - 1],
                "test_release": releases[idx],
            }
        )
    return windows


def run_audit(workspace: Path, output_dir: Path, seeds: tuple[int, ...], ensemble_size: int) -> dict[str, object]:
    pairs = pd.read_csv(workspace / "data" / "processed" / "chembl" / "standardized_pairs.csv")
    pairs["release_number"] = pairs["chembl_release"].map(_release_number)
    windows = _rolling_windows(pairs)
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    design_rows: list[dict[str, object]] = []
    for window in windows:
        split = pairs.copy()
        split["split"] = "excluded"
        split.loc[split["release_number"].isin(window["train_releases"]), "split"] = "train"
        split.loc[split["release_number"] == window["val_release"], "split"] = "val"
        split.loc[split["release_number"] == window["test_release"], "split"] = "test"
        split = split.loc[split["split"] != "excluded"].copy()
        split = add_target_familiarity(split)
        train = split.loc[split["split"] == "train"].copy()
        val = split.loc[split["split"] == "val"].copy()
        test = split.loc[split["split"] == "test"].copy()
        design_rows.append(
            {
                **window,
                "train_releases": ",".join(str(x) for x in window["train_releases"]),
                "num_train": len(train),
                "num_val": len(val),
                "num_test": len(test),
                "num_test_targets": test["target_id"].nunique(),
            }
        )
        for seed in seeds:
            for backbone_name in BACKBONES:
                if backbone_name == "ChemBERTaHybrid":
                    val_mean, val_std, test_mean, test_std = _predict_chemberta_hybrid(
                        train, val, test, workspace=workspace, ensemble_size=ensemble_size, random_state=seed
                    )
                else:
                    val_mean, val_std, test_mean, test_std = _predict_ensemble(
                        backbone_name, train, val, test, ensemble_size=ensemble_size, random_state=seed
                    )
                val_pred = _prediction_frame(val, val_mean, val_std, backbone_name=backbone_name)
                test_pred = _prediction_frame(test, test_mean, test_std, backbone_name=backbone_name)
                scored = _score_posthoc(val_pred, test_pred, random_state=seed)
                scored["window_name"] = window["window_name"]
                scored["rolling_seed"] = seed
                prediction_rows.append(scored)
                meta = {
                    "run_name": f"{window['window_name']}_{backbone_name.lower()}_seed{seed}",
                    "window_name": window["window_name"],
                    "train_releases": ",".join(str(x) for x in window["train_releases"]),
                    "val_release": int(window["val_release"]),
                    "test_release": int(window["test_release"]),
                    "seed": int(seed),
                    "model_type": backbone_name.lower(),
                    "backbone_name": backbone_name,
                    "num_train": int(len(train)),
                    "num_val": int(len(val)),
                    "num_test": int(len(test)),
                }
                for source_name, confidence_col in SOURCES:
                    summary_rows.append(
                        {
                            **meta,
                            "confidence_source": source_name,
                            **selective_metrics(scored, confidence_col=confidence_col),
                        }
                    )
    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    pairwise = _paired_summary(summary) if not summary.empty else pd.DataFrame()
    design = pd.DataFrame(design_rows)
    _write_frame(design, output_dir / "chembl_rolling_release_design.csv")
    _write_frame(summary, output_dir / "chembl_rolling_release_summary.csv")
    _write_frame(pairwise, output_dir / "chembl_rolling_release_pairwise_stats.csv")
    _write_frame(predictions, output_dir / "chembl_rolling_release_predictions.csv")
    status = {
        "windows": len(windows),
        "seeds": list(seeds),
        "backbones": list(BACKBONES),
        "summary_rows": int(len(summary)),
        "prediction_rows": int(len(predictions)),
        "caution": "The first rolling window has a small CHEMBL_27 calibration block and is reported as a stress test.",
    }
    (output_dir / "chembl_rolling_release_status.json").write_text(json.dumps(status, indent=2))
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rolling ChEMBL release temporal audit.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--ensemble-size", type=int, default=3)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else workspace / "reports" / "submission_upgrade_audits"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    status = run_audit(workspace, output_dir, seeds, args.ensemble_size)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
