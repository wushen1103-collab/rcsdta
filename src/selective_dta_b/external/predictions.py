from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from selective_dta_b.data.loading import load_split_frame
from selective_dta_b.external.kanpm import (
    resolve_kanpm_dataset_name,
    resolve_kanpm_running_set_name,
    resolve_kanpm_seeded_running_set_name,
)
from selective_dta_b.external.pmmr import (
    parse_pmmr_run_name,
    resolve_pmmr_report_dir_from_run_name,
)

EXTERNAL_MODEL_TYPES = {"adambind", "deepdtagen", "kanpm", "pmmr"}


@dataclass(frozen=True)
class RunContext:
    workspace: Path
    run_name: str
    run_dir: Path
    summary_path: Path
    summary: dict[str, object]

    @property
    def dataset_name(self) -> str:
        return str(self.summary["dataset_name"])

    @property
    def split_name(self) -> str:
        return str(self.summary["split_name"])

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])

    @property
    def split_seed(self) -> int:
        return int(self.summary.get("split_seed", self.seed))

    @property
    def model_type(self) -> str:
        return str(self.summary.get("model_type", self.summary.get("model_family", "baseline")))


def _normalize_summary(summary: dict[str, object], *, run_name: str) -> dict[str, object]:
    normalized = dict(summary)
    normalized.setdefault("run_name", run_name)
    if "model_type" not in normalized and "model_family" in normalized:
        normalized["model_type"] = normalized["model_family"]
    return normalized


def _load_context_from_summary(summary_path: Path, *, workspace: Path, run_name: str) -> RunContext:
    summary = _normalize_summary(json.loads(summary_path.read_text()), run_name=run_name)
    return RunContext(
        workspace=workspace,
        run_name=run_name,
        run_dir=summary_path.parent,
        summary_path=summary_path,
        summary=summary,
    )


def _find_existing_external_summary(workspace: Path, run_name: str) -> Path | None:
    base = workspace / "artifacts" / "external_runs"
    if not base.exists():
        return None
    matches = sorted(base.glob(f"*/runs/{run_name}/run_summary.json"))
    return matches[0] if matches else None


def _materialize_kanpm_context(workspace: Path, run_name: str) -> RunContext:
    match = re.fullmatch(r"kanpm_ep15_([^_]+)_(.+)_seed(\d+)", run_name)
    if match is None:
        raise FileNotFoundError(f"Unable to resolve external run summary for: {run_name}")

    dataset_name = match.group(1)
    split_name = match.group(2)
    seed = int(match.group(3))
    external_dataset = resolve_kanpm_dataset_name(dataset_name)
    running_set = resolve_kanpm_seeded_running_set_name(split_name, seed)
    split_prefix = f"{external_dataset}-{resolve_kanpm_running_set_name(split_name)}-seed{seed}-"

    external_root = workspace / "artifacts" / "external_runs" / "kanpm_ep15_seed42"
    model_dir = external_root / "models"
    csv_dir = external_root / "csv"
    model_matches = sorted(model_dir.glob(f"{split_prefix}*.pth"), key=lambda path: path.stat().st_mtime)
    if not model_matches:
        raise FileNotFoundError(f"No KANPM checkpoint matched {split_prefix!r} under {model_dir}")
    model_path = model_matches[-1]
    run_slug = model_path.stem
    metrics_path = csv_dir / f"Test-{run_slug}.csv"

    metrics_payload: dict[str, float] = {}
    if metrics_path.exists():
        metrics_frame = pd.read_csv(metrics_path)
        if not metrics_frame.empty:
            metrics_payload = {
                "test_mse": float(metrics_frame.loc[0, "mse"]),
                "test_ci": float(metrics_frame.loc[0, "ci"]),
                "test_rm2": float(metrics_frame.loc[0, "rm2"]),
            }

    run_dir = external_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    summary = {
        "run_name": run_name,
        "dataset_name": dataset_name,
        "split_name": split_name,
        "seed": seed,
        "split_seed": seed,
        "model_type": "kanpm",
        "status": "finished" if metrics_path.exists() else "checkpoint_only",
        "run_dir": str(run_dir),
        "metrics": metrics_payload,
        "paths": {
            "model": str(model_path),
            "metrics_csv": str(metrics_path),
        },
        "run_slug": run_slug,
        "running_set": running_set,
        "external_dataset_name": external_dataset,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return _load_context_from_summary(summary_path, workspace=workspace, run_name=run_name)


def _materialize_pmmr_context(workspace: Path, run_name: str) -> RunContext:
    parsed = parse_pmmr_run_name(run_name)
    dataset_name = str(parsed["dataset_name"])
    split_name = str(parsed["split_name"])
    seed = int(parsed["seed"])
    source_run_dir = resolve_pmmr_report_dir_from_run_name(workspace, run_name)
    source_summary_path = source_run_dir / "summary.json"
    if not source_summary_path.exists():
        raise FileNotFoundError(f"No PMMR summary found for {run_name!r} under {source_run_dir}")

    source_summary = json.loads(source_summary_path.read_text())
    run_dir = workspace / "artifacts" / "external_runs" / "pmmr" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    summary = {
        "run_name": run_name,
        "dataset_name": dataset_name,
        "split_name": split_name,
        "seed": seed,
        "split_seed": seed,
        "model_type": "pmmr",
        "status": "finished",
        "run_dir": str(run_dir),
        "source_run_dir": str(source_run_dir),
        "source_summary_path": str(source_summary_path),
        "checkpoint_path": source_summary.get("checkpoint_path"),
        "asset_root": source_summary.get("asset_root"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return _load_context_from_summary(summary_path, workspace=workspace, run_name=run_name)


def resolve_run_context(workspace: str | Path, run_name: str) -> RunContext:
    workspace_path = Path(workspace).resolve()
    summary_path = workspace_path / "artifacts" / "runs" / run_name / "run_summary.json"
    if summary_path.exists():
        return _load_context_from_summary(summary_path, workspace=workspace_path, run_name=run_name)

    external_summary = _find_existing_external_summary(workspace_path, run_name)
    if external_summary is not None:
        return _load_context_from_summary(external_summary, workspace=workspace_path, run_name=run_name)

    if run_name.startswith("kanpm_ep15_"):
        return _materialize_kanpm_context(workspace_path, run_name)
    if run_name.startswith("pmmr_"):
        return _materialize_pmmr_context(workspace_path, run_name)

    raise FileNotFoundError(f"Run summary not found for run {run_name!r}")


def _split_file_stem(split_value: str) -> str:
    return "validation" if split_value == "val" else "test"


def _resolve_prediction_cache_path(context: RunContext, split_value: str) -> Path | None:
    stem = _split_file_stem(split_value)
    candidates = [
        context.run_dir / f"{stem}_predictions.csv",
        context.run_dir / "external_predictions" / f"{stem}_predictions.csv",
    ]
    if context.model_type == "pmmr" and context.summary.get("source_run_dir"):
        source_run_dir = Path(str(context.summary["source_run_dir"]))
        candidates.insert(0, source_run_dir / f"{stem}_predictions.csv")
    if split_value == "test" and context.model_type == "deepdtagen":
        candidates.insert(0, context.run_dir / "test_predictions.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_external_predictions(raw_frame: pd.DataFrame, split_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    if "row_id" not in frame.columns:
        if len(frame) != len(split_frame):
            raise ValueError(
                f"Prediction row count mismatch: predictions={len(frame)} split={len(split_frame)}"
            )
        frame.insert(0, "row_id", split_frame["row_id"].tolist())

    if "prediction_mean" not in frame.columns:
        if "prediction" in frame.columns:
            frame["prediction_mean"] = frame["prediction"]
        elif "y_pred" in frame.columns:
            frame["prediction_mean"] = frame["y_pred"]
        else:
            raise KeyError("Prediction frame is missing prediction_mean/y_pred/prediction column")

    if "target" not in frame.columns:
        if "y_true" in frame.columns:
            frame["target"] = frame["y_true"]
        else:
            frame["target"] = split_frame["affinity_model_target"].tolist()

    if "prediction_std_mc_dropout" not in frame.columns:
        if "prediction_std" in frame.columns:
            frame["prediction_std_mc_dropout"] = frame["prediction_std"]
        else:
            frame["prediction_std_mc_dropout"] = 0.0

    if "prediction_std" not in frame.columns:
        frame["prediction_std"] = frame["prediction_std_mc_dropout"]

    return frame


def _generate_external_predictions(
    *,
    context: RunContext,
    split_value: str,
    accelerator: str,
    batch_size: int,
    num_workers: int,
    num_mc_samples: int,
    output_path: Path,
) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    script_name = {
        "adambind": "export_adambind_external_predictions.py",
        "deepdtagen": "predict_deepdtagen_external.py",
        "kanpm": "predict_kanpm_external.py",
        "pmmr": "export_pmmr_external_predictions.py",
    }[context.model_type]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / script_name),
            "--workspace",
            str(context.workspace),
            "--run-name",
            context.run_name,
            "--split",
            split_value,
            "--accelerator",
            accelerator,
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--num-mc-samples",
            str(num_mc_samples),
            "--output-path",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
    )
    return output_path


def load_external_prediction_frame(
    *,
    workspace: str | Path,
    run_name: str,
    split_value: str,
    accelerator: str,
    batch_size: int,
    num_workers: int,
    num_mc_samples: int,
) -> tuple[RunContext, pd.DataFrame]:
    context = resolve_run_context(workspace, run_name)
    split_frame = load_split_frame(
        workspace=context.workspace,
        dataset_name=context.dataset_name,
        split_name=context.split_name,
        seed=context.split_seed,
    )
    subset = split_frame.loc[split_frame["split"] == split_value].reset_index(drop=True)
    cache_path = _resolve_prediction_cache_path(context, split_value)
    if cache_path is None:
        cache_path = context.run_dir / "external_predictions" / f"{_split_file_stem(split_value)}_predictions.csv"
        _generate_external_predictions(
            context=context,
            split_value=split_value,
            accelerator=accelerator,
            batch_size=batch_size,
            num_workers=num_workers,
            num_mc_samples=num_mc_samples,
            output_path=cache_path,
        )
    frame = _normalize_external_predictions(pd.read_csv(cache_path), subset)
    return context, frame


__all__ = [
    "EXTERNAL_MODEL_TYPES",
    "RunContext",
    "load_external_prediction_frame",
    "resolve_run_context",
]


