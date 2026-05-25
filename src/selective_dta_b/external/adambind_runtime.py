from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from selective_dta_b.external.adambind import stage_adambind_split


DEFAULT_ADAMBIND_RUN_GROUP = "adambind_formal"
DEFAULT_ADAMBIND_GNN = "gat_gcn"
DEFAULT_ADAMBIND_NUMS = 10
ADAMBIND_RESULT_KEYS = ("mse", "ci", "r2", "spearman", "pearson")

_DATASET_PRIORITY = {
    "davis": 0,
    "kiba": 1,
    "bindingdb": 2,
}
_SPLIT_PRIORITY = {
    "unseen_target": 0,
    "similarity_aware_unseen_target": 1,
}


@dataclass(frozen=True)
class AdaMBindRunSpec:
    dataset_name: str
    split_name: str
    seed: int

    def slug(self, *, prefix: str = "adambind") -> str:
        return f"{prefix}_{self.dataset_name}_{self.split_name}_seed{self.seed}"


def _coerce_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_filter(values: Iterable[str | int] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value).strip() for value in values if str(value).strip()}
    return normalized or None


def _compatibility_csv_path(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / "reports" / "summary" / "adambind_split_compatibility.csv"


def list_compatible_adambind_runs(
    workspace: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    splits: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[AdaMBindRunSpec]:
    compatibility_path = _compatibility_csv_path(workspace)
    if not compatibility_path.exists():
        raise FileNotFoundError(f"AdaMBind compatibility audit not found: {compatibility_path}")

    dataset_filter = _normalize_filter(datasets)
    split_filter = _normalize_filter(splits)
    seed_filter = _normalize_filter(seeds)

    frame = pd.read_csv(compatibility_path)
    compatible_rows = []
    for row in frame.to_dict(orient="records"):
        if not _coerce_truthy(row.get("representable")):
            continue
        dataset_name = str(row["dataset"]).strip()
        split_name = str(row["split"]).strip()
        seed = int(row["seed"])
        if dataset_filter is not None and dataset_name not in dataset_filter:
            continue
        if split_filter is not None and split_name not in split_filter:
            continue
        if seed_filter is not None and str(seed) not in seed_filter:
            continue
        compatible_rows.append(AdaMBindRunSpec(dataset_name=dataset_name, split_name=split_name, seed=seed))

    compatible_rows.sort(
        key=lambda spec: (
            _DATASET_PRIORITY.get(spec.dataset_name, 999),
            spec.dataset_name,
            _SPLIT_PRIORITY.get(spec.split_name, 999),
            spec.split_name,
            spec.seed,
        )
    )
    return compatible_rows


def _link_or_copy_file(source: Path, destination: Path) -> str:
    if not source.exists():
        raise FileNotFoundError(f"Required AdaMBind asset missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def prepare_adambind_run_root(
    workspace: str | Path,
    external_root: str | Path,
    run_spec: AdaMBindRunSpec,
    run_root: str | Path,
    *,
    min_total_interactions: int | None = None,
) -> dict[str, object]:
    workspace_path = Path(workspace).resolve()
    external_root_path = Path(external_root).resolve()
    run_root_path = Path(run_root).resolve()
    shared_data_root = external_root_path / "data"
    shared_processed_root = shared_data_root / "processed"
    run_data_root = run_root_path / "data"

    # Link the shared AdaMBind assets before staging splits so viability
    # filtering can inspect the run-local `{dataset}-full-data.csv`.
    linked_assets = {
        "dataset_csv": _link_or_copy_file(
            shared_data_root / f"{run_spec.dataset_name}-full-data.csv",
            run_data_root / f"{run_spec.dataset_name}-full-data.csv",
        ),
        "processed_dataset": _link_or_copy_file(
            shared_processed_root / f"{run_spec.dataset_name}-full-data.pt",
            run_data_root / "processed" / f"{run_spec.dataset_name}-full-data.pt",
        ),
    }

    staged_manifest = stage_adambind_split(
        workspace=workspace_path,
        dataset_name=run_spec.dataset_name,
        split_name=run_spec.split_name,
        seed=run_spec.seed,
        external_root=run_root_path,
        activate=True,
        min_total_interactions=min_total_interactions,
    )

    manifest = {
        "run_spec": asdict(run_spec),
        "run_root": str(run_root_path),
        "shared_external_root": str(external_root_path),
        "min_total_interactions": min_total_interactions,
        "linked_assets": linked_assets,
        "staged_manifest": staged_manifest,
    }
    manifest_path = run_root_path / "adambind_run_root_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def run_dir_for_spec(
    workspace: str | Path,
    run_spec: AdaMBindRunSpec,
    *,
    run_group: str = DEFAULT_ADAMBIND_RUN_GROUP,
) -> Path:
    return Path(workspace).resolve() / "artifacts" / "external_runs" / run_group / "runs" / run_spec.slug(prefix=run_group)


def default_adambind_result_filename(run_spec: AdaMBindRunSpec, *, nums: int = DEFAULT_ADAMBIND_NUMS) -> str:
    return f"{run_spec.dataset_name}_{nums}_{run_spec.seed}.txt"


def allocate_adambind_gpus(
    run_specs: list[AdaMBindRunSpec],
    *,
    gpu_ids: list[int],
    busy_gpu_ids: set[int] | None = None,
) -> list[tuple[AdaMBindRunSpec, int]]:
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")
    busy_gpu_ids = busy_gpu_ids or set()
    preferred_gpu_ids = [gpu for gpu in gpu_ids if gpu not in busy_gpu_ids]
    fallback_gpu_ids = [gpu for gpu in gpu_ids if gpu in busy_gpu_ids]
    ordered_gpu_ids = preferred_gpu_ids + fallback_gpu_ids
    return [
        (run_spec, ordered_gpu_ids[index % len(ordered_gpu_ids)])
        for index, run_spec in enumerate(run_specs)
    ]


def build_adambind_train_argv(
    *,
    train_script: str | Path,
    run_root: str | Path,
    run_dir: str | Path,
    run_spec: AdaMBindRunSpec,
    gnn: str = DEFAULT_ADAMBIND_GNN,
    nums: int = DEFAULT_ADAMBIND_NUMS,
) -> list[str]:
    train_script_path = Path(train_script).resolve()
    run_root_path = Path(run_root).resolve()
    run_dir_path = Path(run_dir).resolve()
    checkpoints_dir = run_dir_path / "checkpoints"
    result_dir = run_dir_path / "result"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    return [
        "--train-script",
        str(train_script_path),
        "--root",
        str(run_root_path),
        "--dataset",
        run_spec.dataset_name,
        "--seed",
        str(run_spec.seed),
        "--nums",
        str(nums),
        "--ckpt_dir",
        str(checkpoints_dir),
        "--result_dir",
        str(result_dir),
        "--gnn",
        gnn,
    ]


def parse_adambind_result_file(result_file: str | Path) -> dict[str, float]:
    result_path = Path(result_file).resolve()
    if not result_path.exists():
        raise FileNotFoundError(f"AdaMBind result file not found: {result_path}")

    lines = [line.strip() for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < len(ADAMBIND_RESULT_KEYS):
        raise ValueError(f"Incomplete AdaMBind result file: {result_path}")
    return {
        key: float(value)
        for key, value in zip(ADAMBIND_RESULT_KEYS, lines[: len(ADAMBIND_RESULT_KEYS)])
    }


__all__ = [
    "ADAMBIND_RESULT_KEYS",
    "DEFAULT_ADAMBIND_GNN",
    "DEFAULT_ADAMBIND_NUMS",
    "DEFAULT_ADAMBIND_RUN_GROUP",
    "AdaMBindRunSpec",
    "allocate_adambind_gpus",
    "build_adambind_train_argv",
    "default_adambind_result_filename",
    "list_compatible_adambind_runs",
    "parse_adambind_result_file",
    "prepare_adambind_run_root",
    "run_dir_for_spec",
]

