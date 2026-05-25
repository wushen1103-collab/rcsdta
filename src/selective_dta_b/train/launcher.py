from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from selective_dta_b.data.materialize import materialize_dataset_split
from selective_dta_b.runtime.resources import recommend_num_workers_per_job


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    model_type: str
    dataset_name: str
    split_name: str
    seed: int
    split_seed: int
    max_epochs: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LaunchAssignment:
    run_spec: RunSpec
    gpu_index: int
    num_workers: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["run_spec"] = self.run_spec.to_dict()
        return payload


def build_baseline_run_specs(
    datasets: list[str],
    splits: list[str],
    seeds: list[int],
    max_epochs: int,
    tag: str | None = None,
    model_type: str = "baseline",
    split_seed: int | None = None,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    prefix_lookup = {
        "baseline": "char_baseline",
        "heteroscedastic": "char_heteroscedastic",
        "deepdta": "deepdta",
        "graphdta": "graphdta",
        "moltrans": "moltrans",
    }
    base_prefix = prefix_lookup.get(model_type, model_type)
    prefix = f"{base_prefix}_{tag}" if tag else base_prefix
    for dataset_name in datasets:
        for split_name in splits:
            for seed in seeds:
                effective_split_seed = seed if split_seed is None else split_seed
                seed_suffix = (
                    f"split{effective_split_seed}_seed{seed}"
                    if effective_split_seed != seed
                    else f"seed{seed}"
                )
                specs.append(
                    RunSpec(
                        run_name=f"{prefix}_{dataset_name}_{split_name}_{seed_suffix}",
                        model_type=model_type,
                        dataset_name=dataset_name,
                        split_name=split_name,
                        seed=seed,
                        split_seed=effective_split_seed,
                        max_epochs=max_epochs,
                    )
                )
    return specs


def assign_runs_to_resources(
    run_specs: list[RunSpec],
    idle_gpu_indices: list[int],
    logical_cpus: int,
    reserve_cpus: int = 8,
    max_workers_per_job: int = 16,
) -> list[LaunchAssignment]:
    if not idle_gpu_indices or not run_specs:
        return []
    concurrency = min(len(run_specs), len(idle_gpu_indices))
    workers = recommend_num_workers_per_job(
        logical_cpus=logical_cpus,
        concurrent_jobs=concurrency,
        reserve_cpus=reserve_cpus,
        max_workers_per_job=max_workers_per_job,
    )
    assignments: list[LaunchAssignment] = []
    for run_spec, gpu_index in zip(run_specs[:concurrency], idle_gpu_indices[:concurrency]):
        assignments.append(
            LaunchAssignment(
                run_spec=run_spec,
                gpu_index=gpu_index,
                num_workers=workers,
            )
        )
    return assignments


def plan_run_waves(
    run_specs: list[RunSpec],
    idle_gpu_indices: list[int],
    logical_cpus: int,
    reserve_cpus: int = 8,
    max_workers_per_job: int = 16,
) -> list[list[LaunchAssignment]]:
    if not idle_gpu_indices:
        return []
    waves: list[list[LaunchAssignment]] = []
    start = 0
    step = len(idle_gpu_indices)
    while start < len(run_specs):
        chunk = run_specs[start:start + step]
        waves.append(
            assign_runs_to_resources(
                run_specs=chunk,
                idle_gpu_indices=idle_gpu_indices,
                logical_cpus=logical_cpus,
                reserve_cpus=reserve_cpus,
                max_workers_per_job=max_workers_per_job,
            )
        )
        start += step
    return waves


def filter_completed_runs(workspace: str | Path, run_specs: list[RunSpec]) -> list[RunSpec]:
    workspace_path = Path(workspace)
    pending: list[RunSpec] = []
    for spec in run_specs:
        summary_path = workspace_path / "artifacts" / "runs" / spec.run_name / "run_summary.json"
        if not summary_path.exists():
            pending.append(spec)
            continue
        try:
            payload = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pending.append(spec)
            continue
        if payload.get("status") != "finished":
            pending.append(spec)
    return pending


def ensure_run_splits_materialized(
    workspace: str | Path,
    run_specs: list[RunSpec],
    overwrite: bool = False,
) -> list[dict[str, object]]:
    materialized: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for spec in run_specs:
        split_seed = getattr(spec, "split_seed", spec.seed)
        key = (spec.dataset_name, spec.split_name, split_seed)
        if key in seen:
            continue
        seen.add(key)
        materialized.append(
            materialize_dataset_split(
                workspace=workspace,
                dataset_name=spec.dataset_name,
                split_name=spec.split_name,
                seed=split_seed,
                overwrite=overwrite,
            )
        )
    return materialized

