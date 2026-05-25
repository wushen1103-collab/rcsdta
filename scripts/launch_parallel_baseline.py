#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from selective_dta_b.runtime.resources import build_resource_snapshot
from selective_dta_b.train.launcher import (
    assign_runs_to_resources,
    build_baseline_run_specs,
    ensure_run_splits_materialized,
    filter_completed_runs,
)


def _parse_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch char-baseline runs on idle GPUs")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--datasets", default="bindingdb,davis,kiba")
    parser.add_argument("--splits", default="unseen_target,similarity_aware_unseen_target")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--allowed-gpus", default=None, help="Comma-separated physical GPU indices to use")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--model-type", default="baseline")
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reserve-cpus", type=int, default=24)
    parser.add_argument("--max-workers-per-job", type=int, default=16)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _build_command(
    workspace: Path,
    assignment,
    batch_size: int,
) -> list[str]:
    run_dir = workspace / "artifacts" / "runs" / assignment.run_spec.run_name
    return [
        str(workspace / ".venv" / "bin" / "python"),
        "scripts/train_char_baseline.py",
        "--workspace",
        str(workspace),
        "--dataset-name",
        assignment.run_spec.dataset_name,
        "--split-name",
        assignment.run_spec.split_name,
        "--seed",
        str(assignment.run_spec.seed),
        "--split-seed",
        str(assignment.run_spec.split_seed),
        "--run-name",
        assignment.run_spec.run_name,
        "--default-root-dir",
        str(run_dir),
        "--model-type",
        assignment.run_spec.model_type,
        "--accelerator",
        "gpu",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(assignment.num_workers),
        "--max-epochs",
        str(assignment.run_spec.max_epochs),
    ]


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    allowed_gpus = None if args.allowed_gpus is None else [int(part.strip()) for part in args.allowed_gpus.split(",") if part.strip()]

    requested_specs = build_baseline_run_specs(
        datasets=_parse_csv_arg(args.datasets),
        splits=_parse_csv_arg(args.splits),
        seeds=[int(seed) for seed in _parse_csv_arg(args.seeds)],
        max_epochs=args.max_epochs,
        tag=args.tag,
        model_type=args.model_type,
        split_seed=args.split_seed,
    )
    if args.max_concurrent is not None:
        requested_specs = requested_specs[: args.max_concurrent]
    pending_specs = filter_completed_runs(workspace, requested_specs) if args.skip_completed else requested_specs
    materialized_splits = ensure_run_splits_materialized(workspace, pending_specs) if pending_specs else []

    launch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_dir = workspace / "artifacts" / "launches" / launch_id
    launch_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "launch_id": launch_id,
        "workspace": str(workspace),
        "requested_runs": [spec.to_dict() for spec in requested_specs],
        "pending_runs": [spec.to_dict() for spec in pending_specs],
        "materialized_splits": materialized_splits,
        "waves": [],
        "dry_run": args.dry_run,
        "skip_completed": args.skip_completed,
    }

    remaining = pending_specs[:]
    wave_index = 0
    while remaining:
        snapshot = build_resource_snapshot(workspace)
        idle_gpu_indices = list(snapshot["gpu"]["idle_gpu_indices"])
        if allowed_gpus is not None:
            idle_gpu_indices = [gpu_index for gpu_index in idle_gpu_indices if gpu_index in allowed_gpus]
        logical_cpus = int(snapshot["cpu"]["logical_cores"])
        assignments = assign_runs_to_resources(
            run_specs=remaining,
            idle_gpu_indices=idle_gpu_indices,
            logical_cpus=logical_cpus,
            reserve_cpus=args.reserve_cpus,
            max_workers_per_job=args.max_workers_per_job,
        )
        if not assignments:
            break

        wave_payload = {
            "wave_index": wave_index,
            "idle_gpu_indices": idle_gpu_indices,
            "assignments": [],
        }
        processes: list[tuple[subprocess.Popen, object, dict[str, object]]] = []
        for assignment in assignments:
            log_path = workspace / "logs" / "training" / f"{assignment.run_spec.run_name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = _build_command(workspace, assignment, args.batch_size)
            launched = {
                "run_name": assignment.run_spec.run_name,
                "gpu_index": assignment.gpu_index,
                "num_workers": assignment.num_workers,
                "command": command,
                "log_path": str(log_path),
                "run_dir": str(workspace / "artifacts" / "runs" / assignment.run_spec.run_name),
            }
            if not args.dry_run:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(assignment.gpu_index)
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                log_handle = open(log_path, "a", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=workspace,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                launched["pid"] = process.pid
                processes.append((process, log_handle, launched))
            wave_payload["assignments"].append(launched)

        manifest["waves"].append(wave_payload)
        manifest_path = launch_dir / "launch_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        for process, log_handle, launched in processes:
            return_code = process.wait()
            log_handle.close()
            launched["return_code"] = return_code

        completed_names = {assignment.run_spec.run_name for assignment in assignments}
        remaining = [spec for spec in remaining if spec.run_name not in completed_names]
        wave_index += 1

        if args.dry_run:
            break

    manifest["remaining_runs"] = [spec.to_dict() for spec in remaining]
    manifest_path = launch_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"launch_manifest": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

