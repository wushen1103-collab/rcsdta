#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from selective_dta_b.external.adambind_runtime import (
    AdaMBindRunSpec,
    allocate_adambind_gpus,
    list_compatible_adambind_runs,
    run_dir_for_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue compatible AdaMBind runs onto available GPUs")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--gpu-ids", default="2,3,4,5,6,7")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gnn", default="gat_gcn")
    parser.add_argument("--nums", type=int, default=10)
    parser.add_argument("--min-total-interactions", type=int, default=None)
    parser.add_argument("--run-group", default="adambind_formal")
    parser.add_argument("--force", action="store_true")
    return parser


def _split_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _is_running(run_dir: Path) -> bool:
    running_status = _read_running_status(run_dir)
    return running_status is not None


def _read_running_status(run_dir: Path) -> dict[str, object] | None:
    status_path = run_dir / "status.json"
    pid_path = run_dir / "launcher.pid"
    if not status_path.exists() or not pid_path.exists():
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, json.JSONDecodeError):
        return None
    if status.get("status") != "running" or not _pid_is_alive(pid):
        return None
    status["pid"] = pid
    return status


def _dataset_is_ready(external_root: Path, dataset_name: str) -> bool:
    shared_data_root = external_root / "data"
    return (
        (shared_data_root / f"{dataset_name}-full-data.csv").exists()
        and (shared_data_root / "processed" / f"{dataset_name}-full-data.pt").exists()
    )


def _launch_run(
    *,
    workspace: Path,
    external_root: Path,
    run_spec: AdaMBindRunSpec,
    gpu: int,
    gnn: str,
    nums: int,
    min_total_interactions: int | None,
    run_group: str,
    force: bool,
) -> dict[str, object]:
    runner_script = workspace / "scripts" / "run_adambind_experiment.py"
    log_dir = workspace / "logs" / "batch"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_spec.slug(prefix=run_group)}.log"
    command = [
        sys.executable,
        str(runner_script),
        "--workspace",
        str(workspace),
        "--external-root",
        str(external_root),
        "--dataset-name",
        run_spec.dataset_name,
        "--split-name",
        run_spec.split_name,
        "--seed",
        str(run_spec.seed),
        "--gpu",
        str(gpu),
        "--gnn",
        gnn,
        "--nums",
        str(nums),
        "--run-group",
        run_group,
    ]
    if min_total_interactions is not None:
        command.extend(["--min-total-interactions", str(min_total_interactions)])
    if force:
        command.append("--force")

    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {
        "run_spec": run_spec.__dict__,
        "gpu": gpu,
        "pid": process.pid,
        "log_path": str(log_path),
        "run_dir": str(run_dir_for_spec(workspace=workspace, run_spec=run_spec, run_group=run_group)),
    }


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "AdaMBind"
    gpu_ids = [int(item) for item in _split_csv_arg(args.gpu_ids) or []]
    if not gpu_ids:
        raise ValueError("No GPU ids provided")

    run_specs = list_compatible_adambind_runs(
        workspace=workspace,
        datasets=_split_csv_arg(args.datasets),
        splits=_split_csv_arg(args.splits),
        seeds=_split_csv_arg(args.seeds),
    )

    ready_specs = []
    skipped = []
    busy_gpu_ids: set[int] = set()
    for run_spec in run_specs:
        run_dir = run_dir_for_spec(workspace=workspace, run_spec=run_spec, run_group=args.run_group)
        if not _dataset_is_ready(external_root, run_spec.dataset_name):
            skipped.append({"run_spec": run_spec.__dict__, "reason": "dataset_not_ready"})
            continue
        if (run_dir / "run_summary.json").exists() and not args.force:
            skipped.append({"run_spec": run_spec.__dict__, "reason": "already_completed"})
            continue
        running_status = _read_running_status(run_dir)
        if running_status is not None:
            gpu = running_status.get("gpu")
            if isinstance(gpu, int):
                busy_gpu_ids.add(gpu)
            skipped.append({"run_spec": run_spec.__dict__, "reason": "already_running"})
            continue
        ready_specs.append(run_spec)

    if args.limit is not None:
        ready_specs = ready_specs[: args.limit]

    launched = []
    assignments = allocate_adambind_gpus(ready_specs, gpu_ids=gpu_ids, busy_gpu_ids=busy_gpu_ids)
    for run_spec, gpu in assignments:
        launched.append(
            _launch_run(
                workspace=workspace,
                external_root=external_root,
                run_spec=run_spec,
                gpu=gpu,
                gnn=args.gnn,
                nums=args.nums,
                min_total_interactions=args.min_total_interactions,
                run_group=args.run_group,
                force=args.force,
            )
        )

    payload = {
        "workspace": str(workspace),
        "external_root": str(external_root),
        "gpu_ids": gpu_ids,
        "busy_gpu_ids": sorted(busy_gpu_ids),
        "min_total_interactions": args.min_total_interactions,
        "launched": launched,
        "skipped": skipped,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

