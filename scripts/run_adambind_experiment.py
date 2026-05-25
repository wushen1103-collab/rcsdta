#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

from selective_dta_b.external.adambind_runtime import (
    AdaMBindRunSpec,
    build_adambind_train_argv,
    default_adambind_result_filename,
    parse_adambind_result_file,
    prepare_adambind_run_root,
    run_dir_for_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one formal AdaMBind experiment in an isolated run root")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--external-python", default=None)
    parser.add_argument("--wrapper-script", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gnn", default="gat_gcn")
    parser.add_argument("--nums", type=int, default=10)
    parser.add_argument("--min-total-interactions", type=int, default=None)
    parser.add_argument("--run-group", default="adambind_formal")
    parser.add_argument("--force", action="store_true")
    return parser


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "AdaMBind"
    external_python = Path(args.external_python).resolve() if args.external_python else external_root / ".venv" / "bin" / "python"
    wrapper_script = Path(args.wrapper_script).resolve() if args.wrapper_script else workspace / "scripts" / "adambind_train_entry.py"
    train_script = external_root / "train.py"

    run_spec = AdaMBindRunSpec(
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        seed=args.seed,
    )
    run_dir = run_dir_for_spec(workspace=workspace, run_spec=run_spec, run_group=args.run_group)
    run_root = run_dir / "adambind_root"
    summary_path = run_dir / "run_summary.json"
    status_path = run_dir / "status.json"
    pid_path = run_dir / "launcher.pid"

    if summary_path.exists() and not args.force:
        print(json.dumps({"status": "already_completed", "run_summary_path": str(summary_path)}, ensure_ascii=False))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    _write_json(
        status_path,
        {
            "status": "running",
            "pid": os.getpid(),
            "gpu": args.gpu,
            "run_spec": run_spec.__dict__,
            "min_total_interactions": args.min_total_interactions,
        },
    )

    try:
        run_root_manifest = prepare_adambind_run_root(
            workspace=workspace,
            external_root=external_root,
            run_spec=run_spec,
            run_root=run_root,
            min_total_interactions=args.min_total_interactions,
        )

        train_argv = build_adambind_train_argv(
            train_script=train_script,
            run_root=run_root,
            run_dir=run_dir,
            run_spec=run_spec,
            gnn=args.gnn,
            nums=args.nums,
        )
        launch_payload = {
            "external_python": str(external_python),
            "wrapper_script": str(wrapper_script),
            "train_script": str(train_script),
            "train_argv": train_argv,
            "gpu": args.gpu,
            "run_root_manifest_path": run_root_manifest["manifest_path"],
            "min_total_interactions": args.min_total_interactions,
        }
        _write_json(run_dir / "launch_request.json", launch_payload)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["PYTHONUNBUFFERED"] = "1"
        command = [str(external_python), str(wrapper_script), *train_argv]
        completed = subprocess.run(
            command,
            cwd=str(external_root),
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"AdaMBind training exited with code {completed.returncode}")

        result_file = run_dir / "result" / default_adambind_result_filename(run_spec, nums=args.nums)
        metrics = parse_adambind_result_file(result_file)
        summary_payload = {
            "status": "completed",
            "model_family": "adambind",
            "dataset_name": run_spec.dataset_name,
            "split_name": run_spec.split_name,
            "seed": run_spec.seed,
            "gpu": args.gpu,
            "gnn": args.gnn,
            "nums": args.nums,
            "min_total_interactions": args.min_total_interactions,
            "metrics": metrics,
            "result_file": str(result_file),
            "checkpoints_dir": str(run_dir / "checkpoints"),
            "run_root": str(run_root),
        }
        _write_json(summary_path, summary_payload)
        _write_json(status_path, summary_payload)
        print(json.dumps(summary_payload, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover - failure path exercised operationally
        failure_payload = {
            "status": "failed",
            "run_spec": run_spec.__dict__,
            "gpu": args.gpu,
            "min_total_interactions": args.min_total_interactions,
            "error": str(exc),
        }
        _write_json(status_path, failure_payload)
        traceback.print_exc()
        return 1
    finally:
        if pid_path.exists():
            pid_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())

