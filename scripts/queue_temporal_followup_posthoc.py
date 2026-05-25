#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wait for temporal-proxy training and export selective/posthoc predictions")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--sleep-seconds", type=int, default=180)
    parser.add_argument("--max-wait-rounds", type=int, default=240)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-mc-samples", type=int, default=16)
    return parser


def _training_active() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "train_char_baseline.py .*temporal_proxy"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _run(command: list[str], workspace: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=workspace, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"return_code={result.returncode}\n")
        return result.returncode


def _finished_temporal_runs(workspace: Path) -> list[str]:
    run_names: list[str] = []
    for summary_path in sorted((workspace / "artifacts" / "runs").glob("*temporal_proxy_ep15*/run_summary.json")):
        try:
            payload = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "finished":
            run_names.append(str(payload["run_name"]))
    return run_names


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    python = str(workspace / ".venv" / "bin" / "python")
    log_path = workspace / "logs" / "temporal_followup_posthoc_20260421.log"

    for _ in range(args.max_wait_rounds):
        if not _training_active():
            break
        time.sleep(args.sleep_seconds)

    processed: list[str] = []
    failed: list[str] = []
    for run_name in _finished_temporal_runs(workspace):
        metrics_path = workspace / "artifacts" / "runs" / run_name / "posthoc_selector" / f"{run_name}_posthoc_metrics.json"
        if metrics_path.exists():
            processed.append(run_name)
            continue
        export_code = _run(
            [
                python,
                "scripts/export_selective_predictions.py",
                "--workspace",
                str(workspace),
                "--run-name",
                run_name,
                "--accelerator",
                args.accelerator,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--num-mc-samples",
                str(args.num_mc_samples),
            ],
            workspace,
            log_path,
        )
        posthoc_code = _run(
            [
                python,
                "scripts/run_posthoc_selector.py",
                "--workspace",
                str(workspace),
                "--run-name",
                run_name,
                "--accelerator",
                args.accelerator,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--num-mc-samples",
                str(args.num_mc_samples),
                "--regressor-type",
                "ridge",
                "--feature-set",
                "enriched9",
            ],
            workspace,
            log_path,
        )
        if export_code == 0 and posthoc_code == 0:
            processed.append(run_name)
        else:
            failed.append(run_name)

    summary = {
        "processed": processed,
        "failed": failed,
        "num_processed": len(processed),
        "num_failed": len(failed),
    }
    (workspace / "reports" / "followup_experiments" / "temporal_posthoc_queue_status.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
