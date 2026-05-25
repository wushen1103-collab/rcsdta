#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.eval.deployment_upgrade_experiments import (
    BINDINGDB_DEFAULT_SOURCE_URL,
    run_deployment_upgrade_experiments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deployment-oriented upgrades for the selective DTA paper")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-dir", default="reports/deployment_upgrade_experiments")
    parser.add_argument("--paper-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--bindingdb-source-path", default=None)
    parser.add_argument("--bindingdb-source-url", default=None)
    parser.add_argument("--download-bindingdb", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir

    bindingdb_source_url = args.bindingdb_source_url
    if bindingdb_source_url is None and args.download_bindingdb:
        bindingdb_source_url = BINDINGDB_DEFAULT_SOURCE_URL

    status = run_deployment_upgrade_experiments(
        workspace,
        output_dir=output_dir,
        bindingdb_source_path=args.bindingdb_source_path,
        bindingdb_source_url=bindingdb_source_url,
        paper_only=args.paper_only,
        max_runs=args.max_runs,
    )
    print(json.dumps({"workspace": str(workspace), "output_dir": str(output_dir), **status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

