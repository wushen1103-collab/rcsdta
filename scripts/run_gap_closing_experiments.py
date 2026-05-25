#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.eval.gap_closing_experiments import run_gap_closing_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run six gap-closing experiment analyses for the selective DTA paper")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-dir", default="reports/gap_closing_experiments")
    parser.add_argument("--paper-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir
    status = run_gap_closing_experiments(
        workspace,
        output_dir=output_dir,
        paper_only=args.paper_only,
        max_runs=args.max_runs,
    )
    print(json.dumps({"workspace": str(workspace), "output_dir": str(output_dir), **status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

