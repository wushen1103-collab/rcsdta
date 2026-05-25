#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.eval.maximal_trans_experiments import run_maximal_trans_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run maximal Trans-grade selective DTA experiments")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sections", default="chembl_release,vs_utility,paired_stats")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sections = tuple(item.strip() for item in args.sections.split(",") if item.strip())
    status = run_maximal_trans_experiments(
        workspace=Path(args.workspace),
        output_dir=args.output_dir,
        sections=sections,
        max_runs=args.max_runs,
    )
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
