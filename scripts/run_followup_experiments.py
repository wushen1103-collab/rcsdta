#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.eval.followup_experiments import run_followup_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run follow-up selective DTA analyses from cached predictions")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--all-runs", action="store_true", help="Use all discovered posthoc predictions instead of paper matrix only")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--min-compounds-per-target", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    status = run_followup_experiments(
        workspace=Path(args.workspace),
        output_dir=args.output_dir,
        paper_only=not args.all_runs,
        max_runs=args.max_runs,
        min_compounds_per_target=args.min_compounds_per_target,
    )
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
