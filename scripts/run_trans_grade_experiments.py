#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.eval.trans_grade_experiments import run_trans_grade_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Trans-grade selective DTA follow-up experiments")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--all-runs", action="store_true", help="Use all discovered posthoc prediction files")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--sections",
        default="modern,risk,vs,failure,chembl",
        help="Comma-separated subset: modern,risk,vs,failure,chembl",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--bootstrap-sample-cap", type=int, default=5000)
    parser.add_argument("--write-modern-predictions", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sections = tuple(item.strip() for item in args.sections.split(",") if item.strip())
    status = run_trans_grade_experiments(
        workspace=Path(args.workspace),
        output_dir=args.output_dir,
        paper_only=not args.all_runs,
        max_runs=args.max_runs,
        sections=sections,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_sample_cap=args.bootstrap_sample_cap,
        write_modern_predictions=args.write_modern_predictions,
    )
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
