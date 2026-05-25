#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.eval.chembl_temporal_backtest import build_chembl_split_rules, run_chembl_publication_year_backtest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a ChEMBL publication-year temporal DTA backtest")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--refresh", action="store_true", help="Refresh ChEMBL API caches")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-max-rows", type=int, default=2400)
    parser.add_argument("--val-max-rows", type=int, default=900)
    parser.add_argument("--test-max-rows", type=int, default=1200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    status = run_chembl_publication_year_backtest(
        workspace=Path(args.workspace),
        output_dir=args.output_dir,
        refresh=args.refresh,
        ensemble_size=args.ensemble_size,
        random_state=args.random_state,
        split_rules=build_chembl_split_rules(
            train_max_rows=args.train_max_rows,
            val_max_rows=args.val_max_rows,
            test_max_rows=args.test_max_rows,
        ),
    )
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
