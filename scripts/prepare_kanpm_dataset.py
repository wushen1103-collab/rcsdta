#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.external.kanpm import materialize_kanpm_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize selective_dta_b splits into KANPM-DTA dataset layout")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--external-root", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "KANPM-DTA"
    payload = materialize_kanpm_dataset(
        workspace=workspace,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        seed=args.seed,
        external_root=external_root,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

