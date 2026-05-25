#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.external.adambind import stage_adambind_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage selective_dta_b splits into AdaMBind target-list files")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--min-total-interactions", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "AdaMBind"
    payload = stage_adambind_split(
        workspace=workspace,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        seed=args.seed,
        external_root=external_root,
        activate=args.activate,
        min_total_interactions=args.min_total_interactions,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

