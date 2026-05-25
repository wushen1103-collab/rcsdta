from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.data.prep import prepare_dataset_workspace
from selective_dta_b.data.registry import get_default_registry
from selective_dta_b.data.splits import build_split_plan


def cmd_list_datasets() -> int:
    payload = {"datasets": sorted(get_default_registry())}
    print(json.dumps(payload, indent=2))
    return 0


def cmd_init_split(dataset: str, split_type: str, seed: int, output: str) -> int:
    plan = build_split_plan(dataset_name=dataset, split_type=split_type, random_seed=seed)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2))
    print(json.dumps({"output": str(output_path), "split_type": split_type, "dataset": dataset}, indent=2))
    return 0


def cmd_prepare_dataset(dataset: str, workspace: str) -> int:
    registry = get_default_registry()
    prepared = prepare_dataset_workspace(workspace=workspace, dataset=registry[dataset])
    print(json.dumps({"dataset": dataset, **prepared}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selective DTA B workspace CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-datasets")

    split_parser = subparsers.add_parser("init-split")
    split_parser.add_argument("--dataset", required=True)
    split_parser.add_argument("--split-type", required=True)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--output", required=True)

    prepare_parser = subparsers.add_parser("prepare-dataset")
    prepare_parser.add_argument("--dataset", required=True, choices=sorted(get_default_registry()))
    prepare_parser.add_argument("--workspace", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-datasets":
        return cmd_list_datasets()
    if args.command == "init-split":
        return cmd_init_split(args.dataset, args.split_type, args.seed, args.output)
    if args.command == "prepare-dataset":
        return cmd_prepare_dataset(args.dataset, args.workspace)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
