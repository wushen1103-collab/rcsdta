#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from selective_dta_b.external.pmmr import materialize_pmmr_assets, materialize_pmmr_dataset


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize PMMR-compatible CSV splits from the selective DTA workspace")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--materialize-assets", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compound-model-name", default=None)
    parser.add_argument("--protein-model-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "PMMR"
    if args.materialize_assets:
        payload = materialize_pmmr_assets(
            workspace=workspace,
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            seed=args.seed,
            external_root=external_root,
            device=args.device,
            compound_model_name=args.compound_model_name
            or "DeepChem/ChemBERTa-77M-MLM",
            protein_model_name=args.protein_model_name
            or "facebook/esm2_t12_35M_UR50D",
        )
    else:
        payload = materialize_pmmr_dataset(
            workspace=workspace,
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            seed=args.seed,
            external_root=external_root,
        )
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
    if "protein_split_dirs" in serializable:
        serializable["protein_split_dirs"] = {
            key: str(value) for key, value in payload["protein_split_dirs"].items()
        }
    print(json.dumps(serializable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

