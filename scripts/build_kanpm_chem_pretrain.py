#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel

from selective_dta_b.external.kanpm_pretrain import (
    build_embedding_payload,
    load_kanpm_drug_records,
    resolve_kanpm_pretrained_paths,
    save_pickle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ChemBERTa features for KANPM-DTA datasets")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-name", default="DeepChem/ChemBERTa-77M-MTR")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-smiles-length", type=int, default=220)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "KANPM-DTA"
    device = resolve_device(args.device)
    paths = resolve_kanpm_pretrained_paths(external_root, args.dataset_name)
    drug_csv = external_root / "datasets" / args.dataset_name / f"{args.dataset_name}_drugs.csv"

    records = load_kanpm_drug_records(drug_csv)
    if args.limit is not None:
        records = records[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RobertaModel.from_pretrained(args.model_name).to(device).eval()

    def encode_fn(_: str, smiles: str) -> torch.Tensor:
        tokenized = tokenizer(smiles, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokenized)
        return outputs.last_hidden_state[0][1:-1].detach().cpu()

    payload = build_embedding_payload(
        dataset_name=args.dataset_name,
        records=tqdm(records, desc=f"ChemBERTa[{args.dataset_name}]"),
        encode_fn=encode_fn,
        max_length=args.max_smiles_length,
    )
    output_path = save_pickle(payload, paths["chem"])
    summary = {
        "dataset_name": args.dataset_name,
        "output_path": str(output_path),
        "records": len(payload["vec_dict"]),
        "device": str(device),
        "model_name": args.model_name,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

