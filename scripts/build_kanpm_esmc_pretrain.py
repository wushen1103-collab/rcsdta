#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from selective_dta_b.external.kanpm_pretrain import (
    build_embedding_payload,
    load_kanpm_protein_records,
    resolve_kanpm_pretrained_paths,
    save_pickle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ESM-C residue features for KANPM-DTA datasets")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-length", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "KANPM-DTA"
    device = resolve_device(args.device)
    paths = resolve_kanpm_pretrained_paths(external_root, args.dataset_name)
    protein_csv = external_root / "datasets" / args.dataset_name / f"{args.dataset_name}_prots.csv"

    records = load_kanpm_protein_records(protein_csv)
    if args.limit is not None:
        records = records[: args.limit]

    model = ESMC.from_pretrained(args.model_name).to(device).eval()

    def encode_fn(_: str, sequence: str):
        protein = ESMProtein(sequence=sequence)
        with torch.no_grad():
            logits_output = model.logits(
                model.encode(protein),
                LogitsConfig(sequence=True, return_embeddings=True),
            )
        return logits_output.embeddings[0].detach().cpu().numpy()

    payload = build_embedding_payload(
        dataset_name=args.dataset_name,
        records=tqdm(records, desc=f"ESMC[{args.dataset_name}]"),
        encode_fn=encode_fn,
        max_length=args.max_seq_length,
    )
    output_path = save_pickle(payload, paths["esmc"])
    summary = {
        "dataset_name": args.dataset_name,
        "output_path": str(output_path),
        "records": len(payload["vec_dict"]),
        "device": device,
        "model_name": args.model_name,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

