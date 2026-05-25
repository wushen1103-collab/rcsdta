#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import esm
import torch
from tqdm import tqdm

from selective_dta_b.external.kanpm_pretrain import (
    build_contact_map_payload,
    load_kanpm_protein_records,
    merge_contact_map_payloads,
    resolve_kanpm_pretrained_paths,
    save_pickle,
    shard_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ESM2 contact maps for KANPM-DTA datasets")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-fn", default="esm2_t36_3B_UR50D")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-length", type=int, default=1200)
    parser.add_argument("--pad-square-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_shard_output_path(base_path: Path, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}.shard{shard_index:02d}of{num_shards:02d}{base_path.suffix}")


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
    records = shard_records(records, num_shards=args.num_shards, shard_index=args.shard_index)

    if args.merge_shards:
        shard_paths = [
            resolve_shard_output_path(paths["esm2_contact_map"], num_shards=args.num_shards, shard_index=index)
            for index in range(args.num_shards)
        ]
        payloads = []
        for shard_path in shard_paths:
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard payload: {shard_path}")
            with shard_path.open("rb") as handle:
                payloads.append(pickle.load(handle))
        merged_payload = merge_contact_map_payloads(args.dataset_name, payloads)
        output_path = save_pickle(merged_payload, paths["esm2_contact_map"])
        summary = {
            "dataset_name": args.dataset_name,
            "output_path": str(output_path),
            "records": len(merged_payload["contact_map"]),
            "num_shards": args.num_shards,
            "merged": True,
        }
        print(json.dumps(summary, indent=2))
        return 0

    model_loader = getattr(esm.pretrained, args.model_fn)
    model, alphabet = model_loader()
    use_half = device.type == "cuda"
    if use_half:
        model = model.half()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    def contact_map_fn(record_id: str, sequence: str):
        _, _, batch_tokens = batch_converter([(record_id, sequence)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_half):
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return results["contacts"][0].detach().cpu().numpy()

    payload = build_contact_map_payload(
        dataset_name=args.dataset_name,
        records=tqdm(records, desc=f"ESM2Contact[{args.dataset_name}]"),
        contact_map_fn=contact_map_fn,
        max_length=args.max_seq_length,
        pad_square_size=args.pad_square_size,
    )
    output_path = save_pickle(
        payload,
        resolve_shard_output_path(paths["esm2_contact_map"], num_shards=args.num_shards, shard_index=args.shard_index),
    )
    summary = {
        "dataset_name": args.dataset_name,
        "output_path": str(output_path),
        "records": len(payload["contact_map"]),
        "device": str(device),
        "model_fn": args.model_fn,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "merged": False,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

