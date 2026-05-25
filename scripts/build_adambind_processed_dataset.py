#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem


SEQ_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {token: index + 1 for index, token in enumerate(SEQ_VOCAB)}
MAX_SEQ_LEN = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build AdaMBind processed dataset for one benchmark")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def one_of_k_encoding(value: object, allowable_set: list[object]) -> list[bool]:
    if value not in allowable_set:
        raise ValueError(f"input {value!r} not in allowable set {allowable_set!r}")
    return [value == candidate for candidate in allowable_set]


def one_of_k_encoding_unk(value: object, allowable_set: list[object]) -> list[bool]:
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == candidate for candidate in allowable_set]


def atom_features(atom) -> np.ndarray:
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()]
    )


def smile_to_graph(smile: str) -> tuple[int, list[np.ndarray], list[list[int]]]:
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        raise ValueError(f"Failed to parse SMILES: {smile}")
    atom_count = molecule.GetNumAtoms()
    features = []
    for atom in molecule.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / feature.sum())

    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in molecule.GetBonds()]
    graph = nx.Graph(edges).to_directed()
    edge_index = [[start, end] for start, end in graph.edges]
    return atom_count, features, edge_index


def seq_cat(protein_sequence: str) -> np.ndarray:
    encoding = np.zeros(MAX_SEQ_LEN)
    for index, token in enumerate(protein_sequence[:MAX_SEQ_LEN]):
        encoding[index] = SEQ_DICT[token]
    return encoding


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(args.workspace).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else workspace / "external" / "AdaMBind"
    dataset_name = args.dataset_name

    data_root = external_root / "data"
    csv_path = data_root / f"{dataset_name}-full-data.csv"
    processed_path = data_root / "processed" / f"{dataset_name}-full-data.pt"
    if processed_path.exists() and not args.force:
        print(
            json.dumps(
                {
                    "status": "already_exists",
                    "dataset_name": dataset_name,
                    "processed_path": str(processed_path),
                },
                ensure_ascii=False,
            )
        )
        return 0

    sys.path.insert(0, str(external_root))
    from utils.TestbedDataset import TestbedDataset

    original_torch_load = torch.load

    def patched_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = patched_torch_load

    frame = pd.read_csv(csv_path)
    smiles = sorted(set(frame["compound_iso_smiles"].tolist()))
    smile_graph = {smile: smile_to_graph(smile) for smile in smiles}

    drug_array = np.asarray(frame["compound_iso_smiles"].tolist())
    protein_array = np.asarray([seq_cat(sequence) for sequence in frame["target_sequence"].tolist()])
    affinity_array = np.asarray(frame["affinity"].tolist())

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    TestbedDataset(
        root=str(data_root),
        dataset=f"{dataset_name}-full-data",
        xd=drug_array,
        xt=protein_array,
        y=affinity_array,
        smile_graph=smile_graph,
    )

    print(
        json.dumps(
            {
                "status": "built",
                "dataset_name": dataset_name,
                "rows": int(len(frame)),
                "unique_smiles": int(len(smiles)),
                "processed_path": str(processed_path),
                "size_bytes": processed_path.stat().st_size,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

