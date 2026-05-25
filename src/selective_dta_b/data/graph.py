from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data


def _atom_features(atom: Chem.Atom) -> list[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(int(atom.GetIsAromatic())),
    ]


def smiles_to_pyg_graph(smiles: str) -> Data:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        molecule = Chem.MolFromSmiles("C")

    node_features = torch.tensor(
        [_atom_features(atom) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )
    edges: list[list[int]] = []
    for bond in molecule.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edges.append([start, end])
        edges.append([end, start])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=node_features, edge_index=edge_index)


__all__ = ["smiles_to_pyg_graph"]

