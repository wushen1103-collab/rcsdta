from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import pickle
from pathlib import Path
import shutil
import sys
import types
from types import ModuleType

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
import torch


@dataclass(frozen=True)
class DeepDTAGenStagedRun:
    external_root: Path
    run_dir: Path
    data_dir: Path
    train_csv: Path
    test_csv: Path
    tokenizer_path: Path
    processed_train_path: Path
    processed_test_path: Path
    checkpoint_path: Path
    summary_path: Path
    predictions_path: Path
    history_path: Path
    train_rows: int
    test_rows: int


SEQ_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {value: index + 1 for index, value in enumerate(SEQ_VOCAB)}
MAX_SEQ_LEN = 1000


def _one_of_k_encoding(value, allowable_set: list) -> list[bool]:
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == item for item in allowable_set]


def _one_of_k_encoding_unk(value, allowable_set: list) -> list[bool]:
    unknown = value not in allowable_set
    if unknown:
        value = allowable_set[-1]
    return [value == item for item in allowable_set] + [unknown]


def _atom_features(atom) -> np.ndarray:
    return np.array(
        _one_of_k_encoding_unk(
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
        + _one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + _one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + _one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + _one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        + _one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        + [atom.GetIsAromatic()]
        + [atom.IsInRing()]
    )


def _bond_features(bond) -> np.ndarray:
    bond_type = bond.GetBondType()
    features = [0, 0, 0, 0, bond.GetBondTypeAsDouble()]
    if bond_type == Chem.rdchem.BondType.SINGLE:
        features = [1, 0, 0, 0, bond.GetBondTypeAsDouble()]
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        features = [0, 1, 0, 0, bond.GetBondTypeAsDouble()]
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        features = [0, 0, 1, 0, bond.GetBondTypeAsDouble()]
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        features = [0, 0, 0, 1, bond.GetBondTypeAsDouble()]
    return np.array(features)


def smile_to_graph(smile: str) -> tuple[int, list[np.ndarray], list[list[int]], list[np.ndarray]]:
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        raise ValueError(f"Unable to parse SMILES string: {smile!r}")

    node_count = molecule.GetNumAtoms()
    node_features = []
    for atom in molecule.GetAtoms():
        feature = _atom_features(atom)
        node_features.append(feature / max(float(feature.sum()), 1.0))

    graph = nx.Graph()
    for bond in molecule.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            edge_feats=_bond_features(bond),
        )

    edge_index: list[list[int]] = []
    edge_features: list[np.ndarray] = []
    for source, target, payload in graph.to_directed().edges(data=True):
        edge_index.append([source, target])
        edge_features.append(payload["edge_feats"])
    return node_count, node_features, edge_index, edge_features


def seq_cat(protein_sequence: str) -> np.ndarray:
    encoded = np.zeros(MAX_SEQ_LEN)
    for index, residue in enumerate(protein_sequence[:MAX_SEQ_LEN]):
        encoded[index] = SEQ_DICT.get(residue, 0)
    return encoded


def ensure_deepdtagen_sequence_capacity(model, model_module: ModuleType, sequence_length: int, device: torch.device | str):
    current_length = int(model.pos_encoding.pe.size(0))
    if sequence_length <= current_length:
        return model
    target_length = max(sequence_length + 8, current_length * 2)
    model.pos_encoding = model_module.PositionalEncoding(model.hidden_dim, max_len=target_length).to(device)
    model.max_len = max(int(getattr(model, "max_len", 0)), target_length)
    return model


def ensure_deepdtagen_state_dict_capacity(
    model,
    model_module: ModuleType,
    *,
    state_dict: dict[str, torch.Tensor],
    sequence_length: int,
    device: torch.device | str,
):
    checkpoint_length = 0
    positional_encoding = state_dict.get("pos_encoding.pe")
    if isinstance(positional_encoding, torch.Tensor):
        checkpoint_length = int(positional_encoding.shape[0])
    current_length = int(model.pos_encoding.pe.size(0))
    if checkpoint_length > current_length:
        model.pos_encoding = model_module.PositionalEncoding(model.hidden_dim, max_len=checkpoint_length).to(device)
        model.max_len = max(int(getattr(model, "max_len", 0)), checkpoint_length)
        current_length = checkpoint_length
    if int(sequence_length) > current_length:
        return ensure_deepdtagen_sequence_capacity(
            model,
            model_module,
            sequence_length=int(sequence_length),
            device=device,
        )
    return model


@contextmanager
def legacy_torch_load_context():
    original_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _module_file(module: ModuleType | None) -> str | None:
    if module is None:
        return None
    return getattr(module, "__file__", None)


def install_fairseq_shim() -> None:
    try:
        import fairseq.models  # type: ignore  # pragma: no cover
        import fairseq.modules  # type: ignore  # pragma: no cover

        return
    except Exception:
        pass

    fairseq_module = types.ModuleType("fairseq")
    fairseq_models = types.ModuleType("fairseq.models")
    fairseq_modules = types.ModuleType("fairseq.modules")

    class FairseqIncrementalDecoder(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class TransformerEncoderLayer(torch.nn.Module):
        def __init__(self, args) -> None:
            super().__init__()
            self.layer = torch.nn.TransformerEncoderLayer(
                d_model=args.encoder_embed_dim,
                nhead=args.encoder_attention_heads,
                dim_feedforward=args.encoder_ffn_embed_dim,
                dropout=args.dropout,
                batch_first=False,
                norm_first=bool(getattr(args, "encoder_normalize_before", False)),
            )

        def forward(self, x, encoder_padding_mask=None):
            return self.layer(x, src_key_padding_mask=encoder_padding_mask)

    class TransformerDecoderLayer(torch.nn.Module):
        def __init__(self, args) -> None:
            super().__init__()
            self.layer = torch.nn.TransformerDecoderLayer(
                d_model=args.decoder_embed_dim,
                nhead=args.decoder_attention_heads,
                dim_feedforward=args.decoder_ffn_embed_dim,
                dropout=args.dropout,
                batch_first=False,
                norm_first=bool(getattr(args, "decoder_normalize_before", False)),
            )

        def forward(
            self,
            x,
            mem,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            encoder_padding_mask=None,
            incremental_state=None,
        ):
            output = self.layer(
                tgt=x,
                memory=mem,
                tgt_mask=self_attn_mask,
                tgt_key_padding_mask=self_attn_padding_mask,
                memory_key_padding_mask=encoder_padding_mask,
            )
            return (output,)

    fairseq_models.FairseqIncrementalDecoder = FairseqIncrementalDecoder
    fairseq_modules.TransformerEncoderLayer = TransformerEncoderLayer
    fairseq_modules.TransformerDecoderLayer = TransformerDecoderLayer
    fairseq_module.models = fairseq_models
    fairseq_module.modules = fairseq_modules

    sys.modules["fairseq"] = fairseq_module
    sys.modules["fairseq.models"] = fairseq_models
    sys.modules["fairseq.modules"] = fairseq_modules


def _load_module(module_name: str, module_path: Path) -> ModuleType:
    existing = sys.modules.get(module_name)
    if _module_file(existing) == str(module_path):
        assert existing is not None
        return existing

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_deepdtagen_modules(external_root: str | Path) -> dict[str, ModuleType]:
    external_root_path = Path(external_root).resolve()
    install_fairseq_shim()
    utils_module = _load_module("utils", external_root_path / "utils.py")
    fettergrad_module = _load_module("FetterGrad", external_root_path / "FetterGrad.py")
    model_module = _load_module("model", external_root_path / "model.py")
    return {
        "utils": utils_module,
        "fettergrad": fettergrad_module,
        "model": model_module,
    }


def stage_deepdtagen_split(
    external_root: str | Path,
    dataset_name: str,
    split_name: str,
    seed: int,
    run_dir: str | Path,
) -> DeepDTAGenStagedRun:
    external_root_path = Path(external_root).resolve()
    run_dir_path = Path(run_dir).resolve()
    split_root = external_root_path / "selective_data" / dataset_name / f"{split_name}_seed{seed}"
    train_source = split_root / "train.csv"
    test_source = split_root / "test.csv"
    if not train_source.exists() or not test_source.exists():
        raise FileNotFoundError(f"Missing DeepDTAGen split files under {split_root}")

    data_dir = run_dir_path / "data"
    checkpoint_dir = run_dir_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_target = data_dir / f"{dataset_name}_train.csv"
    test_target = data_dir / f"{dataset_name}_test.csv"
    shutil.copy2(train_source, train_target)
    shutil.copy2(test_source, test_target)

    train_rows = len(pd.read_csv(train_target))
    test_rows = len(pd.read_csv(test_target))
    return DeepDTAGenStagedRun(
        external_root=external_root_path,
        run_dir=run_dir_path,
        data_dir=data_dir,
        train_csv=train_target,
        test_csv=test_target,
        tokenizer_path=data_dir / f"{dataset_name}_tokenizer.pkl",
        processed_train_path=data_dir / "processed" / f"{dataset_name}_train.pt",
        processed_test_path=data_dir / "processed" / f"{dataset_name}_test.pt",
        checkpoint_path=checkpoint_dir / "model_last.pt",
        summary_path=run_dir_path / "run_summary.json",
        predictions_path=run_dir_path / "test_predictions.csv",
        history_path=run_dir_path / "history.csv",
        train_rows=train_rows,
        test_rows=test_rows,
    )


def prepare_deepdtagen_data(
    staged: DeepDTAGenStagedRun,
    dataset_name: str,
    modules: dict[str, ModuleType],
):
    utils_module = modules["utils"]
    if not staged.tokenizer_path.exists():
        train_df = pd.read_csv(staged.train_csv)
        test_df = pd.read_csv(staged.test_csv)
        all_smiles = set(train_df["target_smiles"]).union(set(test_df["target_smiles"]))
        tokenizer = utils_module.Tokenizer(utils_module.Tokenizer.gen_vocabs(all_smiles))
        with staged.tokenizer_path.open("wb") as handle:
            pickle.dump(tokenizer, handle)
    else:
        with staged.tokenizer_path.open("rb") as handle:
            tokenizer = pickle.load(handle)

    if not staged.processed_train_path.exists() or not staged.processed_test_path.exists():
        train_df = pd.read_csv(staged.train_csv)
        test_df = pd.read_csv(staged.test_csv)
        compound_smiles = set(train_df["compound_iso_smiles"]).union(set(test_df["compound_iso_smiles"]))
        smile_graph = {smile: smile_to_graph(smile) for smile in compound_smiles}

        def _build_dataset(frame: pd.DataFrame, split_label: str):
            proteins = [seq_cat(sequence) for sequence in frame["target_sequence"]]
            tokenized = [torch.LongTensor(tokenizer.parse(smiles)) for smiles in frame["target_smiles"]]
            return utils_module.TestbedDataset(
                root=str(staged.data_dir),
                dataset=f"{dataset_name}_{split_label}",
                xd=frame["compound_iso_smiles"].to_numpy(),
                xdt=tokenized,
                xt=np.asarray(proteins),
                y=frame["affinity"].to_numpy(),
                smile_graph=smile_graph,
            )

        with legacy_torch_load_context():
            _build_dataset(train_df, "train")
            _build_dataset(test_df, "test")

    with legacy_torch_load_context():
        train_data = utils_module.TestbedDataset(root=str(staged.data_dir), dataset=f"{dataset_name}_train")
        test_data = utils_module.TestbedDataset(root=str(staged.data_dir), dataset=f"{dataset_name}_test")
    return tokenizer, train_data, test_data

