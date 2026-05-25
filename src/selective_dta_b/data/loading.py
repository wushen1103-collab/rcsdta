from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from selective_dta_b.data.graph import smiles_to_pyg_graph


def resolve_split_path(workspace: str | Path, dataset_name: str, split_name: str, seed: int) -> Path:
    workspace_path = Path(workspace)
    return workspace_path / "data" / "processed" / dataset_name / "splits" / f"{split_name}_seed{seed}.csv"


def load_split_frame(workspace: str | Path, dataset_name: str, split_name: str, seed: int) -> pd.DataFrame:
    split_path = resolve_split_path(
        workspace=workspace,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=seed,
    )
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    frame = pd.read_csv(split_path)
    if "split" not in frame.columns:
        raise ValueError(f"Split file missing 'split' column: {split_path}")
    return frame


class SelectiveDTADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        target_column: str = "affinity_model_target",
        include_drug_graph: bool = False,
    ) -> None:
        if target_column not in frame.columns:
            raise KeyError(f"Target column not found: {target_column}")
        self.frame = frame.reset_index(drop=True).copy()
        self.target_column = target_column
        self.include_drug_graph = include_drug_graph
        self._graph_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        item = row.to_dict()
        item["target"] = float(row[self.target_column])
        if self.target_column == "affinity_model_target" and "affinity_model_target_name" in row.index:
            item["target_name"] = str(row["affinity_model_target_name"])
        else:
            item["target_name"] = self.target_column
        if self.include_drug_graph:
            smiles = str(row["drug_smiles"])
            if smiles not in self._graph_cache:
                self._graph_cache[smiles] = smiles_to_pyg_graph(smiles)
            item["drug_graph"] = self._graph_cache[smiles]
        return item


def collate_selective_dta_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key == "target":
            collated[key] = torch.tensor(values, dtype=torch.float32)
        elif key == "drug_graph":
            collated[key] = Batch.from_data_list(values)
        elif all(isinstance(value, bool) for value in values):
            collated[key] = torch.tensor(values, dtype=torch.bool)
        else:
            collated[key] = values
    return collated


class SelectiveDTADataModule(L.LightningDataModule):
    def __init__(
        self,
        workspace: str | Path,
        dataset_name: str,
        split_name: str,
        seed: int = 42,
        target_column: str = "affinity_model_target",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last_train: bool = False,
        include_drug_graph: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.workspace = Path(workspace)
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.seed = seed
        self.target_column = target_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train
        self.include_drug_graph = include_drug_graph
        self.train_dataset: SelectiveDTADataset | None = None
        self.val_dataset: SelectiveDTADataset | None = None
        self.test_dataset: SelectiveDTADataset | None = None

    def setup(self, stage: str | None = None) -> None:
        frame = load_split_frame(
            workspace=self.workspace,
            dataset_name=self.dataset_name,
            split_name=self.split_name,
            seed=self.seed,
        )
        self.train_dataset = SelectiveDTADataset(
            frame.loc[frame["split"] == "train"].reset_index(drop=True),
            target_column=self.target_column,
            include_drug_graph=self.include_drug_graph,
        )
        self.val_dataset = SelectiveDTADataset(
            frame.loc[frame["split"] == "val"].reset_index(drop=True),
            target_column=self.target_column,
            include_drug_graph=self.include_drug_graph,
        )
        self.test_dataset = SelectiveDTADataset(
            frame.loc[frame["split"] == "test"].reset_index(drop=True),
            target_column=self.target_column,
            include_drug_graph=self.include_drug_graph,
        )

    def _build_dataloader(
        self,
        dataset: SelectiveDTADataset | None,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting dataloaders")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=collate_selective_dta_batch,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.train_dataset, shuffle=True, drop_last=self.drop_last_train)

    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.test_dataset, shuffle=False, drop_last=False)


__all__ = [
    "SelectiveDTADataset",
    "SelectiveDTADataModule",
    "collate_selective_dta_batch",
    "load_split_frame",
    "resolve_split_path",
]

