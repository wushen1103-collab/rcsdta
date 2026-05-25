from __future__ import annotations

from typing import Iterable

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SMILES_ALPHABET = "#%()+-./0123456789:=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz\\"
PROTEIN_ALPHABET = "ABCDEFGHIKLMNPQRSTVWXYZOUBJ*-"


def build_character_vocab(alphabet: str | Iterable[str]) -> dict[str, int]:
    symbols = list(dict.fromkeys(alphabet))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for symbol in symbols:
        if symbol not in vocab:
            vocab[symbol] = len(vocab)
    return vocab


def tokenize_character_sequences(
    sequences: list[str],
    vocab: dict[str, int],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.full((len(sequences), max_length), vocab[PAD_TOKEN], dtype=torch.long, device=device)
    token_mask = torch.zeros((len(sequences), max_length), dtype=torch.bool, device=device)

    for row_index, sequence in enumerate(sequences):
        clipped = str(sequence)[:max_length]
        if not clipped:
            continue
        encoded = [vocab.get(symbol, vocab[UNK_TOKEN]) for symbol in clipped]
        token_ids[row_index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=device)
        token_mask[row_index, : len(encoded)] = True

    return token_ids, token_mask


class MaskedMeanSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, char_embed_dim: int, encoder_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(char_embed_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        mask = token_mask.unsqueeze(-1).float()
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)


class CharBaselineDTAModule(L.LightningModule):
    def __init__(
        self,
        drug_max_length: int = 128,
        protein_max_length: int = 512,
        char_embed_dim: int = 64,
        encoder_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.drug_vocab = build_character_vocab(SMILES_ALPHABET)
        self.protein_vocab = build_character_vocab(PROTEIN_ALPHABET)

        self.drug_encoder = MaskedMeanSequenceEncoder(
            vocab_size=len(self.drug_vocab),
            char_embed_dim=char_embed_dim,
            encoder_dim=encoder_dim,
            dropout=dropout,
        )
        self.protein_encoder = MaskedMeanSequenceEncoder(
            vocab_size=len(self.protein_vocab),
            char_embed_dim=char_embed_dim,
            encoder_dim=encoder_dim,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        device = self.device
        drug_ids, drug_mask = tokenize_character_sequences(
            sequences=list(batch["drug_smiles"]),
            vocab=self.drug_vocab,
            max_length=self.hparams.drug_max_length,
            device=device,
        )
        protein_ids, protein_mask = tokenize_character_sequences(
            sequences=list(batch["target_sequence"]),
            vocab=self.protein_vocab,
            max_length=self.hparams.protein_max_length,
            device=device,
        )

        drug_features = self.drug_encoder(drug_ids, drug_mask)
        protein_features = self.protein_encoder(protein_ids, protein_mask)
        predictions = self.regressor(torch.cat([drug_features, protein_features], dim=-1))
        return predictions.squeeze(-1)

    def _shared_step(self, batch: dict[str, object], stage: str) -> torch.Tensor:
        targets = batch["target"].to(self.device)
        predictions = self(batch)
        mse_loss = F.mse_loss(predictions, targets)
        mae = F.l1_loss(predictions, targets)
        rmse = torch.sqrt(mse_loss)

        self.log(f"{stage}_loss", mse_loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True, batch_size=targets.shape[0])
        self.log(f"{stage}_mae", mae, prog_bar=False, on_step=False, on_epoch=True, batch_size=targets.shape[0])
        self.log(f"{stage}_rmse", rmse, prog_bar=(stage != "train"), on_step=False, on_epoch=True, batch_size=targets.shape[0])
        return mse_loss

    def training_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def predict_step(self, batch: dict[str, object], batch_idx: int) -> dict[str, object]:
        predictions = self(batch)
        return {
            "row_id": batch["row_id"],
            "prediction": predictions.detach().cpu(),
            "target": batch["target"].detach().cpu(),
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
