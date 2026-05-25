from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from selective_dta_b.models.char_baseline import (
    PROTEIN_ALPHABET,
    SMILES_ALPHABET,
    build_character_vocab,
    tokenize_character_sequences,
)


class DeepDTASequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        char_embed_dim: int,
        encoder_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.Sequential(
            nn.Conv1d(char_embed_dim, encoder_dim, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=6, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        encoded = self.convs(embedded)
        return self.pool(encoded).squeeze(-1)


class DeepDTDAModule(L.LightningModule):
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

        self.drug_encoder = DeepDTASequenceEncoder(
            vocab_size=len(self.drug_vocab),
            char_embed_dim=char_embed_dim,
            encoder_dim=encoder_dim,
            dropout=dropout,
        )
        self.protein_encoder = DeepDTASequenceEncoder(
            vocab_size=len(self.protein_vocab),
            char_embed_dim=char_embed_dim,
            encoder_dim=encoder_dim,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        device = self.device
        drug_ids, _ = tokenize_character_sequences(
            sequences=list(batch["drug_smiles"]),
            vocab=self.drug_vocab,
            max_length=self.hparams.drug_max_length,
            device=device,
        )
        protein_ids, _ = tokenize_character_sequences(
            sequences=list(batch["target_sequence"]),
            vocab=self.protein_vocab,
            max_length=self.hparams.protein_max_length,
            device=device,
        )

        drug_features = self.drug_encoder(drug_ids)
        protein_features = self.protein_encoder(protein_ids)
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

