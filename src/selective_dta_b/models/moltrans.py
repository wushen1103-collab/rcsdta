from __future__ import annotations

import math

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


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_length, embedding_dim)

    def forward(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device)
        return self.embedding(positions)


class MolTransSequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = LearnedPositionalEncoding(max_length=max_length, embedding_dim=embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]
        token_features = self.token_embedding(token_ids)
        position_features = self.position_embedding(sequence_length=sequence_length, device=token_ids.device)
        encoded = token_features + position_features.unsqueeze(0)
        encoded = self.dropout(encoded)
        return self.encoder(encoded, src_key_padding_mask=~token_mask)


class MolTransInteractionHead(nn.Module):
    def __init__(self, embedding_dim: int, interaction_dim: int, dropout: float) -> None:
        super().__init__()
        self.drug_projection = nn.Linear(embedding_dim, interaction_dim)
        self.protein_projection = nn.Linear(embedding_dim, interaction_dim)
        self.interaction_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AdaptiveMaxPool2d((4, 4)),
        )

    def forward(
        self,
        *,
        drug_encoded: torch.Tensor,
        protein_encoded: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        drug_features = self.drug_projection(drug_encoded)
        protein_features = self.protein_projection(protein_encoded)
        interaction_map = torch.einsum("bid,bjd->bij", drug_features, protein_features) / math.sqrt(drug_features.shape[-1])
        pair_mask = drug_mask.unsqueeze(2) & protein_mask.unsqueeze(1)
        interaction_map = interaction_map.masked_fill(~pair_mask, 0.0)
        interaction_features = self.interaction_conv(interaction_map.unsqueeze(1))
        return interaction_features.flatten(start_dim=1)


class MolTransDTAModule(L.LightningModule):
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
        num_heads = 4 if encoder_dim >= 64 else 2
        interaction_dim = max(16, encoder_dim // 4)

        self.drug_encoder = MolTransSequenceEncoder(
            vocab_size=len(self.drug_vocab),
            max_length=drug_max_length,
            embedding_dim=encoder_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout,
        )
        self.protein_encoder = MolTransSequenceEncoder(
            vocab_size=len(self.protein_vocab),
            max_length=protein_max_length,
            embedding_dim=encoder_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout,
        )
        self.interaction_head = MolTransInteractionHead(
            embedding_dim=encoder_dim,
            interaction_dim=interaction_dim,
            dropout=dropout,
        )
        interaction_feature_dim = 8 * 4 * 4
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim * 2 + interaction_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean_pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

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

        drug_encoded = self.drug_encoder(drug_ids, drug_mask)
        protein_encoded = self.protein_encoder(protein_ids, protein_mask)
        interaction_features = self.interaction_head(
            drug_encoded=drug_encoded,
            protein_encoded=protein_encoded,
            drug_mask=drug_mask,
            protein_mask=protein_mask,
        )
        drug_pooled = self._masked_mean_pool(drug_encoded, drug_mask)
        protein_pooled = self._masked_mean_pool(protein_encoded, protein_mask)
        predictions = self.regressor(torch.cat([drug_pooled, protein_pooled, interaction_features], dim=-1))
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

