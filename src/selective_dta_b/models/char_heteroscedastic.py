from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from selective_dta_b.models.char_baseline import (
    MaskedMeanSequenceEncoder,
    PROTEIN_ALPHABET,
    SMILES_ALPHABET,
    build_character_vocab,
    tokenize_character_sequences,
)


class CharHeteroscedasticDTAModule(L.LightningModule):
    supports_aleatoric_uncertainty = True

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
        min_log_variance: float = -6.0,
        max_log_variance: float = 4.0,
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
        self.trunk = nn.Sequential(
            nn.Linear(encoder_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_variance_head = nn.Linear(hidden_dim, 1)

    def _encode_batch(self, batch: dict[str, object]) -> torch.Tensor:
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
        return self.trunk(torch.cat([drug_features, protein_features], dim=-1))

    def predict_distribution(self, batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode_batch(batch)
        prediction_mean = self.mean_head(hidden).squeeze(-1)
        log_variance = self.log_variance_head(hidden).squeeze(-1)
        log_variance = torch.clamp(
            log_variance,
            min=self.hparams.min_log_variance,
            max=self.hparams.max_log_variance,
        )
        return prediction_mean, log_variance

    def predict_with_uncertainty(self, batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_mean, log_variance = self.predict_distribution(batch)
        prediction_std = torch.exp(0.5 * log_variance)
        return prediction_mean, prediction_std

    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        prediction_mean, _ = self.predict_distribution(batch)
        return prediction_mean

    def _shared_step(self, batch: dict[str, object], stage: str) -> torch.Tensor:
        targets = batch["target"].to(self.device)
        prediction_mean, log_variance = self.predict_distribution(batch)
        mse_loss = F.mse_loss(prediction_mean, targets)
        mae = F.l1_loss(prediction_mean, targets)
        rmse = torch.sqrt(mse_loss)
        gaussian_nll = 0.5 * (torch.exp(-log_variance) * (targets - prediction_mean) ** 2 + log_variance)
        loss = gaussian_nll.mean()

        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True, batch_size=targets.shape[0])
        self.log(f"{stage}_mae", mae, prog_bar=False, on_step=False, on_epoch=True, batch_size=targets.shape[0])
        self.log(f"{stage}_rmse", rmse, prog_bar=(stage != "train"), on_step=False, on_epoch=True, batch_size=targets.shape[0])
        return loss

    def training_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def predict_step(self, batch: dict[str, object], batch_idx: int) -> dict[str, object]:
        prediction_mean, prediction_std = self.predict_with_uncertainty(batch)
        return {
            "row_id": batch["row_id"],
            "prediction": prediction_mean.detach().cpu(),
            "prediction_std": prediction_std.detach().cpu(),
            "target": batch["target"].detach().cpu(),
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

