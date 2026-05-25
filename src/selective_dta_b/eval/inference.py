from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn


def resolve_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found under: {checkpoint_dir}")
    non_last = [path for path in checkpoints if path.name != "last.ckpt"]
    return non_last[0] if non_last else checkpoints[0]


def _enable_dropout_only(module: nn.Module) -> None:
    module.eval()
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.train()


def _forward_mean_for_sampling(
    model,
    batch: dict[str, object],
) -> torch.Tensor:
    if hasattr(model, "predict_distribution"):
        prediction_mean, _ = model.predict_distribution(batch)
        return prediction_mean
    return model(batch)


def collect_model_predictions(
    model,
    dataloader,
    *,
    device: torch.device,
    num_mc_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model.to(device)

    with torch.inference_mode():
        for batch in dataloader:
            model.eval()
            if hasattr(model, "predict_with_uncertainty"):
                prediction_mean_device, prediction_std_aleatoric_device = model.predict_with_uncertainty(batch)
                prediction_mean = prediction_mean_device.detach().cpu()
                prediction_std_aleatoric = prediction_std_aleatoric_device.detach().cpu()
            else:
                prediction_mean_device = model(batch)
                prediction_mean = prediction_mean_device.detach().cpu()
                prediction_std_aleatoric = None

            if num_mc_samples <= 1:
                prediction_std_mc = torch.zeros_like(prediction_mean)
            else:
                _enable_dropout_only(model)
                samples = [_forward_mean_for_sampling(model, batch).detach().cpu() for _ in range(num_mc_samples)]
                stacked = torch.stack(samples, dim=0)
                prediction_std_mc = stacked.std(dim=0, unbiased=False)

            targets = batch["target"].detach().cpu()
            for index, row_id in enumerate(batch["row_id"]):
                row = {
                    "row_id": row_id,
                    "prediction_mean": float(prediction_mean[index].item()),
                    "prediction_std": float(prediction_std_mc[index].item()),
                    "prediction_std_mc_dropout": float(prediction_std_mc[index].item()),
                    "target": float(targets[index].item()),
                }
                if prediction_std_aleatoric is not None:
                    row["prediction_std_aleatoric"] = float(prediction_std_aleatoric[index].item())
                rows.append(row)
    model.eval()
    return pd.DataFrame(rows)


__all__ = [
    "collect_model_predictions",
    "resolve_checkpoint_path",
    "resolve_device",
]

