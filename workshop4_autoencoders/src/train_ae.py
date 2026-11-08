"""Controlled training utilities for the standard autoencoder experiments."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import Autoencoder


@dataclass(frozen=True)
class AETrainingConfig:
    latent_dim: int
    hidden_dims: tuple[int, int] = (192, 128)
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 50
    min_epochs: int = 30
    patience: int = 10
    seed: int = 42

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.max_epochs < self.min_epochs:
            raise ValueError("max_epochs must be at least min_epochs")
        if self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("patience and batch_size must be positive")


@dataclass
class AETrainingResult:
    config: AETrainingConfig
    parameter_count: int
    best_epoch: int
    best_validation_mse: float
    training_time_seconds: float
    checkpoint_path: Path
    history: list[dict[str, float]]
    checkpoint_reload_exact: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _mse_on_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total_squared_error = 0.0
    total_values = 0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            total_squared_error += torch.sum((reconstruction - batch) ** 2).item()
            total_values += batch.numel()
    return total_squared_error / total_values


def save_autoencoder_checkpoint(
    path: Path,
    model: Autoencoder,
    config: AETrainingConfig,
    data_metadata: dict[str, Any],
    best_epoch: int,
    best_validation_mse: float,
    training_time_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_type": "standard_autoencoder",
            "model_state_dict": state_dict,
            "architecture": {
                "input_dim": model.input_dim,
                "hidden_dims": list(model.hidden_dims),
                "latent_dim": model.latent_dim,
                "output_activation": "linear",
            },
            "training_config": asdict(config),
            "reconstruction_loss": "mean_squared_error",
            "parameter_count": model.trainable_parameters,
            "best_epoch": int(best_epoch),
            "best_validation_mse": float(best_validation_mse),
            "training_time_seconds": float(training_time_seconds),
            "selection_partition": "validation",
            "test_partition_used": False,
            "data_metadata": data_metadata,
        },
        path,
    )


def load_autoencoder_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[Autoencoder, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    architecture = payload["architecture"]
    model = Autoencoder(
        input_dim=int(architecture["input_dim"]),
        latent_dim=int(architecture["latent_dim"]),
        hidden_dims=tuple(architecture["hidden_dims"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, payload


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    data_metadata: dict[str, Any],
    config: AETrainingConfig,
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> AETrainingResult:
    """Train one AE and select its checkpoint using validation MSE only."""
    config.validate()
    set_seed(config.seed)
    device = torch.device(device)

    train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    val_tensor = torch.as_tensor(X_val, dtype=torch.float32)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = Autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss(reduction="mean")

    best_validation_mse = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_squared_error = 0.0
        total_values = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_squared_error += torch.sum((reconstruction - batch) ** 2).item()
            total_values += batch.numel()

        train_mse = total_squared_error / total_values
        validation_mse = _mse_on_loader(model, validation_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": float(train_mse),
                "validation_mse": float(validation_mse),
            }
        )

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        elif epoch >= config.min_epochs:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == config.max_epochs:
            print(
                f"latent={config.latent_dim:3d} epoch={epoch:02d} "
                f"train_mse={train_mse:.6f} val_mse={validation_mse:.6f}"
            )

        if (
            epoch >= config.min_epochs
            and epochs_without_improvement >= config.patience
        ):
            print(
                f"Early stopping for latent={config.latent_dim} at epoch {epoch}."
            )
            break

    training_time_seconds = time.perf_counter() - started
    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state_dict)
    model.to("cpu")
    model.eval()
    save_autoencoder_checkpoint(
        checkpoint_path,
        model,
        config,
        data_metadata,
        best_epoch,
        best_validation_mse,
        training_time_seconds,
    )

    reference_batch = val_tensor[: min(32, len(val_tensor))]
    with torch.no_grad():
        reference_reconstruction = model(reference_batch)
    loaded_model, _ = load_autoencoder_checkpoint(checkpoint_path)
    with torch.no_grad():
        loaded_reconstruction = loaded_model(reference_batch)
    checkpoint_reload_exact = torch.equal(
        reference_reconstruction, loaded_reconstruction
    )
    if not checkpoint_reload_exact:
        raise AssertionError("Checkpoint reload changed autoencoder reconstructions")

    return AETrainingResult(
        config=config,
        parameter_count=model.trainable_parameters,
        best_epoch=best_epoch,
        best_validation_mse=best_validation_mse,
        training_time_seconds=training_time_seconds,
        checkpoint_path=checkpoint_path,
        history=history,
        checkpoint_reload_exact=checkpoint_reload_exact,
    )
