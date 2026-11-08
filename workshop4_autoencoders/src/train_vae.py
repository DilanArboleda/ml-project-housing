"""Training utilities for beta-VAE experiments on Ames Housing."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import VariationalAutoencoder


@dataclass(frozen=True)
class VAETrainingConfig:
    target_beta: float
    latent_dim: int = 64
    hidden_dims: tuple[int, int] = (192, 128)
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 50
    min_epochs: int = 30
    patience: int = 10
    warmup_epochs: int = 10
    seed: int = 42

    def validate(self) -> None:
        if self.target_beta < 0:
            raise ValueError("target_beta must be non-negative")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.max_epochs < self.min_epochs:
            raise ValueError("max_epochs must be at least min_epochs")
        if self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("patience and batch_size must be positive")
        if self.warmup_epochs <= 0:
            raise ValueError("warmup_epochs must be positive")


@dataclass
class VAETrainingResult:
    config: VAETrainingConfig
    parameter_count: int
    best_epoch: int
    best_validation_metrics: dict[str, float | int]
    training_time_seconds: float
    checkpoint_path: Path
    history: list[dict[str, float | int]]
    checkpoint_reload_exact: bool
    posterior_collapse_detected: bool


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


def effective_beta(target_beta: float, epoch: int, warmup_epochs: int) -> float:
    """Linearly increase beta to its target during the warm-up period."""
    return float(target_beta * min(epoch / warmup_epochs, 1.0))


def vae_loss_components(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Return consistently normalized reconstruction and KL components.

    Reconstruction MSE is averaged over observations and input features. The
    true Gaussian KL is summed over latent dimensions and averaged over
    observations. For optimization, KL is divided by latent dimension so the
    beta-weighted term is on a per-latent scale comparable across experiments.
    The resulting total objective is not labelled as an ELBO.
    """
    reconstruction_mse = torch.mean((reconstruction - target) ** 2)
    kl_elements = -0.5 * (
        1.0 + logvar - mu.pow(2) - torch.exp(logvar)
    )
    kl_per_observation = kl_elements.sum(dim=1).mean()
    kl_per_latent = kl_elements.mean()
    total_objective = reconstruction_mse + beta * kl_per_latent
    return {
        "reconstruction_mse": reconstruction_mse,
        "kl_per_observation": kl_per_observation,
        "kl_per_latent": kl_per_latent,
        "total_objective": total_objective,
    }


def evaluate_vae(
    model: VariationalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict[str, float | int]:
    """Evaluate deterministic posterior-mean reconstructions."""
    model.eval()
    total_squared_error = 0.0
    total_values = 0
    total_kl = 0.0
    total_observations = 0
    posterior_means = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            reconstruction = model.decode(mu)
            kl_elements = -0.5 * (
                1.0 + logvar - mu.pow(2) - torch.exp(logvar)
            )
            total_squared_error += torch.sum((reconstruction - batch) ** 2).item()
            total_values += batch.numel()
            total_kl += kl_elements.sum().item()
            total_observations += len(batch)
            posterior_means.append(mu.cpu())

    reconstruction_mse = total_squared_error / total_values
    kl_per_observation = total_kl / total_observations
    kl_per_latent = kl_per_observation / model.latent_dim
    means = torch.cat(posterior_means, dim=0)
    latent_variance = torch.var(means, dim=0, unbiased=False)
    active_units = int(torch.sum(latent_variance > 1e-2).item())
    return {
        "reconstruction_mse": float(reconstruction_mse),
        "kl_per_observation": float(kl_per_observation),
        "kl_per_latent": float(kl_per_latent),
        "total_objective": float(reconstruction_mse + beta * kl_per_latent),
        "active_units": active_units,
        "mean_latent_variance": float(latent_variance.mean().item()),
    }


def save_vae_checkpoint(
    path: Path,
    model: VariationalAutoencoder,
    config: VAETrainingConfig,
    data_metadata: dict[str, Any],
    best_epoch: int,
    best_validation_metrics: dict[str, float | int],
    training_time_seconds: float,
    posterior_collapse_detected: bool,
    collapse_diagnostics: dict[str, float | int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_type": "variational_autoencoder",
            "model_state_dict": state_dict,
            "architecture": {
                "input_dim": model.input_dim,
                "hidden_dims": list(model.hidden_dims),
                "latent_dim": model.latent_dim,
                "posterior": "diagonal Gaussian",
                "output_activation": "linear",
            },
            "training_config": asdict(config),
            "loss_definition": {
                "reconstruction": (
                    "MSE averaged over observations and input features"
                ),
                "kl_reported": (
                    "Gaussian KL summed over latent dimensions and averaged "
                    "over observations"
                ),
                "kl_in_objective": (
                    "reported KL divided by latent dimension"
                ),
                "objective": (
                    "reconstruction_mse + effective_beta * kl_per_latent"
                ),
                "is_exact_elbo": False,
            },
            "warmup": {
                "type": "linear",
                "epochs": config.warmup_epochs,
                "target_beta": config.target_beta,
            },
            "parameter_count": model.trainable_parameters,
            "best_epoch": int(best_epoch),
            "best_validation_metrics": best_validation_metrics,
            "training_time_seconds": float(training_time_seconds),
            "posterior_collapse_detected": posterior_collapse_detected,
            "collapse_diagnostics": collapse_diagnostics or {},
            "selection_partition": "validation",
            "selection_metric": "minimum validation reconstruction MSE",
            "test_partition_used": False,
            "data_metadata": data_metadata,
        },
        path,
    )


def load_vae_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[VariationalAutoencoder, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    architecture = payload["architecture"]
    model = VariationalAutoencoder(
        input_dim=int(architecture["input_dim"]),
        latent_dim=int(architecture["latent_dim"]),
        hidden_dims=tuple(architecture["hidden_dims"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, payload


def train_vae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    data_metadata: dict[str, Any],
    config: VAETrainingConfig,
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> VAETrainingResult:
    """Train one beta-VAE and select it using validation reconstruction MSE."""
    config.validate()
    set_seed(config.seed)
    device = torch.device(device)

    train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    val_tensor = torch.as_tensor(X_val, dtype=torch.float32)
    generator = torch.Generator().manual_seed(config.seed)
    training_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    training_evaluation_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=config.batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = VariationalAutoencoder(
        input_dim=X_train.shape[1],
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_validation_mse = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, float | int] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        beta = effective_beta(config.target_beta, epoch, config.warmup_epochs)
        model.train()
        for (batch,) in training_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(batch)
            components = vae_loss_components(
                reconstruction, batch, mu, logvar, beta
            )
            components["total_objective"].backward()
            optimizer.step()

        train_metrics = evaluate_vae(
            model, training_evaluation_loader, device, beta
        )
        validation_metrics = evaluate_vae(
            model, validation_loader, device, beta
        )
        history.append(
            {
                "epoch": epoch,
                "effective_beta": beta,
                "train_reconstruction_mse": train_metrics["reconstruction_mse"],
                "validation_reconstruction_mse": validation_metrics[
                    "reconstruction_mse"
                ],
                "train_kl_per_observation": train_metrics[
                    "kl_per_observation"
                ],
                "validation_kl_per_observation": validation_metrics[
                    "kl_per_observation"
                ],
                "train_kl_per_latent": train_metrics["kl_per_latent"],
                "validation_kl_per_latent": validation_metrics[
                    "kl_per_latent"
                ],
                "train_total_objective": train_metrics["total_objective"],
                "validation_total_objective": validation_metrics[
                    "total_objective"
                ],
                "validation_active_units": validation_metrics["active_units"],
                "validation_mean_latent_variance": validation_metrics[
                    "mean_latent_variance"
                ],
            }
        )

        current_validation_mse = float(
            validation_metrics["reconstruction_mse"]
        )
        # Checkpoint selection starts only after the required minimum training
        # period, when all runs have already reached their target beta.
        if (
            epoch >= config.min_epochs
            and current_validation_mse < best_validation_mse
        ):
            best_validation_mse = current_validation_mse
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_validation_metrics = dict(validation_metrics)
            best_validation_metrics["effective_beta"] = beta
            epochs_without_improvement = 0
        elif epoch >= config.min_epochs:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == config.max_epochs:
            print(
                f"beta={config.target_beta:g} epoch={epoch:02d} "
                f"effective_beta={beta:.3f} "
                f"val_recon={current_validation_mse:.6f} "
                f"val_kl={validation_metrics['kl_per_observation']:.4f} "
                f"active={validation_metrics['active_units']}"
            )

        if (
            epoch >= config.min_epochs
            and epochs_without_improvement >= config.patience
        ):
            print(
                f"Early stopping for beta={config.target_beta:g} at epoch {epoch}."
            )
            break

    training_time_seconds = time.perf_counter() - started
    if best_state_dict is None or best_validation_metrics is None:
        raise RuntimeError("Training did not produce a valid VAE checkpoint")

    model.load_state_dict(best_state_dict)
    model.to("cpu")
    model.eval()
    full_beta_history = [
        row for row in history if int(row["epoch"]) >= config.warmup_epochs
    ]
    final_metrics = full_beta_history[-1]
    minimum_active_units = min(
        int(row["validation_active_units"]) for row in full_beta_history
    )
    active_unit_threshold = max(1, int(np.ceil(0.1 * config.latent_dim)))
    collapse_diagnostics = {
        "active_unit_threshold": active_unit_threshold,
        "minimum_active_units_at_full_beta": minimum_active_units,
        "final_active_units": int(final_metrics["validation_active_units"]),
        "final_kl_per_latent": float(
            final_metrics["validation_kl_per_latent"]
        ),
    }
    posterior_collapse_detected = bool(
        collapse_diagnostics["final_kl_per_latent"] < 1e-3
        or collapse_diagnostics["final_active_units"] < active_unit_threshold
    )
    save_vae_checkpoint(
        checkpoint_path,
        model,
        config,
        data_metadata,
        best_epoch,
        best_validation_metrics,
        training_time_seconds,
        posterior_collapse_detected,
        collapse_diagnostics,
    )

    reference_batch = val_tensor[: min(32, len(val_tensor))]
    with torch.no_grad():
        reference_reconstruction = model.reconstruct_with_mean(reference_batch)
    loaded_model, _ = load_vae_checkpoint(checkpoint_path)
    with torch.no_grad():
        loaded_reconstruction = loaded_model.reconstruct_with_mean(reference_batch)
    checkpoint_reload_exact = torch.equal(
        reference_reconstruction, loaded_reconstruction
    )
    if not checkpoint_reload_exact:
        raise AssertionError("Checkpoint reload changed deterministic reconstructions")

    return VAETrainingResult(
        config=config,
        parameter_count=model.trainable_parameters,
        best_epoch=best_epoch,
        best_validation_metrics=best_validation_metrics,
        training_time_seconds=training_time_seconds,
        checkpoint_path=checkpoint_path,
        history=history,
        checkpoint_reload_exact=checkpoint_reload_exact,
        posterior_collapse_detected=posterior_collapse_detected,
    )
