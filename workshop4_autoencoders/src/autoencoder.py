"""Fully connected autoencoder for transformed Ames Housing features."""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Symmetric MLP autoencoder with a linear latent code and output."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, int] = (192, 128),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        if len(hidden_dims) != 2 or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain two positive widths")

        first, second = hidden_dims
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, second),
            nn.ReLU(),
            nn.Linear(second, first),
            nn.ReLU(),
            nn.Linear(first, self.input_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        encoder_linears = [
            layer for layer in self.encoder if isinstance(layer, nn.Linear)
        ]
        decoder_linears = [
            layer for layer in self.decoder if isinstance(layer, nn.Linear)
        ]
        for layer in encoder_linears[:-1] + decoder_linears[:-1]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(encoder_linears[-1].weight)
        nn.init.zeros_(encoder_linears[-1].bias)
        nn.init.xavier_uniform_(decoder_linears[-1].weight)
        nn.init.zeros_(decoder_linears[-1].bias)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(values))

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class VariationalAutoencoder(nn.Module):
    """Variational autoencoder with a Gaussian 64-dimensional latent space."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: tuple[int, int] = (192, 128),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        if len(hidden_dims) != 2 or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain two positive widths")

        first, second = hidden_dims
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)

        self.encoder_backbone = nn.Sequential(
            nn.Linear(self.input_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(second, self.latent_dim)
        self.logvar_head = nn.Linear(second, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, second),
            nn.ReLU(),
            nn.Linear(second, first),
            nn.ReLU(),
            nn.Linear(first, self.input_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        hidden_linears = [
            layer
            for layer in list(self.encoder_backbone) + list(self.decoder[:-1])
            if isinstance(layer, nn.Linear)
        ]
        for layer in hidden_linears:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        for head in (self.mu_head, self.logvar_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

        output_layer = self.decoder[-1]
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder_backbone(values)
        mu = self.mu_head(hidden)
        logvar = torch.clamp(self.logvar_head(hidden), min=-20.0, max=10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn_like(standard_deviation)
        return mu + standard_deviation * noise

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def reconstruct_with_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Return a deterministic reconstruction using posterior means."""
        mu, _ = self.encode(values)
        return self.decode(mu)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(values)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


if __name__ == "__main__":
    model = Autoencoder(input_dim=262, latent_dim=64)
    sample = torch.randn(8, 262)
    print("Latent shape:", tuple(model.encode(sample).shape))
    print("Reconstruction shape:", tuple(model(sample).shape))
    print("Trainable parameters:", model.trainable_parameters)
