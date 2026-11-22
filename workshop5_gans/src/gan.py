"""
Generative Adversarial Network Architectures for Ames Housing Tabular Data.
Workshop No. 5 — Generative Adversarial Networks.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero Ordeñez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)

Implements:
1. TabularGenerator: Maps latent noise z -> synthetic tabular features in [-1, 1] with Tanh.
2. TabularDiscriminator (Vanilla GAN): Binary classifier with Sigmoid output and LeakyReLU.
3. TabularCritic (WGAN-GP): Real-valued critic with LayerNorm (no BatchNorm) and no Sigmoid.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class TabularGenerator(nn.Module):
    """
    Generator Network for tabular real estate attributes.
    Maps latent noise z ~ N(0, I) to a normalized feature vector in [-1, 1].
    """

    def __init__(
        self,
        latent_dim: int = 64,
        output_dim: int = 261,
        hidden_dims: tuple[int, ...] = (128, 256, 256),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = h_dim

        # Final projection to output dimension with Tanh
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class TabularDiscriminator(nn.Module):
    """
    Vanilla GAN Discriminator.
    Maps tabular feature vector x in [-1, 1] to probability P(real) in [0, 1] via Sigmoid.
    """

    def __init__(
        self,
        input_dim: int = 261,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final binary classification layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class TabularCritic(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP) Critic.
    Maps tabular vector x to unconstrained real-valued score.
    Crucial: Uses LayerNorm instead of BatchNorm to satisfy 1-Lipschitz gradient penalty.
    """

    def __init__(
        self,
        input_dim: int = 261,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = h_dim

        # Final scalar score (NO Sigmoid)
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def compute_gradient_penalty(
    critic: TabularCritic,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes gradient penalty for WGAN-GP:
    GP = E[(||grad_{x_hat} D(x_hat)||_2 - 1)^2]
    where x_hat is interpolated uniformly between real and fake samples.
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    critic_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(critic_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
    return gradient_penalty


if __name__ == "__main__":
    device = torch.device("cpu")
    gen = TabularGenerator(latent_dim=64, output_dim=261)
    disc = TabularDiscriminator(input_dim=261)
    critic = TabularCritic(input_dim=261)

    z = torch.randn(16, 64)
    fake_x = gen(z)
    prob_real = disc(fake_x)
    critic_score = critic(fake_x)

    print("TabularGenerator output shape:", fake_x.shape)
    print("TabularDiscriminator output shape:", prob_real.shape)
    print("TabularCritic output shape:", critic_score.shape)
    print("Output min:", fake_x.min().item(), "max:", fake_x.max().item())
