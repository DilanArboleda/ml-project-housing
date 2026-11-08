"""Tests for VAE architecture, loss scaling, and checkpoint recovery."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
WORKSHOP_SRC = ROOT / "workshop4_autoencoders" / "src"
sys.path.insert(0, str(WORKSHOP_SRC))

from autoencoder import VariationalAutoencoder  # noqa: E402
from train_vae import (  # noqa: E402
    VAETrainingConfig,
    effective_beta,
    load_vae_checkpoint,
    save_vae_checkpoint,
    vae_loss_components,
)


class VariationalAutoencoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_mu_logvar_reconstruction_shapes(self):
        model = VariationalAutoencoder(input_dim=262, latent_dim=64)
        batch = torch.randn(8, 262)
        mu, logvar = model.encode(batch)
        reconstruction, forward_mu, forward_logvar = model(batch)
        self.assertEqual(tuple(mu.shape), (8, 64))
        self.assertEqual(tuple(logvar.shape), (8, 64))
        self.assertEqual(tuple(reconstruction.shape), (8, 262))
        self.assertEqual(tuple(forward_mu.shape), (8, 64))
        self.assertEqual(tuple(forward_logvar.shape), (8, 64))

    def test_reparameterization_is_differentiable(self):
        mu = torch.zeros(5, 64, requires_grad=True)
        logvar = torch.zeros(5, 64, requires_grad=True)
        latent = VariationalAutoencoder.reparameterize(mu, logvar)
        self.assertEqual(tuple(latent.shape), (5, 64))
        self.assertFalse(torch.equal(latent, mu))
        latent.sum().backward()
        self.assertTrue(torch.isfinite(mu.grad).all())
        self.assertTrue(torch.isfinite(logvar.grad).all())

    def test_kl_is_finite_and_non_negative(self):
        target = torch.randn(7, 262)
        reconstruction = target.clone()
        mu = torch.randn(7, 64) * 0.1
        logvar = torch.randn(7, 64) * 0.1
        components = vae_loss_components(
            reconstruction, target, mu, logvar, beta=1.0
        )
        for value in components.values():
            self.assertTrue(torch.isfinite(value))
        self.assertGreaterEqual(components["kl_per_observation"].item(), -1e-7)
        self.assertGreaterEqual(components["kl_per_latent"].item(), -1e-7)

        zero_kl = vae_loss_components(
            reconstruction,
            target,
            torch.zeros_like(mu),
            torch.zeros_like(logvar),
            beta=4.0,
        )
        self.assertAlmostEqual(zero_kl["kl_per_observation"].item(), 0.0)
        self.assertAlmostEqual(zero_kl["total_objective"].item(), 0.0)

    def test_forward_backward_has_finite_values(self):
        model = VariationalAutoencoder(input_dim=262, latent_dim=64)
        batch = torch.randn(16, 262)
        reconstruction, mu, logvar = model(batch)
        components = vae_loss_components(
            reconstruction, batch, mu, logvar, beta=1.0
        )
        components["total_objective"].backward()
        self.assertTrue(torch.isfinite(reconstruction).all())
        self.assertTrue(torch.isfinite(mu).all())
        self.assertTrue(torch.isfinite(logvar).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_mean_reconstruction_is_deterministic(self):
        model = VariationalAutoencoder(input_dim=262, latent_dim=64).eval()
        batch = torch.randn(10, 262)
        with torch.no_grad():
            first = model.reconstruct_with_mean(batch)
            second = model.reconstruct_with_mean(batch)
        self.assertTrue(torch.equal(first, second))

    def test_linear_beta_warmup(self):
        self.assertAlmostEqual(effective_beta(4.0, 1, 10), 0.4)
        self.assertAlmostEqual(effective_beta(4.0, 5, 10), 2.0)
        self.assertAlmostEqual(effective_beta(4.0, 10, 10), 4.0)
        self.assertAlmostEqual(effective_beta(4.0, 50, 10), 4.0)

    def test_checkpoint_round_trip_is_exact(self):
        model = VariationalAutoencoder(input_dim=262, latent_dim=64).eval()
        config = VAETrainingConfig(target_beta=1.0)
        metadata = {
            "input_dim": 262,
            "feature_names": [f"feature_{index}" for index in range(262)],
            "input_mean": np.zeros(262, dtype=np.float32),
            "input_scale": np.ones(262, dtype=np.float32),
            "split_file": "data/splits.json",
            "split_file_sha256": "test-hash",
        }
        validation_metrics = {
            "reconstruction_mse": 0.5,
            "kl_per_observation": 2.0,
            "kl_per_latent": 0.03125,
            "total_objective": 0.53125,
            "active_units": 20,
            "mean_latent_variance": 0.1,
            "effective_beta": 1.0,
        }
        batch = torch.randn(12, 262)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "vae.pt"
            save_vae_checkpoint(
                checkpoint_path,
                model,
                config,
                metadata,
                best_epoch=9,
                best_validation_metrics=validation_metrics,
                training_time_seconds=1.0,
                posterior_collapse_detected=False,
            )
            loaded, payload = load_vae_checkpoint(checkpoint_path)
            with torch.no_grad():
                expected = model.reconstruct_with_mean(batch)
                actual = loaded.reconstruct_with_mean(batch)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(payload["architecture"]["latent_dim"], 64)
        self.assertEqual(payload["warmup"]["epochs"], 10)
        self.assertFalse(payload["loss_definition"]["is_exact_elbo"])
        self.assertFalse(payload["test_partition_used"])


if __name__ == "__main__":
    unittest.main()
