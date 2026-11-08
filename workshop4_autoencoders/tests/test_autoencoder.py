"""Architecture, gradient, determinism, and checkpoint tests for the AE."""

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

from autoencoder import Autoencoder  # noqa: E402
from train_ae import (  # noqa: E402
    AETrainingConfig,
    load_autoencoder_checkpoint,
    save_autoencoder_checkpoint,
)


class AutoencoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_encode_decode_and_forward_shapes(self):
        batch = torch.randn(8, 262)
        for latent_dim in (32, 64, 128):
            model = Autoencoder(input_dim=262, latent_dim=latent_dim)
            latent = model.encode(batch)
            reconstruction = model.decode(latent)
            self.assertEqual(tuple(latent.shape), (8, latent_dim))
            self.assertEqual(tuple(reconstruction.shape), (8, 262))
            self.assertEqual(tuple(model(batch).shape), (8, 262))
            self.assertGreater(model.trainable_parameters, 0)

    def test_backpropagation_produces_finite_gradients(self):
        model = Autoencoder(input_dim=262, latent_dim=64)
        batch = torch.randn(16, 262)
        loss = torch.mean((model(batch) - batch) ** 2)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        )

    def test_eval_reconstruction_is_deterministic(self):
        model = Autoencoder(input_dim=262, latent_dim=64).eval()
        batch = torch.randn(10, 262)
        with torch.no_grad():
            first = model(batch)
            second = model(batch)
        self.assertTrue(torch.equal(first, second))

    def test_checkpoint_round_trip_is_exact(self):
        model = Autoencoder(input_dim=262, latent_dim=64).eval()
        config = AETrainingConfig(latent_dim=64)
        metadata = {
            "input_dim": 262,
            "feature_names": [f"feature_{index}" for index in range(262)],
            "input_mean": np.zeros(262, dtype=np.float32),
            "input_scale": np.ones(262, dtype=np.float32),
            "split_file": "data/splits.json",
            "split_file_sha256": "test-hash",
        }
        batch = torch.randn(12, 262)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "ae.pt"
            save_autoencoder_checkpoint(
                checkpoint_path,
                model,
                config,
                metadata,
                best_epoch=7,
                best_validation_mse=0.5,
                training_time_seconds=1.0,
            )
            loaded, payload = load_autoencoder_checkpoint(checkpoint_path)
            with torch.no_grad():
                expected = model(batch)
                actual = loaded(batch)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(payload["architecture"]["input_dim"], 262)
        self.assertEqual(payload["architecture"]["latent_dim"], 64)
        self.assertEqual(payload["data_metadata"]["feature_names"], metadata["feature_names"])
        np.testing.assert_array_equal(
            payload["data_metadata"]["input_mean"], metadata["input_mean"]
        )
        self.assertFalse(payload["test_partition_used"])
        self.assertEqual(payload["selection_partition"], "validation")


if __name__ == "__main__":
    unittest.main()
