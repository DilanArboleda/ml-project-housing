"""Tests for frozen reconstruction and independent anomaly protocols."""

from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "workshop4_autoencoders" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anomaly_data_adapter import EXCLUDED_COLUMNS, load_anomaly_data  # noqa: E402
from data_adapter import load_unsupervised_data  # noqa: E402
from train_ae import load_autoencoder_checkpoint  # noqa: E402
from train_vae import load_vae_checkpoint  # noqa: E402
from run_anomaly_experiment import _ae_scores, _vae_scores  # noqa: E402


class FinalReconstructionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_unsupervised_data()
        checkpoint_dir = ROOT / "workshop4_autoencoders" / "checkpoints"
        cls.ae, cls.ae_payload = load_autoencoder_checkpoint(checkpoint_dir / "ae_best.pt")
        cls.vae, cls.vae_payload = load_vae_checkpoint(checkpoint_dir / "vae_best.pt")

    def test_original_test_partition_has_440_rows(self) -> None:
        self.assertEqual(self.data.X_test.shape[0], 440)
        self.assertEqual(len(self.data.split_indices["test"]), 440)

    def test_selected_checkpoint_contracts_match(self) -> None:
        for payload in (self.ae_payload, self.vae_payload):
            self.assertEqual(payload["data_metadata"]["split_file_sha256"], self.data.split_file_sha256)
            self.assertEqual(payload["data_metadata"]["feature_names"], self.data.feature_names)
            self.assertFalse(payload["test_partition_used"])
        self.assertEqual(self.ae.latent_dim, 128)
        self.assertEqual(self.vae.latent_dim, 64)
        self.assertEqual(self.vae_payload["training_config"]["target_beta"], 0.5)

    def test_frozen_reconstructions_are_reproducible(self) -> None:
        values = torch.as_tensor(self.data.X_test[:32], dtype=torch.float32)
        with torch.no_grad():
            first_ae, second_ae = self.ae(values), self.ae(values)
            first_vae = self.vae.reconstruct_with_mean(values)
            second_vae = self.vae.reconstruct_with_mean(values)
        self.assertTrue(torch.equal(first_ae, second_ae))
        self.assertTrue(torch.equal(first_vae, second_vae))


class AnomalyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_anomaly_data()

    def test_sale_condition_and_identifiers_do_not_leak(self) -> None:
        for name in self.data.feature_names:
            for forbidden in EXCLUDED_COLUMNS:
                self.assertNotEqual(name, forbidden)
                self.assertFalse(name.startswith(f"{forbidden}_"))

    def test_anomaly_preprocessing_uses_normal_train_only(self) -> None:
        self.assertEqual(self.data.metadata["training_rows"], "Normal Sale Condition only")
        self.assertEqual(self.data.metadata["preprocessing_fit_partition"], "Normal training transactions only")
        self.assertGreater(len(self.data.X_train_normal), 0)
        self.assertEqual(len(self.data.X_val), 440)
        self.assertEqual(len(self.data.X_test), 440)

    def test_anomaly_matrices_are_finite_and_aligned(self) -> None:
        for matrix in (self.data.X_train_normal, self.data.X_val, self.data.X_test):
            self.assertTrue(np.isfinite(matrix).all())
            self.assertEqual(matrix.shape[1], self.data.input_dim)
        self.assertEqual(len(self.data.val_labels), len(self.data.X_val))
        self.assertEqual(len(self.data.test_labels), len(self.data.X_test))

    def test_split_hash_matches_general_protocol(self) -> None:
        self.assertEqual(self.data.split_file_sha256, load_unsupervised_data().split_file_sha256)


class CompletedAnomalyArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_anomaly_data()
        checkpoint_dir = ROOT / "workshop4_autoencoders" / "checkpoints"
        cls.ae, cls.ae_payload = load_autoencoder_checkpoint(checkpoint_dir / "anomaly_ae_best.pt")
        cls.vae, cls.vae_payload = load_vae_checkpoint(checkpoint_dir / "anomaly_vae_best.pt")
        cls.protocol = json.loads((ROOT / "workshop4_autoencoders" / "results" / "anomaly_protocol_metadata.json").read_text(encoding="utf-8"))

    def test_anomaly_checkpoints_are_separate_and_auditable(self) -> None:
        for payload in (self.ae_payload, self.vae_payload):
            self.assertEqual(payload["data_metadata"]["input_dim"], self.data.input_dim)
            self.assertEqual(payload["data_metadata"]["split_file_sha256"], self.data.split_file_sha256)
            self.assertEqual(payload["selection_partition"], "validation")
            self.assertFalse(payload["test_partition_used"])

    def test_anomaly_scores_are_finite_and_deterministic(self) -> None:
        first = _ae_scores(self.ae, self.data.X_test)
        second = _ae_scores(self.ae, self.data.X_test)
        vae_recon, vae_kl, vae_combined = _vae_scores(self.vae, self.data.X_test, 0.5)
        self.assertTrue(np.array_equal(first, second))
        for scores in (first, vae_recon, vae_kl, vae_combined):
            self.assertEqual(len(scores), 440)
            self.assertTrue(np.isfinite(scores).all())

    def test_threshold_is_reproducible_from_normal_validation_only(self) -> None:
        validation_scores = _ae_scores(self.ae, self.data.X_val)
        expected = float(np.quantile(validation_scores[self.data.val_labels == "Normal"], 0.95))
        observed = self.protocol["thresholds"]["AE reconstruction MSE"]
        self.assertAlmostEqual(expected, observed, places=12)
        self.assertFalse(self.protocol["test_used_for_training_selection_or_threshold"])


if __name__ == "__main__":
    unittest.main()
