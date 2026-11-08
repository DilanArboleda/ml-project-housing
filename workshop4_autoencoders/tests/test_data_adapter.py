"""Tests for the unsupervised Ames Housing data adapter."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSHOP_SRC = ROOT / "workshop4_autoencoders" / "src"
sys.path.insert(0, str(WORKSHOP_SRC))

from data_adapter import (  # noqa: E402
    EXCLUDED_INPUT_COLUMNS,
    load_unsupervised_data,
)


class DataAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_path = ROOT / "data" / "splits.json"
        cls.split_bytes_before = cls.split_path.read_bytes()
        cls.data = load_unsupervised_data()

    def test_fixed_partitions_are_intact_and_disjoint(self):
        self.assertEqual(self.split_bytes_before, self.split_path.read_bytes())
        self.assertEqual(
            self.data.split_file_sha256,
            hashlib.sha256(self.split_bytes_before).hexdigest(),
        )
        train = set(self.data.split_indices["train"])
        val = set(self.data.split_indices["val"])
        test = set(self.data.split_indices["test"])
        self.assertEqual((len(train), len(val), len(test)), (2050, 440, 440))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(len(train | val | test), 2930)

    def test_verified_shapes_and_feature_order(self):
        self.assertEqual(self.data.X_train.shape, (2050, 262))
        self.assertEqual(self.data.X_val.shape, (440, 262))
        self.assertEqual(self.data.X_test.shape, (440, 262))
        self.assertEqual(len(self.data.feature_names), 262)

    def test_all_matrices_are_finite(self):
        for matrix in (
            self.data.X_train,
            self.data.X_val,
            self.data.X_test,
            self.data.X_train_before_standardization,
            self.data.X_val_before_standardization,
            self.data.X_test_before_standardization,
        ):
            self.assertTrue(np.isfinite(matrix).all())

    def test_target_identifiers_and_artificial_index_are_excluded(self):
        for forbidden in EXCLUDED_INPUT_COLUMNS:
            self.assertNotIn(forbidden, self.data.feature_names)
            self.assertFalse(
                any(
                    name.startswith(f"{forbidden}_")
                    for name in self.data.feature_names
                )
            )

    def test_additional_scaler_uses_training_statistics_only(self):
        raw_train = self.data.X_train_before_standardization.astype(np.float64)
        expected_mean = raw_train.mean(axis=0)
        expected_scale = raw_train.std(axis=0)
        expected_scale = np.where(expected_scale < 1e-8, 1.0, expected_scale)
        np.testing.assert_allclose(
            self.data.input_mean, expected_mean, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            self.data.input_scale, expected_scale, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            self.data.X_train.mean(axis=0), 0.0, atol=2e-5
        )
        np.testing.assert_allclose(
            self.data.X_train.std(axis=0), 1.0, rtol=2e-5, atol=2e-5
        )

    def test_no_target_informed_outlier_filter_is_enabled(self):
        metadata = self.data.preprocessing_metadata
        self.assertFalse(metadata["remove_train_outliers"])
        self.assertEqual(metadata["fit_partition"], "train only")


if __name__ == "__main__":
    unittest.main()
