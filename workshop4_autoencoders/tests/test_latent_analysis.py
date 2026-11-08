"""Tests for deterministic frozen latent-space extraction and analysis."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "workshop4_autoencoders" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_latent import (  # noqa: E402
    MODEL_ORDER,
    compute_latent_metrics,
    extract_representations,
    standardize_representations,
)
from extract_mlp_features import (  # noqa: E402
    MLP_CHECKPOINT,
    extract_mlp_test_features,
    file_sha256,
    load_frozen_mlp,
)


class FrozenLatentExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.representations, cls.labels, cls.audit = extract_representations()

    def test_embedding_dimensions_and_finite_values(self) -> None:
        expected = {"AE z=128": (440, 128), "AE z=64": (440, 64), "VAE z=64 beta=0.5": (440, 64), "MLP z=64": (440, 64)}
        self.assertEqual(tuple(self.representations), MODEL_ORDER)
        for name, shape in expected.items():
            self.assertEqual(self.representations[name].shape, shape)
            self.assertTrue(np.isfinite(self.representations[name]).all())

    def test_test_index_order_is_exact(self) -> None:
        self.assertEqual(len(self.labels), 440)
        self.assertTrue(self.audit["test_order_exact_across_models"])
        self.assertTrue(np.all(np.diff(self.labels["original_dataset_index"].to_numpy()) > 0))

    def test_repeated_frozen_extraction_is_exact(self) -> None:
        repeated, repeated_labels, repeated_audit = extract_representations()
        for name in MODEL_ORDER:
            self.assertTrue(np.array_equal(self.representations[name], repeated[name]))
        self.assertTrue(np.array_equal(self.labels["original_dataset_index"], repeated_labels["original_dataset_index"]))
        self.assertEqual(self.audit["checkpoint_hashes"], repeated_audit["checkpoint_hashes"])

    def test_silhouette_uses_original_latent_space(self) -> None:
        standardized = standardize_representations(self.representations)
        metrics = compute_latent_metrics(standardized, self.labels)
        self.assertEqual(len(metrics), 12)
        self.assertFalse(metrics["computed_on_2d_projection"].any())
        self.assertTrue((metrics["space"] == "per-model standardized original latent representation").all())


class SupervisedFeatureExtractionTests(unittest.TestCase):
    def test_second_relu_is_the_extraction_layer(self) -> None:
        result = extract_mlp_test_features()
        model, _ = load_frozen_mlp()
        self.assertEqual(result.embeddings.shape, (440, 64))
        self.assertEqual(result.metadata["extraction_layer"], "model.network[:5], output after second ReLU and before second Dropout")
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_mlp_checkpoint_remains_unchanged(self) -> None:
        before = file_sha256(MLP_CHECKPOINT)
        result = extract_mlp_test_features()
        after = file_sha256(MLP_CHECKPOINT)
        self.assertEqual(before, after)
        self.assertEqual(result.checkpoint_hash_before, result.checkpoint_hash_after)


class CompletedLatentArtifactsTests(unittest.TestCase):
    def test_saved_reductions_are_reproducible_and_aligned(self) -> None:
        metadata_path = ROOT / "workshop4_autoencoders" / "results" / "latent_embedding_metadata.json"
        if not metadata_path.exists():
            self.skipTest("Latent analysis artifacts have not been generated yet")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(metadata["no_training_performed"])
        self.assertEqual(metadata["test_observations"], 440)
        for details in metadata["reducer_reproducibility"].values():
            self.assertLessEqual(details["tsne_max_absolute_repeat_difference"], 1e-6)
            self.assertLessEqual(details["umap_max_absolute_repeat_difference"], 1e-6)
        for filename in ("latent_tsne_embeddings.npz", "latent_umap_embeddings.npz"):
            with np.load(ROOT / "workshop4_autoencoders" / "results" / filename) as archive:
                for name in MODEL_ORDER:
                    self.assertEqual(archive[name].shape, (440, 2))
                    self.assertTrue(np.isfinite(archive[name]).all())


if __name__ == "__main__":
    unittest.main()
