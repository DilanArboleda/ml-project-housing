"""Extract frozen supervised MLP features for the Workshop 4 comparison."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
ROOT_SRC = ROOT / "src"
WORKSHOP2_SRC = ROOT / "workshop2_deep_learning" / "src"
for path in (WORKSHOP2_SRC, ROOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_loader import get_dataset_partitions  # noqa: E402
from features import AmesPreprocessor  # noqa: E402
from model import HousingMLP  # noqa: E402


MLP_CHECKPOINT = ROOT / "checkpoints" / "part_a" / "model_e4_wide_128_64_seed42.pt"


@dataclass
class MLPFeatureExtraction:
    embeddings: np.ndarray
    test_indices: np.ndarray
    feature_names: list[str]
    checkpoint_hash_before: str
    checkpoint_hash_after: str
    metadata: dict[str, Any]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def load_frozen_mlp() -> tuple[HousingMLP, dict[str, Any]]:
    """Reconstruct the selected Workshop 2 MLP without modifying its state."""
    checkpoint = torch.load(MLP_CHECKPOINT, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint["input_dim"] != 261:
        raise AssertionError("The selected MLP does not have 261 inputs")
    if tuple(config["hidden_dims"]) != (128, 64) or config["dropout"] != 0.05:
        raise AssertionError("Unexpected selected MLP architecture")
    if checkpoint["seed"] != 42 or checkpoint["test_partition_used"]:
        raise AssertionError("Unexpected MLP selection metadata")

    model = HousingMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    expected_types = (nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear)
    if len(model.network) != len(expected_types) or any(
        not isinstance(layer, expected) for layer, expected in zip(model.network, expected_types)
    ):
        raise AssertionError("model.network does not match the audited 128-64 structure")
    return model, checkpoint


def extract_mlp_test_features() -> MLPFeatureExtraction:
    """Return the deterministic 64-D activation after the second ReLU."""
    hash_before = file_sha256(MLP_CHECKPOINT)
    train_df, _, test_df, splits = get_dataset_partitions()
    preprocessor = AmesPreprocessor(remove_train_outliers=True)
    preprocessor.fit(train_df)
    X_train, _, _ = preprocessor.transform(train_df, is_train=True)
    X_test, _, _ = preprocessor.transform(test_df, is_train=False)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if X_test.shape != (440, 261):
        raise AssertionError(f"Unexpected MLP test matrix shape: {X_test.shape}")

    model, checkpoint = load_frozen_mlp()
    if checkpoint["feature_names"] != preprocessor.feature_names:
        raise AssertionError("MLP feature names or order changed")
    if checkpoint["split_metadata"] != splits["metadata"]:
        raise AssertionError("MLP checkpoint split metadata changed")
    regenerated_mean = X_train.mean(axis=0, dtype=np.float64)
    regenerated_scale = X_train.std(axis=0, dtype=np.float64)
    regenerated_scale = np.where(regenerated_scale < 1e-8, 1.0, regenerated_scale)
    np.testing.assert_allclose(checkpoint["input_mean"], regenerated_mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(checkpoint["input_scale"], regenerated_scale, rtol=1e-5, atol=1e-6)

    X_scaled = _standardize(X_test, np.asarray(checkpoint["input_mean"]), np.asarray(checkpoint["input_scale"]))
    tensor = torch.from_numpy(X_scaled)
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    feature_extractor = model.network[:5]
    feature_extractor.eval()
    with torch.no_grad():
        first = feature_extractor(tensor).cpu()
        second = feature_extractor(tensor).cpu()
    if not torch.equal(first, second):
        raise AssertionError("Supervised feature extraction is not deterministic")
    if first.shape != (440, 64) or not torch.isfinite(first).all():
        raise AssertionError(f"Invalid MLP embedding matrix: {tuple(first.shape)}")
    for key, value in model.state_dict().items():
        if not torch.equal(value, state_before[key]):
            raise AssertionError(f"MLP parameter changed during extraction: {key}")
    hash_after = file_sha256(MLP_CHECKPOINT)
    if hash_before != hash_after:
        raise AssertionError("MLP checkpoint file changed during extraction")

    return MLPFeatureExtraction(
        embeddings=first.numpy(),
        test_indices=np.asarray(splits["test"], dtype=np.int64),
        feature_names=list(preprocessor.feature_names),
        checkpoint_hash_before=hash_before,
        checkpoint_hash_after=hash_after,
        metadata={
            "checkpoint": str(MLP_CHECKPOINT.relative_to(ROOT)),
            "input_dim": 261,
            "hidden_dims": [128, 64],
            "dropout": 0.05,
            "seed": 42,
            "extraction_layer": "model.network[:5], output after second ReLU and before second Dropout",
            "embedding_shape": [440, 64],
            "model_eval": True,
            "parameters_frozen": True,
            "torch_no_grad": True,
            "checkpoint_unchanged": True,
        },
    )


if __name__ == "__main__":
    result = extract_mlp_test_features()
    print("MLP embedding shape:", result.embeddings.shape)
    print("Checkpoint SHA-256:", result.checkpoint_hash_after)
