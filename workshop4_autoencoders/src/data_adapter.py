"""Leakage-aware unsupervised data preparation for Workshop 4."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ROOT_SRC = ROOT / "src"
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from data_loader import get_dataset_partitions  # noqa: E402
from features import AmesPreprocessor  # noqa: E402


EXCLUDED_INPUT_COLUMNS = ("Unnamed: 0", "Order", "PID", "SalePrice")
MIN_SCALE = 1e-8


@dataclass
class UnsupervisedDataBundle:
    """Processed matrices and train-fitted transformation metadata."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    X_train_before_standardization: np.ndarray
    X_val_before_standardization: np.ndarray
    X_test_before_standardization: np.ndarray
    feature_names: list[str]
    input_mean: np.ndarray
    input_scale: np.ndarray
    split_indices: dict[str, list[int]]
    split_metadata: dict[str, Any]
    split_file_sha256: str
    preprocessing_metadata: dict[str, Any]

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[1])

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return metadata needed to audit and reproduce preprocessing."""
        return {
            "input_dim": self.input_dim,
            "feature_names": list(self.feature_names),
            "input_mean": self.input_mean.copy(),
            "input_scale": self.input_scale.copy(),
            "split_metadata": dict(self.split_metadata),
            "split_file": "data/splits.json",
            "split_file_sha256": self.split_file_sha256,
            "excluded_input_columns": list(EXCLUDED_INPUT_COLUMNS),
            "preprocessing": self.preprocessing_metadata,
        }


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a mean-standard-deviation transform on training values only."""
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < MIN_SCALE, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply previously fitted standardization statistics."""
    return ((values - mean) / scale).astype(np.float32)


def _predictor_frame(frame):
    """Remove target, identifiers, and the exported row index."""
    return frame.drop(columns=list(EXCLUDED_INPUT_COLUMNS), errors="ignore")


def _split_hash() -> str:
    return hashlib.sha256((ROOT / "data" / "splits.json").read_bytes()).hexdigest()


def _preprocessor_metadata(preprocessor: AmesPreprocessor) -> dict[str, Any]:
    encoder_categories = []
    if preprocessor.one_hot_encoder is not None:
        encoder_categories = [
            [str(value) for value in category]
            for category in preprocessor.one_hot_encoder.categories_
        ]

    return {
        "source": "src.features.AmesPreprocessor",
        "remove_train_outliers": False,
        "fit_partition": "train only",
        "continuous_columns": list(preprocessor.continuous_cols),
        "nominal_columns": list(preprocessor.nominal_cols),
        "lot_frontage_global_median": float(
            preprocessor.lot_frontage_global_median
        ),
        "lot_frontage_neighborhood_medians": {
            str(key): float(value)
            for key, value in preprocessor.lot_frontage_medians.items()
        },
        "categorical_modes": dict(preprocessor.categorical_modes),
        "numerical_medians": dict(preprocessor.numerical_medians),
        "robust_scaler_center": preprocessor.scaler.center_.astype(float).tolist(),
        "robust_scaler_scale": preprocessor.scaler.scale_.astype(float).tolist(),
        "one_hot_categories": encoder_categories,
        "additional_standardizer": "per-feature mean/std fitted on train only",
        "reconstruction_loss_note": (
            "MSE is computed in the standardized feature space. It balances "
            "features by training variance, but one-hot reconstructions remain "
            "continuous scores rather than valid categorical probabilities."
        ),
    }


def _validate_bundle(bundle: UnsupervisedDataBundle) -> None:
    expected_rows = {"train": 2050, "val": 440, "test": 440}
    matrices = {
        "train": bundle.X_train,
        "val": bundle.X_val,
        "test": bundle.X_test,
    }
    for partition, matrix in matrices.items():
        if matrix.shape[0] != expected_rows[partition]:
            raise AssertionError(
                f"Unexpected {partition} row count: {matrix.shape[0]}"
            )
        if matrix.shape[1] != bundle.input_dim:
            raise AssertionError("Feature dimensions differ across partitions")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{partition} contains NaN or infinite values")

    forbidden = set(EXCLUDED_INPUT_COLUMNS)
    invalid_features = [
        name
        for name in bundle.feature_names
        if name in forbidden
        or any(name.startswith(f"{column}_") for column in forbidden)
    ]
    if invalid_features:
        raise AssertionError(f"Forbidden input features found: {invalid_features}")

    train_indices = set(bundle.split_indices["train"])
    val_indices = set(bundle.split_indices["val"])
    test_indices = set(bundle.split_indices["test"])
    if train_indices & val_indices or train_indices & test_indices or val_indices & test_indices:
        raise AssertionError("Fixed partitions overlap")
    if len(train_indices | val_indices | test_indices) != 2930:
        raise AssertionError("Fixed partitions do not cover all observations")


def load_unsupervised_data() -> UnsupervisedDataBundle:
    """Create train/validation/test matrices without target-informed filtering."""
    train_df, val_df, test_df, splits = get_dataset_partitions()

    train_predictors = _predictor_frame(train_df)
    val_predictors = _predictor_frame(val_df)
    test_predictors = _predictor_frame(test_df)

    preprocessor = AmesPreprocessor(remove_train_outliers=False)
    preprocessor.fit(train_predictors)

    X_train_raw = np.asarray(
        preprocessor.transform(train_predictors, is_train=True), dtype=np.float32
    )
    X_val_raw = np.asarray(preprocessor.transform(val_predictors), dtype=np.float32)
    X_test_raw = np.asarray(preprocessor.transform(test_predictors), dtype=np.float32)

    input_mean, input_scale = fit_standardizer(X_train_raw)
    bundle = UnsupervisedDataBundle(
        X_train=standardize(X_train_raw, input_mean, input_scale),
        X_val=standardize(X_val_raw, input_mean, input_scale),
        X_test=standardize(X_test_raw, input_mean, input_scale),
        X_train_before_standardization=X_train_raw,
        X_val_before_standardization=X_val_raw,
        X_test_before_standardization=X_test_raw,
        feature_names=list(preprocessor.feature_names),
        input_mean=input_mean,
        input_scale=input_scale,
        split_indices={
            "train": list(splits["train"]),
            "val": list(splits["val"]),
            "test": list(splits["test"]),
        },
        split_metadata=dict(splits["metadata"]),
        split_file_sha256=_split_hash(),
        preprocessing_metadata=_preprocessor_metadata(preprocessor),
    )
    _validate_bundle(bundle)
    return bundle


if __name__ == "__main__":
    data = load_unsupervised_data()
    print("Input dimension:", data.input_dim)
    print("Train/validation/test:", len(data.X_train), len(data.X_val), len(data.X_test))
    print("All matrices are finite.")
