"""Leakage-safe Ames transaction data for the anomaly experiment."""

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

from data_adapter import fit_standardizer, standardize  # noqa: E402


LABEL_COLUMN = "Sale Condition"
EXCLUDED_COLUMNS = ("Unnamed: 0", "Order", "PID", "SalePrice", LABEL_COLUMN)
PRIMARY_LABELS = ("Normal", "Abnorml")


@dataclass
class AnomalyDataBundle:
    X_train_normal: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
    feature_names: list[str]
    continuous_feature_count: int
    input_mean: np.ndarray
    input_scale: np.ndarray
    split_indices: dict[str, list[int]]
    split_file_sha256: str
    metadata: dict[str, Any]

    @property
    def input_dim(self) -> int:
        return int(self.X_train_normal.shape[1])

    @property
    def X_val_normal(self) -> np.ndarray:
        return self.X_val[self.val_labels == "Normal"]

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "input_dim": self.input_dim,
            "feature_names": list(self.feature_names),
            "continuous_feature_count": self.continuous_feature_count,
            "input_mean": self.input_mean.copy(),
            "input_scale": self.input_scale.copy(),
            "split_file": "data/splits.json",
            "split_file_sha256": self.split_file_sha256,
        }


def _predictors(frame):
    return frame.drop(columns=list(EXCLUDED_COLUMNS), errors="ignore")


def load_anomaly_data() -> AnomalyDataBundle:
    """Fit preprocessing only on Normal training transactions."""
    train_df, val_df, test_df, splits = get_dataset_partitions()
    for frame_name, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if LABEL_COLUMN not in frame.columns:
            raise KeyError(f"{LABEL_COLUMN!r} is missing from {frame_name}")

    train_labels = train_df[LABEL_COLUMN].astype(str).to_numpy()
    val_labels = val_df[LABEL_COLUMN].astype(str).to_numpy()
    test_labels = test_df[LABEL_COLUMN].astype(str).to_numpy()
    normal_train_mask = train_labels == "Normal"
    if not normal_train_mask.any():
        raise ValueError("No Normal transactions are available in training")

    train_normal_predictors = _predictors(train_df.loc[normal_train_mask])
    val_predictors = _predictors(val_df)
    test_predictors = _predictors(test_df)
    preprocessor = AmesPreprocessor(remove_train_outliers=False)
    preprocessor.fit(train_normal_predictors)

    X_train_raw = np.asarray(preprocessor.transform(train_normal_predictors), dtype=np.float32)
    X_val_raw = np.asarray(preprocessor.transform(val_predictors), dtype=np.float32)
    X_test_raw = np.asarray(preprocessor.transform(test_predictors), dtype=np.float32)
    input_mean, input_scale = fit_standardizer(X_train_raw)

    feature_names = list(preprocessor.feature_names)
    forbidden = [
        name for name in feature_names
        if name == LABEL_COLUMN or name.startswith(f"{LABEL_COLUMN}_")
    ]
    if forbidden:
        raise AssertionError(f"Anomaly label leaked into predictors: {forbidden}")

    bundle = AnomalyDataBundle(
        X_train_normal=standardize(X_train_raw, input_mean, input_scale),
        X_val=standardize(X_val_raw, input_mean, input_scale),
        X_test=standardize(X_test_raw, input_mean, input_scale),
        val_labels=val_labels,
        test_labels=test_labels,
        feature_names=feature_names,
        continuous_feature_count=len(preprocessor.continuous_cols),
        input_mean=input_mean,
        input_scale=input_scale,
        split_indices={key: list(splits[key]) for key in ("train", "val", "test")},
        split_file_sha256=hashlib.sha256((ROOT / "data" / "splits.json").read_bytes()).hexdigest(),
        metadata={
            "protocol": "transaction anomaly detection",
            "training_rows": "Normal Sale Condition only",
            "preprocessing_fit_partition": "Normal training transactions only",
            "excluded_columns": list(EXCLUDED_COLUMNS),
            "primary_binary_labels": {"Normal": 0, "Abnorml": 1},
            "excluded_primary_evaluation_labels": ["Partial", "Family", "Alloca", "AdjLand"],
            "train_normal_count": int(normal_train_mask.sum()),
            "validation_label_counts": {
                str(label): int((val_labels == label).sum()) for label in np.unique(val_labels)
            },
            "test_label_counts": {
                str(label): int((test_labels == label).sum()) for label in np.unique(test_labels)
            },
            "test_partition_used_for_training_selection_or_threshold": False,
        },
    )
    for name, matrix, expected in (
        ("validation", bundle.X_val, 440), ("test", bundle.X_test, 440)
    ):
        if matrix.shape != (expected, bundle.input_dim) or not np.isfinite(matrix).all():
            raise AssertionError(f"Invalid {name} anomaly matrix: {matrix.shape}")
    if not np.isfinite(bundle.X_train_normal).all():
        raise AssertionError("Anomaly training matrix contains non-finite values")
    return bundle


if __name__ == "__main__":
    data = load_anomaly_data()
    print("Input dimension:", data.input_dim)
    print("Normal training observations:", len(data.X_train_normal))
    print("Validation/test observations:", len(data.X_val), len(data.X_test))
