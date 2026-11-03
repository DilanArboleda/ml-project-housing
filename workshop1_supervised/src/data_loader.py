"""
Data Loader and Splitting Protocol for Ames Housing.
Workshop No. 1 â€” Data Preparation & Supervised Learning Baselines.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero OrdeÃ±ez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)
"""

import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Ensure determinism across numpy and python standard library."""
    random.seed(seed)
    np.random.seed(seed)

def get_base_dir() -> Path:
    """Find repository root directory."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "data" / "raw").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent

def create_or_load_splits(df: pd.DataFrame, splits_path: Path = None, test_size: float = 0.15, val_size: float = 0.15, seed: int = RANDOM_SEED) -> dict:
    """
    Generate or load deterministic 70% / 15% / 15% train/val/test split indices.
    Prevents leakage by preserving split assignments permanently in data/splits.json.
    """
    if splits_path is None:
        base_dir = get_base_dir()
        splits_path = base_dir / "data" / "splits.json"

    splits_path = Path(splits_path)

    if splits_path.exists():
        print(f"Loading existing split indices from {splits_path}")
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        return splits

    print(f"Generating new reproducible splits (70/15/15) with random seed {seed}...")
    set_seed(seed)
    n_samples = len(df)
    indices = np.arange(n_samples)

    # First split off test set (15%)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True
    )

    # From remaining 85%, split off validation set (15% of total = 0.15 / 0.85 approx 0.17647)
    val_relative_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_relative_size, random_state=seed, shuffle=True
    )

    splits = {
        "metadata": {
            "total_samples": n_samples,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
            "train_pct": round(len(train_idx) / n_samples * 100, 2),
            "val_pct": round(len(val_idx) / n_samples * 100, 2),
            "test_pct": round(len(test_idx) / n_samples * 100, 2),
            "random_seed": seed
        },
        "train": [int(i) for i in sorted(train_idx)],
        "val": [int(i) for i in sorted(val_idx)],
        "test": [int(i) for i in sorted(test_idx)]
    }

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Splits saved to {splits_path}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return splits


def load_raw_dataset(csv_path: Path = None) -> pd.DataFrame:
    """Load raw Ames Housing dataset."""
    if csv_path is None:
        base_dir = get_base_dir()
        csv_path = base_dir / "data" / "raw" / "AmesHousing.csv"
        if not csv_path.exists():
            csv_path = base_dir / "data" / "raw" / "AmesHousing.tsv"

    csv_path = Path(csv_path)
    sep = "\t" if csv_path.suffix == ".tsv" else ","
    return pd.read_csv(csv_path, sep=sep)


def get_dataset_partitions(csv_path: Path = None, splits_path: Path = None):
    """
    Load dataset and return train, validation, and test subsets based on fixed splits.
    """
    df = load_raw_dataset(csv_path)
    splits = create_or_load_splits(df, splits_path)

    train_df = df.iloc[splits["train"]].copy().reset_index(drop=True)
    val_df = df.iloc[splits["val"]].copy().reset_index(drop=True)
    test_df = df.iloc[splits["test"]].copy().reset_index(drop=True)

    return train_df, val_df, test_df, splits


if __name__ == "__main__":
    train_df, val_df, test_df, splits = get_dataset_partitions()
    print("Partition summary:")
    print(f"  Train: {train_df.shape}")
    print(f"  Val:   {val_df.shape}")
    print(f"  Test:  {test_df.shape}")

