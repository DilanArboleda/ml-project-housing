"""
Ames Housing Dataset Integrity and Summary Scanner.
Workshop No. 1 â€” Data Preparation & Supervised Learning Baselines.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero OrdeÃ±ez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)
"""

import os
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file."""
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_dataset(data_path: str = None):
    """Scan dataset directory, compute checksums, and print structured summary."""
    if data_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_path = base_dir / "data" / "raw" / "AmesHousing.csv"
        if not data_path.exists():
            data_path = base_dir / "data" / "raw" / "AmesHousing.tsv"

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    sep = "\t" if data_path.suffix == ".tsv" else ","
    df = pd.read_csv(data_path, sep=sep)

    print("=" * 80)
    print("AMES HOUSING DATASET â€” INTEGRITY & STRUCTURED SUMMARY REPORT")
    print("=" * 80)
    print(f"File Path:       {data_path.resolve()}")
    print(f"File Size:       {data_path.stat().st_size:,} bytes")
    print(f"MD5 Checksum:    {compute_file_hash(data_path, 'md5')}")
    print(f"SHA-256 Checksum:{compute_file_hash(data_path, 'sha256')}")
    print(f"Total Rows:      {len(df)}")
    print(f"Total Columns:   {df.shape[1]}")
    print("-" * 80)

    # Data Types breakdown
    dtypes_counts = df.dtypes.value_counts()
    print("COLUMN DATA TYPES:")
    for dtype, count in dtypes_counts.items():
        print(f"  - {str(dtype):<15}: {count} columns")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Numerical Features:   {len(num_cols)} (including target 'SalePrice' and identifiers)")
    print(f"Categorical Features: {len(cat_cols)}")
    print("-" * 80)

    # Target Variable Statistics
    if "SalePrice" in df.columns:
        sp = df["SalePrice"]
        print("TARGET VARIABLE ('SalePrice') SUMMARY:")
        print(f"  Count:          {sp.count():,}")
        print(f"  Mean:           ${sp.mean():,.2f}")
        print(f"  Median:         ${sp.median():,.2f}")
        print(f"  Std Dev:        ${sp.std():,.2f}")
        print(f"  Min:            ${sp.min():,}")
        print(f"  25th Percentile:${sp.quantile(0.25):,}")
        print(f"  75th Percentile:${sp.quantile(0.75):,}")
        print(f"  Max:            ${sp.max():,}")
        print(f"  Skewness:       {sp.skew():.3f} (positive/right-skewed)")
        print(f"  Kurtosis:       {sp.kurtosis():.3f}")

        # Binned price tiers (quintiles)
        print("\n  Price Distribution by Quintiles:")
        labels = ["Very Low", "Low-Mid", "Middle", "Mid-High", "Luxury"]
        quintiles = pd.qcut(sp, q=5, labels=labels)
        q_counts = quintiles.value_counts().sort_index()
        for tier, count in q_counts.items():
            pct = 100 * count / len(df)
            print(f"    - {tier:<10}: {count} samples ({pct:.1f}%)")
        print("-" * 80)

    # Missing Values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
    print(f"MISSING VALUE ANALYSIS ({len(null_cols)} columns with missing entries):")
    for col, n_missing in null_cols.items():
        pct = 100 * n_missing / len(df)
        print(f"  - {col:<20}: {n_missing:>5} missing ({pct:>5.1f}%)")
    print("-" * 80)

    # Feature Ranges (Numerical)
    print("KEY NUMERICAL FEATURE RANGES:")
    key_features = ["Gr Liv Area", "Total Bsmt SF", "Lot Area", "Year Built", "Overall Qual", "Garage Cars"]
    for col in key_features:
        if col in df.columns:
            s = df[col].dropna()
            print(f"  - {col:<18}: Min={s.min():>8.0f} | Median={s.median():>8.0f} | Max={s.max():>8.0f} | Std={s.std():>8.2f}")

    # Categorical breakdown of primary attributes
    print("-" * 80)
    print("CATEGORICAL SAMPLES BREAKDOWN (Top distributions):")
    if "Bldg Type" in df.columns:
        print("  Bldg Type:")
        for k, v in df["Bldg Type"].value_counts().items():
            print(f"    * {k:<15}: {v:>5} ({100*v/len(df):.1f}%)")
    if "MS Zoning" in df.columns:
        print("  MS Zoning:")
        for k, v in df["MS Zoning"].value_counts().items():
            print(f"    * {k:<15}: {v:>5} ({100*v/len(df):.1f}%)")

    print("=" * 80)
    print("SCAN COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    scan_dataset()

