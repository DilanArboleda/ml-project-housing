"""
Medical preprocessing pipeline for Workshop 6 (MIT-BIH Arrhythmia).

This module provides the complete data pipeline to:
    1. Download the MIT-BIH Arrhythmia Database from PhysioNet (with cache).
    2. Load ECG signals and annotations for a given record.
    3. Segment heartbeats around R-peaks.
    4. Extract tabular features per beat (RR, morphology, HRV).
    5. Build a leakage-aware feature matrix with train-only scalers.
    6. Create patient-level splits (train/val/test) to avoid leakage.

Design notes
------------
- Splits are made at the *record* (patient) level, never at the beat level.
  This is critical in clinical ML: beats from the same patient share
  morphology and would leak information across splits.
- All scalers (RobustScaler + StandardScaler) are fitted on TRAIN ONLY.
- Labels follow AAMI superclass grouping (N, S, V, F, Q) and are reduced
  to a binary target (Normal vs Abnormal) by default.

Reference
---------
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import RobustScaler, StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# AAMI superclass mapping (standard for MIT-BIH arrhythmia classification).
# See: de Chazal et al., "Automatic classification of heartbeats using ECG
# morphology and heartbeat interval features", IEEE TBME 2004, and
# ANSI/AAMI EC57:2012.
#
# IMPORTANT: class Q (which includes "/", "f", "U") is deliberately EXCLUDED.
# Those symbols correspond to paced beats, fusion of paced and normal beats,
# and unclassifiable beats. They are NOT pathological arrhythmias and would
# bias the binary target (e.g., a pacemaker patient would be flagged as
# "abnormal" for every beat, producing systematic false positives).
AAMI_MAP: Dict[str, str] = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
}

# Final records used in the experiments (all 48 records from MIT-BIH).
MITDB_RECORDS: List[str] = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# Sampling frequency for all MIT-BIH records.
FS: int = 360

# Beat window: 200 samples before + 200 after the R-peak (~1.11 s).
PRE_SAMPLES: int = 200
POST_SAMPLES: int = 200


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BeatFeatures:
    """Container for tabular features extracted per beat."""
    rr_prev: float = np.nan
    rr_next: float = np.nan
    rr_ratio: float = np.nan
    qrs_width: float = np.nan
    qrs_amplitude: float = np.nan
    qrs_energy: float = np.nan
    rr_sdnn: float = np.nan
    rr_rmssd: float = np.nan
    label_aami: str = "N"
    label_binary: int = 0
    record_id: str = ""
    beat_index: int = -1


@dataclass
class SplitResult:
    """Container for the result of make_splits()."""
    train_records: List[str] = field(default_factory=list)
    val_records: List[str] = field(default_factory=list)
    test_records: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Dataset download
# ---------------------------------------------------------------------------

def download_mitdb(
    data_dir: str | Path = "data/mitdb",
    force: bool = False,
) -> Path:
    """
    Download the MIT-BIH Arrhythmia Database into ``data_dir`` if not present.

    Parameters
    ----------
    data_dir : str | Path
        Target directory for the raw PhysioNet files.
    force : bool
        If True, re-download even if files exist.

    Returns
    -------
    Path
        Path to the directory containing the MIT-BIH records.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    marker = data_dir / ".downloaded"
    if marker.exists() and not force:
        print(f"[download_mitdb] Cache found at {data_dir}, skipping download.")
        return data_dir

    print(f"[download_mitdb] Downloading MIT-BIH to {data_dir} ...")
    # wfdb.dl_database downloads the whole MIT-BIH database (~100 MB).
    wfdb.dl_database("mitdb", dl_dir=str(data_dir))
    marker.touch()
    print("[download_mitdb] Download complete.")
    return data_dir


# ---------------------------------------------------------------------------
# 2. Loading records
# ---------------------------------------------------------------------------

def load_record(
    record_id: str,
    data_dir: str | Path = "data/mitdb",
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, int]:
    """
    Load a MIT-BIH record and its beat annotations.

    Returns
    -------
    signal : np.ndarray of shape (N, 2)
    fs : int
    r_peaks : np.ndarray
    labels : np.ndarray of str
    n_excluded : int
        Number of annotated beats excluded because their symbol is not in
        AAMI_MAP (paced, unclassifiable, etc.).
    """
    data_dir = Path(data_dir)
    record_path = data_dir / record_id
    if not (record_path.with_suffix(".dat")).exists():
        raise FileNotFoundError(
            f"Record {record_id} not found in {data_dir}. "
            f"Did you call download_mitdb() first?"
        )

    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    signal = record.p_signal
    fs = int(record.fs)

    r_peaks: List[int] = []
    labels: List[str] = []
    n_excluded = 0
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol in AAMI_MAP:
            r_peaks.append(int(sample))
            labels.append(AAMI_MAP[symbol])
        else:
            n_excluded += 1

    return signal, fs, np.array(r_peaks), np.array(labels), n_excluded


# ---------------------------------------------------------------------------
# 3. Beat segmentation
# ---------------------------------------------------------------------------

def extract_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    pre: int = PRE_SAMPLES,
    post: int = POST_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed-length beat windows around each R-peak.

    Parameters
    ----------
    signal : np.ndarray of shape (N, 2)
        ECG signal (two leads).
    r_peaks : np.ndarray
        Sample indices of R-peaks.
    pre, post : int
        Samples before and after the R-peak.

    Returns
    -------
    beats : np.ndarray of shape (n_beats, pre + post, 2)
        Beat windows (both leads).
    valid_idx : np.ndarray
        Indices of ``r_peaks`` that produced valid windows (those too close
        to the borders are dropped).
    """
    n_samples = signal.shape[0]
    window = pre + post

    valid = []
    for i, peak in enumerate(r_peaks):
        if peak - pre >= 0 and peak + post <= n_samples:
            valid.append(i)

    valid_idx = np.array(valid, dtype=int)
    beats = np.zeros((len(valid_idx), window, signal.shape[1]), dtype=np.float32)

    for j, i in enumerate(valid_idx):
        peak = r_peaks[i]
        beats[j] = signal[peak - pre : peak + post, :]

    return beats, valid_idx


# ---------------------------------------------------------------------------
# 4. Feature extraction
# ---------------------------------------------------------------------------

def _qrs_width(beat: np.ndarray, lead: int = 0) -> float:
    """Estimate QRS width (in samples) as the width at 50% of the peak."""
    x = beat[:, lead]
    # Simple baseline correction: subtract the median
    x = x - np.median(x)
    peak_val = np.max(np.abs(x))
    if peak_val < 1e-6:
        return np.nan
    threshold = 0.5 * peak_val
    above = np.where(np.abs(x) >= threshold)[0]
    if len(above) < 2:
        return np.nan
    return float(above[-1] - above[0] + 1)


def _hrv_features(rr_intervals: np.ndarray) -> Tuple[float, float]:
    """Compute SDNN and RMSSD over a rolling window of RR intervals."""
    if len(rr_intervals) < 3:
        return np.nan, np.nan
    sdnn = float(np.std(rr_intervals, ddof=1))
    diffs = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    return sdnn, rmssd


def extract_beat_features(
    beats: np.ndarray,
    r_peaks: np.ndarray,
    labels: np.ndarray,
    fs: int = FS,
    record_id: str = "",
    valid_idx: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Extract tabular features for each valid beat.

    Parameters
    ----------
    beats : np.ndarray of shape (n_beats, window, n_leads)
        Beat windows.
    r_peaks : np.ndarray
        Sample indices of R-peaks (same order as ``labels``).
    labels : np.ndarray of str
        AAMI labels per beat.
    fs : int
        Sampling frequency.
    record_id : str
        Identifier of the parent record.
    valid_idx : np.ndarray, optional
        Indices of ``r_peaks`` that survived boundary checks.

    Returns
    -------
    pd.DataFrame
        One row per beat with engineered features.
    """
    if valid_idx is None:
        valid_idx = np.arange(len(r_peaks))

    # RR intervals (in seconds) between consecutive valid R-peaks
    rr = np.diff(r_peaks[valid_idx]) / fs  # length = n_beats - 1

    rows: List[BeatFeatures] = []
    for j in range(len(valid_idx)):
        bf = BeatFeatures()
        bf.record_id = record_id
        bf.beat_index = int(valid_idx[j])
        bf.label_aami = str(labels[valid_idx[j]])
        bf.label_binary = 0 if bf.label_aami == "N" else 1

        # RR features
        bf.rr_prev = float(rr[j - 1]) if j - 1 >= 0 else np.nan
        bf.rr_next = float(rr[j]) if j < len(rr) else np.nan
        if not np.isnan(bf.rr_prev) and not np.isnan(bf.rr_next) and bf.rr_next > 0:
            bf.rr_ratio = float(bf.rr_prev / bf.rr_next)

        # QRS features (lead 0 = MLII typically)
        beat_window = beats[j]
        bf.qrs_width = _qrs_width(beat_window, lead=0)
        bf.qrs_amplitude = float(np.max(np.abs(beat_window[:, 0] - np.median(beat_window[:, 0]))))
        bf.qrs_energy = float(np.sum(beat_window[:, 0] ** 2))

        rows.append(bf)

    # HRV features: computed once per record over all RR intervals
    sdnn, rmssd = _hrv_features(rr)
    df = pd.DataFrame([bf.__dict__ for bf in rows])
    df["rr_sdnn"] = sdnn
    df["rr_rmssd"] = rmssd
    return df


# ---------------------------------------------------------------------------
# 5. Build full feature matrix
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    "rr_prev",
    "rr_next",
    "rr_ratio",
    "qrs_width",
    "qrs_amplitude",
    "qrs_energy",
    "rr_sdnn",
    "rr_rmssd",
]


def build_feature_matrix(
    record_ids: List[str],
    data_dir: str | Path = "data/mitdb",
) -> pd.DataFrame:
    """
    Build the full tabular feature matrix for a list of records.

    Beats whose annotation symbol is not in AAMI_MAP (paced, unclassifiable,
    etc.) are excluded. Boundary beats lacking a complete RR window are
    dropped (no imputation is applied because RR intervals are clinically
    informative and synthetic values would distort the signal).
    """
    all_frames: List[pd.DataFrame] = []
    total_excluded = 0

    for rid in record_ids:
        try:
            signal, fs, r_peaks, labels, n_excl = load_record(rid, data_dir)
        except FileNotFoundError as e:
            print(f"[build_feature_matrix] Skipping {rid}: {e}")
            continue

        total_excluded += n_excl
        beats, valid_idx = extract_beats(signal, r_peaks)
        df = extract_beat_features(
            beats, r_peaks, labels, fs=fs, record_id=rid, valid_idx=valid_idx,
        )
        all_frames.append(df)
        print(
            f"[build_feature_matrix] {rid}: {len(df)} beats, "
            f"{df['label_binary'].mean():.4f} abnormal rate, "
            f"{n_excl} non-AAMI beats excluded."
        )

    if not all_frames:
        raise RuntimeError("No records were successfully processed.")

    full = pd.concat(all_frames, ignore_index=True)

    # Drop boundary beats with incomplete RR windows (no imputation).
    n_before = len(full)
    full = full.dropna(
        subset=["rr_prev", "rr_next", "rr_ratio"]
    ).reset_index(drop=True)
    n_dropped = n_before - len(full)

    print(
        f"[build_feature_matrix] Total: {len(full)} beats. "
        f"Dropped {n_dropped} boundary beats with incomplete RR. "
        f"Excluded {total_excluded} non-AAMI beats (paced/unclassifiable)."
    )
    print(
        f"[build_feature_matrix] Final class balance: "
        f"{(full['label_binary'] == 1).mean():.4%} abnormal."
    )
    return full


# ---------------------------------------------------------------------------
# 6. Splits (patient-level)
# ---------------------------------------------------------------------------

def make_splits(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SplitResult:
    """
    Stratified patient-level split by per-record abnormality rate,
    with size-aware assignment to preserve the target 70/15/15 beat ratio.

    Records are assigned to four strata based on their individual
    abnormality rate:
        - Low:       rate < 2%
        - Moderate:  2% <= rate < 10%
        - High:      10% <= rate < 30%
        - Extreme:   rate >= 30%

    Within each stratum, records are sorted by descending beat count and
    assigned to whichever split is furthest below its target beat quota.
    This preserves both the class balance and the 70/15/15 proportion.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    per_record = df.groupby("record_id").agg(
        n_beats=("label_binary", "size"),
        rate=("label_binary", "mean"),
    )

    strata = {
        "low":      per_record[per_record["rate"] < 0.02],
        "moderate": per_record[(per_record["rate"] >= 0.02) & (per_record["rate"] < 0.10)],
        "high":     per_record[(per_record["rate"] >= 0.10) & (per_record["rate"] < 0.30)],
        "extreme":  per_record[per_record["rate"] >= 0.30],
    }

    total_beats = per_record["n_beats"].sum()
    targets = {
        "train": train_frac * total_beats,
        "val":   val_frac   * total_beats,
        "test":  test_frac  * total_beats,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    assigned = {"train": [], "val": [], "test": []}

    rng = np.random.default_rng(seed)

    for stratum_name, sub in strata.items():
        # Shuffle within stratum, then sort by size descending (largest first
        # to avoid the "last record doesn't fit" problem)
        records = sub.sample(frac=1.0, random_state=seed).sort_values(
            "n_beats", ascending=False
        )
        for rid, row in records.iterrows():
            n = int(row["n_beats"])
            # Pick split furthest below its quota (relative deficit)
            deficits = {
                s: (targets[s] - counts[s]) / targets[s]
                for s in counts
            }
            target_split = max(deficits, key=deficits.get)
            assigned[target_split].append(str(rid))
            counts[target_split] += n

        print(
            f"[make_splits] {stratum_name:9s}: {len(records):2d} records -> "
            f"train={len(assigned['train'])}, "
            f"val={len(assigned['val'])}, "
            f"test={len(assigned['test'])}"
        )

    print(
        f"[make_splits] Beat counts: "
        f"train={counts['train']} ({counts['train']/total_beats:.1%}), "
        f"val={counts['val']} ({counts['val']/total_beats:.1%}), "
        f"test={counts['test']} ({counts['test']/total_beats:.1%})"
    )

    return SplitResult(
        train_records=sorted(assigned["train"]),
        val_records=sorted(assigned["val"]),
        test_records=sorted(assigned["test"]),
    )

def save_splits(splits: SplitResult, path: str | Path) -> None:
    """Persist splits as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": splits.train_records,
        "val": splits.val_records,
        "test": splits.test_records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[save_splits] Saved to {path}")


def load_splits(path: str | Path) -> SplitResult:
    """Load splits from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return SplitResult(
        train_records=payload["train"],
        val_records=payload["val"],
        test_records=payload["test"],
    )


# ---------------------------------------------------------------------------
# 7. Leakage-aware scaling
# ---------------------------------------------------------------------------

@dataclass
class Scalers:
    """Container for the two-stage scaler (Robust + Standard)."""
    robust: RobustScaler
    standard: StandardScaler


def fit_scalers(X_train: np.ndarray) -> Scalers:
    """
    Fit RobustScaler + StandardScaler on TRAIN ONLY.

    Two-stage scaling mirrors Workshop 2, where the extra mean-std
    transform was essential to stabilize neural network training.
    """
    robust = RobustScaler().fit(X_train)
    X_r = robust.transform(X_train)
    standard = StandardScaler().fit(X_r)
    return Scalers(robust=robust, standard=standard)


def transform(X: np.ndarray, scalers: Scalers) -> np.ndarray:
    """Apply fitted scalers to any split (train / val / test)."""
    return scalers.standard.transform(scalers.robust.transform(X))