"""
Visualization utilities for Workshop 6 (MIT-BIH Arrhythmia).

This module provides publication-quality plots for the medical EDA:
    1. Class distribution (binary and AAMI superclass).
    2. Raw ECG segments with R-peaks annotated by class.
    3. Representative beat grids (Normal vs Abnormal).
    4. Per-feature distributions by class.
    5. Correlation heatmap of tabular features.

All plots use matplotlib + seaborn to remain consistent with the
visual style of Workshops 1-5. Every function returns the figure
object so that notebooks can display or save it explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Default palette used across the workshop.
PALETTE = {
    "normal":   "#2E86AB",
    "abnormal": "#D1495B",
}
AAMI_COLORS = {
    "N": "#2E86AB",  # Normal
    "S": "#F4A261",  # Supraventricular ectopic
    "V": "#D1495B",  # Ventricular ectopic
    "F": "#8E6C88",  # Fusion
}
FEATURE_COLUMNS = [
    "rr_prev", "rr_next", "rr_ratio",
    "qrs_width", "qrs_amplitude", "qrs_energy",
    "rr_sdnn", "rr_rmssd",
]


# ---------------------------------------------------------------------------
# 1. Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """
    Plot class balance (binary and AAMI superclass) side by side.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with 'label_binary' and 'label_aami' columns.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Binary
    binary_counts = df["label_binary"].value_counts().sort_index()
    labels = ["Normal (0)", "Abnormal (1)"]
    colors = [PALETTE["normal"], PALETTE["abnormal"]]
    bars = axes[0].bar(labels, binary_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Binary Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Number of beats")
    axes[0].grid(axis="y", alpha=0.3)

    total = binary_counts.sum()
    for bar, val in zip(bars, binary_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{val:,}\n({val/total:.2%})",
            ha="center", va="bottom", fontsize=10,
        )

    # AAMI
    aami_counts = df["label_aami"].value_counts().reindex(
        ["N", "S", "V", "F"]
    ).fillna(0).astype(int)
    colors_aami = [AAMI_COLORS[k] for k in aami_counts.index]
    bars = axes[1].bar(
        aami_counts.index, aami_counts.values,
        color=colors_aami, edgecolor="black",
    )
    axes[1].set_title("AAMI Superclass Distribution", fontweight="bold")
    axes[1].set_ylabel("Number of beats")
    axes[1].set_xlabel("AAMI class")
    axes[1].grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, aami_counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Raw ECG segments with R-peaks
# ---------------------------------------------------------------------------

def plot_ecg_segment(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    beat_labels: np.ndarray,
    fs: int,
    start_s: float = 0.0,
    duration_s: float = 10.0,
    lead: int = 0,
    figsize: tuple[float, float] = (14, 3),
) -> plt.Figure:
    """
    Plot a raw ECG segment with R-peaks colored by AAMI class.

    Parameters
    ----------
    signal : np.ndarray of shape (N, n_leads)
        Raw ECG signal.
    r_peaks : np.ndarray
        Sample indices of R-peaks.
    beat_labels : np.ndarray of str
        AAMI label for each R-peak.
    fs : int
        Sampling frequency (Hz).
    start_s, duration_s : float
        Time window (seconds) to display.
    lead : int
        Which lead to plot (0 = MLII, 1 = V1/V5).

    Returns
    -------
    plt.Figure
    """
    start_sample = int(start_s * fs)
    end_sample = int((start_s + duration_s) * fs)
    end_sample = min(end_sample, len(signal))

    time = np.arange(start_sample, end_sample) / fs
    segment = signal[start_sample:end_sample, lead]

    # Filter R-peaks within the window
    mask = (r_peaks >= start_sample) & (r_peaks < end_sample)
    peaks_in_window = r_peaks[mask]
    labels_in_window = beat_labels[mask]
    peak_times = peaks_in_window / fs

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, segment, color="#333333", linewidth=0.8, label="ECG")

    # Color peaks by class
    for peak_t, label in zip(peak_times, labels_in_window):
        color = AAMI_COLORS.get(label, "#999999")
        peak_val = segment[int(peak_t * fs) - start_sample]
        ax.axvline(peak_t, color=color, alpha=0.4, linewidth=0.8)
        ax.scatter(
            peak_t, peak_val, color=color, s=30, zorder=5,
            edgecolor="black", linewidth=0.4,
        )

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=AAMI_COLORS[k], markersize=8,
                   label=f"{k} ({k})")
        for k in ["N", "S", "V", "F"]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(
        f"ECG segment [{start_s:.1f}s - {start_s + duration_s:.1f}s] — "
        f"lead {lead}",
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Representative beats grid
# ---------------------------------------------------------------------------

def plot_representative_beats(
    beats: np.ndarray,
    labels: np.ndarray,
    fs: int,
    n_per_class: int = 4,
    pre_samples: int = 200,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """
    Grid of representative beats per AAMI class.

    Parameters
    ----------
    beats : np.ndarray of shape (n_beats, window, n_leads)
        Beat windows extracted by extract_beats().
    labels : np.ndarray of str
        AAMI label per beat.
    fs : int
        Sampling frequency.
    n_per_class : int
        How many beats to show per class.
    pre_samples : int
        Samples before R-peak (used for x-axis centering).

    Returns
    -------
    plt.Figure
    """
    classes = ["N", "S", "V", "F"]
    classes = [c for c in classes if c in labels]  # only present ones
    fig, axes = plt.subplots(
        len(classes), n_per_class,
        figsize=figsize, sharex=True, sharey=True,
    )
    if len(classes) == 1:
        axes = axes[np.newaxis, :]

    time = (np.arange(beats.shape[1]) - pre_samples) / fs

    for row, cls in enumerate(classes):
        idxs = np.where(labels == cls)[0]
        # Take a few evenly spaced samples
        if len(idxs) > n_per_class:
            chosen = idxs[np.linspace(0, len(idxs) - 1, n_per_class).astype(int)]
        else:
            chosen = idxs

        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(chosen):
                beat = beats[chosen[col], :, 0]
                ax.plot(time, beat, color=AAMI_COLORS[cls], linewidth=1.2)
                ax.axvline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
                ax.set_title(f"{cls} #{col + 1}", fontsize=9)
                ax.grid(alpha=0.3)
            else:
                ax.axis("off")

        axes[row, 0].set_ylabel(f"Class {cls}\nAmplitude", fontsize=9)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time relative to R-peak (s)")

    fig.suptitle("Representative Beats per AAMI Class", fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Feature distributions by class
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Histograms of each tabular feature, split by binary class.

    Parameters
    ----------
    df : pd.DataFrame
    features : sequence of str, optional
        Which feature columns to plot. Defaults to FEATURE_COLUMNS.

    Returns
    -------
    plt.Figure
    """
    if features is None:
        features = FEATURE_COLUMNS

    n = len(features)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for cls_val, name, color in [
            (0, "Normal",   PALETTE["normal"]),
            (1, "Abnormal", PALETTE["abnormal"]),
        ]:
            subset = df[df["label_binary"] == cls_val][feat].dropna()
            if len(subset) > 0:
                ax.hist(
                    subset, bins=40, alpha=0.55,
                    label=f"{name} (n={len(subset):,})",
                    color=color, density=True,
                )
        ax.set_title(feat, fontweight="bold")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Tabular Feature Distributions by Class",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_feature_correlation(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Correlation heatmap of the tabular features.

    Parameters
    ----------
    df : pd.DataFrame
    features : sequence of str, optional

    Returns
    -------
    plt.Figure
    """
    if features is None:
        features = FEATURE_COLUMNS

    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Utility: save figure
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save a figure to disk with consistent formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[save_figure] Saved to {path}")