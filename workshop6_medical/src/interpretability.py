"""
Interpretability utilities for Workshop 6 (MIT-BIH Arrhythmia).

Provides:
    1. Saliency maps for tabular MLP (gradient of output w.r.t. input).
    2. Permutation importance for feature attribution.
    3. RF feature importance.
    4. ROC and Precision-Recall curves comparison.
    5. Threshold sensitivity/specificity curves.
    6. Confusion matrices per model.
    7. Beat-to-spectrogram conversion for the image branch (FASE 3 advanced).

All plots use matplotlib + seaborn for consistency with Workshops 1-5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
)

# Reuse the MLP from transfer.py
import sys
sys.path.append(str(Path(__file__).parent))
from transfer import MedicalMLP, predict_mlp_proba

PALETTE = {
    "normal":   "#2E86AB",
    "abnormal": "#D1495B",
}
MODEL_COLORS = {
    "LogReg":           "#264653",
    "RandomForest":     "#2A9D8F",
    "GradientBoosting": "#E9C46A",
    "MLP":              "#E76F51",
}


# ---------------------------------------------------------------------------
# 1. Saliency map for tabular MLP
# ---------------------------------------------------------------------------

def compute_saliency_mlp(
    model: MedicalMLP,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute saliency (absolute gradient of class-1 logit wrt input).

    Parameters
    ----------
    model : MedicalMLP
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    -------
    saliency : np.ndarray of shape (n_samples, n_features)
        Absolute gradient per sample and feature.
    """
    device = next(model.parameters()).device
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    logits = model(X_t)
    # Gradient of the abnormal-class logit wrt input
    score = logits[:, 1].sum()
    score.backward()
    saliency = X_t.grad.abs().cpu().numpy()
    return saliency


def plot_saliency_heatmap(
    model: MedicalMLP,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_per_class: int = 5,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """
    Heatmap of saliency values for representative samples per class.

    Shows which features most strongly drive the MLP's decision
    (absolute gradient) for Normal vs Abnormal predictions.
    """
    saliency = compute_saliency_mlp(model, X)

    normal_idx = np.where(y == 0)[0][:n_per_class]
    abnormal_idx = np.where(y == 1)[0][:n_per_class]
    chosen = np.concatenate([normal_idx, abnormal_idx])
    labels = ["N"] * len(normal_idx) + ["Ab"] * len(abnormal_idx)

    mat = saliency[chosen]  # (2*n, n_features)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=feature_names,
        yticklabels=[f"{lab} #{i}" for i, lab in enumerate(labels)],
        cbar_kws={"label": "|∂logit_abnormal / ∂x|"},
        ax=ax,
    )
    ax.set_title("MLP Saliency Maps: Feature Attribution per Prediction",
                 fontweight="bold")
    ax.set_xlabel("Tabular feature")
    plt.tight_layout()
    return fig


def aggregate_saliency_by_class(
    model: MedicalMLP,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Bar plot of mean saliency per feature, separated by true class.

    Reveals which features are globally most important for
    distinguishing Normal vs Abnormal predictions.
    """
    saliency = compute_saliency_mlp(model, X)
    means_normal = saliency[y == 0].mean(axis=0)
    means_abnormal = saliency[y == 1].mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(feature_names))
    width = 0.4
    ax.bar(x - width/2, means_normal, width, label="Normal", color=PALETTE["normal"])
    ax.bar(x + width/2, means_abnormal, width, label="Abnormal", color=PALETTE["abnormal"])
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_ylabel("Mean |gradient|")
    ax.set_title("Aggregate Saliency by Feature and True Class", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Permutation importance (model-agnostic)
# ---------------------------------------------------------------------------

def permutation_importance_proba(
    predict_proba_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Permutation importance: average drop in PR-AUC when each feature
    is randomly shuffled (breaks its association with the label).

    Higher drop = more important feature.
    """
    rng = np.random.default_rng(seed)
    baseline_proba = predict_proba_fn(X)
    baseline_pr_auc = average_precision_score(y, baseline_proba)

    importances = []
    n_features = X.shape[1]
    for i in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, i])
            proba = predict_proba_fn(X_perm)
            pr_auc = average_precision_score(y, proba)
            drops.append(baseline_pr_auc - pr_auc)
        importances.append(float(np.mean(drops)))
    return {"baseline_pr_auc": float(baseline_pr_auc), "importances": importances}


def plot_permutation_importance(
    importances_dict: Dict,
    feature_names: Sequence[str],
    title: str = "Permutation Importance (drop in PR-AUC)",
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    imp = importances_dict["importances"]
    order = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        np.arange(len(imp)), np.array(imp)[order],
        color=PALETTE["abnormal"], edgecolor="black",
    )
    ax.set_xticks(np.arange(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("Mean drop in PR-AUC")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. ROC and PR curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    probas: Dict[str, np.ndarray],
    figsize: tuple[float, float] = (6, 6),
) -> plt.Figure:
    """Overlay ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, y_proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(name, "gray")
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})",
                color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
            label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves — Test Set", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    probas: Dict[str, np.ndarray],
    figsize: tuple[float, float] = (6, 6),
) -> plt.Figure:
    """Overlay Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, y_proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        color = MODEL_COLORS.get(name, "gray")
        ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.4f})",
                color=color, linewidth=2)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curves — Test Set", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Threshold sensitivity/specificity plot
# ---------------------------------------------------------------------------

def plot_threshold_curves(
    sweep_df: pd.DataFrame,
    title: str = "Sensitivity / Specificity vs Threshold",
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sweep_df["threshold"], sweep_df["sensitivity"],
            marker="o", label="Sensitivity", color=PALETTE["abnormal"])
    ax.plot(sweep_df["threshold"], sweep_df["specificity"],
            marker="s", label="Specificity", color=PALETTE["normal"])
    ax.plot(sweep_df["threshold"], sweep_df["f1"],
            marker="^", label="F1", color="gray", linestyle="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    y_true: np.ndarray,
    probas: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    n = len(probas)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, y_proba) in zip(axes, probas.items()):
        thr = thresholds[name]
        y_pred = (y_proba >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Pred Normal", "Pred Ab"],
            yticklabels=["True Normal", "True Ab"],
        )
        ax.set_title(f"{name}\n(threshold={thr:.2f})", fontweight="bold")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Spectrogram (image branch — for optional Grad-CAM CNN)
# ---------------------------------------------------------------------------

def beat_to_spectrogram(
    beat: np.ndarray,
    fs: int = 360,
    size: int = 224,
) -> np.ndarray:
    """
    Convert a single-lead beat window into a normalized spectrogram.

    Parameters
    ----------
    beat : np.ndarray of shape (n_samples,) or (n_samples, n_leads)
        Beat signal. If 2D, lead 0 is used.
    fs : int
        Sampling frequency.
    size : int
        Output image size (size x size).

    Returns
    -------
    spec : np.ndarray of shape (size, size)
        Normalized spectrogram in [0, 1].
    """
    from scipy.signal import spectrogram

    if beat.ndim == 2:
        beat = beat[:, 0]
    f, t, Sxx = spectrogram(beat, fs=fs, nperseg=64, noverlap=48)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    Sxx_db = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-10)

    # Resize to (size, size) using PIL for simplicity
    from PIL import Image
    img = Image.fromarray((Sxx_db * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def plot_beat_spectrograms(
    beats: np.ndarray,
    labels: np.ndarray,
    fs: int = 360,
    n_per_class: int = 3,
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """
    Grid of spectrograms for representative Normal vs Abnormal beats.
    Useful for the image branch (Grad-CAM on a CNN).
    """
    classes = ["N", "Ab"]
    fig, axes = plt.subplots(2, n_per_class, figsize=figsize)

    for row, cls in enumerate(classes):
        if cls == "N":
            idxs = np.where(labels == "N")[0][:n_per_class]
        else:
            idxs = np.where(labels != "N")[0][:n_per_class]

        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(idxs):
                spec = beat_to_spectrogram(beats[idxs[col]], fs=fs)
                ax.imshow(spec, cmap="viridis", aspect="auto", origin="lower")
                ax.set_title(f"{cls} #{col+1}", fontsize=9)
            ax.axis("off")

    fig.suptitle("Beat Spectrograms (224×224) — Image Branch for Grad-CAM",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Save helper
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[save_figure] Saved to {path}")