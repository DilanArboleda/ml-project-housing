"""
Generate dedicated report figures for Workshop 5 LaTeX Report.
Workshop No. 5 — Generative Adversarial Networks.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero Ordeñez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)
"""

from __future__ import annotations
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
W5_DIR = ROOT / "workshop5_gans"
RES_DIR = W5_DIR / "results"
FIG_DIR = RES_DIR / "figures"
REP_FIG_DIR = W5_DIR / "report" / "figures"
REP_FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_augmentation_benchmark():
    """Plot bar charts of downstream performance under the 3 conditions."""
    with open(RES_DIR / "augmentation_results.json", "r") as f:
        aug_data = json.load(f)

    conds = ["Real Only\n(Baseline)", "Real +\nVanilla GAN", "Real +\nWGAN-GP"]
    maes = [
        aug_data["condition_a_real_only"]["test_metrics"]["MAE"],
        aug_data["condition_b_real_plus_vanilla"]["test_metrics"]["MAE"],
        aug_data["condition_c_real_plus_wgan"]["test_metrics"]["MAE"],
    ]
    rmses = [
        aug_data["condition_a_real_only"]["test_metrics"]["RMSE"],
        aug_data["condition_b_real_plus_vanilla"]["test_metrics"]["RMSE"],
        aug_data["condition_c_real_plus_wgan"]["test_metrics"]["RMSE"],
    ]
    r2s = [
        aug_data["condition_a_real_only"]["test_metrics"]["R2"] * 100,
        aug_data["condition_b_real_plus_vanilla"]["test_metrics"]["R2"] * 100,
        aug_data["condition_c_real_plus_wgan"]["test_metrics"]["R2"] * 100,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE
    bars1 = axes[0].bar(conds, maes, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
    axes[0].set_title("Downstream Test MAE ($ USD)")
    axes[0].set_ylabel("Mean Absolute Error ($)")
    axes[0].yaxis.set_major_formatter("${x:,.0f}")
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 300, f"${b.get_height():,.0f}", ha="center", fontsize=9.5, fontweight="bold")
    axes[0].set_ylim(0, max(maes) * 1.18)

    # RMSE
    bars2 = axes[1].bar(conds, rmses, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
    axes[1].set_title("Downstream Test RMSE ($ USD)")
    axes[1].set_ylabel("Root Mean Squared Error ($)")
    axes[1].yaxis.set_major_formatter("${x:,.0f}")
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 600, f"${b.get_height():,.0f}", ha="center", fontsize=9.5, fontweight="bold")
    axes[1].set_ylim(0, max(rmses) * 1.18)

    # R2
    bars3 = axes[2].bar(conds, r2s, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
    axes[2].set_title("Downstream Test $R^2$ Variance Explained (%)")
    axes[2].set_ylabel("$R^2$ Score (%)")
    for b in bars3:
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.2f}%", ha="center", fontsize=9.5, fontweight="bold")
    axes[2].set_ylim(0, 100)

    plt.tight_layout()
    out_path = REP_FIG_DIR / "downstream_augmentation_benchmark.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def copy_all_figures():
    """Copy all generated figures to the report figure directory."""
    for img in FIG_DIR.glob("*.png"):
        shutil.copy(img, REP_FIG_DIR / img.name)
    print("All figures synced to report/figures/!")


if __name__ == "__main__":
    plot_augmentation_benchmark()
    copy_all_figures()
