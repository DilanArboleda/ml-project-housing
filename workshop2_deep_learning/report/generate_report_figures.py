"""Generate compact report figures from saved Workshop 2 artifacts only."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "workshop2_deep_learning" / "results"
OUTPUT = Path(__file__).resolve().parent / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#264653",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#E76F51",
    "blue": "#4C78A8",
    "gray": "#F4F5F7",
    "dark": "#263238",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def box(ax, x, y, width, height, text, color, fontsize=7, edge="white"):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", color="white",
        fontsize=fontsize, fontweight="semibold",
    )


def arrow(ax, start, end, color="#6B7280"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9,
            linewidth=1.0, color=color, shrinkA=2, shrinkB=2,
        )
    )


def methodology_figure():
    fig, axes = plt.subplots(3, 1, figsize=(7.15, 4.15))
    fig.subplots_adjust(hspace=0.50, left=0.025, right=0.985, top=0.95, bottom=0.05)

    # Panel A: leakage-aware experimental workflow.
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.02, "(a) Leakage-aware experimental workflow", fontweight="bold", va="bottom")
    labels = [
        "Raw Ames\nHousing",
        "Fit Workshop 1\npreprocessor\non train only",
        "Fixed split\n2050 / 440 / 440",
        "Fit X and y log\nstandardizers\non train only",
        "Train MLP",
        "Select by\nvalidation MAE",
        "One frozen\ntest evaluation",
    ]
    widths = [0.105, 0.145, 0.13, 0.15, 0.09, 0.11, 0.11]
    xs = [0.005, 0.135, 0.31, 0.465, 0.645, 0.755, 0.885]
    palette = [COLORS["navy"], COLORS["blue"], COLORS["gold"], COLORS["teal"], COLORS["orange"], COLORS["red"], COLORS["navy"]]
    for x, width, label, color in zip(xs, widths, labels, palette):
        box(ax, x, 0.23, width, 0.50, label, color, fontsize=6.4)
    for index in range(len(xs) - 1):
        arrow(ax, (xs[index] + widths[index], 0.48), (xs[index + 1], 0.48))
    ax.text(0.54, 0.05, "Validation and test never fit preprocessing or scaling statistics", ha="center", color=COLORS["red"], fontsize=7)

    # Panel B: selected network.
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.02, "(b) Selected MLP architecture (41,857 trainable parameters)", fontweight="bold", va="bottom")
    labels = ["261 standardized\nfeatures", "Dense 128", "ReLU", "Dropout\np = 0.05", "Dense 64", "ReLU", "Dropout\np = 0.05", "Linear output\n1 value"]
    xs = [0.015, 0.17, 0.30, 0.40, 0.535, 0.65, 0.75, 0.88]
    widths = [0.12, 0.10, 0.07, 0.10, 0.09, 0.07, 0.10, 0.105]
    palette = [COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["red"]]
    for x, width, label, color in zip(xs, widths, labels, palette):
        box(ax, x, 0.28, width, 0.44, label, color, fontsize=6.8)
    for index in range(len(xs) - 1):
        arrow(ax, (xs[index] + widths[index], 0.50), (xs[index + 1], 0.50))

    # Panel C: target transformation and inverse.
    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.02, "(c) Target transformation and dollar-scale inverse", fontweight="bold", va="bottom")
    labels = [
        "SalePrice y\n(USD)",
        "ylog = log(1 + y)",
        "zy = (ylog - 12.013993)\n/ 0.395353",
        "MLP predicts zy",
        "yhat = exp(zy x 0.395353\n+ 12.013993) - 1",
        "Metrics in USD",
    ]
    xs = [0.03, 0.19, 0.35, 0.55, 0.69, 0.89]
    widths = [0.105, 0.12, 0.155, 0.105, 0.17, 0.09]
    palette = [COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["red"], COLORS["navy"]]
    for x, width, label, color in zip(xs, widths, labels, palette):
        box(ax, x, 0.28, width, 0.44, label, color, fontsize=6.7)
    for index in range(len(xs) - 1):
        arrow(ax, (xs[index] + widths[index], 0.50), (xs[index + 1], 0.50))
    ax.text(0.425, 0.08, "Training", ha="center", color=COLORS["teal"], fontweight="bold")
    ax.text(0.785, 0.08, "Inference", ha="center", color=COLORS["red"], fontweight="bold")

    fig.savefig(OUTPUT / "methodology_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def experiment_figure():
    histories = {
        "Initial": json.loads((RESULTS / "training_history.json").read_text(encoding="utf-8")),
        "Experiment 2": json.loads((RESULTS / "training_history_experiment2.json").read_text(encoding="utf-8")),
    }
    runs = pd.read_csv(RESULTS / "part_a" / "comparison_all_runs.csv")
    stability = pd.DataFrame(
        json.loads((RESULTS / "part_a" / "stability_summary.json").read_text(encoding="utf-8"))
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))
    fig.subplots_adjust(wspace=0.50, left=0.07, right=0.985, top=0.86, bottom=0.24)

    ax = axes[0]
    for name, history in histories.items():
        ax.plot(
            [row["epoch"] for row in history],
            [row["val_mae"] for row in history],
            linewidth=1.4,
            label=name,
            color=COLORS["red"] if name == "Initial" else COLORS["blue"],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (USD)")
    ax.set_title("(a) Conditioning correction")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    seed42 = runs[runs["seed"] == 42].copy()
    short = {
        "experiment2_reference": "Reference",
        "e3_no_dropout": "No dropout",
        "e4_wide_128_64": "128-64",
        "e5_weight_decay_1e4": "WD 1e-4",
        "e6_learning_rate_5e4": "LR 5e-4",
        "a1_no_input_standardization": "No X std.",
        "a2_no_target_standardization": "No y std.",
    }
    seed42["label"] = seed42["config_name"].map(short)
    seed42 = seed42.sort_values("best_val_mae", ascending=True)
    colors = [COLORS["teal"] if name == "128-64" else COLORS["blue"] for name in seed42["label"]]
    ax.barh(seed42["label"], seed42["best_val_mae"], color=colors)
    ax.axvline(14101.86, color=COLORS["red"], linestyle="--", linewidth=1.0, label="Ridge val.")
    ax.set_xlabel("Best validation MAE (USD)")
    ax.set_title("(b) Controlled runs, seed 42")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[2]
    multi = stability[stability["n_seeds"] == 3].copy()
    order = ["e4_wide_128_64", "e5_weight_decay_1e4", "experiment2_reference"]
    multi = multi.set_index("config_name").loc[order].reset_index()
    names = ["128-64", "WD 1e-4", "Reference"]
    x = np.arange(len(names))
    ax.errorbar(
        x, multi["mae_mean"], yerr=multi["mae_std"], fmt="o",
        markersize=5, capsize=4, color=COLORS["navy"], ecolor=COLORS["orange"],
    )
    ax.axhline(14101.86, color=COLORS["red"], linestyle="--", linewidth=1.0)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel("Validation MAE (USD)")
    ax.set_title("(c) Mean +/- SD across seeds")
    ax.text(0.02, 0.02, "Seeds: 7, 42, 2026", transform=ax.transAxes, fontsize=7)

    fig.savefig(OUTPUT / "experimental_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def final_diagnostics_figure():
    predictions = pd.read_csv(RESULTS / "final_test" / "test_predictions.csv")
    comparison = pd.read_csv(RESULTS / "final_test" / "model_comparison_test.csv")
    actual = predictions["SalePrice_actual"].to_numpy()
    predicted = predictions["deep_learning_mlp_predicted"].to_numpy()
    residual = predictions["deep_learning_mlp_residual"].to_numpy()
    extreme = int(np.argmax(np.abs(residual)))

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.7))
    fig.subplots_adjust(wspace=0.34, hspace=0.46, left=0.09, right=0.985, top=0.94, bottom=0.10)

    ax = axes[0, 0]
    ax.scatter(actual, predicted, s=10, alpha=0.48, color=COLORS["navy"], edgecolors="none")
    limit = max(actual.max(), predicted.max()) * 1.03
    ax.plot([0, limit], [0, limit], "--", color=COLORS["red"], linewidth=1.0)
    ax.scatter(actual[extreme], predicted[extreme], s=32, color=COLORS["red"], marker="x", linewidth=1.5)
    ax.set_xlabel("Actual SalePrice (USD)")
    ax.set_ylabel("Predicted SalePrice (USD)")
    ax.set_title("(a) MLP actual vs. predicted")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    ax = axes[0, 1]
    ax.scatter(predicted, residual, s=10, alpha=0.48, color=COLORS["teal"], edgecolors="none")
    ax.axhline(0, color=COLORS["dark"], linestyle="--", linewidth=1.0)
    ax.scatter(predicted[extreme], residual[extreme], s=32, color=COLORS["red"], marker="x", linewidth=1.5)
    ax.set_xlabel("Predicted SalePrice (USD)")
    ax.set_ylabel("Residual: actual - predicted (USD)")
    ax.set_title("(b) Residuals vs. predictions")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    ax = axes[1, 0]
    ax.hist(residual, bins=34, color=COLORS["blue"], alpha=0.86, edgecolor="white")
    ax.axvline(0, color=COLORS["dark"], linestyle="--", linewidth=1.0)
    ax.axvline(residual[extreme], color=COLORS["red"], linestyle=":", linewidth=1.2)
    ax.set_xlabel("Residual (USD)")
    ax.set_ylabel("Count")
    ax.set_title("(c) MLP residual distribution")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax = axes[1, 1]
    names = ["MLP", "Ridge", "RF", "GB"]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, comparison["MAE"], width, label="MAE", color=COLORS["teal"])
    ax.bar(x + width / 2, comparison["RMSE"], width, label="RMSE", color=COLORS["orange"])
    ax.set_xticks(x, names)
    ax.set_ylabel("Error (USD)")
    ax.set_title("(d) Fixed-test dollar errors")
    ax.legend(frameon=False, ncol=2, loc="upper right")

    fig.savefig(OUTPUT / "final_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    methodology_figure()
    experiment_figure()
    final_diagnostics_figure()
    print(f"Generated report figures in {OUTPUT}")
