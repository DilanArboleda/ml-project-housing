"""
Sample generation and diagnostic visualization for Vanilla GAN and WGAN-GP.
Workshop No. 5 — Generative Adversarial Networks.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero Ordeñez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_and_preprocess_data  # noqa: E402
from gan import TabularGenerator  # noqa: E402
from train_gan import TabularDataBundle, set_seed  # noqa: E402

# Publication style
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

RANDOM_SEED = 42
DEVICE = torch.device("cpu")


def load_trained_generator(ckpt_path: Path) -> tuple[TabularGenerator, int, int]:
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    latent_dim = checkpoint["latent_dim"]
    output_dim = checkpoint["output_dim"]

    gen = TabularGenerator(latent_dim=latent_dim, output_dim=output_dim).to(DEVICE)
    gen.load_state_dict(checkpoint["generator"])
    gen.eval()
    return gen, latent_dim, output_dim


def generate_synthetic_samples(
    generator: TabularGenerator,
    latent_dim: int,
    bundle: TabularDataBundle,
    num_samples: int = 2000,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic samples: returns (norm_samples, X_features, y_log, y_dollar)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    z = torch.randn(num_samples, latent_dim, device=DEVICE)
    with torch.no_grad():
        fake_norm = generator(z).cpu().numpy()

    # Inverse transform
    X_features, y_log, y_dollar = bundle.inverse_transform(fake_norm)
    return fake_norm, X_features, y_log, y_dollar


def plot_training_curves(results_dir: Path, fig_dir: Path):
    """Plot training curves comparing Vanilla GAN and WGAN-GP."""
    with open(results_dir / "vanilla_gan_history.json", "r") as f:
        v_hist = json.load(f)
    with open(results_dir / "wgan_gp_history.json", "r") as f:
        w_hist = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Vanilla GAN Losses
    axes[0].plot(v_hist["epoch"], v_hist["d_loss"], label="Discriminator Loss", color="#1f77b4", lw=2)
    axes[0].plot(v_hist["epoch"], v_hist["g_loss"], label="Generator Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("Vanilla GAN: Loss Trajectories (BCE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # 2. Vanilla GAN Accuracies
    axes[1].plot(v_hist["epoch"], [a * 100 for a in v_hist["d_acc_real"]], label="D(Real) Accuracy", color="#2ca02c", lw=1.8)
    axes[1].plot(v_hist["epoch"], [a * 100 for a in v_hist["d_acc_fake"]], label="D(Fake) Accuracy", color="#d62728", lw=1.8)
    axes[1].axhline(50, color="gray", linestyle="--", label="Equilibrium (50%)")
    axes[1].set_title("Vanilla GAN: Discriminator Discrimination Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    # 3. WGAN-GP Critic & Wasserstein Estimate
    axes[2].plot(w_hist["epoch"], w_hist["wasserstein_distance"], label="Est. Wasserstein Distance", color="#9467bd", lw=2)
    axes[2].plot(w_hist["epoch"], w_hist["gp"], label="Gradient Penalty (GP)", color="#8c564b", linestyle=":", lw=1.5)
    axes[2].set_title("WGAN-GP: Wasserstein Metric & Gradient Penalty")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Metric Value")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = fig_dir / "training_curves_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_feature_distributions(
    real_df: pd.DataFrame,
    vanilla_df: pd.DataFrame,
    wgan_df: pd.DataFrame,
    fig_dir: Path,
):
    """Plot KDE overlays of key continuous and ordinal features."""
    features = [
        ("SalePrice", "SalePrice ($ USD)", True),
        ("TotalSF", "Total Living Footprint (sq ft)", False),
        ("Gr Liv Area", "Above Ground Living Area (sq ft)", False),
        ("Overall Qual", "Overall Quality Rating (1–10)", False),
        ("Year Built", "Construction Year", False),
        ("Garage Area", "Garage Size (sq ft)", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (col, label, is_currency) in enumerate(features):
        ax = axes[idx]
        if col not in real_df.columns:
            continue

        sns.kdeplot(real_df[col].dropna(), ax=ax, label="Real (Train)", color="#1f77b4", lw=2.5, fill=True, alpha=0.15)
        sns.kdeplot(vanilla_df[col].dropna(), ax=ax, label="Vanilla GAN", color="#ff7f0e", lw=2.0, linestyle="--")
        sns.kdeplot(wgan_df[col].dropna(), ax=ax, label="WGAN-GP", color="#2ca02c", lw=2.2)

        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        if is_currency:
            ax.xaxis.set_major_formatter("${x:,.0f}")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = fig_dir / "feature_distributions_kde.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_correlation_fidelity(
    real_df: pd.DataFrame,
    vanilla_df: pd.DataFrame,
    wgan_df: pd.DataFrame,
    fig_dir: Path,
):
    """Plot correlation heatmaps and Frobenius distance error matrix."""
    cols = ["SalePrice", "TotalSF", "Gr Liv Area", "Overall Qual", "Year Built", "Garage Area", "TotalBath", "HouseAge"]
    cols = [c for c in cols if c in real_df.columns and c in vanilla_df.columns and c in wgan_df.columns]

    corr_real = real_df[cols].corr()
    corr_vanilla = vanilla_df[cols].corr()
    corr_wgan = wgan_df[cols].corr()

    diff_vanilla = (corr_real - corr_vanilla).abs()
    diff_wgan = (corr_real - corr_wgan).abs()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_real, ax=axes[0], cmap=cmap, vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", cbar=False, square=True)
    axes[0].set_title("Real Data Correlation Matrix")

    sns.heatmap(corr_vanilla, ax=axes[1], cmap=cmap, vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", cbar=False, square=True)
    axes[1].set_title(f"Vanilla GAN Correlations (Mean Abs Err: {diff_vanilla.values.mean():.3f})")

    sns.heatmap(corr_wgan, ax=axes[2], cmap=cmap, vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", cbar=True, square=True)
    axes[2].set_title(f"WGAN-GP Correlations (Mean Abs Err: {diff_wgan.values.mean():.3f})")

    plt.tight_layout()
    out_path = fig_dir / "correlation_matrix_fidelity.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_pca_manifold_projection(
    real_norm: np.ndarray,
    vanilla_norm: np.ndarray,
    wgan_norm: np.ndarray,
    fig_dir: Path,
    n_samples: int = 600,
):
    """Plot 2D PCA projection of real vs generated samples."""
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca.fit(real_norm)

    r_pca = pca.transform(real_norm[:n_samples])
    v_pca = pca.transform(vanilla_norm[:n_samples])
    w_pca = pca.transform(wgan_norm[:n_samples])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Real vs Vanilla
    axes[0].scatter(r_pca[:, 0], r_pca[:, 1], alpha=0.45, label="Real Data", color="#1f77b4", s=25)
    axes[0].scatter(v_pca[:, 0], v_pca[:, 1], alpha=0.55, label="Vanilla GAN", color="#ff7f0e", s=25, marker="^")
    axes[0].set_title(f"2D PCA Projection: Real vs. Vanilla GAN\nExplained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Real vs WGAN-GP
    axes[1].scatter(r_pca[:, 0], r_pca[:, 1], alpha=0.45, label="Real Data", color="#1f77b4", s=25)
    axes[1].scatter(w_pca[:, 0], w_pca[:, 1], alpha=0.55, label="WGAN-GP", color="#2ca02c", s=25, marker="s")
    axes[1].set_title(f"2D PCA Projection: Real vs. WGAN-GP\nExplained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = fig_dir / "pca_manifold_projection.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    set_seed(RANDOM_SEED)
    bundle = TabularDataBundle(seed=RANDOM_SEED)

    w5_dir = ROOT / "workshop5_gans"
    ckpt_dir = w5_dir / "checkpoints"
    results_dir = w5_dir / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trained generators...")
    gen_vanilla, lat_v, _ = load_trained_generator(ckpt_dir / "gan_vanilla_best.pt")
    gen_wgan, lat_w, _ = load_trained_generator(ckpt_dir / "gan_wgan_best.pt")

    print("Generating 2,000 synthetic samples from each model...")
    v_norm, v_X, v_ylog, v_ydollar = generate_synthetic_samples(gen_vanilla, lat_v, bundle, num_samples=2000, seed=42)
    w_norm, w_X, w_ylog, w_ydollar = generate_synthetic_samples(gen_wgan, lat_w, bundle, num_samples=2000, seed=42)

    # Reconstruct readable DataFrames
    feat_names = bundle.preprocessor.feature_names
    cont_names = bundle.preprocessor.continuous_cols
    
    # Real train DataFrame
    raw_train_X, raw_train_ylog, raw_train_ydollar = bundle.inverse_transform(bundle.train_norm)
    real_df = pd.DataFrame(raw_train_X[:, :len(cont_names)], columns=cont_names)
    real_df["SalePrice"] = raw_train_ydollar

    # Vanilla GAN DataFrame
    vanilla_df = pd.DataFrame(v_X[:, :len(cont_names)], columns=cont_names)
    vanilla_df["SalePrice"] = v_ydollar

    # WGAN-GP DataFrame
    wgan_df = pd.DataFrame(w_X[:, :len(cont_names)], columns=cont_names)
    wgan_df["SalePrice"] = w_ydollar

    # Save synthetic samples to NPZ and CSV
    np.savez_compressed(
        results_dir / "synthetic_samples.npz",
        vanilla_norm=v_norm,
        vanilla_X=v_X,
        vanilla_y_log=v_ylog,
        vanilla_y_dollar=v_ydollar,
        wgan_norm=w_norm,
        wgan_X=w_X,
        wgan_y_log=w_ylog,
        wgan_y_dollar=w_ydollar,
        real_norm=bundle.train_norm,
        real_X=raw_train_X,
        real_y_log=raw_train_ylog,
        real_y_dollar=raw_train_ydollar,
    )
    vanilla_df.head(100).to_csv(results_dir / "vanilla_samples_preview.csv", index=False)
    wgan_df.head(100).to_csv(results_dir / "wgan_samples_preview.csv", index=False)
    print("Saved synthetic sample files to results/synthetic_samples.npz and preview CSVs.")

    # Generate Figures
    print("Generating diagnostic plots...")
    plot_training_curves(results_dir, fig_dir)
    plot_feature_distributions(real_df, vanilla_df, wgan_df, fig_dir)
    plot_correlation_fidelity(real_df, vanilla_df, wgan_df, fig_dir)
    plot_pca_manifold_projection(bundle.train_norm, v_norm, w_norm, fig_dir)

    print("Sample generation and plotting complete!")


if __name__ == "__main__":
    main()
