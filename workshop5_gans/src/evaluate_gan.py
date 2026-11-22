"""
Quantitative generation quality evaluation for Vanilla GAN and WGAN-GP.
Workshop No. 5 — Generative Adversarial Networks.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero Ordeñez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)

Computes:
1. Fréchet Distance (FD) between real and synthetic multivariate distributions.
2. Wasserstein Distance (Earth Mover's Distance) on marginals and PCA components.
3. Correlation Matrix Distance (Frobenius norm ||C_real - C_synth||_F).
4. Kolmogorov-Smirnov (KS) marginal distribution discrepancy test.
5. Diversity / Mode Collapse Diagnostic: Pairwise Euclidean sample dispersion.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy import stats
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]


def calculate_frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6
) -> float:
    """
    Computes Fréchet Distance between two Gaussians N(mu1, sigma1) and N(mu2, sigma2):
    FD = ||mu1 - mu2||_2^2 + Tr(sigma1 + sigma2 - 2*(sigma1 * sigma2)^{1/2})
    """
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical imaginary artifact correction
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # Just take real part
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def evaluate_dataset(real_mat: np.ndarray, synth_mat: np.ndarray, name: str) -> dict:
    """Compute comprehensive evaluation metrics for a synthetic tabular sample."""
    # 1. PCA embedding for stable Fréchet distance
    pca = PCA(n_components=10, random_state=42)
    real_pca = pca.fit_transform(real_mat)
    synth_pca = pca.transform(synth_mat)

    mu_r, sig_r = np.mean(real_pca, axis=0), np.cov(real_pca, rowvar=False)
    mu_s, sig_s = np.mean(synth_pca, axis=0), np.cov(synth_pca, rowvar=False)
    frechet_dist = calculate_frechet_distance(mu_r, sig_r, mu_s, sig_s)

    # 2. Correlation Matrix Distance (Frobenius norm)
    # Using top 20 PCA components or features
    c_real = np.corrcoef(real_mat[:, :30], rowvar=False)
    c_synth = np.corrcoef(synth_mat[:, :30], rowvar=False)
    # Replace NaNs if any constant columns
    c_real = np.nan_to_num(c_real)
    c_synth = np.nan_to_num(c_synth)
    frob_corr_dist = float(np.linalg.norm(c_real - c_synth, ord="fro"))
    mean_abs_corr_diff = float(np.mean(np.abs(c_real - c_synth)))

    # 3. Kolmogorov-Smirnov Test (Average KS statistic across features)
    ks_stats = []
    p_vals = []
    for col in range(min(50, real_mat.shape[1])):
        ks_res = stats.ks_2samp(real_mat[:, col], synth_mat[:, col])
        ks_stats.append(ks_res.statistic)
        p_vals.append(ks_res.pvalue)
    mean_ks = float(np.mean(ks_stats))

    # 4. 1-Wasserstein Distance (Earth Mover's Distance)
    w_dists = []
    for col in range(min(50, real_mat.shape[1])):
        w_dists.append(stats.wasserstein_distance(real_mat[:, col], synth_mat[:, col]))
    mean_wasserstein = float(np.mean(w_dists))

    # 5. Diversity / Dispersion Metric (Pairwise distance)
    # Subsample 400 for speed
    sub_real = real_mat[np.random.choice(len(real_mat), 400, replace=False)]
    sub_synth = synth_mat[np.random.choice(len(synth_mat), 400, replace=False)]

    pw_real = scipy.spatial.distance.pdist(sub_real, metric="euclidean")
    pw_synth = scipy.spatial.distance.pdist(sub_synth, metric="euclidean")

    mean_pw_dist = float(np.mean(pw_synth))
    ratio_pw_dist = float(mean_pw_dist / np.mean(pw_real))

    # Mode collapse indicator: if ratio is < 0.60 or std is very low
    std_pw_dist = float(np.std(pw_synth))

    return {
        "model_name": name,
        "frechet_distance_pca10": frechet_dist,
        "correlation_frobenius_norm": frob_corr_dist,
        "correlation_mean_abs_diff": mean_abs_corr_diff,
        "mean_ks_statistic": mean_ks,
        "mean_wasserstein_distance": mean_wasserstein,
        "mean_pairwise_sample_distance": mean_pw_dist,
        "real_pairwise_sample_distance": float(np.mean(pw_real)),
        "pairwise_distance_ratio": ratio_pw_dist,
        "pairwise_distance_std": std_pw_dist,
        "mode_collapse_detected": bool(ratio_pw_dist < 0.65 or std_pw_dist < 0.5),
    }


def main():
    w5_dir = ROOT / "workshop5_gans"
    results_dir = w5_dir / "results"
    data_path = results_dir / "synthetic_samples.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run generate_samples.py first.")

    np.random.seed(42)
    npz = np.load(data_path)
    real_norm = npz["real_norm"]
    vanilla_norm = npz["vanilla_norm"]
    wgan_norm = npz["wgan_norm"]

    print("=" * 80)
    print("EVALUATING GENERATION QUALITY: VANILLA GAN vs. WGAN-GP")
    print("=" * 80)

    eval_vanilla = evaluate_dataset(real_norm, vanilla_norm, "Vanilla GAN")
    eval_wgan = evaluate_dataset(real_norm, wgan_norm, "WGAN-GP")

    results = {
        "Vanilla GAN": eval_vanilla,
        "WGAN-GP": eval_wgan,
        "comparison": {
            "lower_frechet_distance": "WGAN-GP" if eval_wgan["frechet_distance_pca10"] < eval_vanilla["frechet_distance_pca10"] else "Vanilla GAN",
            "lower_correlation_error": "WGAN-GP" if eval_wgan["correlation_frobenius_norm"] < eval_vanilla["correlation_frobenius_norm"] else "Vanilla GAN",
            "lower_ks_statistic": "WGAN-GP" if eval_wgan["mean_ks_statistic"] < eval_vanilla["mean_ks_statistic"] else "Vanilla GAN",
            "higher_diversity": "WGAN-GP" if eval_wgan["mean_pairwise_sample_distance"] > eval_vanilla["mean_pairwise_sample_distance"] else "Vanilla GAN",
        }
    }

    out_json = results_dir / "generation_quality_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- Summary of Generation Quality Metrics ---")
    print(f"{'Metric':<35} | {'Vanilla GAN':<15} | {'WGAN-GP':<15}")
    print("-" * 70)
    print(f"{'Fréchet Distance (PCA-10)':<35} | {eval_vanilla['frechet_distance_pca10']:<15.4f} | {eval_wgan['frechet_distance_pca10']:<15.4f}")
    print(f"{'Correlation Frobenius Dist':<35} | {eval_vanilla['correlation_frobenius_norm']:<15.4f} | {eval_wgan['correlation_frobenius_norm']:<15.4f}")
    print(f"{'Correlation Mean Abs Diff':<35} | {eval_vanilla['correlation_mean_abs_diff']:<15.4f} | {eval_wgan['correlation_mean_abs_diff']:<15.4f}")
    print(f"{'Mean KS Statistic':<35} | {eval_vanilla['mean_ks_statistic']:<15.4f} | {eval_wgan['mean_ks_statistic']:<15.4f}")
    print(f"{'Mean Wasserstein Dist':<35} | {eval_vanilla['mean_wasserstein_distance']:<15.4f} | {eval_wgan['mean_wasserstein_distance']:<15.4f}")
    print(f"{'Pairwise Sample Distance':<35} | {eval_vanilla['mean_pairwise_sample_distance']:<15.4f} | {eval_wgan['mean_pairwise_sample_distance']:<15.4f}")
    print(f"{'Pairwise Diversity Ratio':<35} | {eval_vanilla['pairwise_distance_ratio']:<15.4f} | {eval_wgan['pairwise_distance_ratio']:<15.4f}")
    print(f"{'Mode Collapse Detected':<35} | {str(eval_vanilla['mode_collapse_detected']):<15} | {str(eval_wgan['mode_collapse_detected']):<15}")
    print("-" * 70)
    print(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
