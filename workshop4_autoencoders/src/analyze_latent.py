"""Frozen latent-space analysis for AE, VAE, and supervised MLP models."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import argparse
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ml_project_housing_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data_loader import get_dataset_partitions  # noqa: E402
from data_adapter import load_unsupervised_data  # noqa: E402
from extract_mlp_features import extract_mlp_test_features, file_sha256  # noqa: E402
from train_ae import load_autoencoder_checkpoint  # noqa: E402
from train_vae import load_vae_checkpoint  # noqa: E402


BASE = ROOT / "workshop4_autoencoders"
CHECKPOINTS = BASE / "checkpoints"
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures" / "latent"
RANDOM_STATE = 42
MODEL_ORDER = ("AE z=128", "AE z=64", "VAE z=64 beta=0.5", "MLP z=64")


def _index_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _verify_unsupervised_checkpoint(payload: dict[str, Any], data, name: str) -> None:
    metadata = payload["data_metadata"]
    if metadata["split_file_sha256"] != data.split_file_sha256:
        raise AssertionError(f"{name} split hash mismatch")
    if list(metadata["feature_names"]) != data.feature_names:
        raise AssertionError(f"{name} feature order mismatch")
    if int(payload["architecture"]["input_dim"]) != data.input_dim:
        raise AssertionError(f"{name} input dimension mismatch")
    if not np.array_equal(np.asarray(metadata["input_mean"]), data.input_mean):
        raise AssertionError(f"{name} input mean mismatch")
    if not np.array_equal(np.asarray(metadata["input_scale"]), data.input_scale):
        raise AssertionError(f"{name} input scale mismatch")


def extract_representations() -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    """Extract aligned deterministic test representations without training."""
    data = load_unsupervised_data()
    if data.X_test.shape != (440, 262):
        raise AssertionError(f"Unexpected Workshop 4 test matrix: {data.X_test.shape}")
    indices = np.asarray(data.split_indices["test"], dtype=np.int64)
    test_tensor = torch.as_tensor(data.X_test, dtype=torch.float32)

    checkpoint_paths = {
        "AE z=128": CHECKPOINTS / "ae_best.pt",
        "AE z=64": CHECKPOINTS / "ae_latent_64.pt",
        "VAE z=64 beta=0.5": CHECKPOINTS / "vae_best.pt",
    }
    hashes_before = {name: file_sha256(path) for name, path in checkpoint_paths.items()}
    ae128, ae128_payload = load_autoencoder_checkpoint(checkpoint_paths["AE z=128"])
    ae64, ae64_payload = load_autoencoder_checkpoint(checkpoint_paths["AE z=64"])
    vae, vae_payload = load_vae_checkpoint(checkpoint_paths["VAE z=64 beta=0.5"])
    for name, payload in (("AE z=128", ae128_payload), ("AE z=64", ae64_payload), ("VAE z=64 beta=0.5", vae_payload)):
        _verify_unsupervised_checkpoint(payload, data, name)
    if ae128.latent_dim != 128 or ae64.latent_dim != 64 or vae.latent_dim != 64:
        raise AssertionError("Unsupervised latent dimensions differ from the selected checkpoints")
    if float(vae_payload["training_config"]["target_beta"]) != 0.5:
        raise AssertionError("The selected VAE is not beta=0.5")

    for model in (ae128, ae64, vae):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    with torch.no_grad():
        ae128_first = ae128.encode(test_tensor).cpu().numpy()
        ae128_second = ae128.encode(test_tensor).cpu().numpy()
        ae64_first = ae64.encode(test_tensor).cpu().numpy()
        ae64_second = ae64.encode(test_tensor).cpu().numpy()
        vae_first = vae.encode(test_tensor)[0].cpu().numpy()
        vae_second = vae.encode(test_tensor)[0].cpu().numpy()
    for name, first, second in (
        ("AE z=128", ae128_first, ae128_second),
        ("AE z=64", ae64_first, ae64_second),
        ("VAE z=64 beta=0.5", vae_first, vae_second),
    ):
        if not np.array_equal(first, second) or not np.isfinite(first).all():
            raise AssertionError(f"{name} extraction is non-deterministic or non-finite")

    mlp = extract_mlp_test_features()
    if not np.array_equal(indices, mlp.test_indices):
        raise AssertionError("MLP and autoencoder test observation order differs")
    representations = {
        "AE z=128": ae128_first,
        "AE z=64": ae64_first,
        "VAE z=64 beta=0.5": vae_first,
        "MLP z=64": mlp.embeddings,
    }
    expected_shapes = {"AE z=128": (440, 128), "AE z=64": (440, 64), "VAE z=64 beta=0.5": (440, 64), "MLP z=64": (440, 64)}
    for name, values in representations.items():
        if values.shape != expected_shapes[name] or not np.isfinite(values).all():
            raise AssertionError(f"Invalid {name} representation shape or values")

    _, _, test_df, splits = get_dataset_partitions()
    if not np.array_equal(indices, np.asarray(splits["test"], dtype=np.int64)):
        raise AssertionError("Raw labels and latent matrices use different test order")
    labels = pd.DataFrame({
        "test_row": np.arange(440),
        "original_dataset_index": indices,
        "SalePrice": test_df["SalePrice"].to_numpy(dtype=float),
        "Overall Qual": test_df["Overall Qual"].to_numpy(dtype=int),
        "Neighborhood": test_df["Neighborhood"].astype(str).to_numpy(),
        "Sale Condition": test_df["Sale Condition"].astype(str).to_numpy(),
    })

    hashes_after = {name: file_sha256(path) for name, path in checkpoint_paths.items()}
    if hashes_before != hashes_after:
        raise AssertionError("An unsupervised checkpoint changed during extraction")
    audit = {
        "test_observations": 440,
        "test_indices_sha256": _index_sha256(indices),
        "test_order_exact_across_models": True,
        "deterministic_extraction": True,
        "finite_representations": True,
        "checkpoint_hashes": {
            **{name: hashes_after[name] for name in checkpoint_paths},
            "MLP z=64": mlp.checkpoint_hash_after,
        },
        "checkpoint_paths": {
            **{name: str(path.relative_to(ROOT)) for name, path in checkpoint_paths.items()},
            "MLP z=64": mlp.metadata["checkpoint"],
        },
        "mlp": mlp.metadata,
        "preprocessing": {
            "AE_and_VAE": "262 inputs; Workshop 4 train-only preprocessor and standardizer",
            "MLP": "261 inputs; Workshop 2 train-only preprocessor and saved standardizer",
        },
    }
    return representations, labels, audit


def standardize_representations(representations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: StandardScaler().fit_transform(values).astype(np.float32)
        for name, values in representations.items()
    }


def compute_reductions(standardized: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    tsne_results: dict[str, np.ndarray] = {}
    umap_results: dict[str, np.ndarray] = {}
    reproducibility: dict[str, Any] = {}
    for name in MODEL_ORDER:
        values = standardized[name]
        tsne_config = dict(n_components=2, random_state=RANDOM_STATE, init="pca", learning_rate="auto", perplexity=30, max_iter=1000)
        umap_config = dict(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=RANDOM_STATE, transform_seed=RANDOM_STATE, n_jobs=1)
        tsne_first = TSNE(**tsne_config).fit_transform(values).astype(np.float32)
        tsne_second = TSNE(**tsne_config).fit_transform(values).astype(np.float32)
        umap_first = umap.UMAP(**umap_config).fit_transform(values).astype(np.float32)
        umap_second = umap.UMAP(**umap_config).fit_transform(values).astype(np.float32)
        tsne_difference = float(np.max(np.abs(tsne_first - tsne_second)))
        umap_difference = float(np.max(np.abs(umap_first - umap_second)))
        if tsne_difference > 1e-6 or umap_difference > 1e-6:
            raise AssertionError(f"Reducer reproducibility failed for {name}")
        tsne_results[name] = tsne_first
        umap_results[name] = umap_first
        reproducibility[name] = {
            "tsne_max_absolute_repeat_difference": tsne_difference,
            "umap_max_absolute_repeat_difference": umap_difference,
            "tsne_exact": bool(np.array_equal(tsne_first, tsne_second)),
            "umap_exact": bool(np.array_equal(umap_first, umap_second)),
        }
    return tsne_results, umap_results, reproducibility


def _grouping(labels: pd.DataFrame, scheme: str) -> tuple[np.ndarray, np.ndarray, str]:
    if scheme == "Overall Qual":
        raw = labels[scheme].astype(str)
        counts = raw.value_counts()
        keep = raw.isin(counts[counts >= 5].index)
        return keep.to_numpy(), raw[keep].to_numpy(), "Categories with fewer than 5 test observations excluded"
    if scheme == "Neighborhood":
        raw = labels[scheme].astype(str)
        counts = raw.value_counts()
        grouped = raw.where(raw.isin(counts[counts >= 10].index), "Other (<10)")
        keep = np.ones(len(raw), dtype=bool)
        return keep, grouped.to_numpy(), "Neighborhoods with fewer than 10 test observations grouped as Other"
    if scheme == "Normal vs Abnorml":
        raw = labels["Sale Condition"].astype(str)
        keep = raw.isin(["Normal", "Abnorml"])
        return keep.to_numpy(), raw[keep].to_numpy(), "Only native Normal and Abnorml categories included"
    raise ValueError(f"Unknown grouping scheme: {scheme}")


def compute_latent_metrics(standardized: dict[str, np.ndarray], labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        for scheme in ("Overall Qual", "Neighborhood", "Normal vs Abnorml"):
            mask, group_labels, handling = _grouping(labels, scheme)
            values = standardized[name][mask]
            counts = pd.Series(group_labels).value_counts()
            score = silhouette_score(values, group_labels, metric="euclidean")
            rows.append({
                "model": name,
                "latent_dimension": standardized[name].shape[1],
                "grouping": scheme,
                "silhouette_score": float(score),
                "observations": len(values),
                "clusters": int(len(counts)),
                "minimum_cluster_size": int(counts.min()),
                "group_handling": handling,
                "metric": "euclidean",
                "space": "per-model standardized original latent representation",
                "computed_on_2d_projection": False,
            })
    return pd.DataFrame(rows)


def compute_embedding_quality(standardized: dict[str, np.ndarray], tsne_results: dict[str, np.ndarray], umap_results: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        for method, embedding in (("t-SNE", tsne_results[name]), ("UMAP", umap_results[name])):
            rows.append({"model": name, "method": method, "trustworthiness_k15": float(trustworthiness(standardized[name], embedding, n_neighbors=15)), "observations": len(embedding)})
    return pd.DataFrame(rows)


def _plot_continuous(embeddings: dict[str, np.ndarray], labels: pd.DataFrame, method: str, variable: str, path: Path) -> None:
    values = labels[variable].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.07, top=0.86, wspace=0.25, hspace=0.27)
    vmin, vmax = (np.quantile(values, 0.01), np.quantile(values, 0.99)) if variable == "SalePrice" else (values.min(), values.max())
    scatter = None
    for axis, name in zip(axes.flat, MODEL_ORDER):
        coordinates = embeddings[name]
        scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=values, cmap="viridis", vmin=vmin, vmax=vmax, s=24, alpha=0.8, edgecolors="none")
        axis.set(title=name, xlabel=f"{method} 1", ylabel=f"{method} 2")
    label = "SalePrice (USD)" if variable == "SalePrice" else variable
    color_axis = fig.add_axes([0.90, 0.16, 0.018, 0.66])
    fig.colorbar(scatter, cax=color_axis, label=label)
    fig.suptitle(f"{method} projections colored by {label}\nEach reduction is fitted separately to one standardized latent space", fontsize=14, y=0.97)
    fig.savefig(path, dpi=180); plt.close(fig)


def _plot_categorical(embeddings: dict[str, np.ndarray], labels: pd.DataFrame, method: str, variable: str, path: Path) -> None:
    raw = labels[variable].astype(str)
    if variable == "Neighborhood":
        top = raw.value_counts().nlargest(8).index
        plotted = raw.where(raw.isin(top), "Other neighborhoods")
    else:
        plotted = raw
    categories = sorted(plotted.unique())
    palette = dict(zip(categories, sns.color_palette("tab10", n_colors=len(categories))))
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(left=0.07, right=0.79, bottom=0.07, top=0.86, wspace=0.25, hspace=0.27)
    for axis, name in zip(axes.flat, MODEL_ORDER):
        coordinates = embeddings[name]
        for category in categories:
            mask = plotted.to_numpy() == category
            axis.scatter(coordinates[mask, 0], coordinates[mask, 1], color=palette[category], label=category, s=22, alpha=0.78, edgecolors="none")
        axis.set(title=name, xlabel=f"{method} 1", ylabel=f"{method} 2")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=palette[category], label=category) for category in categories]
    fig.legend(handles=handles, labels=categories, loc="center left", bbox_to_anchor=(0.81, 0.5), title=variable, fontsize=8)
    suffix = " (eight most frequent categories; remaining grouped)" if variable == "Neighborhood" else ""
    fig.suptitle(f"{method} projections colored by {variable}{suffix}\nEach reduction is fitted separately to one standardized latent space", fontsize=14, y=0.97)
    fig.savefig(path, dpi=180); plt.close(fig)


def generate_figures(tsne_results: dict[str, np.ndarray], umap_results: dict[str, np.ndarray], labels: pd.DataFrame) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    for method, embeddings, stem in (("t-SNE", tsne_results, "tsne"), ("UMAP", umap_results, "umap")):
        for variable, variable_stem in (("SalePrice", "saleprice"), ("Overall Qual", "overall_quality")):
            path = FIGURES / f"{stem}_{variable_stem}.png"
            _plot_continuous(embeddings, labels, method, variable, path); outputs.append(str(path.relative_to(ROOT)))
        for variable, variable_stem in (("Neighborhood", "neighborhood"), ("Sale Condition", "sale_condition")):
            path = FIGURES / f"{stem}_{variable_stem}.png"
            _plot_categorical(embeddings, labels, method, variable, path); outputs.append(str(path.relative_to(ROOT)))
    return outputs


def regenerate_figures_from_saved() -> list[str]:
    """Regenerate plots without recomputing or refitting any embedding."""
    labels = pd.read_csv(RESULTS / "latent_test_labels.csv")
    with np.load(RESULTS / "latent_tsne_embeddings.npz") as archive:
        tsne_results = {name: archive[name].copy() for name in MODEL_ORDER}
    with np.load(RESULTS / "latent_umap_embeddings.npz") as archive:
        umap_results = {name: archive[name].copy() for name in MODEL_ORDER}
    return generate_figures(tsne_results, umap_results, labels)


def _write_report(metrics: pd.DataFrame, quality: pd.DataFrame, metadata: dict[str, Any]) -> None:
    pivot = metrics.pivot(index="model", columns="grouping", values="silhouette_score").reindex(MODEL_ORDER)
    lines = [
        "# Frozen Latent-Space Comparison", "",
        "All results use the same 440 test observations in their original split order. No model was trained or selected in this analysis. AE and VAE representations were learned through reconstruction, while the MLP representation was learned with the supervised standardized log-price objective. SalePrice is therefore an interpretation variable unseen by AE/VAE training but part of the MLP training target.", "",
        "## Silhouette scores in standardized original latent spaces", "",
        "| Model | Overall Qual | Neighborhood | Normal vs Abnorml |", "|---|---:|---:|---:|",
    ]
    for name in MODEL_ORDER:
        lines.append(f"| {name} | {pivot.loc[name, 'Overall Qual']:.4f} | {pivot.loc[name, 'Neighborhood']:.4f} | {pivot.loc[name, 'Normal vs Abnorml']:.4f} |")
    lines += [
        "", "Silhouette is computed in each model's per-dimension standardized original latent space, never in the two-dimensional projection. Rare Overall Qual categories with fewer than five observations are excluded; neighborhoods with fewer than ten observations are grouped into `Other (<10)`; the transaction-condition analysis uses only the native `Normal` and `Abnorml` labels.", "",
        "## Interpretation", "",
        "The scores quantify association with existing descriptive categories, not predictive superiority. Positive values indicate some local category consistency, values near zero indicate overlap, and negative values indicate that many observations are closer to another labeled group. The Normal/Abnorml result is exploratory because `Abnorml` is a sale-condition label rather than a structural anomaly label.", "",
        "The supervised MLP may organize its representation around price-related quality signals because SalePrice determined its training gradient. By contrast, AE and VAE must preserve broad input information useful for reconstruction. The VAE's KL regularization encourages a smoother posterior but can suppress case-specific structure, consistent with its weaker reconstruction results. Differences between AE z=64 and AE z=128 reflect both capacity and independently trained weights; they are not a causal estimate of latent dimension alone.", "",
        "t-SNE and UMAP are fitted separately to each standardized latent matrix. Their axes, rotation, scale, and inter-cluster distances are not directly comparable across panels. Apparent two-dimensional clusters can be introduced or exaggerated by the reducers, so conclusions are cross-checked against original-space Silhouette scores. Trustworthiness values are reported in `latent_embedding_quality.csv` as a limited local-neighborhood diagnostic, not as proof that a projection preserves all geometry.", "",
        "## Reproducibility and limitations", "",
        f"Both reducers use random state {RANDOM_STATE}. Every reduction was repeated and matched within 1e-6; exact-repeat details are stored in `latent_embedding_metadata.json`. Checkpoint hashes remained unchanged. The analysis uses held-out labels only for post-hoc interpretation and cannot be used to revise the previously frozen model selection.", "",
        "Additional limitations include the small `Abnorml` group, imbalance among neighborhoods, sensitivity of t-SNE/UMAP to their fixed hyperparameters, and the fact that separately standardized spaces remove absolute activation-scale information. No projection demonstrates global model superiority, regression accuracy, or anomaly-detection performance.",
    ]
    (RESULTS / "latent_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_latent_analysis() -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    representations, labels, audit = extract_representations()
    standardized = standardize_representations(representations)
    tsne_results, umap_results, reproducibility = compute_reductions(standardized)
    metrics = compute_latent_metrics(standardized, labels)
    quality = compute_embedding_quality(standardized, tsne_results, umap_results)
    metrics.to_csv(RESULTS / "latent_metrics.csv", index=False)
    quality.to_csv(RESULTS / "latent_embedding_quality.csv", index=False)
    labels.to_csv(RESULTS / "latent_test_labels.csv", index=False)
    np.savez_compressed(RESULTS / "latent_representations.npz", **{name: values for name, values in representations.items()}, test_indices=labels["original_dataset_index"].to_numpy())
    np.savez_compressed(RESULTS / "latent_standardized_representations.npz", **standardized)
    np.savez_compressed(RESULTS / "latent_tsne_embeddings.npz", **tsne_results)
    np.savez_compressed(RESULTS / "latent_umap_embeddings.npz", **umap_results)
    figures = generate_figures(tsne_results, umap_results, labels)

    metadata = {
        **audit,
        "representation_shapes": {name: list(values.shape) for name, values in representations.items()},
        "latent_preparation": "StandardScaler fitted independently to each 440-row latent matrix before distance-based analysis",
        "tsne": {"random_state": 42, "init": "pca", "learning_rate": "auto", "perplexity": 30, "max_iter": 1000, "fit_separately_per_model": True},
        "umap": {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 42, "n_jobs": 1, "fit_separately_per_model": True},
        "reducer_reproducibility": reproducibility,
        "interpretation_variables": {"SalePrice": "MLP training target; post-hoc only for AE/VAE", "Overall Qual": "post-hoc descriptive variable", "Neighborhood": "post-hoc descriptive variable", "Sale Condition": "post-hoc descriptive variable"},
        "library_versions": {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "torch", "scikit-learn", "umap-learn", "matplotlib", "seaborn")},
        "figures": figures,
        "no_training_performed": True,
        "test_results_not_used_for_model_selection": True,
    }
    (RESULTS / "latent_embedding_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report(metrics, quality, metadata)
    return {"metrics": metrics, "quality": quality, "metadata": metadata}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true", help="Regenerate figures from saved embeddings without refitting reducers")
    arguments = parser.parse_args()
    if arguments.figures_only:
        for figure in regenerate_figures_from_saved():
            print(figure)
    else:
        output = run_latent_analysis()
        print(output["metrics"].to_string(index=False))
