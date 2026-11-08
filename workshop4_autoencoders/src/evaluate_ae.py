"""Final frozen-checkpoint reconstruction evaluation for Workshop 4."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ml_project_housing_matplotlib"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_adapter import load_unsupervised_data  # noqa: E402
from train_ae import load_autoencoder_checkpoint  # noqa: E402
from train_vae import load_vae_checkpoint  # noqa: E402


CHECKPOINT_DIR = ROOT / "workshop4_autoencoders" / "checkpoints"
RESULTS_DIR = ROOT / "workshop4_autoencoders" / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "reconstruction"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_checkpoint_contract(payload: dict[str, Any], data, model: str) -> None:
    metadata = payload["data_metadata"]
    architecture = payload["architecture"]
    if metadata["split_file_sha256"] != data.split_file_sha256:
        raise AssertionError(f"{model} checkpoint uses a different split file")
    if list(metadata["feature_names"]) != data.feature_names:
        raise AssertionError(f"{model} checkpoint feature order differs")
    if int(architecture["input_dim"]) != data.input_dim:
        raise AssertionError(f"{model} checkpoint input dimension differs")
    if not np.array_equal(np.asarray(metadata["input_mean"]), data.input_mean):
        raise AssertionError(f"{model} checkpoint input means differ")
    if not np.array_equal(np.asarray(metadata["input_scale"]), data.input_scale):
        raise AssertionError(f"{model} checkpoint input scales differ")
    if payload.get("selection_partition") != "validation":
        raise AssertionError(f"{model} was not selected on validation")
    if payload.get("test_partition_used") is not False:
        raise AssertionError(f"{model} checkpoint reports prior test use")


def _pearson(original: np.ndarray, reconstruction: np.ndarray) -> float:
    x = original.reshape(-1).astype(np.float64)
    y = reconstruction.reshape(-1).astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def _aggregate_metrics(
    original: np.ndarray,
    reconstruction: np.ndarray,
    continuous_count: int,
    model_name: str,
) -> list[dict[str, Any]]:
    groups = {
        "all": slice(None),
        "continuous": slice(0, continuous_count),
        "one_hot": slice(continuous_count, original.shape[1]),
    }
    rows = []
    for group, columns in groups.items():
        truth = original[:, columns]
        predicted = reconstruction[:, columns]
        rows.append(
            {
                "model": model_name,
                "feature_group": group,
                "feature_count": int(truth.shape[1]),
                "reconstruction_mse": float(np.mean((predicted - truth) ** 2)),
                "reconstruction_mae": float(np.mean(np.abs(predicted - truth))),
                "pearson_correlation": _pearson(truth, predicted),
                "evaluation_observations": int(len(truth)),
                "scale": "train-standardized transformed feature space",
            }
        )
    return rows


def _plot_results(
    original: np.ndarray,
    ae_reconstruction: np.ndarray,
    vae_reconstruction: np.ndarray,
    observation_scores: pd.DataFrame,
    feature_metrics: pd.DataFrame,
    feature_names: list[str],
) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figures = []

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(observation_scores["ae_mse"], stat="density", kde=True, label="AE", ax=ax, alpha=0.45)
    sns.histplot(observation_scores["vae_mse"], stat="density", kde=True, label="VAE", ax=ax, alpha=0.45)
    ax.set(title="Test reconstruction error distribution", xlabel="Per-observation MSE (standardized feature space)", ylabel="Density")
    ax.legend()
    path = FIGURES_DIR / "reconstruction_error_distribution.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); figures.append(str(path.relative_to(ROOT)))

    combined = observation_scores[["ae_mse", "vae_mse"]].mean(axis=1).to_numpy()
    order = np.argsort(combined)
    quantiles = np.linspace(0, 1, 8)
    selected = np.unique(order[np.rint(quantiles * (len(order) - 1)).astype(int)])
    if len(selected) < 8:
        selected = order[np.linspace(0, len(order) - 1, 8, dtype=int)]
    top_features = (
        feature_metrics.groupby("feature_index")["mse"].max().nlargest(20).index.to_numpy(dtype=int)
    )
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=True)
    for axis, row_index, quantile in zip(axes.flat, selected, quantiles):
        axis.plot(original[row_index, top_features], marker="o", markersize=3, label="Original")
        axis.plot(ae_reconstruction[row_index, top_features], label="AE")
        axis.plot(vae_reconstruction[row_index, top_features], label="VAE")
        axis.set_title(f"Error quantile {quantile:.2f} — test row {row_index}")
        axis.set_ylabel("Standardized value")
    axes[-1, 0].set_xticks(range(len(top_features)), [feature_names[i] for i in top_features], rotation=75, ha="right", fontsize=7)
    axes[-1, 1].set_xticks(range(len(top_features)), [feature_names[i] for i in top_features], rotation=75, ha="right", fontsize=7)
    axes[0, 0].legend(ncol=3)
    fig.suptitle("Quantile-selected reconstruction profiles (no manual case selection)")
    path = FIGURES_DIR / "quantile_reconstruction_profiles.png"
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(path, dpi=180); plt.close(fig); figures.append(str(path.relative_to(ROOT)))

    matrices = [
        original[selected][:, top_features], ae_reconstruction[selected][:, top_features],
        vae_reconstruction[selected][:, top_features],
        np.abs(ae_reconstruction[selected][:, top_features] - original[selected][:, top_features]),
        np.abs(vae_reconstruction[selected][:, top_features] - original[selected][:, top_features]),
    ]
    titles = ["Original", "AE reconstruction", "VAE reconstruction", "AE absolute difference", "VAE absolute difference"]
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    for axis, matrix, title in zip(axes, matrices, titles):
        sns.heatmap(matrix, cmap="viridis" if "difference" not in title else "magma", ax=axis, cbar_kws={"label": "Standardized value"})
        axis.set_title(title); axis.set_ylabel("Selected test case")
    axes[-1].set_xticks(np.arange(len(top_features)) + 0.5, [feature_names[i] for i in top_features], rotation=70, ha="right", fontsize=8)
    fig.suptitle("Quantile-selected observations and reconstruction differences")
    path = FIGURES_DIR / "reconstruction_heatmaps.png"
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(path, dpi=180); plt.close(fig); figures.append(str(path.relative_to(ROOT)))

    top = feature_metrics.nlargest(15, "mse").sort_values("mse")
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = top["model"] + " — " + top["feature_name"]
    ax.barh(labels, top["mse"], color=np.where(top["model"].eq("AE"), "#f4a261", "#457b9d"))
    ax.set(title="Features with the highest mean reconstruction error", xlabel="Feature MSE (standardized feature space)", ylabel="Model and feature")
    path = FIGURES_DIR / "highest_error_features.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); figures.append(str(path.relative_to(ROOT)))

    summary = observation_scores[["ae_mse", "vae_mse", "ae_mae", "vae_mae"]].mean()
    plot_frame = pd.DataFrame({"Model": ["AE", "VAE"], "MSE": [summary.ae_mse, summary.vae_mse], "MAE": [summary.ae_mae, summary.vae_mae]}).melt("Model", var_name="Metric", value_name="Error")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_frame, x="Model", y="Error", hue="Metric", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)
    ax.set(title="Frozen-model test reconstruction", ylabel="Error (standardized feature space)")
    path = FIGURES_DIR / "test_mse_mae_comparison.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); figures.append(str(path.relative_to(ROOT)))
    return figures


def evaluate_reconstruction() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_unsupervised_data()
    if len(data.X_test) != 440:
        raise AssertionError("Final reconstruction evaluation must contain 440 original test rows")

    ae_path = CHECKPOINT_DIR / "ae_best.pt"
    vae_path = CHECKPOINT_DIR / "vae_best.pt"
    ae, ae_payload = load_autoencoder_checkpoint(ae_path)
    vae, vae_payload = load_vae_checkpoint(vae_path)
    _assert_checkpoint_contract(ae_payload, data, "AE")
    _assert_checkpoint_contract(vae_payload, data, "VAE")
    if ae.latent_dim != 128 or vae.latent_dim != 64:
        raise AssertionError("Selected latent dimensions do not match validation selection")
    if float(vae_payload["training_config"]["target_beta"]) != 0.5:
        raise AssertionError("Selected VAE beta does not equal 0.5")
    loss_definition = vae_payload["loss_definition"]
    if "divided by latent dimension" not in loss_definition["kl_in_objective"]:
        raise AssertionError("VAE checkpoint does not document KL normalization")

    test_tensor = torch.as_tensor(data.X_test, dtype=torch.float32)
    with torch.no_grad():
        ae_reconstruction = ae(test_tensor).numpy()
        vae_reconstruction = vae.reconstruct_with_mean(test_tensor).numpy()
        vae_repeated = vae.reconstruct_with_mean(test_tensor).numpy()
    if not np.array_equal(vae_reconstruction, vae_repeated):
        raise AssertionError("Posterior-mean VAE reconstruction is not deterministic")

    continuous_count = len(data.preprocessing_metadata["continuous_columns"])
    metric_rows = _aggregate_metrics(data.X_test, ae_reconstruction, continuous_count, "AE")
    metric_rows += _aggregate_metrics(data.X_test, vae_reconstruction, continuous_count, "VAE")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_DIR / "reconstruction_metrics.csv", index=False)

    errors = {}
    feature_rows = []
    for model_name, reconstruction in (("AE", ae_reconstruction), ("VAE", vae_reconstruction)):
        difference = reconstruction - data.X_test
        errors[model_name] = difference
        mse_feature = np.mean(difference ** 2, axis=0)
        mae_feature = np.mean(np.abs(difference), axis=0)
        for index, name in enumerate(data.feature_names):
            feature_rows.append({"model": model_name, "feature_index": index, "feature_name": name, "feature_type": "continuous" if index < continuous_count else "one_hot", "mse": float(mse_feature[index]), "mae": float(mae_feature[index]), "pearson_correlation": _pearson(data.X_test[:, index:index+1], reconstruction[:, index:index+1])})
    feature_metrics = pd.DataFrame(feature_rows)
    feature_metrics.to_csv(RESULTS_DIR / "reconstruction_feature_metrics.csv", index=False)

    observation_scores = pd.DataFrame({
        "test_row": np.arange(len(data.X_test)),
        "original_dataset_index": data.split_indices["test"],
        "ae_mse": np.mean(errors["AE"] ** 2, axis=1),
        "ae_mae": np.mean(np.abs(errors["AE"]), axis=1),
        "vae_mse": np.mean(errors["VAE"] ** 2, axis=1),
        "vae_mae": np.mean(np.abs(errors["VAE"]), axis=1),
    })
    observation_scores.to_csv(RESULTS_DIR / "reconstruction_observation_scores.csv", index=False)
    np.savez_compressed(RESULTS_DIR / "reconstruction_predictions.npz", original=data.X_test, ae_reconstruction=ae_reconstruction, vae_reconstruction=vae_reconstruction, feature_names=np.asarray(data.feature_names))
    figures = _plot_results(data.X_test, ae_reconstruction, vae_reconstruction, observation_scores, feature_metrics, data.feature_names)

    audit = {
        "test_observations": 440,
        "same_test_matrix_and_feature_order": True,
        "split_file_sha256": data.split_file_sha256,
        "ae_checkpoint_sha256": _sha256(ae_path),
        "vae_checkpoint_sha256": _sha256(vae_path),
        "ae_configuration": ae_payload["architecture"],
        "vae_configuration": vae_payload["architecture"],
        "vae_target_beta": vae_payload["training_config"]["target_beta"],
        "vae_kl_normalized_by_latent_dimension": True,
        "vae_reconstruction_uses_posterior_mean": True,
        "one_hot_reconstruction_note": "One-hot columns are reconstructed as continuous values, not categorical probabilities.",
        "test_used_after_validation_selection": True,
        "figures": figures,
    }
    (RESULTS_DIR / "reconstruction_evaluation_metadata.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"metrics": metric_rows, "audit": audit}


if __name__ == "__main__":
    result = evaluate_reconstruction()
    print(pd.DataFrame(result["metrics"]).to_string(index=False))
