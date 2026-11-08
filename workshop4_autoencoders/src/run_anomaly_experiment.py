"""Train and evaluate the independent Sale Condition anomaly protocol."""

from __future__ import annotations

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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from anomaly_data_adapter import PRIMARY_LABELS, load_anomaly_data  # noqa: E402
from train_ae import AETrainingConfig, load_autoencoder_checkpoint, train_autoencoder  # noqa: E402
from train_vae import VAETrainingConfig, load_vae_checkpoint, train_vae  # noqa: E402


BASE = ROOT / "workshop4_autoencoders"
CHECKPOINT_DIR = BASE / "checkpoints"
RESULTS_DIR = BASE / "results"
HISTORY_DIR = RESULTS_DIR / "histories"
FIGURE_DIR = RESULTS_DIR / "figures" / "anomaly"
SEED = 42
BOOTSTRAP_REPLICATES = 2000


def _write_history(history: list[dict[str, Any]], stem: str) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(HISTORY_DIR / f"{stem}.csv", index=False)
    (HISTORY_DIR / f"{stem}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def _ae_scores(model, values: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    with torch.no_grad():
        reconstruction = model(tensor)
    return torch.mean((reconstruction - tensor) ** 2, dim=1).numpy()


def _vae_scores(model, values: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    with torch.no_grad():
        mu, logvar = model.encode(tensor)
        reconstruction = model.decode(mu)
        reconstruction_mse = torch.mean((reconstruction - tensor) ** 2, dim=1)
        kl_per_latent = torch.mean(-0.5 * (1.0 + logvar - mu.pow(2) - torch.exp(logvar)), dim=1)
        combined = reconstruction_mse + beta * kl_per_latent
    return reconstruction_mse.numpy(), kl_per_latent.numpy(), combined.numpy()


def _bootstrap_auc(y_true: np.ndarray, scores: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    if len(positives) < 2 or len(negatives) < 2:
        return {"replicates": 0, "roc_auc_ci_low": None, "roc_auc_ci_high": None, "pr_auc_ci_low": None, "pr_auc_ci_high": None, "limitation": "Too few observations in at least one class for stratified bootstrap."}
    rng = np.random.default_rng(SEED)
    roc_values = np.empty(replicates)
    pr_values = np.empty(replicates)
    for index in range(replicates):
        sampled = np.concatenate((rng.choice(negatives, len(negatives), replace=True), rng.choice(positives, len(positives), replace=True)))
        roc_values[index] = roc_auc_score(y_true[sampled], scores[sampled])
        pr_values[index] = average_precision_score(y_true[sampled], scores[sampled])
    return {
        "replicates": replicates,
        "roc_auc_ci_low": float(np.quantile(roc_values, 0.025)),
        "roc_auc_ci_high": float(np.quantile(roc_values, 0.975)),
        "pr_auc_ci_low": float(np.quantile(pr_values, 0.025)),
        "pr_auc_ci_high": float(np.quantile(pr_values, 0.975)),
        "limitation": "Percentile intervals use a small positive class and therefore may be unstable.",
    }


def _binary_metrics(name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    bootstrap = _bootstrap_auc(y_true, scores)
    return {
        "score_definition": name,
        "threshold": float(threshold),
        "threshold_source": "95th percentile of Normal validation scores",
        "normal_test_count": int((y_true == 0).sum()),
        "abnorml_test_count": int((y_true == 1).sum()),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
        **bootstrap,
    }


def _plot_anomaly_results(
    history_ae: list[dict[str, Any]],
    history_vae: list[dict[str, Any]],
    y_true: np.ndarray,
    score_map: dict[str, np.ndarray],
    metrics: list[dict[str, Any]],
) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ae_frame, vae_frame = pd.DataFrame(history_ae), pd.DataFrame(history_vae)
    axes[0].plot(ae_frame.epoch, ae_frame.train_mse, label="Train Normal")
    axes[0].plot(ae_frame.epoch, ae_frame.validation_mse, label="Validation Normal")
    axes[0].set(title="Anomaly-specific AE training", xlabel="Epoch", ylabel="Reconstruction MSE"); axes[0].legend()
    axes[1].plot(vae_frame.epoch, vae_frame.train_reconstruction_mse, label="Train Normal")
    axes[1].plot(vae_frame.epoch, vae_frame.validation_reconstruction_mse, label="Validation Normal")
    axes[1].set(title="Anomaly-specific VAE training", xlabel="Epoch", ylabel="Deterministic reconstruction MSE"); axes[1].legend()
    path = FIGURE_DIR / "anomaly_training_curves.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); outputs.append(str(path.relative_to(ROOT)))

    fig, axes = plt.subplots(1, len(score_map), figsize=(17, 5))
    for axis, (name, scores) in zip(axes, score_map.items()):
        sns.histplot(scores[y_true == 0], stat="density", kde=True, label="Normal", alpha=0.45, ax=axis)
        sns.histplot(scores[y_true == 1], stat="density", kde=True, label="Abnorml", alpha=0.55, ax=axis)
        threshold = next(row["threshold"] for row in metrics if row["score_definition"] == name)
        axis.axvline(threshold, color="black", linestyle="--", label="Validation threshold")
        axis.set(title=name, xlabel="Anomaly score", ylabel="Density"); axis.legend()
    fig.suptitle("Test anomaly-score distributions by native Sale Condition")
    path = FIGURE_DIR / "anomaly_score_distributions.png"; fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=180); plt.close(fig); outputs.append(str(path.relative_to(ROOT)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        row = next(item for item in metrics if item["score_definition"] == name)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={row['roc_auc']:.3f})")
        axes[1].plot(recall, precision, label=f"{name} (AP={row['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1); axes[0].set(title="ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].axhline(float(y_true.mean()), color="black", linestyle="--", linewidth=1, label="Positive prevalence"); axes[1].set(title="Precision–Recall curves", xlabel="Recall", ylabel="Precision")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    path = FIGURE_DIR / "anomaly_roc_precision_recall.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); outputs.append(str(path.relative_to(ROOT)))

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    for axis, row in zip(axes, metrics):
        matrix = np.array([[row["true_negative"], row["false_positive"]], [row["false_negative"], row["true_positive"]]])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis, xticklabels=["Normal", "Abnorml"], yticklabels=["Normal", "Abnorml"])
        axis.set(title=row["score_definition"], xlabel="Predicted", ylabel="Actual")
    fig.suptitle("Test confusion matrices using validation-only thresholds")
    path = FIGURE_DIR / "anomaly_confusion_matrices.png"; fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(path, dpi=180); plt.close(fig); outputs.append(str(path.relative_to(ROOT)))
    return outputs


def run_anomaly_experiment() -> dict[str, Any]:
    data = load_anomaly_data()
    metadata = data.checkpoint_metadata()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ae_checkpoint = CHECKPOINT_DIR / "anomaly_ae_best.pt"
    vae_checkpoint = CHECKPOINT_DIR / "anomaly_vae_best.pt"
    ae_history_path = HISTORY_DIR / "anomaly_ae_history.json"
    vae_history_path = HISTORY_DIR / "anomaly_vae_history.json"
    completed_training = all(path.exists() for path in (ae_checkpoint, vae_checkpoint, ae_history_path, vae_history_path))
    if completed_training:
        # Once final test scores have been observed, frozen validation-selected
        # checkpoints must be reused rather than trained or selected again.
        ae_history = json.loads(ae_history_path.read_text(encoding="utf-8"))
        vae_history = json.loads(vae_history_path.read_text(encoding="utf-8"))
    else:
        ae_config = AETrainingConfig(latent_dim=128, seed=SEED)
        vae_config = VAETrainingConfig(target_beta=0.5, latent_dim=64, seed=SEED)
        ae_result = train_autoencoder(data.X_train_normal, data.X_val_normal, metadata, ae_config, ae_checkpoint)
        vae_result = train_vae(data.X_train_normal, data.X_val_normal, metadata, vae_config, vae_checkpoint)
        ae_history = ae_result.history
        vae_history = vae_result.history
        _write_history(ae_history, "anomaly_ae_history")
        _write_history(vae_history, "anomaly_vae_history")

    ae_model, ae_payload = load_autoencoder_checkpoint(ae_checkpoint)
    vae_model, vae_payload = load_vae_checkpoint(vae_checkpoint)
    if ae_payload["data_metadata"]["split_file_sha256"] != data.split_file_sha256 or vae_payload["data_metadata"]["split_file_sha256"] != data.split_file_sha256:
        raise AssertionError("Anomaly checkpoint split hash mismatch")

    beta = float(vae_payload["training_config"]["target_beta"])
    ae_val_scores = _ae_scores(ae_model, data.X_val)
    ae_test_scores = _ae_scores(ae_model, data.X_test)
    vae_val_recon, vae_val_kl, vae_val_combined = _vae_scores(vae_model, data.X_val, beta)
    vae_test_recon, vae_test_kl, vae_test_combined = _vae_scores(vae_model, data.X_test, beta)
    normal_val = data.val_labels == "Normal"
    thresholds = {
        "AE reconstruction MSE": float(np.quantile(ae_val_scores[normal_val], 0.95)),
        "VAE reconstruction MSE": float(np.quantile(vae_val_recon[normal_val], 0.95)),
        "VAE reconstruction MSE + beta*KL/latent": float(np.quantile(vae_val_combined[normal_val], 0.95)),
    }

    primary_mask = np.isin(data.test_labels, PRIMARY_LABELS)
    primary_labels = data.test_labels[primary_mask]
    y_true = (primary_labels == "Abnorml").astype(int)
    score_map = {
        "AE reconstruction MSE": ae_test_scores[primary_mask],
        "VAE reconstruction MSE": vae_test_recon[primary_mask],
        "VAE reconstruction MSE + beta*KL/latent": vae_test_combined[primary_mask],
    }
    metric_rows = [_binary_metrics(name, y_true, scores, thresholds[name]) for name, scores in score_map.items()]
    pd.DataFrame(metric_rows).to_csv(RESULTS_DIR / "anomaly_metrics.csv", index=False)

    scores = pd.DataFrame({
        "test_row": np.arange(len(data.X_test)),
        "original_dataset_index": data.split_indices["test"],
        "sale_condition": data.test_labels,
        "included_in_primary_binary_evaluation": primary_mask,
        "binary_label": np.where(data.test_labels == "Abnorml", 1, np.where(data.test_labels == "Normal", 0, np.nan)),
        "ae_reconstruction_mse": ae_test_scores,
        "vae_reconstruction_mse": vae_test_recon,
        "vae_kl_per_latent": vae_test_kl,
        "vae_combined_score": vae_test_combined,
    })
    scores.to_csv(RESULTS_DIR / "anomaly_scores.csv", index=False)
    summary_rows = []
    score_columns = ["ae_reconstruction_mse", "vae_reconstruction_mse", "vae_combined_score"]
    for label in PRIMARY_LABELS:
        subset = scores.loc[scores["sale_condition"] == label]
        for column in score_columns:
            values = subset[column].to_numpy(dtype=float)
            summary_rows.append({
                "sale_condition": label,
                "score": column,
                "count": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            })
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "anomaly_score_summary.csv", index=False)
    figures = _plot_anomaly_results(ae_history, vae_history, y_true, score_map, metric_rows)

    protocol = {
        **metadata,
        "input_dim_after_excluding_sale_condition": data.input_dim,
        "models": {
            "AE": {"architecture": ae_payload["architecture"], "best_epoch": ae_payload["best_epoch"], "best_normal_validation_mse": ae_payload["best_validation_mse"], "checkpoint": "checkpoints/anomaly_ae_best.pt"},
            "VAE": {"architecture": vae_payload["architecture"], "target_beta": beta, "best_epoch": vae_payload["best_epoch"], "best_normal_validation_metrics": vae_payload["best_validation_metrics"], "checkpoint": "checkpoints/anomaly_vae_best.pt"},
        },
        "thresholds": thresholds,
        "threshold_fit_data": "Normal validation transactions only",
        "vae_combined_score_definition": "per-observation reconstruction MSE + 0.5 * Gaussian KL averaged over latent dimensions",
        "primary_test_counts": {"Normal": int((y_true == 0).sum()), "Abnorml": int((y_true == 1).sum())},
        "other_sale_conditions_excluded_from_primary_metrics": sorted(set(data.test_labels) - set(PRIMARY_LABELS)),
        "bootstrap": {"method": "stratified percentile bootstrap", "replicates": BOOTSTRAP_REPLICATES, "seed": SEED},
        "interpretation_warning": "Abnorml denotes an abnormal sale condition, not necessarily a structurally anomalous dwelling.",
        "test_used_for_training_selection_or_threshold": False,
        "figures": figures,
    }
    (RESULTS_DIR / "anomaly_protocol_metadata.json").write_text(json.dumps(protocol, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value), encoding="utf-8")
    return {"metrics": metric_rows, "protocol": protocol}


if __name__ == "__main__":
    result = run_anomaly_experiment()
    print(pd.DataFrame(result["metrics"]).to_string(index=False))
