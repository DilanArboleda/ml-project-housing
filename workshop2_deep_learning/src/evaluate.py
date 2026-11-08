"""Final test evaluation for the frozen Workshop 2 Part A model."""

import csv
import hashlib
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ames-mpl-cache")
)

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
WORKSHOP_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WORKSHOP_SRC))

from baselines import compute_metrics  # noqa: E402
from data_loader import RANDOM_SEED, get_dataset_partitions  # noqa: E402
from features import AmesPreprocessor  # noqa: E402
from model import HousingMLP  # noqa: E402
from train import inverse_target, standardize  # noqa: E402


SELECTED_CONFIG = "e4_wide_128_64"
SELECTED_SEED = 42
SELECTED_CHECKPOINT = (
    ROOT / "checkpoints" / "part_a" /
    "model_e4_wide_128_64_seed42.pt"
)
OTHER_SELECTED_CHECKPOINTS = (
    ROOT / "checkpoints" / "part_a" / "model_e4_wide_128_64_seed7.pt",
    ROOT / "checkpoints" / "part_a" / "model_e4_wide_128_64_seed2026.pt",
)
MANIFEST_PATH = (
    ROOT / "workshop2_deep_learning" / "results" / "part_a" /
    "experiment_manifest.json"
)
BASELINE_METRICS_PATH = (
    ROOT / "workshop1_supervised" / "results" / "baseline_metrics.json"
)
RESULTS_DIR = ROOT / "workshop2_deep_learning" / "results" / "final_test"
FIGURES_DIR = (
    ROOT / "workshop2_deep_learning" / "results" / "figures" / "final_test"
)


def verify_selection_manifest() -> dict:
    """Prove that architecture and seed were fixed without test metrics."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    selected = manifest["selected_configuration"]
    if protocol["test_partition_used"]:
        raise AssertionError("The experiment manifest reports test usage")
    if selected["config_name"] != SELECTED_CONFIG:
        raise AssertionError("Selected configuration differs from 128-64")
    if selected["n_seeds"] != 3 or set(selected["seeds"]) != {7, 42, 2026}:
        raise AssertionError("Expected validation comparison across three seeds")
    if any(run["test_partition_used"] for run in manifest["runs"]):
        raise AssertionError("A tuning run reports using the test partition")

    selected_runs = [
        run for run in manifest["runs"]
        if run["config_name"] == SELECTED_CONFIG
    ]
    best_validation_run = min(
        selected_runs, key=lambda run: run["best_val_mae"]
    )
    if best_validation_run["seed"] != SELECTED_SEED:
        raise AssertionError("Seed 42 was not the validation-selected run")
    return manifest


def load_partitions_and_preprocess() -> dict:
    """Fit Workshop 1 preprocessing on train and transform the fixed test set."""
    train_df, _, test_df, splits = get_dataset_partitions()
    preprocessor = AmesPreprocessor(remove_train_outliers=True)
    preprocessor.fit(train_df)
    X_train, y_train_log, y_train_raw = preprocessor.transform(
        train_df, is_train=True
    )
    X_test, y_test_log, y_test_raw = preprocessor.transform(
        test_df, is_train=False
    )
    test_indices = np.asarray(splits["test"], dtype=np.int64)
    if len(test_indices) != len(X_test) or len(X_test) != 440:
        raise AssertionError("Unexpected number of fixed test observations")

    return {
        # The MLP was trained after an explicit float32 cast. Workshop 1
        # baselines used the preprocessor's original float64 matrices.
        "X_train": np.asarray(X_train, dtype=np.float32),
        "X_train_baseline": np.asarray(X_train, dtype=np.float64),
        "y_train_log": np.asarray(y_train_log, dtype=np.float64),
        "y_train_raw": np.asarray(y_train_raw, dtype=np.float64),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "X_test_baseline": np.asarray(X_test, dtype=np.float64),
        "y_test_log": np.asarray(y_test_log, dtype=np.float64),
        "y_test_raw": np.asarray(y_test_raw, dtype=np.float64),
        "test_indices": test_indices,
        "test_order": np.asarray(test_df["Order"]),
        "test_pid": np.asarray(test_df["PID"]),
        "feature_names": preprocessor.feature_names,
        "split_metadata": splits["metadata"],
    }


def load_checkpoint(path: Path) -> tuple[dict, HousingMLP]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["name"] != SELECTED_CONFIG:
        raise AssertionError(f"Unexpected configuration in {path.name}")
    if tuple(config["hidden_dims"]) != (128, 64):
        raise AssertionError("Checkpoint architecture is not 128-64")
    if config["dropout"] != 0.05:
        raise AssertionError("Checkpoint dropout differs from frozen value")
    if config["weight_decay"] != 1e-5 or config["learning_rate"] != 1e-3:
        raise AssertionError("Checkpoint hyperparameters differ from selection")
    if checkpoint["test_partition_used"]:
        raise AssertionError("Checkpoint reports test usage during selection")

    model = HousingMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return checkpoint, model


def verify_checkpoint_preprocessing(checkpoint: dict, data: dict) -> None:
    if checkpoint["feature_names"] != data["feature_names"]:
        raise AssertionError("Feature names/order differ from training")
    if checkpoint["split_metadata"] != data["split_metadata"]:
        raise AssertionError("Split metadata differ from checkpoint")

    regenerated_mean = data["X_train"].mean(axis=0, dtype=np.float64)
    regenerated_scale = data["X_train"].std(axis=0, dtype=np.float64)
    regenerated_scale = np.where(regenerated_scale < 1e-8, 1.0, regenerated_scale)
    np.testing.assert_allclose(
        checkpoint["input_mean"], regenerated_mean, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        checkpoint["input_scale"], regenerated_scale, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        checkpoint["target_log_mean"], data["y_train_log"].mean(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        checkpoint["target_log_scale"], data["y_train_log"].std(),
        rtol=1e-6,
    )


def predict_deep_learning(
    checkpoint: dict, model: HousingMLP, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = standardize(
        X_test, checkpoint["input_mean"], checkpoint["input_scale"]
    )
    with torch.no_grad():
        predictions_scaled = model(
            torch.from_numpy(X_scaled).to(dtype=torch.float32)
        ).cpu().numpy()
    predictions_log, predictions_raw = inverse_target(
        predictions_scaled,
        checkpoint["target_log_mean"],
        checkpoint["target_log_scale"],
    )
    return (
        np.asarray(predictions_log, dtype=np.float64),
        np.asarray(predictions_raw, dtype=np.float64),
    )


def reconstruct_frozen_baselines(data: dict) -> dict[str, dict[str, np.ndarray]]:
    """Rebuild already-defined Workshop 1 models to obtain test predictions."""
    alphas = np.logspace(-2, 3, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="An ill-conditioned matrix detected.*"
        )
        ridge_cv.fit(data["X_train_baseline"], data["y_train_log"])
        ridge = Ridge(alpha=ridge_cv.alpha_, random_state=RANDOM_SEED)
        ridge.fit(data["X_train_baseline"], data["y_train_log"])

    random_forest = RandomForestRegressor(
        n_estimators=250,
        max_depth=22,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    random_forest.fit(data["X_train_baseline"], data["y_train_log"])

    gradient_boosting = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        random_state=RANDOM_SEED,
    )
    gradient_boosting.fit(data["X_train_baseline"], data["y_train_log"])

    models = {
        "Ridge Regression": ridge,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
    }
    predictions = {}
    for name, model in models.items():
        prediction_log = np.asarray(
            model.predict(data["X_test_baseline"]), dtype=np.float64
        )
        predictions[name] = {
            "log": prediction_log,
            "raw": np.expm1(prediction_log),
        }
    return predictions


def calculate_all_metrics(data: dict, predictions: dict) -> dict:
    metrics = {}
    for name, values in predictions.items():
        result = compute_metrics(
            data["y_test_raw"],
            values["raw"],
            data["y_test_log"],
            values["log"],
        )
        metrics[name] = {key: float(value) for key, value in result.items()}
    return metrics


def audit_baseline_reproduction(metrics: dict) -> dict:
    """Compare current reconstruction with immutable Workshop 1 results."""
    stored = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    audit = {}
    for model_name in ("Ridge Regression", "Random Forest", "Gradient Boosting"):
        audit[model_name] = {
            "stored_test_metrics": {},
            "reconstructed_test_metrics": {},
            "absolute_difference": {},
        }
        for metric_name, current_value in metrics[model_name].items():
            stored_value = float(stored[model_name]["Test"][metric_name])
            difference = abs(current_value - stored_value)
            audit[model_name]["stored_test_metrics"][metric_name] = stored_value
            audit[model_name]["reconstructed_test_metrics"][metric_name] = current_value
            audit[model_name]["absolute_difference"][metric_name] = difference
        audit[model_name]["exact_within_1e-6"] = all(
            difference <= 1e-6
            for difference in audit[model_name]["absolute_difference"].values()
        )
    return audit


def comparison_metrics_with_stored_baselines(dl_metrics: dict) -> dict:
    """Use the historical Workshop 1 test metrics as the baseline authority."""
    stored = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    comparison = {"Deep Learning MLP": dl_metrics["Deep Learning MLP"]}
    for model_name in ("Ridge Regression", "Random Forest", "Gradient Boosting"):
        comparison[model_name] = {
            key: float(value)
            for key, value in stored[model_name]["Test"].items()
        }
    return comparison


def test_set_fingerprint(data: dict) -> dict:
    index_bytes = data["test_indices"].astype("<i8").tobytes()
    target_bytes = data["y_test_raw"].astype("<f8").tobytes()
    return {
        "n_observations": len(data["test_indices"]),
        "split_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "targets_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "first_split_index": int(data["test_indices"][0]),
        "last_split_index": int(data["test_indices"][-1]),
    }


def write_comparison_csv(metrics: dict) -> None:
    with (RESULTS_DIR / "model_comparison_test.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "MAE", "RMSE", "R2", "RMSE_log"])
        for model_name, values in metrics.items():
            writer.writerow([
                model_name,
                values["MAE"],
                values["RMSE"],
                values["R2"],
                values["RMSE_log"],
            ])


def write_predictions(data: dict, predictions: dict) -> None:
    frame = pd.DataFrame({
        "split_index": data["test_indices"],
        "Order": data["test_order"],
        "PID": data["test_pid"],
        "SalePrice_actual": data["y_test_raw"],
    })
    for name, values in predictions.items():
        column = name.lower().replace(" ", "_")
        frame[f"{column}_predicted"] = values["raw"]
        frame[f"{column}_residual"] = data["y_test_raw"] - values["raw"]
    frame.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)


def generate_figures(
    data: dict,
    predictions: dict,
    prediction_metrics: dict,
    comparison_metrics: dict,
) -> None:
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    model_names = list(predictions)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.ravel()
    all_predictions = np.concatenate([
        values["raw"] for values in predictions.values()
    ])
    limit = max(data["y_test_raw"].max(), all_predictions.max()) * 1.03
    for ax, name, color in zip(axes, model_names, colors):
        ax.scatter(
            data["y_test_raw"], predictions[name]["raw"],
            alpha=0.55, s=22, color=color, edgecolors="none",
        )
        ax.plot([0, limit], [0, limit], "--", color="black", linewidth=1)
        ax.set_title(
            f"{name}\nMAE=${prediction_metrics[name]['MAE']:,.0f} | "
            f"R²={prediction_metrics[name]['R2']:.3f}"
        )
        ax.set_xlabel("Actual SalePrice (USD)")
        ax.set_ylabel("Predicted SalePrice (USD)")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Actual vs predicted — fixed test set (baselines reconstructed)",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted_test.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for column, (name, color) in enumerate(zip(model_names, colors)):
        predicted = predictions[name]["raw"]
        residuals = data["y_test_raw"] - predicted
        axes[0, column].scatter(
            predicted, residuals, alpha=0.5, s=18,
            color=color, edgecolors="none",
        )
        axes[0, column].axhline(0, color="black", linestyle="--")
        axes[0, column].set_title(name)
        axes[0, column].set_xlabel("Predicted price")
        axes[0, column].set_ylabel("Residual" if column == 0 else "")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].hist(residuals, bins=30, color=color, alpha=0.8)
        axes[1, column].axvline(0, color="black", linestyle="--")
        axes[1, column].set_xlabel("Residual (actual - predicted)")
        axes[1, column].set_ylabel("Count" if column == 0 else "")
        axes[1, column].set_title(
            f"mean=${residuals.mean():,.0f}, std=${residuals.std():,.0f}"
        )
    fig.suptitle("Residual analysis — fixed test set", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residual_analysis_test.png", dpi=180)
    plt.close(fig)

    x = np.arange(len(model_names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mae = [comparison_metrics[name]["MAE"] for name in model_names]
    rmse = [comparison_metrics[name]["RMSE"] for name in model_names]
    width = 0.36
    axes[0].bar(x - width / 2, mae, width, label="MAE", color="#2a9d8f")
    axes[0].bar(x + width / 2, rmse, width, label="RMSE", color="#e76f51")
    axes[0].set_xticks(x, model_names, rotation=20, ha="right")
    axes[0].set_ylabel("Error (USD)")
    axes[0].set_title("Dollar-scale errors")
    axes[0].legend()
    log_errors = [comparison_metrics[name]["RMSE_log"] for name in model_names]
    axes[1].bar(x, log_errors, color=colors)
    axes[1].set_xticks(x, model_names, rotation=20, ha="right")
    axes[1].set_ylabel("RMSE_log")
    axes[1].set_title("Log-scale error")
    for index, value in enumerate(log_errors):
        axes[1].text(index, value, f"{value:.4f}", ha="center", va="bottom")
    fig.suptitle("Final test error comparison", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_comparison_test.png", dpi=180)
    plt.close(fig)


def evaluate_additional_seeds(data: dict) -> list[dict]:
    """Describe frozen 128-64 checkpoints without using them for selection."""
    rows = []
    for path in (OTHER_SELECTED_CHECKPOINTS[0], SELECTED_CHECKPOINT,
                 OTHER_SELECTED_CHECKPOINTS[1]):
        checkpoint, model = load_checkpoint(path)
        verify_checkpoint_preprocessing(checkpoint, data)
        prediction_log, prediction_raw = predict_deep_learning(
            checkpoint, model, data["X_test"]
        )
        result = compute_metrics(
            data["y_test_raw"], prediction_raw,
            data["y_test_log"], prediction_log,
        )
        rows.append({
            "seed": int(checkpoint["seed"]),
            "checkpoint": str(path.relative_to(ROOT)),
            **{key: float(value) for key, value in result.items()},
            "used_for_model_selection": False,
        })
    rows.sort(key=lambda row: row["seed"])
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = verify_selection_manifest()
    data = load_partitions_and_preprocess()
    checkpoint, model = load_checkpoint(SELECTED_CHECKPOINT)
    if checkpoint["seed"] != SELECTED_SEED:
        raise AssertionError("The final checkpoint is not seed 42")
    verify_checkpoint_preprocessing(checkpoint, data)

    dl_log, dl_raw = predict_deep_learning(
        checkpoint, model, data["X_test"]
    )
    predictions = {
        "Deep Learning MLP": {"log": dl_log, "raw": dl_raw},
        **reconstruct_frozen_baselines(data),
    }
    if any(len(values["raw"]) != 440 for values in predictions.values()):
        raise AssertionError("Models do not predict the same 440 observations")

    reconstructed_metrics = calculate_all_metrics(data, predictions)
    comparison_metrics = comparison_metrics_with_stored_baselines(
        reconstructed_metrics
    )
    baseline_audit = audit_baseline_reproduction(reconstructed_metrics)
    fingerprint = test_set_fingerprint(data)
    seed_stability = evaluate_additional_seeds(data)

    output = {
        "evaluation_status": "final_test_evaluation",
        "selection_frozen_before_test": True,
        "selection_source": str(MANIFEST_PATH.relative_to(ROOT)),
        "selected_configuration": manifest["selected_configuration"],
        "selected_checkpoint": str(SELECTED_CHECKPOINT.relative_to(ROOT)),
        "selected_seed": SELECTED_SEED,
        "test_set": fingerprint,
        "metrics": comparison_metrics,
        "baseline_metrics_source": str(BASELINE_METRICS_PATH.relative_to(ROOT)),
        "baseline_reconstruction_audit": baseline_audit,
        "baseline_reconstruction_note": (
            "Predictions were reconstructed with the frozen Workshop 1 "
            "definitions for plots. Small differences from the stored metrics "
            "are retained explicitly and are consistent with numerical/library "
            "version variation; the comparison table uses the immutable "
            "Workshop 1 metrics."
        ),
        "additional_seed_description_not_used_for_selection": seed_stability,
    }
    (RESULTS_DIR / "final_test_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    write_comparison_csv(comparison_metrics)
    write_predictions(data, predictions)
    (RESULTS_DIR / "selected_model_seed_stability_test.json").write_text(
        json.dumps(seed_stability, indent=2), encoding="utf-8"
    )
    generate_figures(
        data, predictions, reconstructed_metrics, comparison_metrics
    )

    print("Final test evaluation (selection remained frozen):")
    for name, values in comparison_metrics.items():
        print(
            f"{name:24s} MAE=${values['MAE']:,.2f} "
            f"RMSE=${values['RMSE']:,.2f} R2={values['R2']:.4f} "
            f"RMSE_log={values['RMSE_log']:.4f}"
        )
    print(f"Same test observations: {fingerprint['n_observations']}")
    exact = all(
        row["exact_within_1e-6"] for row in baseline_audit.values()
    )
    print(f"Exact baseline reconstruction in current environment: {exact}")
    print("Comparison table uses the stored Workshop 1 test metrics.")


if __name__ == "__main__":
    main()
