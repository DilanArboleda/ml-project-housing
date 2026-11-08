"""Controlled MLP comparison for Workshop 2, using validation only."""

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
WORKSHOP_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WORKSHOP_SRC))

from data_loader import get_dataset_partitions  # noqa: E402
from features import AmesPreprocessor  # noqa: E402
from model import HousingMLP  # noqa: E402
from train import fit_standardizer, inverse_target, set_seed, standardize  # noqa: E402


RESULTS_DIR = ROOT / "workshop2_deep_learning" / "results" / "part_a"
FIGURES_DIR = ROOT / "workshop2_deep_learning" / "results" / "figures" / "part_a"
CHECKPOINT_DIR = ROOT / "checkpoints" / "part_a"
REFERENCE_MAE = 15837.9453125
RIDGE_VALIDATION_MAE = 14101.863563482528
MAX_EPOCHS = 250
PATIENCE = 30
BATCH_SIZE = 32


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.05
    weight_decay: float = 1e-5
    learning_rate: float = 1e-3
    standardize_inputs: bool = True
    standardize_target: bool = True


CONFIGS = (
    ExperimentConfig(
        name="e3_no_dropout",
        description="Reference with dropout changed from 0.05 to 0.00",
        dropout=0.0,
    ),
    ExperimentConfig(
        name="e4_wide_128_64",
        description="Reference with hidden layers changed to 128-64",
        hidden_dims=(128, 64),
    ),
    ExperimentConfig(
        name="e5_weight_decay_1e4",
        description="Reference with weight decay changed from 1e-5 to 1e-4",
        weight_decay=1e-4,
    ),
    ExperimentConfig(
        name="e6_learning_rate_5e4",
        description="Reference with learning rate changed from 1e-3 to 5e-4",
        learning_rate=5e-4,
    ),
    ExperimentConfig(
        name="a1_no_input_standardization",
        description="Ablation: keep Workshop 1 X without extra standardization",
        standardize_inputs=False,
    ),
    ExperimentConfig(
        name="a2_no_target_standardization",
        description="Ablation: train directly on log1p(SalePrice)",
        standardize_target=False,
    ),
)

REFERENCE_CONFIG = ExperimentConfig(
    name="experiment2_reference",
    description="Experiment 2 reference configuration",
)


def load_train_validation() -> dict:
    """Load only train/validation using the immutable Workshop 1 split."""
    train_df, val_df, _, splits = get_dataset_partitions()
    preprocessor = AmesPreprocessor(remove_train_outliers=True)
    preprocessor.fit(train_df)
    X_train, y_train_log, y_train_raw = preprocessor.transform(
        train_df, is_train=True
    )
    X_val, y_val_log, y_val_raw = preprocessor.transform(
        val_df, is_train=False
    )
    return {
        "X_train": np.asarray(X_train, dtype=np.float32),
        "X_val": np.asarray(X_val, dtype=np.float32),
        "y_train_log": np.asarray(y_train_log, dtype=np.float32),
        "y_val_log": np.asarray(y_val_log, dtype=np.float32),
        "y_train_raw": np.asarray(y_train_raw, dtype=np.float32),
        "y_val_raw": np.asarray(y_val_raw, dtype=np.float32),
        "feature_names": preprocessor.feature_names,
        "split_metadata": splits["metadata"],
    }


def prepare_arrays(data: dict, config: ExperimentConfig) -> dict:
    """Fit optional transforms on train and apply them to validation."""
    if config.standardize_inputs:
        input_mean, input_scale = fit_standardizer(data["X_train"])
    else:
        input_mean = np.zeros(data["X_train"].shape[1], dtype=np.float32)
        input_scale = np.ones(data["X_train"].shape[1], dtype=np.float32)

    if config.standardize_target:
        target_mean_array, target_scale_array = fit_standardizer(
            data["y_train_log"]
        )
    else:
        target_mean_array = np.asarray(0.0, dtype=np.float32)
        target_scale_array = np.asarray(1.0, dtype=np.float32)

    return {
        "X_train": standardize(
            data["X_train"], input_mean, input_scale
        ),
        "X_val": standardize(data["X_val"], input_mean, input_scale),
        "y_train": standardize(
            data["y_train_log"], target_mean_array, target_scale_array
        ),
        "y_val": standardize(
            data["y_val_log"], target_mean_array, target_scale_array
        ),
        "input_mean": input_mean,
        "input_scale": input_scale,
        "target_mean": float(target_mean_array),
        "target_scale": float(target_scale_array),
    }


def predict(model: nn.Module, values: np.ndarray, device: torch.device):
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        return model(tensor).cpu().numpy()


def metrics_from_predictions(
    y_true_raw: np.ndarray,
    predictions_model_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
) -> tuple[dict[str, float], np.ndarray]:
    predictions_log, predictions_raw = inverse_target(
        predictions_model_scale, target_mean, target_scale
    )
    metrics = {
        "mae": float(mean_absolute_error(y_true_raw, predictions_raw)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_raw, predictions_raw))),
        "r2": float(r2_score(y_true_raw, predictions_raw)),
    }
    return metrics, predictions_log


def run_experiment(
    config: ExperimentConfig,
    seed: int,
    data: dict,
    device: torch.device,
) -> dict:
    """Train one isolated run and persist its history and best checkpoint."""
    run_name = f"{config.name}_seed{seed}"
    history_path = RESULTS_DIR / f"history_{run_name}.json"
    summary_path = RESULTS_DIR / f"summary_{run_name}.json"
    checkpoint_path = CHECKPOINT_DIR / f"model_{run_name}.pt"

    if summary_path.exists() and history_path.exists() and checkpoint_path.exists():
        print(f"[resume] {run_name}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"\n[{run_name}] {config.description}")
    set_seed(seed)
    arrays = prepare_arrays(data, config)
    dataset = TensorDataset(
        torch.from_numpy(arrays["X_train"]),
        torch.from_numpy(arrays["y_train"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = HousingMLP(
        input_dim=arrays["X_train"].shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    criterion = nn.MSELoss()
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    best_mae = float("inf")
    best_epoch = 0
    best_metrics = None
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_loss_sum += loss.item() * len(xb)

        train_predictions = predict(model, arrays["X_train"], device)
        val_predictions = predict(model, arrays["X_val"], device)
        train_metrics, train_predictions_log = metrics_from_predictions(
            data["y_train_raw"],
            train_predictions,
            arrays["target_mean"],
            arrays["target_scale"],
        )
        val_metrics, val_predictions_log = metrics_from_predictions(
            data["y_val_raw"],
            val_predictions,
            arrays["target_mean"],
            arrays["target_scale"],
        )
        train_loss_log = float(np.mean(
            (train_predictions_log - data["y_train_log"]) ** 2
        ))
        val_loss_log = float(np.mean(
            (val_predictions_log - data["y_val_log"]) ** 2
        ))
        val_loss_model_scale = float(np.mean(
            (val_predictions - arrays["y_val"]) ** 2
        ))
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss_model_scale": batch_loss_sum / len(dataset),
            "train_loss_log": train_loss_log,
            "val_loss_log": val_loss_log,
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "learning_rate": learning_rate,
        })

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            torch.save({
                "run_name": run_name,
                "config": asdict(config),
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "input_dim": arrays["X_train"].shape[1],
                "feature_names": data["feature_names"],
                "input_mean": arrays["input_mean"],
                "input_scale": arrays["input_scale"],
                "target_log_mean": arrays["target_mean"],
                "target_log_scale": arrays["target_scale"],
                "best_epoch": best_epoch,
                "best_validation_metrics": best_metrics,
                "split_metadata": data["split_metadata"],
                "test_partition_used": False,
            }, checkpoint_path)
        else:
            patience_counter += 1

        scheduler.step(val_loss_model_scale)
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"  epoch={epoch:03d} val_mae={val_metrics['mae']:.2f} "
                f"val_log_mse={val_loss_log:.5f} lr={learning_rate:.2e}"
            )
        if patience_counter >= PATIENCE:
            break

    history_path.write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    summary = {
        "run_name": run_name,
        "config_name": config.name,
        "description": config.description,
        "seed": seed,
        "hidden_dims": list(config.hidden_dims),
        "dropout": config.dropout,
        "weight_decay": config.weight_decay,
        "learning_rate": config.learning_rate,
        "standardize_inputs": config.standardize_inputs,
        "standardize_target": config.standardize_target,
        "trainable_parameters": parameter_count,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_mae": best_metrics["mae"],
        "best_val_rmse": best_metrics["rmse"],
        "best_val_r2": best_metrics["r2"],
        "selection_partition": "validation",
        "test_partition_used": False,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "history": str(history_path.relative_to(ROOT)),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"  best epoch={best_epoch} MAE={best_metrics['mae']:.2f} "
        f"RMSE={best_metrics['rmse']:.2f} R2={best_metrics['r2']:.4f}"
    )
    return summary


def reference_seed42_summary() -> dict:
    summary = json.loads(
        (ROOT / "workshop2_deep_learning" / "results" /
         "experiment2_summary.json").read_text(encoding="utf-8")
    )
    history = json.loads(
        (ROOT / "workshop2_deep_learning" / "results" /
         "training_history_experiment2.json").read_text(encoding="utf-8")
    )
    best = min(history, key=lambda row: row["val_mae"])
    return {
        "run_name": "experiment2_reference_seed42",
        "config_name": REFERENCE_CONFIG.name,
        "description": REFERENCE_CONFIG.description,
        "seed": 42,
        "hidden_dims": summary["hidden_dims"],
        "dropout": summary["dropout"],
        "weight_decay": summary["weight_decay"],
        "learning_rate": summary["learning_rate"],
        "standardize_inputs": True,
        "standardize_target": True,
        "trainable_parameters": summary["trainable_parameters"],
        "best_epoch": summary["best_epoch"],
        "epochs_run": summary["epochs_run"],
        "best_val_mae": best["val_mae"],
        "best_val_rmse": best["val_rmse"],
        "best_val_r2": best["val_r2"],
        "selection_partition": "validation",
        "test_partition_used": False,
        "checkpoint": "checkpoints/model_best_experiment2.pt",
        "history": (
            "workshop2_deep_learning/results/"
            "training_history_experiment2.json"
        ),
    }


def write_comparison_csv(summaries: list[dict]) -> None:
    fieldnames = [
        "run_name", "config_name", "seed", "hidden_dims", "dropout",
        "weight_decay", "learning_rate", "standardize_inputs",
        "standardize_target", "trainable_parameters", "best_epoch",
        "epochs_run", "best_val_mae", "best_val_rmse", "best_val_r2",
    ]
    with (RESULTS_DIR / "comparison_all_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def aggregate_stability(summaries: list[dict]) -> list[dict]:
    grouped = {}
    for row in summaries:
        grouped.setdefault(row["config_name"], []).append(row)
    aggregates = []
    for config_name, rows in grouped.items():
        maes = np.asarray([row["best_val_mae"] for row in rows])
        rmses = np.asarray([row["best_val_rmse"] for row in rows])
        r2s = np.asarray([row["best_val_r2"] for row in rows])
        aggregates.append({
            "config_name": config_name,
            "n_seeds": len(rows),
            "seeds": [row["seed"] for row in rows],
            "mae_mean": float(maes.mean()),
            "mae_std": float(maes.std(ddof=1)) if len(maes) > 1 else None,
            "mae_min": float(maes.min()),
            "mae_max": float(maes.max()),
            "rmse_mean": float(rmses.mean()),
            "r2_mean": float(r2s.mean()),
        })
    aggregates.sort(key=lambda row: row["mae_mean"])
    (RESULTS_DIR / "stability_summary.json").write_text(
        json.dumps(aggregates, indent=2), encoding="utf-8"
    )
    return aggregates


def load_history(summary: dict) -> list[dict]:
    return json.loads((ROOT / summary["history"]).read_text(encoding="utf-8"))


def plot_seed42_comparison(seed42: list[dict]) -> None:
    plt.figure(figsize=(12, 7))
    for summary in seed42:
        history = load_history(summary)
        epochs = [row["epoch"] for row in history]
        if summary["config_name"] == REFERENCE_CONFIG.name:
            target_scale = 0.39535337686538696
            train_loss = [
                row["train_loss_standardized"] * target_scale ** 2
                for row in history
            ]
            val_loss = [
                row["val_loss_standardized"] * target_scale ** 2
                for row in history
            ]
        else:
            train_loss = [row["train_loss_log"] for row in history]
            val_loss = [row["val_loss_log"] for row in history]
        validation_line, = plt.plot(
            epochs, val_loss, label=f"{summary['config_name']} val"
        )
        plt.plot(
            epochs,
            train_loss,
            alpha=0.35,
            linestyle="--",
            color=validation_line.get_color(),
        )
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE on log1p(SalePrice)")
    plt.title("Training (dashed) and validation (solid) loss — seed 42")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loss_curves_seed42.png", dpi=180)
    plt.close()

    ordered = sorted(seed42, key=lambda row: row["best_val_mae"])
    labels = [row["config_name"] for row in ordered]
    values = [row["best_val_mae"] for row in ordered]
    colors = ["#2a9d8f" if value < RIDGE_VALIDATION_MAE else "#457b9d"
              for value in values]
    plt.figure(figsize=(11, 6))
    bars = plt.barh(labels, values, color=colors)
    plt.axvline(
        RIDGE_VALIDATION_MAE, color="#e76f51", linestyle="--",
        label=f"Ridge: {RIDGE_VALIDATION_MAE:,.0f}",
    )
    plt.bar_label(bars, fmt="%.0f", padding=3)
    plt.xlabel("Best validation MAE (USD)")
    plt.title("Controlled MLP comparison — seed 42")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mae_comparison_seed42.png", dpi=180)
    plt.close()


def plot_stability(summaries: list[dict]) -> None:
    repeated_names = [
        name for name in sorted({row["config_name"] for row in summaries})
        if sum(row["config_name"] == name for row in summaries) == 3
    ]
    values = [
        [row["best_val_mae"] for row in summaries
         if row["config_name"] == name]
        for name in repeated_names
    ]
    plt.figure(figsize=(9, 6))
    plt.boxplot(values, tick_labels=repeated_names, showmeans=True)
    plt.axhline(RIDGE_VALIDATION_MAE, color="#e76f51", linestyle="--")
    plt.ylabel("Best validation MAE (USD)")
    plt.title("Seed stability (7, 42, 2026)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "seed_stability.png", dpi=180)
    plt.close()


def write_manifest(
    summaries: list[dict], aggregates: list[dict], selected: dict
) -> None:
    manifest = {
        "protocol": {
            "selection_metric": "minimum validation MAE in USD",
            "test_partition_used": False,
            "split_file": "data/splits.json",
            "seeds": [7, 42, 2026],
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "scalers_fit_partition": "train only",
            "ridge_validation_mae_reference": RIDGE_VALIDATION_MAE,
        },
        "configurations": [asdict(config) for config in CONFIGS],
        "runs": summaries,
        "stability": aggregates,
        "selected_configuration": selected,
    }
    (RESULTS_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    data = load_train_validation()

    reference42 = reference_seed42_summary()
    seed42_new = [run_experiment(config, 42, data, device) for config in CONFIGS]
    eligible = [
        summary for summary in seed42_new
        if not summary["config_name"].startswith("a")
    ]
    top_two_names = {
        row["config_name"]
        for row in sorted(eligible, key=lambda row: row["best_val_mae"])[:2]
    }
    config_by_name = {config.name: config for config in CONFIGS}

    summaries = [reference42, *seed42_new]
    for seed in (7, 2026):
        summaries.append(
            run_experiment(REFERENCE_CONFIG, seed, data, device)
        )
        for config_name in sorted(top_two_names):
            summaries.append(
                run_experiment(config_by_name[config_name], seed, data, device)
            )

    write_comparison_csv(summaries)
    aggregates = aggregate_stability(summaries)
    repeated = [row for row in aggregates if row["n_seeds"] == 3]
    selected = min(repeated, key=lambda row: row["mae_mean"])
    write_manifest(summaries, aggregates, selected)
    plot_seed42_comparison([reference42, *seed42_new])
    plot_stability(summaries)
    print("\nSelected by mean validation MAE across three seeds:")
    print(json.dumps(selected, indent=2))
    print("Test partition was not evaluated.")


if __name__ == "__main__":
    main()
