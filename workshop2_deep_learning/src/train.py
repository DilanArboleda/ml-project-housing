"""Leakage-free deep-learning training for Ames Housing."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_and_preprocess_data  # noqa: E402
from model import HousingMLP  # noqa: E402


SEED = 42
EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 30
HIDDEN_DIMS = (64, 32)
DROPOUT = 0.05
WEIGHT_DECAY = 1e-5
MIN_SCALE = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean/std on a training matrix only."""
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < MIN_SCALE, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def inverse_target(
    predictions_scaled: np.ndarray, target_mean: float, target_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictions on log1p and original dollar scales."""
    predictions_log = predictions_scaled * target_scale + target_mean
    predictions_raw = np.expm1(np.clip(predictions_log, -20.0, 20.0))
    return predictions_log, predictions_raw


def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        return model(tensor).cpu().numpy()


def calculate_metrics(
    y_true: np.ndarray,
    predictions_scaled: np.ndarray,
    target_mean: float,
    target_scale: float,
) -> dict[str, float]:
    _, y_pred = inverse_target(
        predictions_scaled, target_mean, target_scale
    )
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_diagnostics(
    X_train_raw: np.ndarray,
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    feature_names: list[str],
) -> dict:
    raw_max = np.max(np.abs(X_train_raw), axis=0)
    largest = np.argsort(raw_max)[-10:][::-1]
    return {
        "n_samples": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "target_log": {
            "min": float(np.min(y_train_log)),
            "mean": float(np.mean(y_train_log)),
            "std": float(np.std(y_train_log)),
            "max": float(np.max(y_train_log)),
        },
        "input_before_additional_standardization": {
            "min": float(np.min(X_train_raw)),
            "max": float(np.max(X_train_raw)),
            "global_std": float(np.std(X_train_raw)),
            "values_abs_gt_100": int(np.sum(np.abs(X_train_raw) > 100)),
            "largest_feature_maxima": [
                {
                    "feature": feature_names[index],
                    "max_abs": float(raw_max[index]),
                }
                for index in largest
            ],
        },
        "input_after_additional_standardization": {
            "min": float(np.min(X_train)),
            "max": float(np.max(X_train)),
            "mean": float(np.mean(X_train)),
            "std": float(np.std(X_train)),
        },
    }


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Existing split indices and Workshop 1 preprocessing remain unchanged.
    data = load_and_preprocess_data(SEED)
    X_train_raw = np.asarray(data["X_train"], dtype=np.float32)
    X_val_raw = np.asarray(data["X_val"], dtype=np.float32)
    y_train_log = np.asarray(data["y_train_log"], dtype=np.float32)
    y_val_log = np.asarray(data["y_val_log"], dtype=np.float32)
    y_val_raw = np.asarray(data["y_val"], dtype=np.float32)

    arrays = (X_train_raw, X_val_raw, y_train_log, y_val_log)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Training/validation data contain non-finite values")

    # Both transforms are fitted exclusively on the training partition.
    input_mean, input_scale = fit_standardizer(X_train_raw)
    target_mean_array, target_scale_array = fit_standardizer(y_train_log)
    target_mean = float(target_mean_array)
    target_scale = float(target_scale_array)

    X_train = standardize(X_train_raw, input_mean, input_scale)
    X_val = standardize(X_val_raw, input_mean, input_scale)
    y_train = standardize(y_train_log, target_mean_array, target_scale_array)
    y_val = standardize(y_val_log, target_mean_array, target_scale_array)

    input_dim = X_train.shape[1]
    print("Input features:", input_dim)
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print(f"Target train mean/std: {target_mean:.6f}/{target_scale:.6f}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = HousingMLP(
        input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )

    checkpoint_dir = ROOT / "checkpoints"
    results_dir = ROOT / "workshop2_deep_learning" / "results"
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = build_diagnostics(
        X_train_raw, X_train, y_train_log, data["feature_names"]
    )
    with (results_dir / "data_diagnostics_experiment2.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(diagnostics, file, indent=2)

    best_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("Trainable parameters:", parameter_count)
    print("\nStarting experiment 2 training...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(xb)
            loss = criterion(predictions, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(train_dataset)
        val_predictions = predict(model, X_val, device)
        val_loss = float(np.mean((val_predictions - y_val) ** 2))
        metrics = calculate_metrics(
            y_val_raw, val_predictions, target_mean, target_scale
        )
        val_mae = metrics["MAE"]
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch + 1,
            "train_loss_standardized": train_loss,
            "val_loss_standardized": val_loss,
            "val_mae": val_mae,
            "val_rmse": metrics["RMSE"],
            "val_r2": metrics["R2"],
            "learning_rate": current_lr,
        })

        print(
            f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | Val MAE: {val_mae:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "experiment": 2,
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dims": HIDDEN_DIMS,
                "dropout": DROPOUT,
                "epoch": best_epoch,
                "best_val_mae": best_mae,
                "feature_names": data["feature_names"],
                "input_mean": input_mean,
                "input_scale": input_scale,
                "target_log_mean": target_mean,
                "target_log_scale": target_scale,
            }, checkpoint_dir / "model_best_experiment2.pt")
        else:
            patience_counter += 1

        scheduler.step(val_loss)
        if patience_counter >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    with (results_dir / "training_history_experiment2.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(history, file, indent=2)

    summary = {
        "experiment": 2,
        "selection_partition": "validation",
        "test_partition_used": False,
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "epochs_run": len(history),
        "trainable_parameters": parameter_count,
        "hidden_dims": list(HIDDEN_DIMS),
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "target_log_mean_train_only": target_mean,
        "target_log_scale_train_only": target_scale,
    }
    with (results_dir / "experiment2_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nTraining completed.")
    print(f"Best validation MAE: {best_mae:.2f} (epoch {best_epoch})")
    print("Test partition was not used.")


if __name__ == "__main__":
    main()
