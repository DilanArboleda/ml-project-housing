"""
Model transfer from Workshops 1, 2, 4, 5 to the medical domain.

Transfers implemented:
    - Workshop 1: classical tabular baselines (LogReg, Random Forest, GBR).
    - Workshop 2: MLP classifier (128-64) with class weighting and threshold
      analysis, following the leakage-aware conditioning protocol.
    - Workshop 4: autoencoder trained only on Normal beats for anomaly scoring.
    - Workshop 5: WGAN-GP to generate synthetic abnormal beats (rare classes).

All models receive the 7 tabular features engineered in
`medical_preprocessing.py`, after the two-stage scaler (Robust + Standard)
fitted on TRAIN ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all sources of randomness."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SplitTensors:
    """Torch tensors for one split (already scaled)."""
    X: torch.Tensor
    y: torch.Tensor
    n_normal: int = 0
    n_abnormal: int = 0


@dataclass
class MLPConfig:
    """Hyperparameters for the Workshop 2 MLP transfer."""
    input_dim: int = 8
    hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 15
    grad_clip: float = 5.0


# ---------------------------------------------------------------------------
# Workshop 1: classical baselines
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str = "balanced",
) -> LogisticRegression:
    """Logistic Regression baseline (Workshop 1 transfer)."""
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 250,
    max_depth: int = 20,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    """Random Forest baseline (Workshop 1 transfer)."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.85,
) -> GradientBoostingClassifier:
    """Gradient Boosting baseline (Workshop 1 transfer)."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Workshop 2: MLP classifier with class weighting
# ---------------------------------------------------------------------------

class MedicalMLP(nn.Module):
    """
    MLP adapted from Workshop 2 for binary medical classification.

    Architecture (mirrors W2):
        input (7) -> Dense(128) -> ReLU -> Dropout(0.05)
                  -> Dense(64)  -> ReLU -> Dropout(0.05)
                  -> Dense(2)   (logits for Normal vs Abnormal)
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_tensor_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[MLPConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[MedicalMLP, Dict[str, List[float]]]:
    """
    Train the Medical MLP with class weighting and early stopping on val AUC.

    Returns
    -------
    model : MedicalMLP
        Trained model (best checkpoint loaded).
    history : dict
        Training curves (train_loss, val_loss, val_auc).
    """
    if config is None:
        config = MLPConfig()

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class weights inversely proportional to frequency
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    weight_pos = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, weight_pos], dtype=torch.float32).to(device)

    model = MedicalMLP(config).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    train_loader = _make_tensor_loader(X_train, y_train, config.batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc = -np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, config.max_epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running_loss += loss.item() * len(xb)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_AUC={val_auc:.4f}"
            )

        if epochs_no_improve >= config.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best AUC={best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_mlp_proba(model: MedicalMLP, X: np.ndarray) -> np.ndarray:
    """Return probability of class 1 (Abnormal)."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Workshop 4: autoencoder for anomaly detection
# ---------------------------------------------------------------------------

class MedicalAE(nn.Module):
    """
    Autoencoder adapted from Workshop 4, trained only on Normal beats.

    Architecture (mirrors W4 symmetry with smaller input):
        8 -> 16 -> 8 (latent) -> 16 -> 8
    """

    def __init__(self, input_dim: int = 8, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_autoencoder(
    X_train_normal: np.ndarray,
    latent_dim: int = 8,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> MedicalAE:
    """Train an AE only on Normal beats (Workshop 4 protocol)."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MedicalAE(input_dim=X_train_normal.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train_normal, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        if verbose and epoch % 20 == 0:
            print(f"AE epoch {epoch:3d} | MSE={running / len(X_train_normal):.5f}")

    model.eval()
    return model


def autoencoder_anomaly_score(model: MedicalAE, X: np.ndarray) -> np.ndarray:
    """Reconstruction MSE per sample as anomaly score."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        recon = model(X_t)
        scores = ((recon - X_t) ** 2).mean(dim=1).cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# Evaluation helpers (clinical metrics)
# ---------------------------------------------------------------------------

def evaluate_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute standard + clinical metrics at a given threshold.

    Returns
    -------
    dict with keys: accuracy, sensitivity, specificity, ppv, npv, f1,
    roc_auc, pr_auc, tp, fp, tn, fn.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy":    float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv":         float(ppv),
        "npv":         float(npv),
        "f1":          float(f1),
        "roc_auc":     float(roc_auc_score(y_true, y_proba)),
        "pr_auc":      float(average_precision_score(y_true, y_proba)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute sensitivity/specificity/PPV/NPV/F1 across a range of thresholds.

    Useful for clinicians to choose the operating point based on the
    acceptable trade-off between false negatives (missed disease) and
    false positives (alarm fatigue).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        m = evaluate_binary(y_true, y_proba, threshold=float(t))
        rows.append({
            "threshold":   float(t),
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "ppv":         m["ppv"],
            "npv":         m["npv"],
            "f1":          m["f1"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
        })
    return pd.DataFrame(rows)