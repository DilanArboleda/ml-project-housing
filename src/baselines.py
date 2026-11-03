"""
Supervised Baseline Models and Evaluation for Ames Housing Price Prediction.
Workshop No. 1 â€” Data Preparation & Supervised Learning Baselines.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero OrdeÃ±ez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import load_and_preprocess_data
from data_loader import RANDOM_SEED, set_seed

# Publication plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def compute_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Calculate MAE, RMSE, R2 (dollar scale) and RMSE_log (RMSLE)."""
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    r2 = r2_score(y_true_raw, y_pred_raw)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "RMSE_log": rmse_log
    }

def train_and_evaluate_baselines():
    set_seed(RANDOM_SEED)
    data = load_and_preprocess_data(RANDOM_SEED)

    X_train, y_train_log, y_train_raw = data["X_train"], data["y_train_log"], data["y_train"]
    X_val, y_val_log, y_val_raw = data["X_val"], data["y_val_log"], data["y_val"]
    X_test, y_test_log, y_test_raw = data["X_test"], data["y_test_log"], data["y_test"]
    preprocessor = data["preprocessor"]
    feature_names = data["feature_names"]

    fig_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(__file__).resolve().parent.parent / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAINING SUPERVISED BASELINE MODELS (AMES HOUSING)")
    print("=" * 80)

    # 1. Baseline 1: Regularized Linear Model (Ridge)
    print("\n[1/3] Training Baseline 1: Regularized Linear Model (Ridge)...")
    alphas = np.logspace(-2, 3, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train, y_train_log)
    best_alpha = ridge_cv.alpha_
    print(f"      Optimal Ridge alpha: {best_alpha:.4f}")
    ridge_model = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
    ridge_model.fit(X_train, y_train_log)

    # 2. Baseline 2: Tree Ensemble (Random Forest)
    print("\n[2/3] Training Baseline 2: Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=22,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train_log)

    # 3. Baseline 3: Gradient Boosting Regressor
    print("\n[3/3] Training Baseline 3: Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        random_state=RANDOM_SEED
    )
    gb_model.fit(X_train, y_train_log)

    models = {
        "Ridge Regression": ridge_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model
    }

    # Evaluate across Train, Val, Test
    results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        pred_train_log = model.predict(X_train)
        pred_val_log = model.predict(X_val)
        pred_test_log = model.predict(X_test)

        pred_train_raw = preprocessor.inverse_transform_target(pred_train_log)
        pred_val_raw = preprocessor.inverse_transform_target(pred_val_log)
        pred_test_raw = preprocessor.inverse_transform_target(pred_test_log)

        m_train = compute_metrics(y_train_raw, pred_train_raw, y_train_log, pred_train_log)
        m_val = compute_metrics(y_val_raw, pred_val_raw, y_val_log, pred_val_log)
        m_test = compute_metrics(y_test_raw, pred_test_raw, y_test_log, pred_test_log)

        results[name] = {
            "Train": m_train,
            "Validation": m_val,
            "Test": m_test
        }
        predictions[name] = {
            "val_log": pred_val_log,
            "val_raw": pred_val_raw,
            "test_log": pred_test_log,
            "test_raw": pred_test_raw
        }

        print(f"  [Train] MAE: ${m_train['MAE']:,.2f} | RMSE: ${m_train['RMSE']:,.2f} | R2: {m_train['R2']:.4f} | RMSLE: {m_train['RMSE_log']:.4f}")
        print(f"  [Val]   MAE: ${m_val['MAE']:,.2f} | RMSE: ${m_val['RMSE']:,.2f} | R2: {m_val['R2']:.4f} | RMSLE: {m_val['RMSE_log']:.4f}")
        print(f"  [Test]  MAE: ${m_test['MAE']:,.2f} | RMSE: ${m_test['RMSE']:,.2f} | R2: {m_test['R2']:.4f} | RMSLE: {m_test['RMSE_log']:.4f}")

    # Save metrics to JSON
    metrics_path = res_dir / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Generate Figures
    print("\nGenerating Diagnostic Figures...")

    # Figure 1: Actual vs Predicted (Test Set)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = predictions[name]["test_raw"]
        ax.scatter(y_test_raw, y_pred, alpha=0.5, edgecolors="none", s=25, color="#1f77b4")
        max_val = max(y_test_raw.max(), y_pred.max()) * 1.05
        ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Ideal y = x")
        r2 = results[name]["Test"]["R2"]
        rmse = results[name]["Test"]["RMSE"]
        mae = results[name]["Test"]["MAE"]
        ax.set_title(f"{name} (Test)\n$R^2$={r2:.3f} | RMSE=\${rmse:,.0f} | MAE=\${mae:,.0f}")
        ax.set_xlabel("Actual SalePrice ($)")
        ax.xaxis.set_major_formatter("${x:,.0f}")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Predicted SalePrice ($)")
    axes[0].yaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    f1_path = fig_dir / "actual_vs_predicted.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"Saved: {f1_path}")

    # Figure 2: Residual Analysis (Residuals vs Predicted on Test Set)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = predictions[name]["test_raw"]
        residuals = y_test_raw - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="none", s=25, color="#2ca02c")
        ax.axhline(0, color="red", linestyle="--", lw=1.5)
        ax.set_title(f"Residuals: {name}\nStd: \${residuals.std():,.0f} | Mean: \${residuals.mean():,.0f}")
        ax.set_xlabel("Predicted SalePrice ($)")
        ax.xaxis.set_major_formatter("${x:,.0f}")

    axes[0].set_ylabel("Residual (Actual - Predicted) ($)")
    axes[0].yaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    f2_path = fig_dir / "residuals_plot.png"
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"Saved: {f2_path}")

    # Figure 3: Residual Distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = predictions[name]["test_raw"]
        residuals = y_test_raw - y_pred
        sns.histplot(residuals, kde=True, ax=ax, color="#9467bd", bins=30)
        ax.axvline(0, color="red", linestyle="--", lw=1.5)
        ax.set_title(f"Error Distribution: {name}")
        ax.set_xlabel("Residual ($ USD)")
        ax.xaxis.set_major_formatter("${x:,.0f}")

    axes[0].set_ylabel("Count")
    plt.tight_layout()
    f3_path = fig_dir / "residuals_distribution.png"
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"Saved: {f3_path}")

    # Figure 4: Feature Importance (Gradient Boosting & Random Forest)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Random Forest
    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    sns.barplot(x=rf_imp.values, y=rf_imp.index, ax=axes[0], palette="mako")
    axes[0].set_title("Random Forest â€” Top 15 Feature Importances")
    axes[0].set_xlabel("Relative Importance")

    # Gradient Boosting
    gb_imp = pd.Series(gb_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    sns.barplot(x=gb_imp.values, y=gb_imp.index, ax=axes[1], palette="crest")
    axes[1].set_title("Gradient Boosting â€” Top 15 Feature Importances")
    axes[1].set_xlabel("Relative Importance")

    plt.tight_layout()
    f4_path = fig_dir / "feature_importance.png"
    plt.savefig(f4_path, dpi=300)
    plt.close()
    print(f"Saved: {f4_path}")

    print("\nAll baseline models trained, evaluated, and plotted successfully!")
    return results

if __name__ == "__main__":
    train_and_evaluate_baselines()

