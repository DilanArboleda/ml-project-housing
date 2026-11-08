"""Run beta-VAE comparison and validation-only seed stability experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ames-workshop4-mpl")
)

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_adapter import load_unsupervised_data  # noqa: E402
from train_ae import AETrainingConfig, train_autoencoder  # noqa: E402
from train_vae import VAETrainingConfig, VAETrainingResult, train_vae  # noqa: E402


BETAS = (0.5, 1.0, 4.0)
STABILITY_SEEDS = (7, 42, 2026)
CHECKPOINT_DIR = WORKSHOP_ROOT / "checkpoints"
RESULTS_DIR = WORKSHOP_ROOT / "results"
HISTORY_DIR = RESULTS_DIR / "histories"
FIGURE_DIR = RESULTS_DIR / "figures"


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _beta_tag(beta: float) -> str:
    return f"{beta:.1f}".replace(".", "_")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_history(prefix: str, history: list[dict[str, Any]]) -> None:
    json_path = HISTORY_DIR / f"{prefix}_history.json"
    csv_path = HISTORY_DIR / f"{prefix}_history.csv"
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    _write_rows(csv_path, history)


def _vae_row(result: VAETrainingResult) -> dict[str, Any]:
    metrics = result.best_validation_metrics
    full_beta_history = [
        row
        for row in result.history
        if int(row["epoch"]) >= result.config.warmup_epochs
    ]
    return {
        "model": "VAE",
        "target_beta": result.config.target_beta,
        "latent_dim": result.config.latent_dim,
        "parameter_count": result.parameter_count,
        "seed": result.config.seed,
        "epochs_run": len(result.history),
        "best_epoch": result.best_epoch,
        "validation_reconstruction_mse": metrics["reconstruction_mse"],
        "validation_kl_per_observation": metrics["kl_per_observation"],
        "validation_kl_per_latent": metrics["kl_per_latent"],
        "validation_total_objective": metrics["total_objective"],
        "validation_active_units": metrics["active_units"],
        "validation_mean_latent_variance": metrics["mean_latent_variance"],
        "minimum_active_units_at_full_beta": min(
            int(row["validation_active_units"]) for row in full_beta_history
        ),
        "final_active_units": int(
            full_beta_history[-1]["validation_active_units"]
        ),
        "final_kl_per_latent": float(
            full_beta_history[-1]["validation_kl_per_latent"]
        ),
        "effective_beta_at_best_epoch": metrics["effective_beta"],
        "posterior_collapse_detected": result.posterior_collapse_detected,
        "training_time_seconds": result.training_time_seconds,
        "checkpoint": str(result.checkpoint_path.relative_to(WORKSHOP_ROOT)),
        "checkpoint_reload_exact": result.checkpoint_reload_exact,
        "selection_partition": "validation",
        "test_partition_used": False,
    }


def _plot_metric_histories(
    results: list[VAETrainingResult],
    train_key: str,
    validation_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
    for axis, result in zip(axes, results):
        epochs = [int(row["epoch"]) for row in result.history]
        axis.plot(
            epochs,
            [float(row[train_key]) for row in result.history],
            label="Train",
            linewidth=1.8,
        )
        axis.plot(
            epochs,
            [float(row[validation_key]) for row in result.history],
            label="Validation",
            linewidth=1.8,
        )
        axis.axvline(
            result.best_epoch,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Selected epoch {result.best_epoch}",
        )
        axis.set_title(rf"$\beta={result.config.target_beta:g}$")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    output = FIGURE_DIR / filename
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _plot_beta_reconstruction(rows: list[dict[str, Any]]) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    betas = [str(row["target_beta"]) for row in rows]
    values = [float(row["validation_reconstruction_mse"]) for row in rows]
    best_index = int(np.argmin(values))
    colors = ["#4c78a8"] * len(rows)
    colors[best_index] = "#2a9d8f"
    bars = axis.bar(betas, values, color=colors)
    axis.set_xlabel(r"Target $\beta$")
    axis.set_ylabel("Selected validation reconstruction MSE")
    axis.set_title("VAE reconstruction quality by beta")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.6f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    output = FIGURE_DIR / "vae_validation_reconstruction_by_beta.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _plot_kl_reconstruction(rows: list[dict[str, Any]]) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for row in rows:
        x_value = float(row["validation_kl_per_observation"])
        y_value = float(row["validation_reconstruction_mse"])
        axis.scatter(x_value, y_value, s=90)
        axis.annotate(
            rf"$\beta={row['target_beta']}$",
            (x_value, y_value),
            xytext=(7, 6),
            textcoords="offset points",
        )
    axis.set_xlabel("Validation KL per observation")
    axis.set_ylabel("Validation reconstruction MSE")
    axis.set_title("Latent regularization and reconstruction trade-off")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output = FIGURE_DIR / "vae_kl_reconstruction_tradeoff.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _plot_ae_vae_comparison(
    ae64_validation_mse: float, rows: list[dict[str, Any]]
) -> Path:
    labels = ["AE z=64"] + [f"VAE beta={row['target_beta']}" for row in rows]
    values = [ae64_validation_mse] + [
        float(row["validation_reconstruction_mse"]) for row in rows
    ]
    colors = ["#f4a261"] + ["#4c78a8"] * len(rows)
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    bars = axis.bar(labels, values, color=colors)
    axis.set_ylabel("Selected validation reconstruction MSE")
    axis.set_title("Controlled 64-dimensional AE and VAE comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    output = FIGURE_DIR / "ae64_vae_validation_comparison.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _plot_stability(stability: list[dict[str, Any]]) -> Path:
    labels = [row["model_configuration"] for row in stability]
    means = [float(row["mean_validation_mse"]) for row in stability]
    deviations = [float(row["std_validation_mse"]) for row in stability]
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    positions = np.arange(len(labels))
    axis.errorbar(
        positions,
        means,
        yerr=deviations,
        fmt="o",
        capsize=8,
        markersize=8,
        color="#264653",
        ecolor="#e76f51",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Validation reconstruction MSE")
    axis.set_title("Validation stability across seeds 7, 42, and 2026")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = FIGURE_DIR / "ae_vae_seed_stability.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _load_ae_rows() -> list[dict[str, str]]:
    with (RESULTS_DIR / "ae_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))


def _select_vae(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows if not bool(row["posterior_collapse_detected"])
    ]
    candidates = eligible if eligible else rows
    return min(
        candidates, key=lambda row: float(row["validation_reconstruction_mse"])
    )


def _stability_summary(
    configuration: str, seed_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    values = np.asarray(
        [float(row["validation_reconstruction_mse"]) for row in seed_rows],
        dtype=np.float64,
    )
    return {
        "model_configuration": configuration,
        "seeds": [int(row["seed"]) for row in seed_rows],
        "validation_mse_by_seed": values.tolist(),
        "mean_validation_mse": float(values.mean()),
        "std_validation_mse": float(values.std(ddof=1)),
        "min_validation_mse": float(values.min()),
        "max_validation_mse": float(values.max()),
    }


def _write_report(
    vae_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    ae64_mse: float,
    stability_rows: list[dict[str, Any]],
    figures: list[Path],
) -> Path:
    lines = [
        "# Variational Autoencoder Beta Comparison",
        "",
        "All configurations use latent dimension 64, seed 42, and the same "
        "training and validation observations. Model selection uses validation "
        "reconstruction MSE only; test was not evaluated.",
        "",
        "The optimized objective is reconstruction MSE plus effective beta times "
        "KL normalized per latent dimension. The true KL summed over latent "
        "dimensions is reported per observation. This normalized objective is "
        "not labelled as an ELBO.",
        "",
        "| Beta | Best epoch | Validation MSE | KL/observation | Active units | Collapse |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in vae_rows:
        lines.append(
            "| {target_beta} | {best_epoch} | "
            "{validation_reconstruction_mse:.8f} | "
            "{validation_kl_per_observation:.6f} | "
            "{validation_active_units} | {posterior_collapse_detected} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"The selected beta is **{selected['target_beta']}**, with validation "
            f"reconstruction MSE **{selected['validation_reconstruction_mse']:.8f}**.",
            "",
            f"The existing AE z=64 validation MSE is **{ae64_mse:.8f}**.",
            "",
            "## Seed stability",
            "",
            "| Configuration | Mean validation MSE | Sample standard deviation |",
            "|---|---:|---:|",
        ]
    )
    for row in stability_rows:
        lines.append(
            f"| {row['model_configuration']} | "
            f"{row['mean_validation_mse']:.8f} | "
            f"{row['std_validation_mse']:.8f} |"
        )
    lines.extend(["", "## Figures", ""])
    lines.extend(
        f"- `{path.relative_to(WORKSHOP_ROOT)}`" for path in figures
    )
    output = RESULTS_DIR / "vae_experimental_comparison.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    for directory in (CHECKPOINT_DIR, RESULTS_DIR, HISTORY_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_unsupervised_data()
    if data.input_dim != 262:
        raise AssertionError(f"Expected input dimension 262, got {data.input_dim}")
    metadata = data.checkpoint_metadata()

    original_ae_paths = [
        CHECKPOINT_DIR / "ae_latent_32.pt",
        CHECKPOINT_DIR / "ae_latent_64.pt",
        CHECKPOINT_DIR / "ae_latent_128.pt",
        CHECKPOINT_DIR / "ae_best.pt",
    ]
    original_ae_hashes = {path.name: _sha256(path) for path in original_ae_paths}

    vae_results: list[VAETrainingResult] = []
    for beta in BETAS:
        tag = _beta_tag(beta)
        checkpoint = CHECKPOINT_DIR / f"vae_beta_{tag}.pt"
        print(f"\nTraining VAE with beta={beta:g}, seed=42\n")
        result = train_vae(
            data.X_train,
            data.X_val,
            metadata,
            VAETrainingConfig(target_beta=beta, seed=42),
            checkpoint,
            device="cpu",
        )
        _write_history(f"vae_beta_{tag}_seed42", result.history)
        vae_results.append(result)

    vae_rows = [_vae_row(result) for result in vae_results]
    selected_row = _select_vae(vae_rows)
    selected_result = next(
        result
        for result in vae_results
        if result.config.target_beta == float(selected_row["target_beta"])
    )
    shutil.copy2(selected_result.checkpoint_path, CHECKPOINT_DIR / "vae_best.pt")
    _write_rows(RESULTS_DIR / "vae_comparison.csv", vae_rows)

    ae_rows = _load_ae_rows()
    ae64_row = next(row for row in ae_rows if int(row["latent_dim"]) == 64)
    ae128_seed42 = next(row for row in ae_rows if int(row["latent_dim"]) == 128)
    ae64_mse = float(ae64_row["best_validation_mse"])

    selected_beta = float(selected_row["target_beta"])
    selected_tag = _beta_tag(selected_beta)
    vae_seed_rows = [dict(selected_row)]
    for seed in (7, 2026):
        checkpoint = CHECKPOINT_DIR / f"vae_beta_{selected_tag}_seed{seed}.pt"
        print(f"\nRepeating selected VAE beta={selected_beta:g}, seed={seed}\n")
        result = train_vae(
            data.X_train,
            data.X_val,
            metadata,
            VAETrainingConfig(target_beta=selected_beta, seed=seed),
            checkpoint,
            device="cpu",
        )
        _write_history(
            f"vae_beta_{selected_tag}_seed{seed}", result.history
        )
        vae_seed_rows.append(_vae_row(result))
    vae_seed_rows.sort(key=lambda row: int(row["seed"]))

    ae_seed_rows: list[dict[str, Any]] = [
        {
            "model": "AE",
            "latent_dim": 128,
            "seed": 42,
            "best_epoch": int(ae128_seed42["best_epoch"]),
            "validation_reconstruction_mse": float(
                ae128_seed42["best_validation_mse"]
            ),
            "parameter_count": int(ae128_seed42["parameter_count"]),
            "checkpoint": ae128_seed42["checkpoint"],
            "checkpoint_reload_exact": (
                ae128_seed42["checkpoint_reload_exact"] == "True"
            ),
            "test_partition_used": False,
        }
    ]
    for seed in (7, 2026):
        checkpoint = CHECKPOINT_DIR / f"ae_latent_128_seed{seed}.pt"
        print(f"\nRepeating AE z=128, seed={seed}\n")
        result = train_autoencoder(
            data.X_train,
            data.X_val,
            metadata,
            AETrainingConfig(latent_dim=128, seed=seed),
            checkpoint,
            device="cpu",
        )
        _write_history(f"ae_latent_128_seed{seed}", result.history)
        ae_seed_rows.append(
            {
                "model": "AE",
                "latent_dim": 128,
                "seed": seed,
                "best_epoch": result.best_epoch,
                "validation_reconstruction_mse": result.best_validation_mse,
                "parameter_count": result.parameter_count,
                "checkpoint": str(checkpoint.relative_to(WORKSHOP_ROOT)),
                "checkpoint_reload_exact": result.checkpoint_reload_exact,
                "test_partition_used": False,
            }
        )
    ae_seed_rows.sort(key=lambda row: int(row["seed"]))

    for path in original_ae_paths:
        if _sha256(path) != original_ae_hashes[path.name]:
            raise AssertionError(f"Existing AE checkpoint was modified: {path}")

    _write_rows(RESULTS_DIR / "vae_selected_seed_runs.csv", vae_seed_rows)
    _write_rows(RESULTS_DIR / "ae128_seed_runs.csv", ae_seed_rows)
    stability_rows = [
        _stability_summary("AE z=128", ae_seed_rows),
        _stability_summary(
            f"VAE z=64 beta={selected_beta:g}", vae_seed_rows
        ),
    ]
    (RESULTS_DIR / "ae_vae_stability_summary.json").write_text(
        json.dumps(_json_ready(stability_rows), indent=2), encoding="utf-8"
    )

    figures = [
        _plot_metric_histories(
            vae_results,
            "train_reconstruction_mse",
            "validation_reconstruction_mse",
            "Reconstruction MSE",
            "VAE deterministic reconstruction loss",
            "vae_reconstruction_curves.png",
        ),
        _plot_metric_histories(
            vae_results,
            "train_kl_per_observation",
            "validation_kl_per_observation",
            "KL divergence per observation",
            "VAE latent regularization",
            "vae_kl_curves.png",
        ),
        _plot_metric_histories(
            vae_results,
            "train_total_objective",
            "validation_total_objective",
            "Normalized total objective",
            "VAE beta-specific normalized objectives",
            "vae_total_objective_curves.png",
        ),
        _plot_beta_reconstruction(vae_rows),
        _plot_kl_reconstruction(vae_rows),
        _plot_ae_vae_comparison(ae64_mse, vae_rows),
        _plot_stability(stability_rows),
    ]
    stability_figure = figures[-1]
    report = _write_report(
        vae_rows, selected_row, ae64_mse, stability_rows, figures
    )

    manifest = {
        "protocol": {
            "input_dim": data.input_dim,
            "latent_dim": 64,
            "betas": list(BETAS),
            "warmup_epochs": 10,
            "selection_partition": "validation",
            "selection_rule": (
                "minimum validation reconstruction MSE among runs without "
                "severe posterior collapse"
            ),
            "test_partition_used": False,
            "split_file": "data/splits.json",
            "split_file_sha256": data.split_file_sha256,
            "stability_seeds": list(STABILITY_SEEDS),
            "objective_is_exact_elbo": False,
        },
        "loss_scaling": {
            "reconstruction_mse": (
                "averaged over observations and input features"
            ),
            "reported_kl": (
                "summed over latent dimensions and averaged over observations"
            ),
            "objective_kl": "reported KL divided by latent dimension",
        },
        "beta_runs_seed42": vae_rows,
        "selected_vae": selected_row,
        "ae64_validation_mse": ae64_mse,
        "stability_runs": {
            "vae": vae_seed_rows,
            "ae128": ae_seed_rows,
            "summary": stability_rows,
        },
        "existing_ae_checkpoint_hashes_verified": original_ae_hashes,
        "artifacts": {
            "comparison_table": "results/vae_comparison.csv",
            "stability_figure": str(stability_figure.relative_to(WORKSHOP_ROOT)),
            "report": str(report.relative_to(WORKSHOP_ROOT)),
            "figures": [str(path.relative_to(WORKSHOP_ROOT)) for path in figures],
            "best_checkpoint": "checkpoints/vae_best.pt",
        },
    }
    (RESULTS_DIR / "vae_experiment_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2), encoding="utf-8"
    )

    print("\nVAE experiments completed without test evaluation.")
    print("Selected beta:", selected_beta)
    print(
        "Selected validation reconstruction MSE:",
        f"{float(selected_row['validation_reconstruction_mse']):.8f}",
    )


if __name__ == "__main__":
    main()
