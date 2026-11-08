"""Run the controlled latent-dimension comparison for the standard AE."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
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

from data_adapter import UnsupervisedDataBundle, load_unsupervised_data  # noqa: E402
from train_ae import AETrainingConfig, AETrainingResult, train_autoencoder  # noqa: E402


LATENT_DIMENSIONS = (32, 64, 128)
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


def _write_history(result: AETrainingResult) -> tuple[Path, Path]:
    stem = f"ae_latent_{result.config.latent_dim}_history"
    json_path = HISTORY_DIR / f"{stem}.json"
    csv_path = HISTORY_DIR / f"{stem}.csv"
    json_path.write_text(
        json.dumps(result.history, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["epoch", "train_mse", "validation_mse"]
        )
        writer.writeheader()
        writer.writerows(result.history)
    return json_path, csv_path


def _summary_row(result: AETrainingResult) -> dict[str, Any]:
    return {
        "model": "AE",
        "latent_dim": result.config.latent_dim,
        "hidden_dims": "-".join(map(str, result.config.hidden_dims)),
        "parameter_count": result.parameter_count,
        "seed": result.config.seed,
        "learning_rate": result.config.learning_rate,
        "batch_size": result.config.batch_size,
        "epochs_run": len(result.history),
        "best_epoch": result.best_epoch,
        "best_validation_mse": result.best_validation_mse,
        "training_time_seconds": result.training_time_seconds,
        "checkpoint": str(result.checkpoint_path.relative_to(WORKSHOP_ROOT)),
        "checkpoint_reload_exact": result.checkpoint_reload_exact,
        "selection_partition": "validation",
        "test_partition_used": False,
    }


def _plot_training_curves(results: list[AETrainingResult]) -> Path:
    figure, axes = plt.subplots(1, len(results), figsize=(15, 4.4), sharey=True)
    for axis, result in zip(axes, results):
        epochs = [int(row["epoch"]) for row in result.history]
        axis.plot(
            epochs,
            [row["train_mse"] for row in result.history],
            label="Train",
            linewidth=1.8,
        )
        axis.plot(
            epochs,
            [row["validation_mse"] for row in result.history],
            label="Validation",
            linewidth=1.8,
        )
        axis.axvline(
            result.best_epoch,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Best epoch {result.best_epoch}",
        )
        axis.set_title(f"Latent dimension {result.config.latent_dim}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Reconstruction MSE")
    axes[-1].legend(fontsize=8)
    figure.suptitle("Standard autoencoder training and validation loss")
    figure.tight_layout()
    output = FIGURE_DIR / "ae_training_curves.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _plot_validation_comparison(results: list[AETrainingResult]) -> Path:
    dimensions = [result.config.latent_dim for result in results]
    losses = [result.best_validation_mse for result in results]
    best_index = int(np.argmin(losses))
    colors = ["#4c78a8"] * len(results)
    colors[best_index] = "#2a9d8f"

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar([str(value) for value in dimensions], losses, color=colors)
    axis.set_xlabel("Latent dimension")
    axis.set_ylabel("Best validation MSE")
    axis.set_title("Validation reconstruction error by latent dimension")
    axis.grid(axis="y", alpha=0.25)
    for bar, loss in zip(bars, losses):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{loss:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    output = FIGURE_DIR / "ae_validation_mse_comparison.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _write_comparison(rows: list[dict[str, Any]]) -> Path:
    output = RESULTS_DIR / "ae_comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def _write_report(
    data: UnsupervisedDataBundle,
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    figures: list[Path],
) -> Path:
    lines = [
        "# Standard Autoencoder Latent-Dimension Comparison",
        "",
        "The experiment used the immutable train/validation split. Test inputs "
        "were prepared only for shape and finite-value validation and were not "
        "passed through any trained autoencoder.",
        "",
        f"- Input dimension: {data.input_dim}",
        f"- Training observations: {len(data.X_train)}",
        f"- Validation observations: {len(data.X_val)}",
        "- Selection metric: minimum validation reconstruction MSE",
        "- Test partition used for model selection: no",
        "",
        "| Latent dimension | Parameters | Epochs run | Best epoch | Best validation MSE | Time (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {latent_dim} | {parameter_count} | {epochs_run} | "
            "{best_epoch} | {best_validation_mse:.8f} | "
            "{training_time_seconds:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"The selected model uses latent dimension **{best['latent_dim']}** "
            f"with validation MSE **{best['best_validation_mse']:.8f}** at "
            f"epoch **{best['best_epoch']}**.",
            "",
            "## Generated figures",
            "",
        ]
    )
    lines.extend(
        f"- `{path.relative_to(WORKSHOP_ROOT)}`" for path in figures
    )
    lines.extend(
        [
            "",
            "## Methodological note",
            "",
            "MSE is optimized in a train-standardized mixed feature space. This "
            "gives each feature comparable variance but treats one-hot columns "
            "as continuous reconstruction targets; reconstructed categorical "
            "values are scores, not valid probabilities.",
        ]
    )
    output = RESULTS_DIR / "ae_experimental_comparison.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    for directory in (CHECKPOINT_DIR, RESULTS_DIR, HISTORY_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_unsupervised_data()
    if data.input_dim != 262:
        raise AssertionError(
            f"Verified input dimension changed from 262 to {data.input_dim}"
        )
    data_metadata = data.checkpoint_metadata()
    (RESULTS_DIR / "data_metadata.json").write_text(
        json.dumps(_json_ready(data_metadata), indent=2), encoding="utf-8"
    )

    results: list[AETrainingResult] = []
    for latent_dim in LATENT_DIMENSIONS:
        config = AETrainingConfig(latent_dim=latent_dim)
        checkpoint_path = CHECKPOINT_DIR / f"ae_latent_{latent_dim}.pt"
        print(f"\nTraining standard AE with latent dimension {latent_dim}\n")
        result = train_autoencoder(
            data.X_train,
            data.X_val,
            data_metadata,
            config,
            checkpoint_path,
            device="cpu",
        )
        _write_history(result)
        results.append(result)

    rows = [_summary_row(result) for result in results]
    for row in rows:
        row["input_dim"] = data.input_dim
    best_result = min(results, key=lambda result: result.best_validation_mse)
    best_row = min(rows, key=lambda row: row["best_validation_mse"])
    shutil.copy2(best_result.checkpoint_path, CHECKPOINT_DIR / "ae_best.pt")

    comparison_path = _write_comparison(rows)
    figures = [
        _plot_training_curves(results),
        _plot_validation_comparison(results),
    ]
    report_path = _write_report(data, rows, best_row, figures)
    manifest = {
        "protocol": {
            "split_file": "data/splits.json",
            "split_file_sha256": data.split_file_sha256,
            "selection_partition": "validation",
            "selection_metric": "minimum validation reconstruction MSE",
            "test_partition_used": False,
            "latent_dimensions": list(LATENT_DIMENSIONS),
            "seed": 42,
        },
        "data": {
            "input_dim": data.input_dim,
            "train_rows": len(data.X_train),
            "validation_rows": len(data.X_val),
            "test_rows_prepared_but_not_evaluated": len(data.X_test),
            "excluded_input_columns": data_metadata["excluded_input_columns"],
        },
        "runs": rows,
        "selected_model": best_row,
        "artifacts": {
            "comparison_table": str(comparison_path.relative_to(WORKSHOP_ROOT)),
            "report": str(report_path.relative_to(WORKSHOP_ROOT)),
            "figures": [str(path.relative_to(WORKSHOP_ROOT)) for path in figures],
            "best_checkpoint": "checkpoints/ae_best.pt",
        },
    }
    (RESULTS_DIR / "ae_experiment_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2), encoding="utf-8"
    )

    print("\nExperiment completed without test evaluation.")
    print("Input dimension:", data.input_dim)
    print("Selected latent dimension:", best_result.config.latent_dim)
    print("Best validation MSE:", f"{best_result.best_validation_mse:.8f}")


if __name__ == "__main__":
    main()
