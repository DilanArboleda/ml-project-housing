# Workshop 4 - Autoencoders and Variational Autoencoders

This directory contains the Group 1 implementation for unsupervised
representation learning on Ames Housing. It includes the standard autoencoder
latent-dimension comparison, the controlled variational autoencoder beta study,
multi-seed stability, frozen test reconstruction, the independent transaction-
anomaly protocol, and the supervised/unsupervised latent-space comparison.

## Final academic notebook

The complete executed notebook is available at
[`notebooks/workshop4_autoencoders.ipynb`](notebooks/workshop4_autoencoders.ipynb).
It loads the saved experimental artifacts and embeds all tables and figures for
direct rendering on GitHub. It does not retrain models or refit t-SNE/UMAP.

To open the saved notebook locally from the repository root:

```powershell
.\.venv\Scripts\jupyter.exe notebook `
  workshop4_autoencoders\notebooks\workshop4_autoencoders.ipynb
```

To rebuild and execute it from the repository root:

```powershell
.\.venv\Scripts\python.exe `
  workshop4_autoencoders\notebooks\create_workshop4_notebook.py

.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=600 `
  workshop4_autoencoders\notebooks\workshop4_autoencoders.ipynb

.\.venv\Scripts\python.exe `
  workshop4_autoencoders\notebooks\validate_workshop4_notebook.py
```

## Data protocol

- The immutable `data/splits.json` partitions are reused: 2,050 train, 440
  validation, and 440 test observations.
- `Unnamed: 0`, `Order`, `PID`, and `SalePrice` are excluded from model inputs.
- The Workshop 1 preprocessor is reused with target-informed outlier removal
  disabled.
- Imputation, one-hot encoding, robust scaling, and additional standardization
  are fitted exclusively on training data.
- The verified input dimension is 262.
- Model selection uses only training and validation data. Frozen final
  reconstruction, anomaly, and latent-space analyses use the unchanged test
  partition only after model selection is complete.

MSE is computed in the standardized mixed feature space. This balances each
feature by training variance, but one-hot columns are reconstructed as
continuous scores and should not be interpreted as categorical probabilities.

## Run tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s workshop4_autoencoders\tests -v
```

## Run the AE comparison

```powershell
.\.venv\Scripts\python.exe `
  workshop4_autoencoders\src\run_experiments.py
```

The script trains independent autoencoders with latent dimensions 32, 64, and
128. Selection uses minimum validation reconstruction MSE only. Checkpoints are
written to `workshop4_autoencoders/checkpoints/`; histories, figures, and the
experiment manifest are written to `workshop4_autoencoders/results/`.

## Run the VAE comparison and stability study

```powershell
.\.venv\Scripts\python.exe `
  workshop4_autoencoders\src\run_vae_experiments.py
```

The VAE comparison fixes the latent dimension at 64 and evaluates target beta
values 0.5, 1.0, and 4.0 with a 10-epoch linear beta warmup. The reconstruction
term is MSE averaged over observations and features. The reported KL divergence
is summed over latent dimensions and averaged over observations, while the
optimized objective divides that KL term by the latent dimension before
applying beta. Consequently, the logged normalized objective is not presented
as an exact evidence lower bound (ELBO).

Selection uses validation reconstruction MSE and excludes configurations with
severe posterior collapse. The selected VAE and the AE with latent dimension
128 are repeated with seeds 7, 42, and 2026. The original seed-42 AE artifacts
are reused and verified by SHA-256 rather than overwritten. No test metrics are
computed by this script.

The principal VAE outputs are:

- `results/vae_experiment_manifest.json`: protocol, metrics, diagnostics, and
  artifact paths.
- `results/vae_comparison.csv`: beta comparison for seed 42.
- `results/vae_selected_seed_runs.csv` and `results/ae128_seed_runs.csv`:
  stability runs.
- `results/vae_experimental_comparison.md`: concise experimental summary.
- `results/figures/`: reconstruction, KL, trade-off, model-comparison, and
  seed-stability figures.

All model selection remains validation-only. The AE/VAE comparison scripts use
the test partition only for data-contract checks; trained models see test only
in the frozen final-evaluation commands below.

## Run the final reconstruction evaluation

```powershell
.\.venv\Scripts\python.exe `
  workshop4_autoencoders\src\evaluate_ae.py
```

This command loads `ae_best.pt` and `vae_best.pt` without retraining, audits
their preprocessing metadata, and evaluates deterministic reconstructions for
the 440 fixed test observations. Outputs include aggregate, per-observation,
and per-feature metrics, compressed predictions, and reconstruction figures.

## Run the independent transaction-anomaly protocol

```powershell
.\.venv\Scripts\python.exe `
  workshop4_autoencoders\src\run_anomaly_experiment.py
```

This protocol removes `Sale Condition` before preprocessing and fits all
transformations and anomaly-specific AE/VAE models using only `Normal` training
transactions. Checkpoints are selected on Normal validation transactions and
95th-percentile thresholds are fitted on that same validation subset. The
primary binary evaluation uses only `Normal` and `Abnorml`; the latter denotes
an abnormal sale condition and not necessarily a structurally anomalous home.
Existing completed anomaly checkpoints and histories are reused on subsequent
runs so that final test observations cannot trigger retraining or reselection.

See `results/final_evaluation_report.md` for the reconstruction table, anomaly
metrics, bootstrap intervals, interpretation, and limitations.

## Run the frozen latent-space analysis

Install the project requirements, including `umap-learn`, and run:

```powershell
.\.venv\Scripts\python.exe `
  workshop4_autoencoders\src\analyze_latent.py
```

The script loads the frozen AE z=128, AE z=64, VAE z=64 beta=0.5, and Workshop
2 MLP 128-64 checkpoints. It extracts aligned test representations without
training, standardizes each latent space independently, fits separate t-SNE and
UMAP reductions, and calculates Silhouette scores in the standardized original
latent spaces. The MLP uses its original 261-feature preprocessing and the
activation after the second ReLU; it is never fed the 262-feature Workshop 4
matrix. Every reduction is repeated to verify deterministic reproduction.

To rebuild only the plots from the saved coordinates without refitting either
reducer:

```powershell
.\.venv\Scripts\python.exe `
  workshop4_autoencoders\src\analyze_latent.py --figures-only
```

Results are summarized in `results/latent_analysis_report.md`. SalePrice,
Overall Qual, Neighborhood, and Sale Condition are used only for post-hoc
interpretation. SalePrice was part of MLP training but not AE/VAE training.
