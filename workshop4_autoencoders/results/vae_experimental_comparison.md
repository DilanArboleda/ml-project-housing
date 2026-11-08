# Variational Autoencoder Beta Comparison

All configurations use latent dimension 64, seed 42, and the same training and validation observations. Model selection uses validation reconstruction MSE only; test was not evaluated.

The optimized objective is reconstruction MSE plus effective beta times KL normalized per latent dimension. The true KL summed over latent dimensions is reported per observation. This normalized objective is not labelled as an ELBO.

| Beta | Best epoch | Validation MSE | KL/observation | Active units | Collapse |
|---:|---:|---:|---:|---:|:---:|
| 0.5 | 50 | 1.01274090 | 0.626941 | 34 | False |
| 1.0 | 30 | 1.02157813 | 0.286190 | 9 | True |
| 4.0 | 37 | 1.02156425 | 0.227492 | 3 | True |

The selected beta is **0.5**, with validation reconstruction MSE **1.01274090**.

The existing AE z=64 validation MSE is **0.49225994**.

## Seed stability

| Configuration | Mean validation MSE | Sample standard deviation |
|---|---:|---:|
| AE z=128 | 0.45729156 | 0.01105020 |
| VAE z=64 beta=0.5 | 0.92103272 | 0.08233021 |

## Figures

- `results\figures\vae_reconstruction_curves.png`
- `results\figures\vae_kl_curves.png`
- `results\figures\vae_total_objective_curves.png`
- `results\figures\vae_validation_reconstruction_by_beta.png`
- `results\figures\vae_kl_reconstruction_tradeoff.png`
- `results\figures\ae64_vae_validation_comparison.png`
- `results\figures\ae_vae_seed_stability.png`
