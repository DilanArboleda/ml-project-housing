# Standard Autoencoder Latent-Dimension Comparison

The experiment used the immutable train/validation split. Test inputs were prepared only for shape and finite-value validation and were not passed through any trained autoencoder.

- Input dimension: 262
- Training observations: 2050
- Validation observations: 440
- Selection metric: minimum validation reconstruction MSE
- Test partition used for model selection: no

| Latent dimension | Parameters | Epochs run | Best epoch | Best validation MSE | Time (s) |
|---:|---:|---:|---:|---:|---:|
| 32 | 158886 | 50 | 50 | 0.52512607 | 26.03 |
| 64 | 167110 | 50 | 50 | 0.49225994 | 75.62 |
| 128 | 183558 | 50 | 50 | 0.45277109 | 37.51 |

The selected model uses latent dimension **128** with validation MSE **0.45277109** at epoch **50**.

## Generated figures

- `results\figures\ae_training_curves.png`
- `results\figures\ae_validation_mse_comparison.png`

## Methodological note

MSE is optimized in a train-standardized mixed feature space. This gives each feature comparable variance but treats one-hot columns as continuous reconstruction targets; reconstructed categorical values are scores, not valid probabilities.
