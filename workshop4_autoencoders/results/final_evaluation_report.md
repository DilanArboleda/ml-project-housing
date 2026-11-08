# Final Reconstruction and Transaction-Anomaly Evaluation

## Protocol separation and checkpoint audit

The final reconstruction evaluation loaded the validation-selected checkpoints
`ae_best.pt` and `vae_best.pt` without retraining. Their architectures match the
experimental selection: the AE uses latent dimension 128 and the VAE uses
latent dimension 64 with beta 0.5. Both checkpoints contain the current
`data/splits.json` SHA-256 value, the same 262 ordered feature names, and the
same train-fitted standardization statistics. The VAE reconstruction is
deterministic because it decodes the posterior mean rather than a sampled
latent vector. Its optimized KL contribution is the Gaussian KL averaged over
observations and latent dimensions; the normalized objective is not an exact
ELBO.

The anomaly experiment is independent. `Sale Condition`, `SalePrice`, `Order`,
`PID`, and `Unnamed: 0` were removed before preprocessing. Imputation, encoding,
robust scaling, and additional standardization were fitted on the 1,694
`Normal` training transactions only. This produced 248 inputs. Checkpoint
selection used the 366 `Normal` validation transactions, and each threshold was
the 95th percentile of the corresponding score on those observations. Test was
not used for fitting, checkpoint selection, or threshold calibration.

## Frozen reconstruction results

All 440 original test observations were evaluated in the train-standardized
transformed feature space.

| Model | Feature group | Features | MSE | MAE | Pearson correlation |
|---|---|---:|---:|---:|---:|
| AE | All | 262 | 0.399277 | 0.359611 | 0.762230 |
| AE | Continuous | 62 | 0.442426 | 0.438086 | 0.754514 |
| AE | One-hot | 200 | 0.385901 | 0.335283 | 0.764806 |
| VAE | All | 262 | 0.940321 | 0.419778 | 0.091476 |
| VAE | Continuous | 62 | 1.017783 | 0.709241 | 0.075016 |
| VAE | One-hot | 200 | 0.916308 | 0.330044 | 0.097469 |

The AE clearly reconstructs the held-out data better overall and especially on
continuous features. The VAE has a slightly lower one-hot MAE, but its one-hot
MSE and correlation remain worse; this combination is consistent with outputs
concentrated near common values rather than faithful case-specific
reconstruction. One-hot outputs are unconstrained continuous values and must
not be interpreted as categorical probabilities.

## Transaction-anomaly results

The primary binary test evaluation contains 353 `Normal` and 19 `Abnorml`
transactions. `Partial`, `Family`, `Alloca`, and `AdjLand` are retained in the
score artifact but excluded from the binary metrics.

| Score | ROC-AUC (95% bootstrap CI) | PR-AUC (95% bootstrap CI) | Precision | Recall | F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|
| AE reconstruction MSE | 0.6317 (0.4971–0.7601) | 0.1330 (0.0656–0.3074) | 0.2308 | 0.1579 | 0.1875 | 343 / 10 / 16 / 3 |
| VAE reconstruction MSE | 0.6071 (0.4644–0.7433) | 0.1414 (0.0592–0.2934) | 0.1818 | 0.1053 | 0.1333 | 344 / 9 / 17 / 2 |
| VAE MSE + 0.5 KL/latent | 0.6064 (0.4655–0.7449) | 0.1413 (0.0589–0.2940) | 0.1667 | 0.1053 | 0.1290 | 343 / 10 / 17 / 2 |

`Abnorml` cases have larger average scores than `Normal` cases. For the AE, the
mean MSE rises from 0.4058 to 0.8830 and the median from 0.2541 to 0.4190. For
VAE reconstruction, the corresponding means are 0.8096 and 1.4584, while the
medians are 0.5695 and 0.7918. Nevertheless, the interquartile ranges overlap
substantially, which explains the low recall at the validation-calibrated
thresholds.

The AE provides the highest ROC-AUC, F1, and recall. The VAE reconstruction
score has a marginally higher point estimate for PR-AUC, but the differences
are small relative to the bootstrap uncertainty. Adding the normalized KL term
does not improve discrimination. Thus, the VAE regularization reduces
reconstruction fidelity without providing a measurable anomaly-detection gain
under this protocol.

## Limitations

`Abnorml` identifies an unusual transaction condition, not necessarily a
structurally anomalous dwelling. Only 19 positive test observations are
available, so the 2,000-replicate stratified percentile bootstrap intervals are
wide and themselves unstable. The positive prevalence is about 5.1%, making
PR-AUC and threshold-dependent metrics particularly sensitive to a few cases.
The 95th-percentile threshold is a prespecified operational rule rather than an
optimized classifier threshold.

Reconstruction quality and anomaly discrimination address different
questions. A model can reconstruct common and abnormal cases equally well, or
can assign high errors to rare but legitimate category combinations. Hence the
AE's much lower general reconstruction MSE does not automatically imply strong
anomaly detection; its anomaly ROC-AUC confidence interval includes 0.5.

No comparison is made with the Workshop 2 dollar-scale MAE because these
reconstruction errors are dimensionless quantities in a standardized mixed
feature space.
