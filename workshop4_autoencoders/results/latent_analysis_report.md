# Frozen Latent-Space Comparison

All results use the same 440 test observations in their original split order. No model was trained or selected in this analysis. AE and VAE representations were learned through reconstruction, while the MLP representation was learned with the supervised standardized log-price objective. SalePrice is therefore an interpretation variable unseen by AE/VAE training but part of the MLP training target.

## Silhouette scores in standardized original latent spaces

| Model | Overall Qual | Neighborhood | Normal vs Abnorml |
|---|---:|---:|---:|
| AE z=128 | -0.0465 | -0.0166 | 0.1775 |
| AE z=64 | -0.0360 | -0.0222 | 0.1610 |
| VAE z=64 beta=0.5 | -0.3836 | -0.4668 | 0.5166 |
| MLP z=64 | -0.0123 | -0.1380 | 0.2466 |

Silhouette is computed in each model's per-dimension standardized original latent space, never in the two-dimensional projection. Rare Overall Qual categories with fewer than five observations are excluded; neighborhoods with fewer than ten observations are grouped into `Other (<10)`; the transaction-condition analysis uses only the native `Normal` and `Abnorml` labels.

## Interpretation

The scores quantify association with existing descriptive categories, not predictive superiority. Positive values indicate some local category consistency, values near zero indicate overlap, and negative values indicate that many observations are closer to another labeled group. The Normal/Abnorml result is exploratory because `Abnorml` is a sale-condition label rather than a structural anomaly label.

The supervised MLP displays the clearest monotonic SalePrice and Overall Qual gradients in both t-SNE and UMAP. This is consistent with its supervised objective: SalePrice determined its training gradient. AE z=64 also shows a visible price/quality progression, despite never receiving SalePrice, because size, quality, age, and location are simultaneously useful for reconstruction and strongly associated with price. AE z=128 retains similar structure but with a less compact two-dimensional gradient. The original-space Overall Qual Silhouette scores remain near zero for all three models (MLP -0.0123, AE z=64 -0.0360, and AE z=128 -0.0465), so the visible gradients should not be described as discrete clusters.

Neighborhood colors form some local patches in the AE projections, particularly AE z=64, but remain substantially mixed. The original-space scores support this cautious reading: AE z=128 and AE z=64 are close to zero (-0.0166 and -0.0222), whereas MLP is more negative (-0.1380). The MLP representation is optimized for price rather than geographic category recovery, so its clear price gradient does not imply neighborhood separation.

The VAE projections show a compact concentration plus separated branches rather than a stable price or quality gradient. Their local trustworthiness is only 0.5897 for t-SNE and 0.6178 for UMAP, compared with 0.8128--0.8591 for the AE and MLP projections. Together with the strongly negative Overall Qual and Neighborhood Silhouette scores (-0.3836 and -0.4668), this suggests that KL regularization and partial latent underuse suppressed case-specific geometry relevant to these labels. This observation is consistent with the VAE's weaker reconstruction results, but does not establish a universal disadvantage of variational models.

For the imbalanced Normal/Abnorml comparison, the VAE has the highest original-space Silhouette score (0.5166), followed by the MLP (0.2466), AE z=128 (0.1775), and AE z=64 (0.1610). This apparently strong VAE value must not be treated as anomaly-detection superiority: only 19 `Abnorml` observations are present, the label describes transaction conditions, and the earlier score-based experiment found only modest VAE ROC-AUC. Silhouette measures relative geometry under the supplied labels, not ranking or threshold performance.

Differences between AE z=64 and AE z=128 reflect both capacity and independently trained weights; they are not a causal estimate of latent dimension alone. AE z=64 has slightly higher projection trustworthiness (0.8591 t-SNE and 0.8367 UMAP) than AE z=128 (0.8395 and 0.8130), while AE z=128 has marginally better Neighborhood and Normal/Abnorml Silhouette. There is therefore no consistent basis for declaring either latent dimension globally superior from this analysis.

t-SNE and UMAP are fitted separately to each standardized latent matrix. Their axes, rotation, scale, and inter-cluster distances are not directly comparable across panels. Apparent two-dimensional clusters can be introduced or exaggerated by the reducers, so conclusions are cross-checked against original-space Silhouette scores. Trustworthiness values are reported in `latent_embedding_quality.csv` as a limited local-neighborhood diagnostic, not as proof that a projection preserves all geometry.

## Reproducibility and limitations

Both reducers use random state 42. Every reduction was repeated and matched within 1e-6; exact-repeat details are stored in `latent_embedding_metadata.json`. Checkpoint hashes remained unchanged. The analysis uses held-out labels only for post-hoc interpretation and cannot be used to revise the previously frozen model selection.

Additional limitations include the small `Abnorml` group, imbalance among neighborhoods, sensitivity of t-SNE/UMAP to their fixed hyperparameters, and the fact that separately standardized spaces remove absolute activation-scale information. No projection demonstrates global model superiority, regression accuracy, or anomaly-detection performance.
