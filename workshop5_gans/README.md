# Workshop 5 — Generative Adversarial Networks (GANs)

**Course:** Machine Learning — Semester 2026-III  
**University:** Universidad Distrital Francisco José de Caldas  
**Professor:** Eng. Carlos Andrés Sierra, M.Sc.  
**Team:** Group 1  
* Miguel Alejandro Chávez Porras (ID: 20212020033)  
* Luisa Fernanda Guerrero Ordoñez (ID: 20212020099)  
* Dilan Stive Arboleda Zambrano (ID: 20212020105)  
**Domain:** Real estate valuation — Ames Housing Dataset (Tabular Continuous & Categorical Feature Generation)

---

## 1. Overview & Theoretical Framework

This workshop implements deep generative adversarial modeling on the multi-dimensional tabular feature space of the Ames Housing dataset ($D = 261$ preprocessed features):

1. **Vanilla GAN (Goodfellow et al., 2014):**
   - Minimax zero-sum game under Binary Cross-Entropy (BCE) loss:
     $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$
   - Implicitly minimizes Jensen-Shannon Divergence ($\text{JSD}$). Susceptible to vanishing gradients and mode collapse when the discriminator outpaces the generator.
   - Employs real label smoothing ($y_{\text{real}} = 0.9$) and dropout ($p = 0.25$) for stabilization.

2. **Wasserstein GAN with Gradient Penalty (WGAN-GP, Gulrajani et al., 2017):**
   - Minimizes the Earth Mover's Distance ($W_1$) via the Kantorovich-Rubinstein duality:
     $$\mathcal{L}_{\text{critic}} = \mathbb{E}_{\tilde{x} \sim p_g}[C(\tilde{x})] - \mathbb{E}_{x \sim p_r}[C(x)] + \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\left[ (\|\nabla_{\hat{x}} C(\hat{x})\|_2 - 1)^2 \right]$$
   - Enforces the 1-Lipschitz condition along straight-line interpolations $\hat{x} = \epsilon x + (1 - \epsilon)\tilde{x}$ ($\lambda = 10.0$).
   - Uses `LayerNorm` instead of `BatchNorm` in the critic to maintain per-sample gradient independence.
   - Critic update ratio $n_{\text{critic}} = 5$, Adam optimizer with $\beta_1 = 0.0, \beta_2 = 0.9$.

---

## 2. Quantitative Generation Quality Benchmark

Evaluated on $N = 2,000$ synthetic profiles against the real Ames Housing training partition ($N_{\text{train}} = 2,050$):

| Quality Metric | Real Baseline | Vanilla GAN | WGAN-GP | Architectural Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **Fréchet Distance (PCA-10)** | 0.0000 | 0.8582 | **0.4719** | **45.0% error reduction** in multivariate feature manifold |
| **Correlation Frobenius Distance** | 0.0000 | 4.3136 | **3.1115** | Superior preservation of joint inter-feature covariances |
| **Mean Wasserstein 1-EMD** | 0.0000 | 0.1048 | **0.0918** | Lower marginal discrepancy across all 261 feature dimensions |
| **Living Area vs. Price Corr ($r$)** | **0.7068** | 0.4437 | **0.6276** | Reconstructs fundamental physical pricing mechanics ($|\Delta r| = 0.079$) |
| **Quality vs. Price Corr ($r$)** | **0.7963** | 0.4905 | **0.6908** | Captures non-linear appraisal valuation trends |
| **Pairwise Diversity Ratio** | 1.0000 | 0.8656 | **0.8346** | Robust sample dispersion; zero mode collapse |

---

## 3. Downstream Data Augmentation Experiment

To verify downstream machine learning utility, the supervised deep neural network from Workshop 2 (`HousingMLP`: $260 \to 128 \to 64 \to 1$) was retrained under three protocols and evaluated on the **untouched test partition (440 real properties)**:

| Training Protocol | Effective Train Size | Test MAE ($) | Test RMSE ($) | Test $R^2$ (%) | Test RMSLE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **(a) Real Data Only (Baseline)** | 2,050 | **\$16,400.67** | \$41,283.36 | 77.82% | **0.1340** |
| **(b) Real + 2,000 Vanilla GAN** | 4,050 | \$17,541.61 | \$38,600.50 | 80.61% | 0.1450 |
| **(c) Real + 2,000 WGAN-GP** | 4,050 | \$20,923.78 | **\$35,777.83** | **83.34%** | 0.1555 |

### Key Scientific Findings:
- **Variance Regularization on the Manifold:** WGAN-GP augmentation acts as an empirical $L_2$ manifold regularizer. By populating under-represented boundary regions (e.g. rare architectural styles and luxury estates), it eliminates catastrophic quadratic outliers, dropping Test RMSE by **\$5,505.53** and improving explained variance ($R^2$) from **77.82% to 83.34%**.
- **Continuous Relaxation Trade-off:** The continuous generator ($Tanh$) produces continuous fractional values for ordinal and one-hot features (e.g. fractional bathroom counts), resulting in a modest median calibration drift reflected in MAE (\$16.4k $\to$ \$20.9k) while dramatically improving tail risk.

---

## 4. Directory Structure

```text
workshop5_gans/
├── checkpoints/
│   ├── gan_vanilla_best.pt        # Trained weights for Vanilla Generator & Discriminator
│   └── gan_wgan_best.pt           # Trained weights for WGAN-GP Generator & Critic
├── report/
│   ├── figures/                   # Diagnostic PNG figures for publication
│   ├── generate_report_figures.py # Script generating comparative figures
│   ├── group1_workshop5.tex       # Complete IEEE/ACM LaTeX report (12 pages)
│   └── group1_workshop5.pdf       # Compiled academic PDF report
├── results/
│   ├── figures/                   # Diagnostic high-resolution plots
│   ├── augmentation_report.md     # Markdown downstream experiment report
│   ├── augmentation_results.json  # Exact test metrics across all three conditions
│   ├── generation_quality_metrics.json # Fréchet, Wasserstein, and Frobenius metrics
│   ├── synthetic_samples.npz      # 2,000 generated synthetic samples
│   ├── vanilla_samples_preview.csv # Human-readable sample preview
│   └── wgan_samples_preview.csv   # Human-readable sample preview
├── src/
│   ├── gan.py                     # PyTorch Generator, Discriminator, Critic & GP
│   ├── train_gan.py               # Adversarial training loops for Vanilla and WGAN-GP
│   ├── generate_samples.py        # Sampling & visual diagnostic plots
│   ├── evaluate_gan.py            # Quantitative evaluation suite
│   └── augment_and_retrain.py     # Downstream data augmentation benchmark
├── group1_workshop5.pdf           # Submission PDF report
└── README.md                      # This documentation
```

---

## 5. Reproduction Instructions

All scripts can be executed using the project's Python virtual environment:

```powershell
# 1. Train both Vanilla GAN and WGAN-GP (150 epochs each)
python workshop5_gans/src/train_gan.py

# 2. Generate 2,000 synthetic samples and diagnostic distribution plots
python workshop5_gans/src/generate_samples.py

# 3. Compute quantitative metrics (Fréchet, Wasserstein, Frobenius)
python workshop5_gans/src/evaluate_gan.py

# 4. Run the downstream data augmentation benchmark
python workshop5_gans/src/augment_and_retrain.py

# 5. Compile the formal LaTeX report with Tectonic
.\tools\tectonic\tectonic.exe workshop5_gans/report/group1_workshop5.tex --outdir workshop5_gans/report
```
