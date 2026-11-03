# Machine Learning Project — Housing Price Prediction

**Course:** Machine Learning — Semester 2026-III  
**University:** Universidad Distrital Francisco José de Caldas  
**Professor:** Eng. Carlos Andrés Sierra, M.Sc.  
**Team:** Group 1  

## Members

| Integrante | Código |
|------------|--------|
| Chavez Porras Miguel Alejandro | 20212020033 |
| Guerrero Ordeñez Luisa Fernanda | 20212020099 |
| Arboleda Zambrano Dilan Stive | 20212020105 |

## Domain
Real estate valuation — Housing Price Prediction (Ames Housing Dataset).

---

## Project Status

- [x] Initial Setup & Environment Verification
- [x] **Workshop 1 — Data Preparation & Supervised Learning Baselines**
- [ ] Workshop 2 — Deep Learning
- [ ] Workshop 3 — Reinforcement Learning
- [ ] Workshop 4 — Autoencoders
- [ ] Workshop 5 — GANs
- [ ] Workshop 6 — Medical Adaptation

---

## Repository Structure

```text
ml-project-housing/
├── checkpoints/                  # Saved model weights (future workshops)
├── data/
│   ├── docs/                     # Assessor codebooks and documentation
│   ├── raw/                      # Raw datasets (AmesHousing.csv / TSV)
│   └── splits.json               # Deterministic 70/15/15 train/val/test split indices
├── docs/                         # Cross-cutting architecture and course docs
├── notebooks/
│   └── eda.ipynb                 # Interactive exploratory data analysis
├── runs/                         # Training logs (TensorBoard / wandb)
├── src/                          # Reusable core modules
│   ├── data_loader.py            # Dataset loader and deterministic splitting
│   ├── dataset.py                # Unified data interface
│   ├── dataset_info.py           # Dataset integrity scanner and metadata summary
│   ├── features.py               # Leak-free preprocessor and feature engineering
│   └── baselines.py              # Supervised baseline training and evaluation
├── tools/
│   └── tectonic/                 # Self-contained LaTeX compiler engine
├── workshop1_supervised/
│   ├── notebooks/
│   │   └── eda.ipynb             # Executed EDA notebook with plots and stats
│   ├── report/
│   │   ├── group1_workshop1.tex  # LaTeX source report
│   │   └── group1_workshop1.pdf  # Compiled submission PDF report
│   ├── results/
│   │   ├── baseline_metrics.json # Quantitative test/val/train benchmark metrics
│   │   └── figures/              # High-resolution publication diagnostic figures
│   └── src/                      # Workshop 1 source scripts
├── README.md
└── requirements.txt
```

---

## Workshop 1: Supervised Baselines Benchmark

### Dataset Integrity & Access
- **Source:** Dean De Cock (2011), *Journal of Statistics Education*, Truman State University.
- **Raw TSV URL:** `http://jse.amstat.org/v19n3/decock/AmesHousing.txt`
- **Documentation URL:** `http://jse.amstat.org/v19n3/decock/DataDocumentation.txt`
- **Raw Dimensions:** 2,930 rows, 82 columns.
- **Raw TSV SHA-256:** `6cfe6cb525ba437de428653a1040e2aed7d696640bf75203786a6d7a0e67cfcc`
- **Raw TSV MD5:** `c2997eb69ab4bf946b9fbdb7e6805e36`
- **Canonical CSV SHA-256:** `5d5d25c60c165143749013a89a86c604fd55bac07c2a2e6e0d6c9dacf65e6226`

### Data Preprocessing & Leakage Prevention
1. **Target Stabilization:** Transformed target $y^* = \log(1 + \text{SalePrice})$ to reduce skewness from $+1.744$ to $-0.061$ and eliminate residual heteroscedasticity.
2. **Deterministic Splitting:** 70% Train (2,050 samples), 15% Validation (440 samples), and 15% Test (440 samples) saved permanently in `data/splits.json` (`random_state=42`).
3. **Domain Feature Engineering:** Engineered composite features including `TotalSF` (living area + basement finished space), `TotalBath`, `HouseAge`, `RemodAge`, `TotalPorchSF`, and binary amenity flags.
4. **Leakage Protocol:** Imputation medians (including neighborhood-conditional lot frontage), one-hot categorical dictionaries, and `RobustScaler` parameters were computed exclusively on the training partition. Outlier removal (>4,000 sq ft) was restricted to training.

### Empirical Model Results

| Model | Partition | MAE ($ USD) | RMSE ($ USD) | $R^2$ Score | RMSLE (log) |
|---|---|---|---|---|---|
| **Ridge Regression** ($\alpha = 29.47$) | Train | \$12,508.91 | \$18,368.79 | 0.9417 | 0.1094 |
| | Validation | \$14,101.86 | \$20,057.30 | 0.9481 | 0.1156 |
| | **Test** | **\$15,031.51** | **\$44,511.47** | **0.7421** | **0.1254** |
| **Random Forest** (250 Trees, depth 22) | Train | \$6,279.71 | \$10,485.18 | 0.9810 | 0.0604 |
| | Validation | \$17,779.07 | \$28,992.06 | 0.8916 | 0.1384 |
| | **Test** | **\$16,700.76** | **\$31,325.66** | **0.8723** | **0.1325** |
| **Gradient Boosting** (250 Trees, lr 0.05) | Train | \$7,878.73 | \$10,739.24 | 0.9801 | 0.0591 |
| | Validation | \$14,963.25 | \$22,986.39 | 0.9319 | 0.1199 |
| | **Test** | **\$14,876.80** | **\$31,939.29** | **0.8672** | **0.1208** |

- **Best Baseline:** Gradient Boosting Regressor achieved the lowest Test MAE (\$14,876.80) and lowest Test RMSLE (0.1208, $\approx 12\%$ relative error).
- **Deliverable Report:** Compiled 11-page IEEE/ACM-style PDF available at [`workshop1_supervised/report/group1_workshop1.pdf`](workshop1_supervised/report/group1_workshop1.pdf).

---

## Reproduction Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd ml-project-housing

# Create virtual environment and activate
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset Integrity
```bash
python workshop1_supervised/src/dataset_info.py
```

### 3. Generate Exploratory Visualizations
```bash
python workshop1_supervised/src/eda_generator.py
```

### 4. Train Baselines & Generate Diagnostic Plots
```bash
python workshop1_supervised/src/baselines.py
```

### 5. Recompile LaTeX Report (Optional)
```bash
tools\tectonic\tectonic.exe workshop1_supervised/report/group1_workshop1.tex --outdir workshop1_supervised/report
```
