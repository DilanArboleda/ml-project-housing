# Workshop 6 — Medical Domain Adaptation
## Group 1: Housing Price Prediction → Cardiac Arrhythmia Detection

Transfer of the machine learning techniques developed in Workshops 1–5
(supervised learning, deep learning, reinforcement learning, autoencoders,
GANs) to the **MIT-BIH Arrhythmia Database** for ECG beat classification.

### Project structure

workshop6_medical/
├── src/
│   ├── medical_preprocessing.py    # Data pipeline (MIT-BIH → tabular features)
│   ├── medical_visualization.py    # EDA figures
│   └── transfer.py                 # (FASE 2) Model transfer
├── notebooks/
│   └── 01_medical_eda.ipynb        # Exploratory analysis
├── data/                           # (gitignored) MIT-BIH raw + features CSV
├── results/                        # (gitignored) figures, checkpoints
└── report/                         # Final PDF report (FASE 4)

### Reproducibility

```bash
# 1. Install dependencies
pip install wfdb pandas scikit-learn matplotlib seaborn scipy

# 2. Download MIT-BIH (~100 MB, once)
python -c "from src.medical_preprocessing import download_mitdb; download_mitdb('data/mitdb')"

# 3. Build feature matrix + splits
python -c "from src.medical_preprocessing import MITDB_RECORDS, build_feature_matrix, make_splits, save_splits; \
           df = build_feature_matrix(MITDB_RECORDS, 'data/mitdb'); \
           df.to_csv('data/features_mitdb.csv', index=False); \
           save_splits(make_splits(df, seed=42), 'data/splits_medical.json')"

# 4. Run EDA notebook
jupyter notebook notebooks/01_medical_eda.ipynb



Dataset
MIT-BIH Arrhythmia Database (Moody & Mark, 2001)

48 half-hour 2-lead ECG recordings, 360 Hz

Beat labels: AAMI EC57 superclasses N, S, V, F (Q excluded)

101,297 valid beats after preprocessing

Patient-level stratified split: 34/7/7 records (train/val/test)

Status
☑ FASE 1 — Data pipeline + EDA (Complete)
□ FASE 2 — Model transfer (In progress)
□ FASE 3 — Clinical evaluation + interpretability
□ FASE 4 — Deployment analysis + final report



