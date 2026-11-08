# Workshop 2 - Advanced Learning Models (Part A)

This directory contains the deep learning experiments, final evaluation, and
academic report for Group 1. The model is a PyTorch MLP for Ames Housing price
regression. All splits come from `data/splits.json`; preprocessing and scaling
statistics are fitted on the training partition only.

## Environment

From the repository root on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The raw Ames file must be available as `data/raw/AmesHousing.csv` or
`data/raw/AmesHousing.tsv`. Existing split assignments must not be regenerated.

## Main notebook

The practical documentation for Part A is available at:

```text
workshop2_deep_learning/notebooks/workshop2_deep_learning.ipynb
```

The notebook is written entirely in English and contains the methodology,
architecture summary, initial-training diagnosis, controlled experiments,
ablation study, multi-seed analysis, final test comparison, and residual
analysis. Its tables and figures are stored as notebook outputs so they render
directly on GitHub. Executing the notebook loads existing JSON, CSV, figure,
and checkpoint artifacts; it does not retrain the models or overwrite results.

Run it from the repository root with Jupyter Notebook or JupyterLab after
activating the project environment. The path-discovery cell also supports
execution from the notebook directory.

## Reproduce training and experiments

The original experiment-2 reference can be trained with:

```powershell
python workshop2_deep_learning/src/train.py
```

The controlled architecture, regularization, learning-rate, and ablation study
can be resumed without overwriting completed runs:

```powershell
python -u workshop2_deep_learning/src/run_experiments.py
```

Histories and summaries are written under
`workshop2_deep_learning/results/part_a/`; checkpoints are written under
`checkpoints/part_a/`. The selected configuration is the 128-64 MLP with
dropout 0.05, learning rate `1e-3`, and weight decay `1e-5`.

## Final test evaluation

The final evaluation loads the frozen seed-42 checkpoint and does not retrain
or reselect the model:

```powershell
python workshop2_deep_learning/src/evaluate.py
```

It writes the definitive table, per-observation predictions, reproducibility
audit, and figures under `workshop2_deep_learning/results/final_test/` and
`workshop2_deep_learning/results/figures/final_test/`.

## Tests

```powershell
python workshop2_deep_learning/src/test_model.py
python -m unittest discover -s workshop2_deep_learning/tests -v
```

The first command checks checkpoint metadata, validation-only selection,
train-only scaler statistics, split identity, inverse target transformation,
prediction reproducibility, and saved metrics. The second command tests the
shared standardization utilities and model output.

## Academic report

The IEEE-style report source and bibliography are:

```text
workshop2_deep_learning/report/group1_workshop2.tex
workshop2_deep_learning/report/references.bib
```

Generate the compact report figures from the saved experiment artifacts, then
compile from the report directory so the relative paths resolve:

```powershell
python workshop2_deep_learning/report/generate_report_figures.py
cd workshop2_deep_learning/report
pdflatex group1_workshop2.tex
bibtex group1_workshop2
pdflatex group1_workshop2.tex
pdflatex group1_workshop2.tex
```

The expected output name is `group1_workshop2.pdf`. An IEEEtran-compatible
LaTeX distribution is required. Tectonic can also compile the source directly
and resolves the bibliography automatically. The figure script and report use
only existing experimental artifacts; they do not retrain models or modify
Workshop 1.
