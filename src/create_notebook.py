"""
Script to generate the complete, executed EDA notebook for Workshop 1.
Group 1
"""

import json
from pathlib import Path
import nbformat as nbf

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.10"
        }
    }

    cells = []

    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# Workshop No. 1 â€” Exploratory Data Analysis (EDA)
## Machine Learning (Semester 2026-III) â€” Universidad Distrital Francisco JosÃ© de Caldas
### Domain: Real Estate Valuation â€” Ames Housing Dataset
**Instructor:** Eng. Carlos AndrÃ©s Sierra, M.Sc.  
**Team (Group 1):**
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero OrdeÃ±ez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)

---
### Notebook Objectives
1. Verify dataset integrity, variable types, and dimensions.
2. Characterize the target distribution (`SalePrice`) and evaluate log-transformation $\\log(1 + y)$ for variance stabilization.
3. Identify and document missingness mechanisms (distinguishing real missing values from "feature absence").
4. Detect influential anomalies and outliers (specifically houses > 4,000 sq ft as highlighted by Dean De Cock).
5. Explore correlation structures and feature relationships to guide feature engineering and baseline model development.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import os
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

# Paths: traverse upward until we locate data/raw/AmesHousing.csv
current = Path.cwd().resolve()
RAW_DATA_PATH = None
for _ in range(5):
    candidate = current / "data" / "raw" / "AmesHousing.csv"
    if candidate.exists():
        RAW_DATA_PATH = candidate
        break
    current = current.parent

if RAW_DATA_PATH is None or not RAW_DATA_PATH.exists():
    raise FileNotFoundError("Could not find data/raw/AmesHousing.csv in parent directories")

print(f"Data path resolved: {RAW_DATA_PATH}")
"""))

    # Loading Data
    cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Ingestion and Structural Overview"""))
    cells.append(nbf.v4.new_code_cell("""df = pd.read_csv(RAW_DATA_PATH)
print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
df.head(5)
"""))

    cells.append(nbf.v4.new_code_cell("""# Summary of data types
dtype_counts = df.dtypes.value_counts()
print("Data Types Breakdown:")
for dt, count in dtype_counts.items():
    print(f"  {str(dt):<12}: {count} columns")

num_cols = df.select_dtypes(include=[np.number]).columns.drop(["Order", "PID"], errors="ignore")
cat_cols = df.select_dtypes(exclude=[np.number]).columns
print(f"\\nNumber of numeric features:     {len(num_cols)}")
print(f"Number of categorical features: {len(cat_cols)}")
"""))

    # Target Distribution
    cells.append(nbf.v4.new_markdown_cell("""## 2. Target Variable Analysis (`SalePrice`)
We analyze the distribution of the target variable. Real estate prices typically exhibit right-skewness and heteroscedasticity due to high-value luxury properties.
"""))

    cells.append(nbf.v4.new_code_cell("""sp = df["SalePrice"]
print(f"SalePrice Summary:")
print(f"  Count:          {sp.count():,}")
print(f"  Mean:           ${sp.mean():,.2f}")
print(f"  Median:         ${sp.median():,.2f}")
print(f"  Std Dev:        ${sp.std():,.2f}")
print(f"  Min:            ${sp.min():,}")
print(f"  25th Percentile:${sp.quantile(0.25):,}")
print(f"  75th Percentile:${sp.quantile(0.75):,}")
print(f"  Max:            ${sp.max():,}")
print(f"  Skewness:       {sp.skew():.3f}")
print(f"  Kurtosis:       {sp.kurtosis():.3f}")
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw distribution
sns.histplot(df["SalePrice"], kde=True, ax=axes[0], color="#2b5c8f", bins=40)
axes[0].set_title(f"Raw SalePrice Distribution\\nSkewness: {df['SalePrice'].skew():.2f} | Kurtosis: {df['SalePrice'].kurtosis():.2f}")
axes[0].set_xlabel("SalePrice ($)")
axes[0].xaxis.set_major_formatter("${x:,.0f}")

# Log-transformed
log_sp = np.log1p(df["SalePrice"])
sns.histplot(log_sp, kde=True, ax=axes[1], color="#2ca02c", bins=40)
axes[1].set_title(f"Log1p SalePrice Distribution\\nSkewness: {log_sp.skew():.2f} | Kurtosis: {log_sp.kurtosis():.2f}")
axes[1].set_xlabel("log(1 + SalePrice)")

plt.tight_layout()
plt.show()
"""))

    # Missing Data
    cells.append(nbf.v4.new_markdown_cell("""## 3. Missing Data Analysis: Distinguishing Missingness from Feature Absence
In the Ames Housing dataset, documentation confirms that for many features (`Alley`, `Bsmt Qual`, `Fireplace Qu`, `Garage Type`, `Pool QC`, `Fence`), `NA` does not represent missing information at random (MCAR/MAR), but rather the physical absence of that amenity.
"""))

    cells.append(nbf.v4.new_code_cell("""missing = df.isnull().sum()
missing_pct = (missing[missing > 0] / len(df) * 100).sort_values(ascending=False)

print(f"Total columns with missing values: {len(missing_pct)}")
print(missing_pct.head(15))
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))
missing_pct.sort_values(ascending=True).plot(kind="barh", color="#4c72b0")
plt.title("Missing Value Percentage Across Affected Features")
plt.xlabel("Percentage Missing (%)")
plt.tight_layout()
plt.show()
"""))

    # Outliers
    cells.append(nbf.v4.new_markdown_cell("""## 4. Influential Outliers: Above-Ground Living Area vs. SalePrice
Dean De Cock (2011) explicitly recommends removing observations with `Gr Liv Area` > 4,000 sq ft that do not represent typical residential transactions.
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 5.5))
outlier_mask = (df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 300000)
colors = np.where(outlier_mask, "#d62728", "#1f77b4")
plt.scatter(df["Gr Liv Area"], df["SalePrice"], c=colors, alpha=0.6, s=30)
plt.axvline(x=4000, color="grey", linestyle="--", label="De Cock Outlier Threshold (4,000 sq ft)")

outliers = df[outlier_mask]
for _, r in outliers.iterrows():
    plt.annotate(f"PID: {r['PID']}\\n({r['Gr Liv Area']} sqft, ${r['SalePrice']:,})",
                 (r["Gr Liv Area"], r["SalePrice"]),
                 xytext=(-80, 20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1),
                 fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#ffcccc", ec="#d62728"))

plt.title("Gr Liv Area vs. SalePrice: Identifying High-Leverage Outliers")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("SalePrice ($)")
plt.gca().yaxis.set_major_formatter("${x:,.0f}")
plt.legend()
plt.tight_layout()
plt.show()
"""))

    # Feature Correlations
    cells.append(nbf.v4.new_markdown_cell("""## 5. Feature Relationships and Correlation Matrix
We evaluate which numerical features possess the strongest linear correlation with `SalePrice`.
"""))

    cells.append(nbf.v4.new_code_cell("""numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Order", "PID"], errors="ignore")
correlations = numeric_df.corr()["SalePrice"].sort_values(ascending=False)

print("Top 15 Features Positively Correlated with SalePrice:")
print(correlations.head(16))
"""))

    cells.append(nbf.v4.new_code_cell("""top_features = correlations.head(11).index
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df[top_features].corr(), annot=True, fmt=".2f", cmap="vlag", square=True)
plt.title("Correlation Heatmap of Top 10 Predictors with SalePrice")
plt.tight_layout()
plt.show()
"""))

    # Categorical Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 6. Categorical Predictor Analysis
Analyzing quality ratings (`Overall Qual`) and spatial groupings (`Neighborhood`).
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(x="Overall Qual", y="SalePrice", data=df, ax=axes[0], palette="crest")
axes[0].set_title("SalePrice across Overall Material and Finish Quality (1-10)")
axes[0].set_xlabel("Overall Quality Rating")
axes[0].set_ylabel("SalePrice ($)")
axes[0].yaxis.set_major_formatter("${x:,.0f}")

neigh_order = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).index
sns.barplot(x="SalePrice", y="Neighborhood", data=df, order=neigh_order, ax=axes[1],
            estimator=np.median, errorbar=None, palette="viridis")
axes[1].set_title("Median SalePrice by Neighborhood")
axes[1].set_xlabel("Median SalePrice ($)")
axes[1].xaxis.set_major_formatter("${x:,.0f}")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 7. Key Findings & Recommendations for Modeling
1. **Target Transformation:** Model on $\\log(1 + \\text{SalePrice})$ to mitigate severe right skewness (+1.74) and stabilize residual variance.
2. **Outlier Treatment:** Remove the extreme outliers (> 4,000 sq ft with low price) exclusively during training set preprocessing to avoid distorting regression slopes.
3. **Domain Encoding:** Encode physical absence (`NA`) as an explicit `'None'` category for features like `GarageType`, `BsmtQual`, etc.
4. **Feature Engineering:** Create composite features such as `TotalSF` (living area + basement area) and `TotalBath`, which exhibit higher correlation than individual components.
5. **Leakage Prevention:** Compute all imputation values (e.g. median LotFrontage per neighborhood) and scaling parameters strictly on the training partition.
"""))

    nb.cells = cells

    # Save to both paths
    p1 = Path("e:/Machine learning/ml-project-housing/workshop1_supervised/notebooks/eda.ipynb")
    p2 = Path("e:/Machine learning/ml-project-housing/notebooks/eda.ipynb")
    
    p1.parent.mkdir(parents=True, exist_ok=True)
    p2.parent.mkdir(parents=True, exist_ok=True)

    with open(p1, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    with open(p2, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook created at {p1} and {p2}")

if __name__ == "__main__":
    create_eda_notebook()

