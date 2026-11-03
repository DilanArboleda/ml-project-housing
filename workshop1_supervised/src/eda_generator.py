"""
EDA Script: Generate visualizations, compute statistical findings,
and export figures for Workshop No. 1.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero OrdeÃ±ez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure publication-quality style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_csv = base_dir / "data" / "raw" / "AmesHousing.csv"
    fig_dir = base_dir / "workshop1_supervised" / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {raw_csv}...")
    df = pd.read_csv(raw_csv)
    print(f"Shape: {df.shape}")

    # 1. Target Distribution: Raw vs Log-transformed
    print("Generating Figure 1: Target distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Raw SalePrice
    sns.histplot(df["SalePrice"], kde=True, ax=axes[0], color="#2b5c8f", bins=40)
    raw_skew = df["SalePrice"].skew()
    raw_kurt = df["SalePrice"].kurtosis()
    axes[0].set_title(f"Raw SalePrice Distribution\nSkewness: {raw_skew:.2f} | Kurtosis: {raw_kurt:.2f}")
    axes[0].set_xlabel("SalePrice ($ USD)")
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter("${x:,.0f}")

    # Log-transformed SalePrice
    log_sp = np.log1p(df["SalePrice"])
    sns.histplot(log_sp, kde=True, ax=axes[1], color="#2ca02c", bins=40)
    log_skew = log_sp.skew()
    log_kurt = log_sp.kurtosis()
    axes[1].set_title(f"Log-Transformed log(1 + SalePrice)\nSkewness: {log_skew:.2f} | Kurtosis: {log_kurt:.2f}")
    axes[1].set_xlabel("log(1 + SalePrice)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig1_path = fig_dir / "target_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # 2. Correlation with Target (Top 15 numerical features)
    print("Generating Figure 2: Correlation heatmap...")
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["Order", "PID"], errors="ignore")
    corr_series = num_df.corr()["SalePrice"].sort_values(ascending=False)
    top_corr_features = corr_series.head(16).index.tolist() # include SalePrice
    
    plt.figure(figsize=(12, 10))
    corr_sub = num_df[top_corr_features].corr()
    mask = np.triu(np.ones_like(corr_sub, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_sub, mask=mask, cmap=cmap, vmax=1.0, vmin=-0.2, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix of Top Features with SalePrice", pad=15)
    plt.tight_layout()
    fig2_path = fig_dir / "correlation_heatmap.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # 3. Outlier Analysis: Gr Liv Area vs SalePrice
    print("Generating Figure 3: Outliers scatter...")
    plt.figure(figsize=(9, 6))
    colors = np.where((df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 300000), "#d62728", "#1f77b4")
    plt.scatter(df["Gr Liv Area"], df["SalePrice"], c=colors, alpha=0.6, edgecolors="none", s=35)
    
    # Highlight De Cock outliers
    outliers = df[(df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 300000)]
    for _, r in outliers.iterrows():
        plt.annotate(f"PID: {r['PID']}\n({r['Gr Liv Area']} sqft, ${r['SalePrice']:,})",
                     (r["Gr Liv Area"], r["SalePrice"]),
                     xytext=(-90, 20), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                     fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ffcccc", ec="#d62728", lw=1))

    plt.axvline(x=4000, color="grey", linestyle="--", alpha=0.7, label="De Cock Outlier Threshold (4,000 sq ft)")
    plt.title("Living Area vs. SalePrice (Identifying Partial-Sale Atypical Outliers)")
    plt.xlabel("Above Ground Living Area (Gr Liv Area in sq ft)")
    plt.ylabel("SalePrice ($ USD)")
    plt.gca().yaxis.set_major_formatter("${x:,.0f}")
    plt.legend()
    plt.tight_layout()
    fig3_path = fig_dir / "outliers_grlivarea.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # 4. Missing Values Analysis
    print("Generating Figure 4: Missing values bar...")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts[missing_counts > 0] / len(df) * 100).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    bars = plt.barh(missing_pct.index, missing_pct.values, color="#4c72b0", edgecolor="none")
    plt.xlabel("Percentage Missing (%)")
    plt.title(f"Missing Value Profile across Features ({len(missing_pct)} Columns with Missing Entries)")
    plt.xlim(0, 105)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=8.5)
    plt.tight_layout()
    fig4_path = fig_dir / "missing_values_bar.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    # 5. Price by Overall Quality Boxplot
    print("Generating Figure 5: Overall Quality boxplot...")
    plt.figure(figsize=(10, 5.5))
    sns.boxplot(x="Overall Qual", y="SalePrice", data=df, palette="crest", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    plt.title("SalePrice Distribution Across Overall Material and Finish Quality Ratings (1-10)")
    plt.xlabel("Overall Quality Rating (1 = Very Poor, 10 = Very Excellent)")
    plt.ylabel("SalePrice ($ USD)")
    plt.gca().yaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    fig5_path = fig_dir / "overall_qual_boxplot.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    # 6. Median SalePrice by Neighborhood
    print("Generating Figure 6: Neighborhood price distribution...")
    plt.figure(figsize=(12, 7))
    neigh_order = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).index
    sns.barplot(x="SalePrice", y="Neighborhood", data=df, order=neigh_order,
                estimator=np.median, ci=None, palette="viridis")
    plt.title("Median Property Sale Price by Ames Neighborhood")
    plt.xlabel("Median SalePrice ($ USD)")
    plt.ylabel("Neighborhood")
    plt.gca().xaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    fig6_path = fig_dir / "neighborhood_price.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"Saved: {fig6_path}")

    print("All EDA figures generated successfully!")

if __name__ == "__main__":
    main()

