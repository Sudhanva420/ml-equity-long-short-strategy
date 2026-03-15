#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm


def load_data():
    train_df = pd.read_parquet("/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/splits/train.parquet")

    rank_cols = [col for col in train_df.columns if '_rank' in col]
    train_df = train_df.drop(columns=rank_cols)

    test_df = pd.read_parquet("/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/splits/test.parquet")

    rank_cols = [col for col in test_df.columns if '_rank' in col]
    test_df = test_df.drop(columns=rank_cols)

    val_df = pd.read_parquet("/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/splits/val.parquet")

    rank_cols = [col for col in val_df.columns if '_rank' in col]
    val_df = val_df.drop(columns=rank_cols)

    return train_df, val_df, test_df

train_df, val_df, test_df = load_data()

# Verify splits
def verify_splits(train_df, val_df, test_df):
    print(f"Train period: {train_df['date'].min()} → {train_df['date'].max()}")
    print(f"  Shape: {train_df.shape}")

    print(f"\nVal period: {val_df['date'].min()} → {val_df['date'].max()}")
    print(f"  Shape: {val_df.shape}")

    print(f"\nTest period: {test_df['date'].min()} → {test_df['date'].max()}")
    print(f"  Shape: {test_df.shape}")

    # Verify no overlap
    assert train_df['date'].max() < val_df['date'].min(), "Train/Val overlap"
    assert val_df['date'].max() < test_df['date'].min(), "Val/Test overlap"
    print("\nNo temporal overlap")


feature_cols = [c for c in train_df.columns if c not in ["date", "ticker", "fwd_return_5d"]]
print(len(feature_cols))

def drift_analysis(train_df, val_df, feature_cols):
    drift_stats = []

    for feat in feature_cols:
        train_vals = train_df[feat].dropna()
        val_vals   = val_df[feat].dropna()

        ks_stat, ks_p = ks_2samp(train_vals, val_vals)
        w_dist = wasserstein_distance(train_vals, val_vals)

        drift_stats.append({
            "feature": feat,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "wasserstein": w_dist
        })

    drift_df = pd.DataFrame(drift_stats).sort_values("ks_stat", ascending=False)
    return drift_df

drift_df = drift_analysis(train_df, val_df, feature_cols)


top_drift_feats = drift_df.head(5)["feature"]

for feat in top_drift_feats:
    plt.figure(figsize=(10,4))
    sns.kdeplot(train_df[feat], label="Train", fill=True)
    sns.kdeplot(val_df[feat], label="Validation", fill=True)
    plt.title(f"Distribution Drift: {feat}")
    plt.legend()
    plt.show()
    

def compute_ic(df, feature, target="fwd_return_5d"):
    return df.groupby("date").apply(
        lambda x: x[feature].corr(x[target], method="spearman")
    )


ic_drift = []

for feat in feature_cols:
    ic_train = compute_ic(train_df, feat)
    ic_val   = compute_ic(val_df, feat)

    ic_drift.append({
        "feature": feat,
        "ic_train": ic_train.mean(),
        "ic_val": ic_val.mean(),
        "ic_drop": ic_val.mean() - ic_train.mean()
    })

ic_drift_df = pd.DataFrame(ic_drift).sort_values("ic_drop")
ic_drift_df.head(10)


plt.figure(figsize=(12,6))
sns.histplot(ic_drift_df["ic_drop"], bins=50, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("IC Drift Distribution")
plt.show()


# Statistical Drift Tests (Train vs Val)

def newey_west_tstat(diff_series, lags=5):
    X = np.ones(len(diff_series))
    model = sm.OLS(diff_series, X).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': lags}
    )
    return model.tvalues[0]

def ic_drift_test(train_df, val_df, feature_cols):
    ic_drift_stats = []

    for feat in feature_cols:
        ic_train = compute_ic(train_df, feat).dropna()
        ic_val   = compute_ic(val_df, feat).dropna()

        if len(ic_train) < 30 or len(ic_val) < 30:
            continue

        diff = ic_val - ic_train.mean()

        if len(diff) < 30:
            continue

        t_stat = newey_west_tstat(diff)

        ic_drift_stats.append({
            "feature": feat,
            "nw_tstat": t_stat
        })

    return pd.DataFrame(ic_drift_stats)

ic_drift_stats = ic_drift_test(train_df, val_df, feature_cols)
ic_drift_test_df = pd.DataFrame(ic_drift_stats)
ic_drift_test_df.sort_values("nw_tstat").head(10)


final_drift = drift_df.merge(ic_drift_df, on="feature") \
                       .merge(ic_drift_test_df, on="feature", how="left")

final_drift.sort_values(["ks_stat","ic_drop"], ascending=False).head(15)