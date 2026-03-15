#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm
import statsmodels.api as sm
from config import PROCESSED_DATA_PATH
from pathlib import Path

pd.set_option('display.max_columns', 100)


# LOADING THE FEATURE DATASET AND SETTING IT UP

def load_data():
    features_df = pd.read_parquet("/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/splits/train.parquet")
    features_df.head()


    features_df.shape

    features_df.reset_index(drop=True, inplace=True)

    features_df.shape


    features_df.head()

features_df = load_data()


def clean_data(df):
    rank_cols = [col for col in df.columns if '_rank' in col]
    df2 = features_df.drop(columns=rank_cols)

    cols = ["date", "ticker", "fwd_return_5d"]

    feature_cols = [c for c in df2.columns if c not in cols]

    len(feature_cols)
    return df2, feature_cols

df2, feature_cols = clean_data(features_df)



# Info. Coefficient Function

def compute_daily_ic(df, feature_cols, return_col="fwd_return_5d"):
    ic_records = []

    grouped = df.groupby("date")

    for date, g in tqdm(grouped, total=len(grouped)):
        if g.shape[0] < 30:
            continue 

        for col in feature_cols:
            x = g[col]
            y = g[return_col]

            if x.isna().all():
                continue

            ic, _ = spearmanr(x, y, nan_policy='omit')

            ic_records.append({
                "date": date,
                "feature": col,
                "ic": ic
            })

    return pd.DataFrame(ic_records)


ic_df = compute_daily_ic(df2, feature_cols)
ic_df.head()

cols = ['date','ic', 'feature']
ic_df[cols].sort_values(by = 'ic', ascending=False).head(20)

# Aggregating IC stats

def ic_summary_stats(ic_df):
    ic_summary = (ic_df.groupby("feature")['ic'].agg(['mean', 'std', 'count']).reset_index())
    ic_summary

    cols = ['mean', 'feature']
    ic_summary[cols].sort_values(by = 'mean', ascending=False).head(20)

    ic_summary["t_stat"] = (
        ic_summary["mean"] / 
        (ic_summary["std"] / np.sqrt(ic_summary["count"]))
    )

    ic_summary["IR"] = ic_summary["mean"] / ic_summary["std"]

    ic_summary.sort_values("t_stat", ascending=False).head(10)
    
    return ic_summary

ic_summary = ic_summary_stats(ic_df)


def plot_ic_distribution(ic_summary):
    plt.figure(figsize=(10,5))
    sns.histplot(ic_summary["mean"], bins=50, kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Distribution of Mean IC Across Features")
    plt.show()
    
plot_ic_distribution(ic_summary)

significant_features = ic_summary[ic_summary['t_stat'].abs() > 2.0]
print(f"Significant features: {len(significant_features)}")

# t-stat is high but IC IR is very low, so signal exits overall but not tradeable( requires more stocks to be considered)


top_features = ic_summary.sort_values("t_stat", ascending=False).head(5)["feature"]

def plot_ic_stability(ic_df, top_features):
    plt.figure(figsize=(14,6))

    for feat in top_features:
        tmp = ic_df[ic_df["feature"] == feat]
        tmp = tmp.set_index("date")["ic"].rolling(63).mean()
        plt.plot(tmp, label=feat)

    plt.legend()
    plt.title("6-Month Rolling IC Stability")
    plt.show()

plot_ic_distribution(ic_summary)
plot_ic_stability(ic_df, top_features)


# Graph oscillates up and down for all, shows that the trend isn't consistent. Could be due to regime changes, mean reverting behaviour, weak signal etc

# Statistical Significance Testing of IC 

# 1) Building IC time series
# This requires a wide-form table, not the long-form that is currently there. Having wide-form makes it easier to plot,compute stats directly(now you need to do groupby everytime), ranking etc 


ic_ts = ic_df.pivot(index="date", columns="feature", values="ic")
ic_ts

def newey_west_tstat(series, lags=5):
    
    y = series.dropna().values
    X = np.ones(len(y))   # intercept only

    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})

    return results.tvalues[0], results.pvalues[0]

nw_results = []

for col in ic_ts.columns:
    t_stat, p_val = newey_west_tstat(ic_ts[col], lags=5)

    nw_results.append({
        "feature": col,
        "nw_t": t_stat,
        "nw_p": p_val
    })
nw_df = pd.DataFrame(nw_results)
nw_df.sort_values("nw_t", ascending=False).head(10)


# Compared to the naive-tstat, now only 7/14 have newey-west t-stat value above 2. So we can move on with these more confidently sicne autocorr. and heteroskeadticity have been accounted for

# Bootstrap Significance Testing

def block_bootstrap(series, block = 20, n_samples = 5000):
    
    series = series.dropna().values
    n = len(series)
    means = []
    
    for _ in range(n_samples):
        
        idx = []
        
        while len(idx) < n:
            
            start = np.random.randint(0, n-block)
            idx.extend(range(start, start + block))

        sample = series[idx[:n]]
        means.append(sample.mean())

    return np.array(means)


boot_results = []

for col in ic_ts.columns:

    series = ic_ts[col].dropna()

    actual = series.mean()

    # Enforce null hypothesis: mean = 0
    shifted = series - actual

    dist = block_bootstrap(shifted, block=20, n_samples=3000)

    # Bootstrap distribution of mean under H0
    p_value = (np.abs(dist) >= np.abs(actual)).mean()

    boot_results.append({
        "feature": col,
        "boot_p": p_value
    })

boot_df = pd.DataFrame(boot_results)

ic_stats_full = (
    ic_summary
    .merge(nw_df, on="feature")
    .merge(boot_df, on="feature")
)


def save_features(df):
    
    path = Path(PROCESSED_DATA_PATH) / "ic_df.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved features to {path}")

save_features(ic_stats_full)


# Layer and What Is Filtered:
# 
# 1) Mean IC - Useless signals
# 2) t-stat + IR - Noisy signals
# 3) Stability - Regime fragile
# 4) Newey-West - Autocorrelation illusions
# 5) Bootstrap - Luck-based artifacts

