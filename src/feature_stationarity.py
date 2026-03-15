#!/usr/bin/env python
# coding: utf-8


# We do this after an extensive testing phase in stat_testing. This is done at the end because stat_testing checks different metrics of IC features. So once we have completed the check we can confidently saying that these features actaully add value. Then stationarity testing checks if these signals persist and are reliable
# 
# stats_testing is on IC time series while the stat_stationary is on the feature itself

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from tqdm import tqdm

def load_data():
    features_df = pd.read_parquet("/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/splits/train.parquet")
    features_df.head()
    
    df = features_df

    df["date"] = pd.to_datetime(df["date"])

    rank_cols = [col for col in df.columns if '_rank' in col]
    df = df.drop(columns=rank_cols)
    
    return df

df =load_data()

def cs(group, cols):

    feature_cols = [c for c in df.columns if c not in cols]

    len(feature_cols)
    #Creating the cross-sectional median time series

    cs_median = (df.groupby('date')[feature_cols].median().sort_index())
    return cs_median

cs_median = cs(cols = ["date", "ticker", "fwd_return_5d"])


def run_stationarity_tests(series):
    
    series = series.dropna()
    
    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
    
    return adf_stat, adf_p, kpss_stat, kpss_p

adf_stat, adf_p, kpss_stat, kpss_p = run_stationarity_tests(cs_median["momentum_20d"])

def create_stationarity_df(cs_median, cols):
    
    results = []

    feature_cols = [c for c in cs_median.columns if c not in cols]
    for feature in tqdm(feature_cols):
        
        try:
            
            adf_stat, adf_p, kpss_stat, kpss_p = run_stationarity_tests(cs_median[feature])
            results.append({
                "feature": feature,
                "adf_stat": adf_stat,
                "adf_p": adf_p,
                "kpss_stat": kpss_stat,
                "kpss_p": kpss_p
            })
            
        except:
            continue

    stationarity_df = pd.DataFrame(results)
    stationarity_df.head()
    return stationarity_df

stationarity_df = create_stationarity_df(cs_median, ["date", "ticker", "fwd_return_5d"])

def classify_stationarity(row):
    
    adf_stationary = row['adf_p'] < 0.05
    kpss_stationary = row['kpss_p'] > 0.05

    if adf_stationary and kpss_stationary:
        return "stationary"
    elif not adf_stationary and not kpss_stationary:
        return "non_stationary"
    else:
        return "borderline"

stationarity_df["regime"] = stationarity_df.apply(classify_stationarity, axis=1)
stationarity_df["regime"].value_counts()

stationarity_df.sort_values("adf_p").head(10)


stationarity_df.sort_values("kpss_p").head(10)


# 2. Subsample testing(randomly choose stocks and test) 

def subsample_test(cs_median, feature, n_samples=30):
    
    tickers = df['ticker'].unique()
    sample_tickers = np.random.choice(tickers, 30, replace=False)

    test_feat = stationarity_df.query("regime=='non_stationary'").iloc[0]["feature"]

    bad_series = (
        df[df["ticker"].isin(sample_tickers)]
        .pivot(index="date", columns="ticker", values=test_feat)
    )

    bad_series.head()

    bad_series.head()
    test_feat

    return bad_series

bad_series = subsample_test(cs_median, feature="momentum_20d", n_samples=30)
# We checked median but now we should know whether for some tickers, is it actually stationary

def pvals_check():
    pvals = []

    for col in bad_series:
        series = bad_series[col].dropna()
        if len(series) > 200:
            pvals.append(adfuller(series)[1])

    np.mean(np.array(pvals) < 0.05)
    
pvals_check()
