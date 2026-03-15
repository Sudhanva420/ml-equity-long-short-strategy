#!/usr/bin/env python
# coding: utf-8


from os import link

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

ic_df = pd.read_parquet("/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/ic_df.parquet")
ic_df.head()

def load_data():
    path = "/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/splits/train.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

def get_feature_cols(df):
    return [col for col in df.columns if col not in ["date", "ticker", "fwd_return_5d"] and not col.endswith("_rank")]
feature_cols = get_feature_cols(df)
print(len(feature_cols))


rank_cols = [col for col in df.columns if '_rank' in col]
df = df.drop(columns=rank_cols)

feature_cols = get_feature_cols(df)
print(len(feature_cols))


# 1. Cross Sectional Correlation Matrix

def corr_matrix(df, feature_cols):
    corr_list = []

    for dt, g in df.groupby('date'):
        
        corr = g[feature_cols].corr()
        corr_list.append(corr)

    avg_corr = pd.concat(corr_list).groupby(level=0).mean()

    plt.figure(figsize=(14,12))
    sns.heatmap(avg_corr, cmap="coolwarm", center=0, xticklabels=True, yticklabels=True)
    plt.title("Average Cross-Sectional Feature Correlation")
    plt.show()

    return avg_corr

avg_corr = corr_matrix(df, feature_cols)

# 2. Converting Correlation to Distance matrix

def dist_matrix(avg_corr):
    distance = np.sqrt(0.5 * (1 - avg_corr))

    link = linkage(distance, method="ward")

    plt.figure(figsize=(16,7))
    dendrogram(link, labels=feature_cols, leaf_rotation=90)
    plt.title("Hierarchical Clustering of Features")
    plt.show()

dist_matrix(avg_corr)

# Features that merge very low are very similar, thus redundant. If they merge higher, they are more independant so its important to keep both

# 3. Creating Clusters

def clusters(avg_corr, k=6):
    distance = np.sqrt(0.5 * (1 - avg_corr))

    link = linkage(distance, method="ward")

    k = 6
    clusters = fcluster(link, k, criterion="maxclust")

    cluster_df = pd.DataFrame({
        "feature": feature_cols,
        "cluster": clusters
    })

    cluster_df.sort_values("cluster").head(20)

    for i in range(1, k+1):
        feats = cluster_df[cluster_df["cluster"] == i]["feature"].tolist()
        print(f"\nCluster {i} ({len(feats)} features)")
        print(feats)

clusters(avg_corr, k=6)

# 4. Cluster Representative Selection

# This part involves selecting one representative per cluster. A good method: Feature with highest IC t-stat inside each cluster

def selection(cluster_df, ic_df):


    cluster_ic = cluster_df.merge(ic_df, on="feature", how="left")
    cluster_ic.head()

    cluster_reps = (
        cluster_ic
        .sort_values("t_stat", ascending=False)
        .groupby("cluster")
        .head(1)
    )

    return cluster_ic,cluster_reps

cluster_df = pd.DataFrame({
    "feature": feature_cols,
    "cluster": fcluster(linkage(np.sqrt(0.5 * (1 - avg_corr)), method="ward"), 6, criterion="maxclust")
})
cluster_ic, clusters_reps = selection(cluster_df, ic_df)

idx = cluster_ic.groupby('cluster')['t_stat'].idxmax()
result = cluster_ic.loc[idx]