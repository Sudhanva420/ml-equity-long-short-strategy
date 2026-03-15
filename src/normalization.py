#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from config import PROCESSED_DATA_PATH
from pathlib import Path

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

train_df, val_df, test_df = load_data()

def winsorize_cs(group, cols, clip=5):
    
    for col in cols:
        mu = group[col].mean()
        sigma = group[col].std()
        if sigma > 0:
            group[col] = np.clip(
                group[col],
                mu - clip*sigma,
                mu + clip*sigma
            )
    return group


def zscore_cs(group, cols):
    for col in cols:
        mu = group[col].mean()
        sigma = group[col].std()
        if sigma > 0:
            group[col] = (group[col] - mu) / sigma
    return group


feature_cols = [c for c in train_df.columns 
                if c not in ["date", "ticker", "fwd_return_5d"]]


# Cross-Sectionally applying both functions to all 3 dfs

def normalize_features(df):
    train_df = train_df.groupby("date", group_keys=False)\
        .apply(lambda x: winsorize_cs(x, feature_cols))

    train_df = train_df.groupby("date", group_keys=False)\
        .apply(lambda x: zscore_cs(x, feature_cols))
        
    val_df = val_df.groupby("date", group_keys=False)\
        .apply(lambda x: winsorize_cs(x, feature_cols))

    val_df = val_df.groupby("date", group_keys=False)\
        .apply(lambda x: zscore_cs(x, feature_cols))


    test_df = test_df.groupby("date", group_keys=False)\
        .apply(lambda x: winsorize_cs(x, feature_cols))

    test_df = test_df.groupby("date", group_keys=False)\
        .apply(lambda x: zscore_cs(x, feature_cols))

normalize_features(train_df)
normalize_features(val_df)
normalize_features(test_df)

def checking():
    
    check = train_df.groupby("date")[feature_cols].mean().mean()

    check_std = train_df.groupby("date")[feature_cols].std().mean()

    return check, check_std


train_check, train_check_std = checking()
train_check_std

val_df.shape

val_check, val_check_std = checking()
val_check_std   


test_df.shape
test_check, test_check_std = checking()
test_check_std

def save_features(df, path):
    
    path = Path(PROCESSED_DATA_PATH) / path
    df.to_parquet(path, index=False)
    print(f"Saved features to {path}")

save_features(train_df, "train_normalized_df.parquet")
save_features(val_df, "val_normalized_df.parquet")
save_features(test_df, "test_normalized_df.parquet")
