#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pathlib import Path

from config import PROCESSED_DATA_PATH

def load_clean_data():
    path = Path(PROCESSED_DATA_PATH) / "clean_panel.parquet"
    df = pd.read_parquet(path)
    return df

df = load_clean_data()

# FEATURE ENGINEERING
# 1) MOMENTUM FEATURES


def momentum(df):
    
    df = df.copy()
    
    windows = [5, 10, 21, 63, 252]
    
    #Direct momentum features based on return
    for w in windows:
        
        df[f"return_{w}d"] = (df.groupby('ticker')["adj_close"].pct_change(w))
        
    # These features are a combination of both price and moving average, gives an idea how far price has gone from its mean
    for w in windows:
        
        ma = df.groupby('ticker')["adj_close"].transform(lambda x: x.rolling(w).mean())
        df[f"price_ma_ratio_{w}"] = (df["adj_close"] - ma)/ma
    
    #This is to get what percentage of days was my return greater than 0
    pos_days = (df.groupby('ticker')["return_1d"].transform(lambda x: x>0).rolling(21).mean())
    
    df['pos_days_21'] = pos_days
    
    return df
    

# 2) MEAN REVERSION FEATURES

def mean_reversion(df):
    
    df = df.copy()
    
    #Shift by 1 to prevent look-ahead bias(you want the comparison of current value to how values have changed before, so including today's is wrong)
    
    
    #Z-scores based on price
    roll_mean_21 = df.groupby('ticker')['adj_close'].transform(lambda x :x.shift(1).rolling(21).mean())
    
    roll_std_21 = df.groupby('ticker')['adj_close'].transform(lambda x :x.shift(1).rolling(21).std())
    
    df["z_price_21"] = (df['adj_close'] - roll_mean_21)/roll_std_21
    
    roll_mean_40 = df.groupby('ticker')['adj_close'].transform(lambda x :x.shift(1).rolling(40).mean())
    
    roll_std_40 = df.groupby('ticker')['adj_close'].transform(lambda x :x.shift(1).rolling(40).std())
    
    df["z_price_63"] = (df['adj_close'] - roll_mean_40)/roll_std_40
    
    #Z-scores based on 1d, 5d returns
    roll_mean_ret_5 = df.groupby('ticker')['return_1d'].transform(lambda x : x.shift(1).rolling(5).mean())
    roll_mean_ret_21 = df.groupby('ticker')['return_1d'].transform(lambda x : x.shift(1).rolling(21).mean())
    roll_mean_ret_35 = df.groupby('ticker')['return_1d'].transform(lambda x : x.shift(1).rolling(35).mean())
    
    roll_std_ret_5 = df.groupby('ticker')['return_1d'].transform(lambda x :x.shift(1).rolling(5).std())
    roll_std_ret_21 = df.groupby('ticker')['return_1d'].transform(lambda x :x.shift(1).rolling(21).std())
    roll_std_ret_35 = df.groupby('ticker')['return_1d'].transform(lambda x :x.shift(1).rolling(35).std())
    
    df["z_return_1d_5"] = (df['return_1d']-roll_mean_ret_5)/roll_std_ret_5
    
    df["z_return_1d_21"] = (df['return_1d']-roll_mean_ret_21)/roll_std_ret_21
    
    df["z_return_1d_35"] = (df['return_1d']-roll_mean_ret_35)/roll_std_ret_35
    
    roll_mean_ret_5_5d = df.groupby('ticker')['return_5d'].transform(lambda x : x.shift(1).rolling(5).mean())
    roll_mean_ret_21_5d = df.groupby('ticker')['return_5d'].transform(lambda x : x.shift(1).rolling(21).mean())
    roll_mean_ret_35_5d = df.groupby('ticker')['return_5d'].transform(lambda x : x.shift(1).rolling(35).mean())
    
    roll_std_ret_5_5d = df.groupby('ticker')['return_5d'].transform(lambda x :x.shift(1).rolling(5).std())
    roll_std_ret_21_5d = df.groupby('ticker')['return_5d'].transform(lambda x :x.shift(1).rolling(21).std())
    roll_std_ret_35_5d = df.groupby('ticker')['return_5d'].transform(lambda x :x.shift(1).rolling(35).std())
    
    df["z_return_5d_5"] = (df['return_5d']-roll_mean_ret_5_5d)/roll_std_ret_5_5d
    
    df["z_return_5d_21"] = (df['return_5d']-roll_mean_ret_21_5d)/roll_std_ret_21_5d
    
    df["z_return_5d_35"] = (df['return_5d']-roll_mean_ret_35_5d)/roll_std_ret_35_5d
    
    
    #Simple negative of return features
    df["rev_1d"] = -df["return_1d"]
    df["rev_5d"] = -df['return_5d']

    return df


# 3) VOLATILITY AND RISK FEATURES

def volatility(df):
    
    df = df.copy()
    
    for w in [5,10,21,63]:
        
        df[f"vol_{w}"] = df.groupby('ticker')['return_1d'].transform(lambda x : x.rolling(w).std())
        
    df['vol_ratio_21_63'] =  df['vol_21']/ df['vol_63']
    
    df['vol_ratio_5_21'] = df['vol_5']/df['vol_21']
    
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    downside_1d = df["return_1d"].clip(upper=0)
    
    downside_5d = df['return_5d'].clip(upper=0)
    
    df["downside_vol_1d"] = downside_1d.groupby(df["ticker"]).transform(
        lambda x: x.rolling(21).std()
    )

    df["downside_vol_5d"] = downside_5d.groupby(df["ticker"]).transform(
        lambda x: x.rolling(21).std()
    )
    
    return df
   

# 4) VOLUME AND LIQUIDITY FEATURES


def volume_features(df):
    
    df = df.copy()
    
    df['dollar_volume'] = df['adj_close']*df['volume']
    
    df['vol_avg_5'] = df.groupby('ticker')['volume'].transform(lambda x : x.rolling(5).mean())
    df['vol_avg_21'] = df.groupby('ticker')['volume'].transform(lambda x : x.rolling(21).mean())
    df['vol_avg_63'] = df.groupby('ticker')['volume'].transform(lambda x : x.rolling(63).mean())
    
    df["dollar_vol_log"] = np.log1p(df['dollar_volume'])
    
    #gives slope of the volume trend
    vol_slope = (   
        df.groupby("ticker")["volume"]
        .transform(lambda x: x.rolling(21).apply(
            lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=False))
    )

    df["volume_trend"] = vol_slope

    df["mom_x_vol_5"] = df["return_5d"] * df["vol_avg_5"]
    df["mom_x_vol_21"] = df["return_21d"] * df["vol_avg_21"]
    df["mom_x_vol_63"] = df["return_63d"] * df["vol_avg_63"]
    return df

# CROSS-SECTIONAL NORMALIZATION

def cross_sectional_rank(df, feature_cols):
    
    df = df.copy()

    for col in feature_cols:
        df[f"{col}_rank"] = (
            df.groupby("date")[col]
            .rank(pct=True)
        )

    return df


def get_base_features(df):
    return [
        col for col in df.columns
        if col.startswith((
            "ret_", "price_ma", "pos_", "z_", "rev_",
            "vol_", "hl_", "downside", "vol_ratio",
            "dollar", "volume_", "mom_x_vol"
        ))
        and not col.endswith("_rank")
    ]

def select_features(df):
    
    base_features = get_base_features(df)
    rank_features = [f"{c}_rank" for c in base_features]

    final_features = base_features + rank_features

    return df[["date", "ticker", "fwd_return_5d"] + final_features]


def save_features(df):
    
    path = Path(PROCESSED_DATA_PATH) / "features_df.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved features to {path}")

def run_pipeline(df):
    
    df = load_clean_data()

    df = momentum(df)
    df = mean_reversion(df)
    df = volatility(df)
    df = volume_features(df)
    
    base_features = get_base_features(df)

    df = cross_sectional_rank(df, base_features)
    
    df = select_features(df)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    save_features(df)
    
run_pipeline(df)


def load_clean_data():
    
    path = Path(PROCESSED_DATA_PATH) / "features_df.parquet"
    df = pd.read_parquet(path)
    return df

df = load_clean_data()
