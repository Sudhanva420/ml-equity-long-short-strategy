#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pathlib import Path
import ast
from pathlib import Path
from config import PROCESSED_DATA_PATH

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MIN_PRICE,
    MIN_AVG_DOLLAR_VOLUME,
)


def load_raw_data():
    path = Path(RAW_DATA_PATH) / "us_equities_ohlcv.parquet"
    df = pd.read_parquet(path)
    return df

df_us = load_raw_data()

df_us.head()


df_us.columns



def restore_multicolumns(df):
    new_cols = []

    for col in df.columns:
        if col.startswith("(") and col.endswith(")"):
            new_cols.append(ast.literal_eval(col))
        else:
            # Handles columns like ('date','') or malformed leftovers
            new_cols.append((col, ""))

    df.columns = pd.MultiIndex.from_tuples(new_cols)
    return df

def normalize_price_fields(df):
    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples([
        (field.lower().replace(" ", "_"), ticker)
        for field, ticker in df.columns
    ])
    return df

def wide_to_long_panel(df):
    # Move date to index if needed
    if ("date", "") in df.columns:
        df = df.set_index(("date", ""))

    df.index.name = "date"

    # Stack ticker dimension
    df = df.stack(level=1).reset_index()

    # Rename columns
    df = df.rename(columns={"level_1": "ticker"})

    return df

df = df_us.copy()

df = restore_multicolumns(df)
df = normalize_price_fields(df)
df = wide_to_long_panel(df)

if list(df.columns).count("ticker") > 1:
    df = df.loc[:, ~df.columns.duplicated()]



def basic_cleaning(df):
    df = df.copy()

    df = df[df["close"] > 0]
    df = df[df["volume"] > 0]

    df = df.sort_values(["ticker", "date"])
    df.reset_index(drop=True, inplace=True)

    return df

df_us2 = basic_cleaning(df)

#Now checking if the stocks are liquid enough to be considered

def liquidity_filter(df):
    
    df = df.copy()
    
    df["dollar_volume"] = df["close"]*df["volume"]
    
    liquidity = (df.groupby("ticker")['dollar_volume'].mean().rename("avg_dollar_volume)"))
    
    liquid_tickers = liquidity[liquidity>= MIN_AVG_DOLLAR_VOLUME].index
    
    df[df['ticker'].isin(liquid_tickers)]
    
    return df

df = liquidity_filter(df_us2)

#Now the price filter 

def apply_price_filter(df):
    df = df.copy()

    avg_price = (
        df.groupby("ticker")["close"]
        .mean()
        .rename("avg_price")
    )

    valid_tickers = avg_price[avg_price >= MIN_PRICE].index
    df = df[df["ticker"].isin(valid_tickers)]

    return df

df = apply_price_filter(df)

#Now moving on to creating lables using forward returns

def compute_daily_returns(df):
    df = df.copy()

    df["return_1d"] = (
        df.groupby("ticker")["adj_close"]
        .pct_change()
    )

    return df

def compute_forward_returns(df, horizons=(5, 21)):
    df = df.copy()

    for h in horizons:
        df[f"fwd_return_{h}d"] = (
            df.groupby("ticker")["adj_close"]
            .shift(-h) / df["adj_close"] - 1
        )

    return df

df = compute_daily_returns(df)

df = compute_forward_returns(df)

def final_cleanup(df):
    
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df

df = final_cleanup(df)

def save_processed_data(df):
    Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)

    path = Path(PROCESSED_DATA_PATH) / "clean_panel.parquet"
    df.to_parquet(path, index=False)

    print(f"Saved processed data to {path}")

save_processed_data(df)

