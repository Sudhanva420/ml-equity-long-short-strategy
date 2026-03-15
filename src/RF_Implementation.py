#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import lightgbm as lgb
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

import portfolio_and_backtest
import importlib
importlib.reload(portfolio_and_backtest)
from portfolio_and_backtest import *
import joblib

def create_datasets(train_df, val_df, test_df):
    
    train_df = pd.read_parquet("/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/train_normalized_df.parquet")

    rank_cols = [col for col in train_df.columns if '_rank' in col]
    train_df = train_df.drop(columns=rank_cols)

    test_df = pd.read_parquet("/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/test_normalized_df.parquet")

    rank_cols = [col for col in test_df.columns if '_rank' in col]
    test_df = test_df.drop(columns=rank_cols)

    val_df = pd.read_parquet("/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed/val_normalized_df.parquet")

    rank_cols = [col for col in val_df.columns if '_rank' in col]
    val_df = val_df.drop(columns=rank_cols)

    return train_df, val_df, test_df

train_df, val_df, test_df = create_datasets(None, None, None)

def create_splits(train_df, val_df, test_df):
    
    feature_cols = [
        c for c in train_df.columns
        if c not in ["date","ticker", "fwd_return_5d"]
    ]
    
    x_train = train_df[feature_cols]
    y_train = train_df["fwd_return_5d"]

    x_val = val_df[feature_cols]
    y_val = val_df["fwd_return_5d"]

    x_test = test_df[feature_cols]
    y_test = test_df["fwd_return_5d"]
    
    return x_train, y_train, x_val, y_val, x_test, y_test, train_df, val_df, test_df, feature_cols

x_train, y_train, x_val, y_val, x_test, y_test, train_df, val_df, test_df, feature_cols = create_splits(train_df, val_df, test_df)
val1 = val_df.copy()

def train_rf(x_train, y_train):
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=100,
        n_jobs=-1,
        random_state=42
    )
    
    rf.fit(x_train, y_train)
    
    return rf

rf1 = train_rf(x_train, y_train)

def rf_predict(rf_model, x_val):
    
    rf_pred = rf_model.predict(x_val)
    
    return rf_pred

rf1_pred = rf_predict(rf1, x_val)

val1["rf_alpha"] = rf1_pred


def rf_ic_analysis_val(val_df):
    rf_ic = []

    for d, grp in val1.groupby("date"):
        
        if grp["rf_alpha"].std() == 0:
            continue
            
        ic = spearmanr(grp["rf_alpha"], grp["fwd_return_5d"])[0]
        rf_ic.append(ic)

    rf_ic = np.array(rf_ic)

    print("RF Mean IC:", rf_ic.mean())
    print("RF IR:", rf_ic.mean() / rf_ic.std())


    plt.plot(pd.Series(rf_ic).rolling(63).mean())
    plt.title("Random Forest 63-Day Rolling IC")
    plt.show()

rf_ic_analysis_val(val1)

def rf_analysis_train(rf_model, x_train, train_df):

    rf_pred = rf1.predict(x_train)

    train = train_df.copy()

    train["rf_alpha"] = rf_pred

    rf_ic = []

    for d, grp in train.groupby("date"):
        
        if grp["rf_alpha"].std() == 0:
            continue
            
        ic = spearmanr(grp["rf_alpha"], grp["fwd_return_5d"])[0]
        rf_ic.append(ic)

    rf_ic = np.array(rf_ic)

    print("RF Mean IC:", rf_ic.mean())
    print("RF IR:", rf_ic.mean() / rf_ic.std())


rf_analysis_train(rf1, x_train, train_df)


def rf_feature_importance(rf_model, feature_cols):
    imp = pd.Series(
        rf1.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print(imp.head(10))

rf_feature_importance(rf1, feature_cols)
# Unstable but values are significantly higher than LR, so clearly theres a non-linear relationship that RF is able to capture. Also avg. mean IC and low IR means signal exists but is noisy

# Gap b/w train and val still exists but smaller comparatively


# Portfolio Construction, Backtesting

val_df.columns


val1.columns


def perf_pipeline(val_df):
    
    val_df = construct_portfolio(val_df, pred_col='rf_alpha', top_pct=0.2, bottom_pct=0.2)

    portfolio_returns = stock_pnl_and_returns(val_df)

    portfolio_returns = turnover(val_df, portfolio_returns)

    portfolio_returns = apply_transaction_costs(portfolio_returns)

    metrics = performance_metrics(portfolio_returns, return_col='net_return', freq=52)
    
    plots(portfolio_returns)
    
    return metrics
    
metrics = perf_pipeline(val1)


print(metrics)

def test_performance_on_test(test_df):

    test1 = test_df.copy()

    rf1_pred = rf1.predict(x_test)

    test1["rf_alpha"] = rf1_pred
    
    return perf_pipeline(test1)

test_performance_on_test(test_df)

print(metrics)


# Finetuning the Random Forest model


def calculate_ic(predictions, df_subset):
    
    temp_df = df_subset.copy()
    temp_df['prediction'] = predictions
    
    ic_by_date = temp_df.groupby('date').apply(
        lambda x: spearmanr(x['prediction'], x['fwd_return_5d'])[0]
        if len(x) >= 10 and x['prediction'].std() > 0
        else np.nan
    )
    
    return ic_by_date.mean()

def fine_tune_rf(x_train, y_train, x_val, val_df):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True]
    }

    results = []

    for params in tqdm(ParameterGrid(param_grid), desc='Tuning'):
        
        rf_model = RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=-1
        )
        
        rf_model.fit(x_train, y_train)
        
        joblib.dump(rf_model, "rf_model.pkl")
        
        predictions = rf_model.predict(x_val)
        val_ic = calculate_ic(predictions, val_df)
        
        # Store results
        result = params.copy()
        result['val_ic'] = val_ic
        result['n_estimators_used'] = rf_model.best_iteration_ if hasattr(rf_model, 'best_iteration_') else 100
        
        results.append(result)
        
        return results

fine_tune_rf(x_train, y_train, x_val, val_df)


results = fine_tune_rf(x_train, y_train, x_val, val_df)

def best_parameters(results):
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values('val_ic', ascending=False)

    print("TOP 10 PARAMETER COMBINATIONS")

    print(results_df.head(10).to_string(index=False))

    best_params = results_df.iloc[0].to_dict()
    best_ic = best_params.pop('val_ic')


    best_params


    del best_params['n_estimators_used']
    best_params


    for k in ['max_depth']:
        if k in best_params:
            best_params[k] = int(best_params[k])
        
    print(best_params)

    print(best_ic)
    
    return best_params

best_params = best_parameters(results)

def train_final_rf(x_train, y_train, best_params):
    
    rf_final_model = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
        )

    # Train on full training set
    rf_final_model.fit(x_train, y_train)
    return rf_final_model

rf_final_model = train_final_rf(x_train, y_train, best_params)

def eval_rf_final_model(rf_final_model, x_val, val_df, x_test, test_df):

    test2 = test_df.copy()

    rf2_pred = rf_final_model.predict(x_test)

    test2["rf_alpha"] = rf2_pred

    metrics = perf_pipeline(test2)


    print(metrics)

eval_rf_final_model(rf_final_model, x_val, val_df, x_test, test_df)