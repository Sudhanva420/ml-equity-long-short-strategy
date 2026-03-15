#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

import portfolio_and_backtest
import importlib

importlib.reload(portfolio_and_backtest)
from portfolio_and_backtest import *



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

x_train, y_train, x_val, y_val, x_test, y_test, train_df, val_df, test_df, feature_cols = create_splits(train_df, val_df, test_df)
train_df.shape


# Important point before moving to the models:
# 
# Here we use the IC between the predicted value and the target variable as the metric to judge how good the model is. 
# 
# We do not use MSE, MSME, Accuracy, R^2

# Model 1: Linear Baseline

def lr(x_train, y_train, x_val, y_val):

    lr_model = LinearRegression()

    lr_model.fit(x_train, y_train)

    y_pred = lr_model.predict(x_val)

    return lr_model, y_pred

lr_model, y_pred = lr(x_train, y_train, x_val, y_val)
print("Validation R2:", r2_score(y_val, y_pred))

print(lr_model.coef_)    
print(lr_model.intercept_) 

# Evaluating Alpha with IC


def daily_ic(df):
    
    ic_vals = []

    for d, grp in df.groupby("date"):
        
        if grp["alpha_score"].std() == 0:
            continue
            
        ic = spearmanr(grp["alpha_score"], grp["fwd_return_5d"])[0]
        ic_vals.append(ic)

    return np.array(ic_vals)

def train_ic():
    predict_val = lr_model.predict(x_train)
    print("Validation R2:", r2_score(y_train, predict_val))

    train_df["alpha_score"] = predict_val

    ic_vals = daily_ic(train_df)
    
    return ic_vals

ic_vals = train_ic()
print("Mean IC:", np.mean(ic_vals))
print("IC IR:", np.mean(ic_vals) / np.std(ic_vals))

def plot_train_ic(ic_vals):
    plt.plot(pd.Series(ic_vals).rolling(63).mean())
    plt.title("63-Day Rolling IC")
    plt.show()

plot_train_ic(ic_vals)

def val_ic():
    val = val_df.copy()

    val["alpha_score"] = y_pred

    ic_vals = daily_ic(val)

    print("Mean IC:", np.mean(ic_vals))
    print("IC IR:", np.mean(ic_vals) / np.std(ic_vals))


    plt.plot(pd.Series(ic_vals).rolling(63).mean())
    plt.title("63-Day Rolling IC")
    plt.show()

val_ic()


# Model 2: Ridge


def ridge_reg(x_train, y_train, x_val, y_val):
    ridge = Ridge(alpha = 1.0)

    ridge.fit(x_train, y_train)


    predict_val = ridge.predict(x_val)
    
    return ridge, predict_val

ridge, predict_val = ridge_reg(x_train, y_train, x_val, y_val)
print("Validation R2:", r2_score(y_val, predict_val))

def ridge_ic():
    val = val_df.copy()

    val["alpha_score"] = predict_val

    ic_vals = daily_ic(val)

    print("Mean IC:", np.mean(ic_vals))
    print("IC IR:", np.mean(ic_vals) / np.std(ic_vals))


    plt.plot(pd.Series(ic_vals).rolling(63).mean())
    plt.title("63-Day Rolling IC")
    plt.show()

ridge_ic()

ridge.coef_


# IC is very unstable and value is not significant (good avg. value would be 0.3 to 0.5)

# This makes sense: Selected features were not optimal, model cannot learn non-linear features

# We can clearly see how all metrics collapse (model is underfitting)

# Experimenting with hyperparameters:

def ridge_reg2(x_train, y_train, x_val, y_val):
    ridge2 = Ridge(alpha = 0.01)

    ridge2.fit(x_train, y_train)

    predict_val = ridge2.predict(x_val)
    
    return ridge2, predict_val

ridge2, predict_val = ridge_reg2(x_train, y_train, x_val, y_val)

predict_val = ridge2.predict(x_val)
print("Validation R2:", r2_score(y_val, predict_val))

def ridge_ic2():
    
    val = val_df.copy()

    val["alpha_score"] = predict_val

    ic_vals = daily_ic(val)

    print("Mean IC:", np.mean(ic_vals))
    print("IC IR:", np.mean(ic_vals) / np.std(ic_vals))

    plt.plot(pd.Series(ic_vals).rolling(63).mean())
    plt.title("63-Day Rolling IC")
    plt.show()

ridge_ic2()

def ridge_reg3(x_train, y_train, x_val, y_val):

    ridge3 = Ridge(alpha = 1000)

    ridge3.fit(x_train, y_train)
    predict_val = ridge3.predict(x_val)
    print("Validation R2:", r2_score(y_val, predict_val))
    
    return ridge3, predict_val

ridge3, predict_val = ridge_reg3(x_train, y_train, x_val, y_val)

def ridge_ic3():
    val = val_df.copy()

    val["alpha_score"] = predict_val

    ic_vals = daily_ic(val)

    print("Mean IC:", np.mean(ic_vals))
    print("IC IR:", np.mean(ic_vals) / np.std(ic_vals))

    plt.plot(pd.Series(ic_vals).rolling(63).mean())
    plt.title("63-Day Rolling IC")
    plt.show()

ridge_ic3()

# Trying to find optimal alpha value

alphas = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
best_ic = -np.inf
best_alpha = None

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    ic = spearmanr(preds, y_val).correlation  # cross-sectional IC
    if ic > best_ic:
        best_ic = ic
        best_alpha = a


best_alpha


# Alpha is really big -> Very heavy regularization required. 
# This tracks since multicollieanrity is present, dataset is noisy


# Portfolio Construction, Backtesting

val1 = val_df.copy()

val1['pred'] = lr_model.predict(x_val)



# 1. Baseline LR


#First for simple baseline LR

def perf_pipeline(df):
    
    df = construct_portfolio(df, pred_col='pred', top_pct=0.2, bottom_pct=0.2)

    portfolio_returns = stock_pnl_and_returns(df)

    portfolio_returns = turnover(df, portfolio_returns)

    portfolio_returns = apply_transaction_costs(portfolio_returns)

    metrics = performance_metrics(portfolio_returns, return_col='net_return', freq=52)
    
    plots(portfolio_returns)
    
    return metrics
    
metrics = perf_pipeline(val1)

print(metrics)


def eval_lr_test():
    test1 = test_df.copy()

    test1['pred'] = lr_model.predict(x_test)


    metrics = perf_pipeline(test1)

    print(metrics)

eval_lr_test()

# 2. Ridge Models


def eval_ridge_test():
    val2 = val_df.copy()

    val2['pred'] = ridge.predict(x_val)

    test2 = test_df.copy()

    test2['pred'] = ridge.predict(x_test)


    metrics1 = perf_pipeline(val2)

    metrics2 = perf_pipeline(test2)

    return metrics1, metrics2

metrics1, metrics2 = eval_ridge_test()


# Ridge 2


def eval_ridge2_test():
    val3 = val_df.copy()

    val3['pred'] = ridge2.predict(x_val)

    test3 = test_df.copy()

    test3['pred'] = ridge2.predict(x_test)

    metrics1 = perf_pipeline(val3)

    metrics2 = perf_pipeline(test3)

    return metrics1, metrics2

metrics1, metrics2 = eval_ridge2_test()