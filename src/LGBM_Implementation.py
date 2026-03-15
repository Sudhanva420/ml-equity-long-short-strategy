#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

from datetime import datetime


from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import portfolio_and_backtest
import importlib

importlib.reload(portfolio_and_backtest)

from portfolio_and_backtest import *


def load_data():
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

train_df, val_df, test_df = load_data()

def create_datasets(train_df, val_df, test_df):
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


    val = val_df.copy()

    train = train_df.copy()

    test = test_df.copy()
    
    return x_train, y_train, x_val, y_val, x_test, y_test, train, val, test, feature_cols

x_train, y_train, x_val, y_val, x_test, y_test, train_df, val_df, test_df, feature_cols = create_datasets(train_df, val_df, test_df)

def train_model(x_train, y_train, x_val, y_val):
    # LightGBM parameters
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    }

    train_data = lgb.Dataset(x_train, label=y_train)
    val_data   = lgb.Dataset(x_val, label=y_val, reference=train_data)

    # Train
    lgbm_model1 = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        #early_stopping_rounds=100,
        #verbose_eval=100
    )
    return lgbm_model1

def predict_and_evaluate(lgbm_model1, x_train, x_val, x_test, train_df, val_df, test_df):
    
    train_df['pred'] = lgbm_model1.predict(x_train)
    val_df['pred'] = lgbm_model1.predict(x_val)
    test_df['pred'] = lgbm_model1.predict(x_test)

    train_df1 = train_df.copy()
    val_df1 = val_df.copy()
    test_df1 = test_df.copy()

    return train_df1, val_df1, test_df1

lgbm_model1 = train_model(x_train, y_train, x_val, y_val)
train_df1, val_df1, test_df1 = predict_and_evaluate(lgbm_model1, x_train, x_val, x_test, train_df, val_df, test_df)

def daily_ic(df):
    
    ic_vals = []

    for d, grp in df.groupby("date"):
        
        if grp["pred"].std() == 0:
            continue
            
        ic = spearmanr(grp["pred"], grp["fwd_return_5d"])[0]
        ic_vals.append(ic)

    return np.array(ic_vals)


train_ic = daily_ic(train_df1)

# Summary
print("Training IC mean:", train_ic.mean(), "IC IR:", train_ic.mean()/train_ic.std())


val_ic = daily_ic(val_df1)
test_ic = daily_ic(test_df1)

print("Validation IC mean:", val_ic.mean(), "IC IR:", val_ic.mean()/val_ic.std())
print("Test IC mean:", test_ic.mean(), "IC IR:", test_ic.mean()/test_ic.std())


# Peformed much weaker than RF. Probably because the model is highly complex for a simple dataset like the current one
def feat_imp(model, feature_cols):
# Feature importance
    importance = pd.Series(lgbm_model1.feature_importance(importance_type='gain'), index=feature_cols)
    importance.sort_values(ascending=False, inplace=True)

    plt.figure(figsize=(12,6))
    importance.head(20).plot(kind='bar')
    plt.title("Top 20 Feature Importances (LightGBM)")
    plt.show()


# Trying a simpler LightGBM model:


lgb_params_simple = {
    "objective": "regression",
    "metric": "rmse",          
    "learning_rate": 0.05,     
    "num_leaves": 15,          
    "max_depth": 4,           
    "min_data_in_leaf": 50,    
    "feature_fraction": 0.7,  
    "bagging_fraction": 0.7,  
    "bagging_freq": 1,         
    "lambda_l1": 0.1,        
    "lambda_l2": 0.1,         
    "verbosity": -1,
    "seed": 42
}



train_data = lgb.Dataset(x_train, label=y_train)
val_data   = lgb.Dataset(x_val, label=y_val, reference=train_data)

# Train
lgbm_model2 = lgb.train(
    lgb_params_simple,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    #early_stopping_rounds=100,
    #verbose_eval=100
)

train_df['pred'] = lgbm_model2.predict(x_train)
val_df['pred'] = lgbm_model2.predict(x_val)
test_df['pred'] = lgbm_model2.predict(x_test)



train_df2 = train_df.copy()
val_df2 = val_df.copy()
test_df2 = test_df.copy()


train_ic = daily_ic(train_df2)

print("Training IC mean:", train_ic.mean(), "IC IR:", train_ic.mean()/train_ic.std())

val_ic = daily_ic(val_df2)
test_ic = daily_ic(test_df2)

print("Validation IC mean:", val_ic.mean(), "IC IR:", val_ic.mean()/val_ic.std())
print("Test IC mean:", test_ic.mean(), "IC IR:", test_ic.mean()/test_ic.std())


def perf_pipeline(val_df):
    
    val_df = construct_portfolio(val_df, pred_col='pred', top_pct=0.2, bottom_pct=0.2)

    portfolio_returns = stock_pnl_and_returns(val_df)

    portfolio_returns = turnover(val_df, portfolio_returns)

    portfolio_returns = apply_transaction_costs(portfolio_returns)

    metrics = performance_metrics(portfolio_returns, return_col='net_return', freq=52)
    
    print(metrics)
    
metrics = perf_pipeline(val_df2)



print(metrics)


# Hyperparameter Tuning for better performance


def calculate_ic(predictions, df_subset):

    temp_df = df_subset.copy()
    temp_df['prediction'] = predictions
    
    ic_by_date = temp_df.groupby('date').apply(
        lambda x: spearmanr(x['prediction'], x['fwd_return_5d'])[0]
        if len(x) >= 10 and x['prediction'].std() > 0
        else np.nan
    )
    
    return ic_by_date.mean()


# Coarse Grid Search for tuning

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63],
    'max_depth': [3, 5, 7],
    'min_child_samples': [20, 50, 100],
    'reg_alpha': [0.0, 0.1],
    'reg_lambda': [0.0, 0.1]
}

results = []

for params in tqdm(ParameterGrid(param_grid), desc="Tuning"):
    
    lgbm_model3 = LGBMRegressor(
        **params,
        n_estimators=100,  
        random_state=42,
        verbose=-1,
        force_col_wise=True  
    )
    

    lgbm_model3.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # Evaluating on validation
    predictions = lgbm_model3.predict(x_val)
    val_ic = calculate_ic(predictions, val_df)

    result = params.copy()
    result['val_ic'] = val_ic
    result['n_estimators_used'] = lgbm_model3.best_iteration_ if hasattr(lgbm_model3, 'best_iteration_') else 100
    
    results.append(result)


results_df = pd.DataFrame(results)

results_df = results_df.sort_values('val_ic', ascending=False)

print("TOP 10 PARAMETER COMBINATIONS")

print(results_df.head(10).to_string(index=False))



# Best parameters

best_params = results_df.iloc[0].to_dict()
best_ic = best_params.pop('val_ic')
best_n_estimators = int(best_params.pop('n_estimators_used'))

print("BEST PARAMETERS (Phase 1)")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"  n_estimators: {best_n_estimators}")
print(f"\nValidation IC: {best_ic:.4f}")




# Plotting IC by parameter
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, param in enumerate(param_grid.keys()):
    ax = axes[idx]
    
    # Group by the parameter
    param_impact = results_df.groupby(param)['val_ic'].agg(['mean', 'std', 'count'])

    ax.errorbar(
        param_impact.index,
        param_impact['mean'],
        yerr=param_impact['std'],
        marker='o',
        capsize=5,
        capthick=2
    )
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(param)
    ax.set_ylabel('Mean IC')
    ax.set_title(f'Impact of {param}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hyperparameter_tuning_phase1.png', dpi=300, bbox_inches='tight')
plt.show()



for k in ['num_leaves', 'max_depth', 'min_child_samples']:
    if k in best_params:
        best_params[k] = int(best_params[k])

final_model = LGBMRegressor(
    **best_params,
    n_estimators=best_n_estimators,
    random_state=42,
    verbose=-1,
    force_col_wise=True
)

# Train on full training set
final_model.fit(x_train, y_train)


train_df['pred'] = final_model.predict(x_train)
val_df['pred'] = final_model.predict(x_val)
test_df['pred'] = final_model.predict(x_test)

train_ic = daily_ic(train_df)

print("Training IC mean:", train_ic.mean(), "IC IR:", train_ic.mean()/train_ic.std())

val_ic = daily_ic(val_df)
test_ic = daily_ic(test_df)

print("Validation IC mean:", val_ic.mean(), "IC IR:", val_ic.mean()/val_ic.std())
print("Test IC mean:", test_ic.mean(), "IC IR:", test_ic.mean()/test_ic.std())


train_df4 = train_df.copy()
val_df4 = val_df.copy()
test_df4 = test_df.copy()


# So hyperparameter tuning overfit the validation set noise


def analyze_feature_ic_by_split(train, val, test, features):
    
    results = []
    
    for feature in features:
        
        # Train IC
        train_ic = train.groupby('date').apply(
            lambda x: spearmanr(x[feature], x['fwd_return_5d'])[0]
            if len(x) >= 10 else np.nan
        ).mean()
        
        # Val IC
        val_ic = val.groupby('date').apply(
            lambda x: spearmanr(x[feature], x['fwd_return_5d'])[0]
            if len(x) >= 10 else np.nan
        ).mean()
        
        # Test IC
        test_ic = test.groupby('date').apply(
            lambda x: spearmanr(x[feature], x['fwd_return_5d'])[0]
            if len(x) >= 10 else np.nan
        ).mean()
        
        results.append({
            'feature': feature,
            'train_ic': train_ic,
            'val_ic': val_ic,
            'test_ic': test_ic,
            'val_degradation': val_ic - train_ic,
            'test_degradation': test_ic - val_ic
        })
    
    results_df = pd.DataFrame(results)
    return results_df


    print("="*60)
    print("FEATURE IC ANALYSIS BY SPLIT")
    print("="*60)

    feature_analysis = analyze_feature_ic_by_split(train_df, val_df, test_df, feature_cols)

# Sort by test IC
    feature_analysis_sorted = feature_analysis.sort_values('test_ic', key=abs, ascending=False)

    print(feature_analysis_sorted.to_string(index=False))


    features_with_test_signal = feature_analysis[feature_analysis['test_ic'].abs() > 0.01]

    print(f"\nFeatures with |test_ic| > 0.01: {len(features_with_test_signal)}")
    print(features_with_test_signal[['feature', 'test_ic']])


    feature_analysis[(feature_analysis['test_ic'].abs() > 0.01) & (feature_analysis['val_ic'].abs() < 0.01)]



    # Check for complete degradation
    features_dead_on_test = feature_analysis[feature_analysis['test_ic'].abs() < 0.005]
    print(f"\nFeatures dead on test (|IC| < 0.005): {len(features_dead_on_test)}")
    print(features_dead_on_test[['feature', 'test_ic']])


# Feature Importance for final_model



importance = final_model.feature_importances_
feature_names = feature_cols

# Create dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance (Gain)')
plt.title('LightGBM Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print(importance_df.to_string(index=False))

# Top 5 features
top_5 = importance_df.head(5)['feature'].tolist()
print(f"\nTop 5 features: {top_5}")



def perf_pipeline(test_df):
    
    test_df = construct_portfolio(test_df, pred_col='pred', top_pct=0.2, bottom_pct=0.2)

    portfolio_returns = stock_pnl_and_returns(test_df)

    portfolio_returns = turnover(test_df, portfolio_returns)

    portfolio_returns = apply_transaction_costs(portfolio_returns)

    metrics = performance_metrics(portfolio_returns, return_col='net_return', freq=52)
    
    plots(portfolio_returns)
    
    print(metrics)
    
perf_pipeline(test_df4)


def perf_pipeline(test_df):
    
    test_df = construct_portfolio(test_df, pred_col='pred', top_pct=0.2, bottom_pct=0.2)

    portfolio_returns = stock_pnl_and_returns(test_df)

    portfolio_returns = turnover(test_df, portfolio_returns)

    portfolio_returns = apply_transaction_costs(portfolio_returns)

    metrics = performance_metrics(portfolio_returns, return_col='net_return', freq=52)
    
    plots(portfolio_returns)
    
    print(metrics)
    
perf_pipeline(test_df2)


# Second model despite having better test IC, has much higher turnover. That might be the cause of the lower performance

