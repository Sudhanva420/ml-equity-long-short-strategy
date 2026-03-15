#!/usr/bin/env python
# coding: utf-8

# 1. Constructing Long / Short Portfolio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Determining long and short positions based on ranking of predictions by the model

def construct_portfolio(df, pred_col, top_pct = 0.2, bottom_pct = 0.2):
    
    df = df.copy()
    
    df['pred_rank'] = df.groupby('date')[pred_col].rank(pct = True, method = 'first')
    
    df['position'] = 0
    
    df.loc[df['pred_rank'] >= (1 - top_pct), 'position'] = 1
    df.loc[df['pred_rank'] <= bottom_pct, 'position'] = -1
    
    n_long = df.groupby('date')['position'].transform(lambda x: (x == 1).sum())
    n_short = df.groupby('date')['position'].transform(lambda x: (x == -1).sum())
    
    df.loc[df['position'] == 1, 'position'] = 1.0 / n_long[df['position'] == 1]
    df.loc[df['position'] == -1, 'position'] = -1.0 / n_short[df['position'] == -1]
    
    return df


# Per stock-pnl and aggreagting per date to get portfolio returns

def stock_pnl_and_returns(df):
    
    df = df.copy()
    
    df['stock_pnl'] = df['position'] * df['fwd_return_5d']
    
    position_check = df.groupby('date')['position'].sum()
    print(f"Position sum check (should be ~0): {position_check.abs().max():.6f}")
    
    portfolio_returns = df.groupby('date').agg(
        {
            'stock_pnl': 'sum',
             'position': lambda x: (x != 0).sum()
        }
    ).reset_index()
    
    portfolio_returns.columns = ['date', 'gross_return', 'n_positions']

# Sort by date
    portfolio_returns = portfolio_returns.sort_values('date')

    return portfolio_returns


# Calculate turnover and merge with portfolio returns

def turnover(df, portfolio_returns):
    
    df = df.copy()
    
    df = df.sort_values(['ticker', 'date'])
    
    df['position_lag'] = df.groupby('ticker')['position'].shift(1).fillna(0)
    df['position_change'] = np.abs(df['position'] - df['position_lag'])
    
    turnover = df.groupby('date')['position_change'].sum().reset_index()
    turnover.columns = ['date', 'turnover']

    portfolio_returns = portfolio_returns.merge(turnover, on='date', how='left')
    
    return portfolio_returns

# Adding transaction costs

def apply_transaction_costs(df):
    
    df = df.copy()
    
    COST_BPS = 10
    cost_per_trade = COST_BPS / 10000
    
    df['transaction_cost'] = df['turnover'] * cost_per_trade
    df['net_return'] = df['gross_return'] - df['transaction_cost']
    
    df['cum_gross'] = (1 + df['gross_return']).cumprod()
    df['cum_net'] = (1 + df['net_return']).cumprod()
    
    return df



# Getting all the performance metrics

def performance_metrics(df, return_col='net_return', freq=52):
    
    df = df.copy()
    
    ret = df[return_col]
    cum_ret = (1 + ret).cumprod()
    
    mean_ret = ret.mean() * freq
    std_ret = ret.std() * np.sqrt(freq)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    
    total_return = cum_ret.iloc[-1] - 1
    
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()
    
    # Win rate
    win_rate = (ret > 0).mean()
    
    avg_turnover = df['turnover'].mean()
    avg_cost = df['transaction_cost'].mean()
    
    metrics = {
        'total_return': total_return,
        'ann_return': mean_ret,
        'ann_vol': std_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'avg_turnover': avg_turnover,
        'avg_cost': avg_cost,
        'n_periods': len(ret)
    }
    
    return metrics



def plots(portfolio_returns):
    
        # Equity curve
    plt.figure(figsize=(14, 6))
    plt.plot(portfolio_returns['date'], portfolio_returns['cum_gross'], 
            label='Gross Returns', linewidth=2, alpha=0.7)
    plt.plot(portfolio_returns['date'], portfolio_returns['cum_net'], 
            label='Net Returns (after costs)', linewidth=2)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Drawdown
    running_max = portfolio_returns['cum_net'].cummax()
    drawdown = (portfolio_returns['cum_net'] - running_max) / running_max

    plt.figure(figsize=(14, 6))
    plt.fill_between(portfolio_returns['date'], drawdown, 0, 
                    color='red', alpha=0.3, label='Drawdown')
    plt.plot(portfolio_returns['date'], drawdown, color='red', linewidth=1)
    plt.title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    plt.ylabel('Drawdown')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Rolling Sharpe (50-period)
    rolling_sharpe = (
        portfolio_returns['net_return'].rolling(50).mean() / 
        portfolio_returns['net_return'].rolling(50).std() * 
        np.sqrt(52)
    )

    plt.figure(figsize=(14, 6))
    plt.plot(portfolio_returns['date'], rolling_sharpe, linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', label='Sharpe = 1.0')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.title('Rolling 50-Period Sharpe Ratio', fontsize=14, fontweight='bold')
    plt.ylabel('Sharpe Ratio')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

