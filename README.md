# ML-Based Long-Short Equity Strategy

End-to-end quantitative research pipeline for cross-sectional equity alpha generation using machine learning.

**Project Focus:** Building a complete, rigorous quantitative research workflow with proper temporal validation and statistical testing

---

## Overview

This project implements the full quantitative research lifecycle from data collection through portfolio backtesting. The emphasis is on **methodology and pipeline design** rather than optimizing for maximum performance.

**Universe:** 40 large-cap US equities  
**Period:** 2010-2024  
**Strategy:** Weekly rebalanced long-short (top/bottom 20%)  
**Horizon:** 5-day forward returns

---

## Pipeline Architecture

### 1. Data Collection & Cleaning

- OHLCV data from yfinance
- Data quality filters (minimum price, volume)
- Proper date handling for time-series analysis

### 2. Feature Engineering

- 50+ features across momentum, value, quality, and technical indicators
- Cross-sectional normalization (winsorization + z-scoring)
- Temporal alignment to prevent lookahead bias
- All features properly shifted to use only historical data

### 3. Feature Analysis & Selection

- Information Coefficient (IC) analysis
- Statistical significance testing (t-statistics, bootstrap validation)
- Stationarity testing (ADF, KPSS)
- Redundancy removal via correlation analysis
- Feature selection based on train data only

### 4. Temporal Validation Framework

- **Train:** 2010-2021 (feature selection and model training)
- **Validation:** 2022 - mid 2023 (hyperparameter tuning)
- **Test:** mid 2023-2024 (final evaluation)
- Strict separation to prevent information leakage

### 5. Model Training

Multiple models implemented and compared:

- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Random Forest (ensemble)
- LightGBM (gradient boosting)

Hyperparameter tuning performed on validation set using IC as the optimization metric.

### 6. Evaluation

- **Information Coefficient (IC):** Spearman correlation between predictions and realized returns
- Cross-sectional IC calculated per date, averaged over time
- Evaluated on train, validation, and test sets independently

### 7. Portfolio Backtesting

- Long-short portfolio construction from model predictions
- Equal-weight within long and short baskets
- Transaction cost modeling (10 bps per trade)
- Performance metrics: Sharpe ratio, drawdown, turnover, win rate

## Key Learnings

This project focused on building a realistic research workflow for a trading strategy with proper statistical testing

Observations:

- Train vs validation IC divergence due to noisy financial data
- Regularization effects in Ridge regression
- Differences between model IC performance and portfolio backtest results
- Impact of turnover and trading costs

## Future Improvements

This project establishes the pipeline. Extensions include:

- Expanding universe size for stronger cross-sectional signals
- Testing different return horizons (21-day, monthly)
- Implementing regime-conditional models
- Adding factor neutralization (sector, market)
- Exploring alternative ML architectures
- Developing adaptive rebalancing strategies
