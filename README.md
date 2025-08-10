# README.md — Task 1: Preprocess & Explore Financial Data

## Overview

This repository contains code and resources for **Task 1** of the financial analysis pipeline: *Preprocess and Explore the Data*. The task uses historical market data from Yahoo Finance for:

* **TSLA** — high returns with high volatility
* **BND** — stability and low risk
* **SPY** — diversified, moderate-risk market exposure

The workflow includes:

1. Data extraction using `yfinance`
2. Cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Volatility and outlier analysis
5. Stationarity testing
6. Risk metric computation (Sharpe Ratio, Value at Risk)

## File Structure

```
financial-analysis/
│
├── config.py                # Central configuration (tickers, date range, paths)
├── scripts/
│   ├── utils.py              # Helper functions for download, cleaning, plotting, stats
│   ├── task1_preprocess_eda.py  # Main script for Task 1
│
├── notebooks/
│   ├── Task1_Preprocess_EDA.ipynb  # Jupyter notebook version of Task 1
│
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Cleaned datasets, returns, metrics
│
├── reports/
│   ├── figures/              # Plots from EDA
│   ├── summary/              # Text summaries, metrics tables
│
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

*Required packages: `yfinance`, `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scipy`*

## Usage

**Run as Python script:**

```bash
python scripts/task1_preprocess_eda.py
```

**Run in Jupyter Notebook:**
Open `notebooks/Task1_Preprocess_EDA.ipynb` and run cells sequentially.

## Outputs

* **Cleaned Data:** CSV in `data/processed/`
* **Summary Statistics:** CSV in `data/processed/summary_stats.csv`
* **Plots:** PNG files in `reports/figures/`
* **Risk Metrics:** CSV in `data/processed/risk_metrics.csv`
* **Insights:** Short text summary in `reports/summary/short_insights.txt`

## Key Insights Generated

* Direction and trend of TSLA, SPY, BND over time
* Volatility analysis and high-return/high-risk days
* Stationarity results for prices vs. returns
* Value at Risk and Sharpe Ratios

## Next Steps

These outputs will feed into Task 2 for **modeling and forecasting**
