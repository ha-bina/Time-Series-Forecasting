import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore
from pathlib import Path


def download_data(tickers, start, end):
    import yfinance as yf
    df = yf.download(tickers, start=start, end=end)['Adj Close']
    # If single ticker, ensure DataFrame columns
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df


def clean_data(df):
    # Forward-fill then back-fill missing values
    df = df.ffill().bfill()
    # Force float dtype
    return df.astype(float)


def compute_returns(df):
    return df.pct_change().dropna()


def rolling_stats(df, window=21):
    return df.rolling(window).mean(), df.rolling(window).std()


def detect_outliers(returns_df, z_thresh=3.0):
    z_scores = returns_df.apply(zscore)
    mask = (np.abs(z_scores) > z_thresh).any(axis=1)
    return returns_df[mask]


def adf_test(series, autolag='AIC'):
    series = series.dropna()
    result = adfuller(series, autolag=autolag)
    return {
        'adf_stat': result[0],
        'pvalue': result[1],
        'usedlag': result[2],
        'nobs': result[3],
        'critical_values': result[4]
    }


def value_at_risk(returns, alpha=0.05):
    # Historical VaR (left tail)
    return np.percentile(returns.dropna(), 100 * alpha)


def sharpe_ratio(returns, periods_per_year=252):
    mu = returns.mean()
    sigma = returns.std()
    return (mu / sigma) * np.sqrt(periods_per_year)


def plot_series(df, title, ylabel, out_path: Path = None):
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150)
    plt.close()


def save_dataframe(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)