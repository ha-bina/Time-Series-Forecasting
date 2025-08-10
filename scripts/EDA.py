# =========================
# Task 1: Preprocess & Explore Financial Data
# =========================

import logging
from pathlib import Path
import pandas as pd

from config import TICKERS, START_DATE, END_DATE, PROCESSED_DIR, FIGURES_DIR, ROLLING_WINDOW
from scripts.utils import (
    download_data,
    clean_data,
    compute_returns,
    rolling_stats,
    detect_outliers,
    adf_test,
    value_at_risk,
    sharpe_ratio,
    plot_series,
    save_dataframe,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info('Starting Task 1: Preprocess & EDA')

    # 1) Download
    raw = download_data(TICKERS, START_DATE, END_DATE)
    raw_path = PROCESSED_DIR / 'raw_adj_close.csv'
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path)
    logger.info(f'Raw data saved to {raw_path}')

    # 2) Clean
    df = clean_data(raw)
    cleaned_path = PROCESSED_DIR / 'cleaned_adj_close.csv'
    save_dataframe(df, cleaned_path)
    logger.info(f'Cleaned data saved to {cleaned_path}')

    # 3) Basic stats
    stats = df.describe()
    stats_path = PROCESSED_DIR / 'summary_stats.csv'
    save_dataframe(stats, stats_path)
    logger.info('Summary statistics saved')

    # 4) Returns
    returns = compute_returns(df)
    returns_path = PROCESSED_DIR / 'daily_returns.csv'
    save_dataframe(returns, returns_path)

    # 5) Plots: prices and returns
    plot_series(df, 'Adjusted Closing Prices', 'Price (USD)', out_path=FIGURES_DIR / 'prices.png')
    plot_series(returns, 'Daily Returns', 'Daily Return', out_path=FIGURES_DIR / 'daily_returns.png')
    logger.info('Saved price and returns plots')

    # 6) Rolling stats
    rm, rs = rolling_stats(df, window=ROLLING_WINDOW)
    save_dataframe(rm, PROCESSED_DIR / 'rolling_mean.csv')
    save_dataframe(rs, PROCESSED_DIR / 'rolling_std.csv')

    # Optional: plot TSLA rolling std
    tsla_std = rs[['TSLA']]
    plot_series(tsla_std, f'TSLA {ROLLING_WINDOW}-day Rolling Std Dev', 'Std Dev', out_path=FIGURES_DIR / 'tsla_rolling_std.png')

    # 7) Outlier detection
    outliers = detect_outliers(returns)
    save_dataframe(outliers, PROCESSED_DIR / 'outlier_days.csv')
    logger.info(f'Found {len(outliers)} outlier days (abs(z) > 3)')

    # 8) Stationarity tests
    adf_close = adf_test(df['TSLA'])
    adf_returns = adf_test(returns['TSLA'])
    adf_df = pd.DataFrame([adf_close, adf_returns], index=['TSLA_Close', 'TSLA_Returns'])
    save_dataframe(adf_df, PROCESSED_DIR / 'adf_tests.csv')

    # 9) Risk metrics
    metrics = {}
    for col in returns.columns:
        r = returns[col]
        metrics[col] = {
            'sharpe': sharpe_ratio(r),
            'var_95': value_at_risk(r, alpha=0.05),
            'mean_daily_return': r.mean(),
            'std_daily_return': r.std()
        }
    metrics_df = pd.DataFrame(metrics).T
    save_dataframe(metrics_df, PROCESSED_DIR / 'risk_metrics.csv')

    logger.info('Risk metrics saved')

    # 10) Save a short text summary
    summary_lines = [
        f'TSLA direction (last 30 days): {"up" if df["TSLA"].iloc[-1] > df["TSLA"].iloc[-30] else "down"}',
        f'TSLA daily return std (annualized approx): {returns["TSLA"].std() * (252**0.5):.4f}',
        f'TSLA Sharpe: {metrics_df.loc["TSLA","sharpe"]:.4f}',
        f'TSLA VaR(95%): {metrics_df.loc["TSLA","var_95"]:.4f}',
    ]
    (FIGURES_DIR.parent / 'summary' ).mkdir(parents=True, exist_ok=True)
    summary_path = FIGURES_DIR.parent / 'summary' / 'short_insights.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    logger.info(f'Short summary saved to {summary_path}')
    logger.info('Task 1 complete')


if __name__ == '__main__':
    main()