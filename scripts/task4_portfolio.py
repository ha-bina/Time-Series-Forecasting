import numpy as np
import pandas as pd
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'processed' / 'cleaned_adj_close.csv'
FORECAST_PATH = ROOT / 'data' / 'processed' / 'tsla_arima_forecast_6m.csv'
OUT_DIR = ROOT / 'data' / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from scripts.utils import download_data, clean_data

# Monte Carlo util copied/minimized from previous script
def monte_carlo_portfolios(expected_returns_annual, cov_ann, n_portfolios=20000, risk_free=0.0):
    n = len(expected_returns_annual)
    results = np.zeros((n_portfolios, 3 + n))
    for i in range(n_portfolios):
        w = np.random.random(n)
        w /= np.sum(w)
        port_ret = np.dot(w, expected_returns_annual)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_ann, w)))
        sharpe = (port_ret - risk_free) / port_vol if port_vol != 0 else 0.0
        results[i,0] = port_ret
        results[i,1] = port_vol
        results[i,2] = sharpe
        results[i,3:] = w
    cols = ['ret','vol','sharpe'] + [f'w{i}' for i in range(n)]
    return pd.DataFrame(results, columns=cols)

# Load data
if not DATA_PATH.exists():
    df = download_data(['TSLA','SPY','BND'], start='2015-01-01')
    df = clean_data(df)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

# Load forecast and compute expected TSLA annual return
if not FORECAST_PATH.exists():
    raise FileNotFoundError(f'Forecast file not found: {FORECAST_PATH}')
fc = pd.read_csv(FORECAST_PATH, index_col=0, parse_dates=True)
fc_vals = fc['forecast']
fc_ret = fc_vals.pct_change().dropna()
exp_daily_tsla = fc_ret.mean()
exp_ann_tsla = (1 + exp_daily_tsla) ** 252 - 1

# Historical expected returns for SPY, BND
returns_hist = df.pct_change().dropna()
exp_ann_spy = (1 + returns_hist['SPY'].mean()) ** 252 - 1
exp_ann_bnd = (1 + returns_hist['BND'].mean()) ** 252 - 1

expected_returns = np.array([exp_ann_tsla, exp_ann_spy, exp_ann_bnd])

# Covariance annualized
cov_ann = returns_hist[['TSLA','SPY','BND']].cov() * 252

# Monte Carlo
mc = monte_carlo_portfolios(expected_returns, cov_ann.values, n_portfolios=20000)
max_sharpe = mc.loc[mc['sharpe'].idxmax()]
min_vol = mc.loc[mc['vol'].idxmin()]

# Save results
res = {
    'expected_returns': {
        'TSLA': float(expected_returns[0]),
        'SPY': float(expected_returns[1]),
        'BND': float(expected_returns[2])
    },
    'max_sharpe': max_sharpe.to_dict(),
    'min_vol': min_vol.to_dict()
}
with open(OUT_DIR / 'portfolio_optimization_results.json','w') as f:
    json.dump(res, f, indent=2)

# Also save MC dataframe (sample) and the full frontier CSV
mc.sample(1000).to_csv(OUT_DIR / 'mc_portfolios_sample.csv', index=False)
mc.to_csv(OUT_DIR / 'mc_portfolios_all.csv', index=False)
print('Optimization complete. Results saved to', OUT_DIR)