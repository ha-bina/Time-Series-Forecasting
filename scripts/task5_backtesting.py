import numpy as np
import pandas as pd
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'processed' / 'cleaned_adj_close.csv'
OPT_PATH = ROOT / 'data' / 'processed' / 'portfolio_optimization_results.json'
OUT_DIR = ROOT / 'reports' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load prices
prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)[['TSLA','SPY','BND']]
# Backtest window: last 252 trading days
bt = prices.dropna().tail(252)
start, end = bt.index[0], bt.index[-1]