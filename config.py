# config.py
from pathlib import Path

# Tickers and date range
TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"

# Data directories
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports" / "summary"

# Analysis params
ROLLING_WINDOW = 21  # trading days (~1 month)
ADF_AUTOLAG = 'AIC'

# Ensure directories exist at runtime
for p in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)