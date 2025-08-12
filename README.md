# Time series Forecasting

This project implements a multi-step pipeline for financial data analysis, forecasting, portfolio optimization, and backtesting using Tesla (TSLA) stock data and related assets.

---

## Project Tasks Overview

### **Task 1: Data Collection & Preprocessing**
- **Objective:** Collect Tesla stock price data and prepare it for analysis.
- **Steps:**
  - Fetch historical TSLA price data (e.g., from Yahoo Finance).
  - Clean missing values, handle outliers, and standardize column names.
  - Save the cleaned dataset to `data/processed/`.
- **Output:** `data/processed/tesla_clean.csv`

---

### **Task 2: Time Series Forecasting**
- **Objective:** Forecast Tesla’s future stock prices using two different models.
- **Models Implemented:**
  - **ARIMA/SARIMA** (statistical model)
  - **LSTM** (deep learning model)
- **Steps:**
  - Chronologically split the data into training (2015–2023) and testing (2024–2025).
  - Train both models, tune parameters, and compare performance using MAE, RMSE, and MAPE.
  - Forecast future prices and plot results.
- **Output:** `data/forecasts/tsla_forecast_arima.csv`, `data/forecasts/tsla_forecast_lstm.csv`

---

### **Task 3: Forecast-based Portfolio Construction**
- **Objective:** Use TSLA forecasts to design an optimal investment portfolio.
- **Steps:**
  - Combine TSLA forecasts with other assets.
  - Estimate expected returns and covariance matrix.
  - Use Monte Carlo simulation to identify portfolios with the best Sharpe Ratio.
- **Output:** `data/portfolio/recommended_portfolio.csv`

---

### **Task 4: Portfolio Optimization**
- **Objective:** Optimize the asset allocation to maximize return for a given risk level.
- **Steps:**
  - Run portfolio optimization algorithms (mean-variance optimization, efficient frontier).
  - Plot the efficient frontier curve.
- **Output:** `plots/efficient_frontier.png`, updated recommended portfolio CSV.

---

### **Task 5: Backtesting**
- **Objective:** Evaluate portfolio performance against a benchmark.
- **Steps:**
  - Backtest the recommended portfolio against a 60/40 SPY/BND benchmark.
  - Calculate metrics like CAGR, Sharpe Ratio, and Max Drawdown.
  - Plot performance comparison over time.
- **Output:** `plots/backtest_performance.png`, `data/backtest_results.csv`

---

## **Installation**
```bash
git clone <https://github.com/ha-bina/time-series-forecasting.git>
cd time-series-analysis
pip install -r requirements.txt
stock-analysis-project/
│
├── data/
│   ├── raw/                  # Raw data
│   ├── processed/            # Cleaned datasets
│   ├── forecasts/            # Forecast outputs
│   ├── portfolio/            # Portfolio recommendations
│   └── backtest_results.csv  # Backtest metrics
│
├── scripts/
│   ├── task1_preprocess.py
│   ├── task2_forecast.py
│   ├── task3_forecast.py
│   ├── task4_portfolio.py
│   └── task5_backtesting.py
│
├── plots/                    # Generated plots
├── requirements.txt
├── README.md
└── main_notebook.ipynb       # Runs tasks 
##  **usage**
# each task have a separete script
python scripts/task1_preprocess.py
python scripts/task2_forecast.py
python scripts/task3_forecast.py
python scripts/task4_portfolio_opt.py
python scripts/task5_backtest.py

