# scripts/task2_forecasting.py
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# modeling libraries
try:
    import pmdarima as pm
    HAVE_PMDARIMA = True
except Exception:
    HAVE_PMDARIMA = False

from statsmodels.tsa.arima.model import ARIMA

# Keras for LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAVE_TF = True
except Exception:
    HAVE_TF = False

from scripts.model_utils import train_test_split_time, evaluate_forecast, create_sequences
from scripts.utils import save_dataframe, plot_series

# CONFIG - adjust as needed
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cleaned_adj_close.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Model & training params
TRAIN_END = "2023-12-31"   # train up to this date (inclusive)
TEST_START = "2024-01-01"  # start of test set
LOOKBACK = 30              # days for LSTM lookback window
LSTM_EPOCHS = 50
LSTM_BATCH = 32

def load_tsla_series():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    # Ensure TSLA is present
    if "TSLA" not in df.columns:
        raise ValueError(f"TSLA not found in {DATA_PATH}. Columns: {df.columns.tolist()}")
    series = df["TSLA"].dropna()
    return series

# -------------------------
# ARIMA Model Functions
# -------------------------
def fit_arima_auto(train_series):
    """
    Try using pmdarima.auto_arima if available (faster).
    Fallback: simple grid search with small p,d,q ranges.
    """
    if HAVE_PMDARIMA:
        print("Using pmdarima.auto_arima to find best ARIMA params...")
        model = pm.auto_arima(train_series, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
        print("Best ARIMA order:", model.order)
        return model  # pmdarima model has .predict
    else:
        print("pmdarima not available — doing small grid search for ARIMA(p,d,q)...")
        best_aic = np.inf
        best_order = None
        best_model = None
        # small grid
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        m = ARIMA(train_series, order=(p,d,q)).fit()
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best_order = (p,d,q)
                            best_model = m
                    except Exception:
                        continue
        print("Selected ARIMA order:", best_order, "AIC:", best_aic)
        return best_model

def arima_forecast(model, train_index, steps):
    """
    Accepts either pmdarima model or statsmodels ARIMAResults
    Returns forecast array and index for forecast.
    """
    start_idx = train_index[-1]  # last timestamp in train
    if HAVE_PMDARIMA and isinstance(model, pm.arima.arima.ARIMA):
        # pmdarima returns n_periods forecast
        y_pred = model.predict(n_periods=steps)
    else:
        # statsmodels ARIMAResults
        y_pred = model.forecast(steps=steps)
    return y_pred

# -------------------------
# LSTM Model Functions
# -------------------------
def train_lstm(train_series, val_series=None, lookback=30, epochs=50, batch_size=32):
    if not HAVE_TF:
        raise ImportError("TensorFlow/Keras not available. Install tensorflow to use LSTM.")

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    train_vals = train_series.values.reshape(-1,1)
    scaler.fit(train_vals)
    train_scaled = scaler.transform(train_vals).flatten()

    X_train, y_train = create_sequences(train_scaled, lookback)

    # If validation series provided, prepare val set (for early stopping)
    if val_series is not None:
        val_vals = val_series.values.reshape(-1,1)
        val_scaled = scaler.transform(val_vals).flatten()
        X_val, y_val = create_sequences(val_scaled, lookback)
    else:
        X_val, y_val = None, None

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    callbacks = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)] if X_val is not None else [EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)]
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val) if X_val is not None else None,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1)
    return model, scaler, history

def lstm_predict(model, scaler, full_series, lookback, forecast_steps):
    """
    Recursive multi-step forecast: seed with last lookback window from training+test history,
    predict next step, append to input, and continue.
    full_series: pandas Series of historical values up to (and including) last observed point
    """
    vals = full_series.values.reshape(-1,1)
    scaled = scaler.transform(vals).flatten()
    seq = list(scaled[-lookback:])  # initial seed
    preds_scaled = []
    for _ in range(forecast_steps):
        x_in = np.array(seq[-lookback:]).reshape((1, lookback, 1))
        yhat = model.predict(x_in, verbose=0)[0,0]
        preds_scaled.append(yhat)
        seq.append(yhat)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    return preds

# -------------------------
# Main procedure
# -------------------------
def main():
    series = load_tsla_series()
    # Chronological split: train up to TRAIN_END, test from TEST_START onward
    train, test = train_test_split_time(series, TRAIN_END)
    # Ensure test actually starts after train
    test = test.loc[TEST_START:]

    print(f"Train period: {train.index.min().date()} to {train.index.max().date()} ({len(train)} rows)")
    print(f"Test period: {test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)")

    # ---------- ARIMA ----------
    arima_model = fit_arima_auto(train)
    steps = len(test)
    arima_pred = arima_forecast(arima_model, train.index, steps)

    # build forecast index to align with test
    forecast_index = test.index[:len(arima_pred)]
    arima_pred_series = pd.Series(arima_pred, index=forecast_index)

    arima_metrics = evaluate_forecast(test[:len(arima_pred)], arima_pred_series)
    print("ARIMA performance:", arima_metrics)

    # Save ARIMA results
    pd.DataFrame({"y_true": test[:len(arima_pred)], "y_pred_arima": arima_pred_series}).to_csv(PROCESSED_DIR / "arima_forecast.csv")

    # ---------- LSTM ----------
    if HAVE_TF:
        # We'll train LSTM on train and validate on the last part of train (or small holdout)
        val_split_pct = 0.1
        val_size = int(len(train) * val_split_pct)
        if val_size >= LOOKBACK + 1:
            train_l = train.iloc[:-val_size]
            val_l = train.iloc[-val_size:]
        else:
            train_l = train
            val_l = None

        print("Training LSTM...")
        model, scaler, history = train_lstm(train_l, val_series=val_l, lookback=LOOKBACK, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH)
        # Use both train+test historical upto test start as the 'full_series' seed so predictions are continuous
        seed_series = pd.concat([train, test]).loc[:test.index.max()]  # ensure chronological
        lstm_preds = lstm_predict(model, scaler, seed_series, lookback=LOOKBACK, forecast_steps=steps)
        lstm_pred_series = pd.Series(lstm_preds, index=forecast_index)

        lstm_metrics = evaluate_forecast(test[:len(lstm_pred_series)], lstm_pred_series)
        print("LSTM performance:", lstm_metrics)

        # Save LSTM results
        pd.DataFrame({"y_true": test[:len(lstm_pred_series)], "y_pred_lstm": lstm_pred_series}).to_csv(PROCESSED_DIR / "lstm_forecast.csv")
    else:
        print("TensorFlow/Keras is not installed — skipping LSTM training.")
        lstm_metrics = None

    # ---------- Compare & Plot ----------
    # Combine results into a single DataFrame for plotting
    df_plot = pd.DataFrame({"y_true": test})
    df_plot = df_plot.iloc[:len(arima_pred)]
    df_plot["arima"] = arima_pred_series.values
    if HAVE_TF:
        df_plot["lstm"] = lstm_pred_series.values

    # Plot true vs forecasts
    plt.figure(figsize=(12,6))
    plt.plot(df_plot.index, df_plot["y_true"], label="True")
    plt.plot(df_plot.index, df_plot["arima"], label="ARIMA Pred")
    if HAVE_TF:
        plt.plot(df_plot.index, df_plot["lstm"], label="LSTM Pred")
    plt.title("TSLA: True vs Forecasts")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tsla_forecast_comparison.png", dpi=150)
    plt.close()

    # Save metrics summary
    metrics_summary = {
        "ARIMA": arima_metrics
    }
    if lstm_metrics is not None:
        metrics_summary["LSTM"] = lstm_metrics
    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_df.to_csv(PROCESSED_DIR / "model_performance.csv")
    print("Saved model performance and forecasts.")

if __name__ == "__main__":
    main()
