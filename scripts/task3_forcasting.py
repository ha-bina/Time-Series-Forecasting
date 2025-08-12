import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

try:
    import pmdarima as pm
    HAVE_PMDARIMA = True
except Exception:
    HAVE_PMDARIMA = False

from statsmodels.tsa.arima.model import ARIMA

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    HAVE_TF = True
except Exception:
    HAVE_TF = False

from scripts.utils import download_data, clean_data
from scripts.model_utils import create_sequences

# Config
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'processed' / 'cleaned_adj_close.csv'
FIG_DIR = ROOT / 'reports' / 'figures'
OUT_DIR = ROOT / 'data' / 'processed'
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_MONTHS = 6
FORECAST_DAYS = int(FORECAST_MONTHS * 21)

# Helpers
def load_data():
    if not DATA_PATH.exists():
        df = download_data(['TSLA','SPY','BND'], start='2015-01-01', end=dt.date.today().isoformat())
        df = clean_data(df)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH)
        return df
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

def fit_arima(series):
    if HAVE_PMDARIMA:
        model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
        return model
    else:
        best = None
        best_aic = np.inf
        for p in range(0,3):
            for d in range(0,2):
                for q in range(0,3):
                    try:
                        m = ARIMA(series, order=(p,d,q)).fit()
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best = m
                    except Exception:
                        continue
        return best

def arima_forecast_with_ci(model, series, steps=FORECAST_DAYS, alpha=0.05):
    last_date = series.index[-1]
    if HAVE_PMDARIMA and hasattr(model, 'predict'):
        try:
            fc, conf = model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
            lower, upper = conf[:,0], conf[:,1]
        except Exception:
            fc = model.predict(n_periods=steps)
            resid_std = np.std(model.resid() if hasattr(model, 'resid') else model.resid())
            lower = fc - 1.96 * resid_std
            upper = fc + 1.96 * resid_std
    else:
        res = model
        pred = res.get_forecast(steps=steps)
        fc = pred.predicted_mean.values
        conf = pred.conf_int(alpha=alpha)
        lower = conf.iloc[:,0].values
        upper = conf.iloc[:,1].values

    idx = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(fc))
    return pd.Series(fc, index=idx), pd.DataFrame({'lower': lower, 'upper': upper}, index=idx[:len(lower)])

# Optional simple LSTM training for forecasting (recursive)
def train_lstm(series, lookback=60, epochs=30, batch_size=32):
    if not HAVE_TF:
        raise ImportError('TensorFlow required for LSTM')
    vals = series.values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals).flatten()
    X, y = create_sequences(scaled, lookback)
    val_size = int(len(X)*0.1)
    if val_size > 0:
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    model = Sequential([LSTM(64, input_shape=(X_train.shape[1],1)), Dropout(0.2), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    cb = [EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', patience=8, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_data=(X_val,y_val) if X_val is not None else None, epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)
    return model, scaler, lookback

def lstm_recursive_forecast(model, scaler, series, lookback, steps=FORECAST_DAYS):
    vals = series.values.reshape(-1,1)
    scaled = scaler.transform(vals).flatten()
    seq = list(scaled[-lookback:])
    preds = []
    for _ in range(steps):
        x = np.array(seq[-lookback:]).reshape((1,lookback,1))
        yhat = model.predict(x, verbose=0)[0,0]
        preds.append(yhat)
        seq.append(yhat)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    idx = pd.bdate_range(start=series.index[-1] + pd.Timedelta(days=1), periods=len(preds))
    return pd.Series(preds, index=idx)

# Main
if __name__ == '__main__':
    df = load_data()
    tsla = df['TSLA'].dropna()

    print('Fitting ARIMA...')
    arima_model = fit_arima(tsla)
    arima_fc, arima_ci = arima_forecast_with_ci(arima_model, tsla)
    arima_out = pd.DataFrame({'forecast': arima_fc}).join(arima_ci)
    arima_out.to_csv(OUT_DIR / f'tsla_arima_forecast_{FORECAST_MONTHS}m.csv')

    if HAVE_TF:
        print('Training LSTM (this may take a while)...')
        lstm_model, scaler, lookback = train_lstm(tsla)
        lstm_fc = lstm_recursive_forecast(lstm_model, scaler, tsla, lookback)
        pd.DataFrame({'forecast': lstm_fc}).to_csv(OUT_DIR / f'tsla_lstm_forecast_{FORECAST_MONTHS}m.csv')

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(tsla.index[-504:], tsla[-504:], label='Historical (2 years)')
    plt.plot(arima_fc.index, arima_fc.values, label='ARIMA Forecast')
    plt.fill_between(arima_ci.index, arima_ci['lower'], arima_ci['upper'], alpha=0.2)
    if HAVE_TF:
        plt.plot(lstm_fc.index, lstm_fc.values, label='LSTM Forecast')
    plt.legend(); plt.title('TSLA Forecast'); plt.tight_layout()
    plt.savefig(FIG_DIR / f'tsla_forecast_{FORECAST_MONTHS}m.png', dpi=150); plt.close()

    print('Forecasts saved to', OUT_DIR)