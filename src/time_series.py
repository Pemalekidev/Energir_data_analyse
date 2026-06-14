"""
time_series.py — Analyse et prévision des séries temporelles
CEET Smart Grid · Energy Blackout Prediction Project
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from utils import get_logger, save_model, DATA_PROC

logger = get_logger("time_series")


# ─── Préparation des séries ───────────────────────────────────────────────────
def prepare_timeseries(df: pd.DataFrame, target: str = "total_load_mw",
                       freq: str = "h") -> pd.Series:
    """Crée une série temporelle propre et indexée."""
    ts = df.set_index("datetime")[target].copy()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()
    ts = ts.resample(freq).mean().interpolate()
    logger.info(f"Série temporelle : {len(ts)} points | {ts.index[0]} → {ts.index[-1]}")
    return ts


# ─── Tests de stationnarité ───────────────────────────────────────────────────
def test_stationarity(series: pd.Series, name: str = "série") -> dict:
    """Test ADF de stationnarité."""
    result = adfuller(series.dropna(), autolag="AIC")
    stationary = result[1] < 0.05
    output = {
        "ADF Statistic": round(result[0], 4),
        "p-value":       round(result[1], 6),
        "Lags":          result[2],
        "Observations":  result[3],
        "Critical 1%":   round(result[4]["1%"], 4),
        "Critical 5%":   round(result[4]["5%"], 4),
        "Is stationary": stationary,
    }
    logger.info(f"ADF Test – {name}: {'STATIONNAIRE' if stationary else 'NON STATIONNAIRE'} (p={result[1]:.4f})")
    return output


# ─── Décomposition ────────────────────────────────────────────────────────────
def decompose_series(series: pd.Series, period: int = 24) -> dict:
    """Décompose une série en tendance, saisonnalité, résidu."""
    decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    logger.info(f"Décomposition effectuée (période={period})")
    return {
        "trend":    decomp.trend,
        "seasonal": decomp.seasonal,
        "residual": decomp.resid,
        "observed": decomp.observed,
    }


# ─── SARIMA ───────────────────────────────────────────────────────────────────
def fit_sarima(series: pd.Series,
               order=(1, 1, 1),
               seasonal_order=(1, 1, 1, 24),
               n_forecast: int = 48) -> dict:
    """Ajuste un modèle SARIMA et génère des prévisions."""
    logger.info(f"SARIMA{order} × {seasonal_order}...")
    train = series[:-n_forecast]
    test  = series[-n_forecast:]

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    logger.info(f"  AIC={fitted.aic:.2f} | BIC={fitted.bic:.2f}")

    forecast = fitted.forecast(steps=n_forecast)
    conf_int = fitted.get_forecast(steps=n_forecast).conf_int()

    mae  = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test.values - forecast.values) / (np.abs(test.values) + 1e-9))) * 100

    logger.info(f"  MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%")

    return {
        "model":     fitted,
        "forecast":  forecast,
        "conf_int":  conf_int,
        "test":      test,
        "train":     train,
        "metrics":   {"mae": mae, "rmse": rmse, "mape": mape},
    }


# ─── LSTM / GRU (Numpy pur – sans dépendances lourdes) ───────────────────────
class SimpleLSTMNumpy:
    """
    Implémentation légère d'un modèle LSTM-like basé sur sliding window
    et régression Ridge pour les environnements sans GPU/TensorFlow.
    Simule l'approche séquentielle pour démonstration.
    """

    def __init__(self, window: int = 24, horizon: int = 6):
        from sklearn.linear_model import Ridge
        self.window  = window
        self.horizon = horizon
        self.model   = Ridge(alpha=0.5)
        self.scaler  = MinMaxScaler()
        self.feature_names = None

    def _create_sequences(self, series: np.ndarray):
        """Crée les paires (X, y) par fenêtre glissante."""
        X, y = [], []
        for i in range(len(series) - self.window - self.horizon + 1):
            X.append(series[i : i + self.window])
            y.append(series[i + self.window : i + self.window + self.horizon])
        return np.array(X), np.array(y)

    def fit(self, series: pd.Series):
        """Entraîne le modèle sur une série."""
        vals = self.scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        X, y = self._create_sequences(vals)
        # Aplatir pour la régression Ridge
        self.model.fit(X, y[:, 0] if y.ndim > 1 else y)
        logger.info(f"SimpleLSTM entraîné : window={self.window}, horizon={self.horizon}")
        return self

    def predict(self, series: pd.Series, n_steps: int = None) -> np.ndarray:
        """Génère des prévisions multi-pas."""
        n = n_steps or self.horizon
        vals = self.scaler.transform(series.values.reshape(-1, 1)).flatten()
        preds = []
        window = vals[-self.window:].tolist()
        for _ in range(n):
            x = np.array(window[-self.window:]).reshape(1, -1)
            p = self.model.predict(x)[0]
            preds.append(p)
            window.append(p)
        preds_arr = np.array(preds).reshape(-1, 1)
        return self.scaler.inverse_transform(preds_arr).flatten()

    def evaluate(self, test_series: pd.Series, train_series: pd.Series) -> dict:
        """Évalue les prévisions sur la série de test."""
        forecasts = self.predict(train_series, n_steps=len(test_series))
        actual    = test_series.values[:len(forecasts)]
        mae  = mean_absolute_error(actual, forecasts)
        rmse = np.sqrt(mean_squared_error(actual, forecasts))
        mape = np.mean(np.abs((actual - forecasts) / (np.abs(actual) + 1e-9))) * 100
        return {"mae": mae, "rmse": rmse, "mape": mape}


# ─── Prévision par région ──────────────────────────────────────────────────────
def forecast_by_region(df: pd.DataFrame, target: str = "total_load_mw",
                       n_forecast: int = 24) -> dict:
    """Génère des prévisions SARIMA pour chaque région."""
    results = {}
    regions = df["region"].unique() if "region" in df.columns else ["All"]

    for region in regions:
        logger.info(f"\nRégion : {region}")
        region_df = df[df["region"] == region] if region != "All" else df
        ts = prepare_timeseries(region_df, target)

        if len(ts) < 100:
            logger.warning(f"  Données insuffisantes ({len(ts)} points) — région ignorée")
            continue

        try:
            result = fit_sarima(ts, order=(1,1,1), seasonal_order=(1,1,1,24),
                                n_forecast=min(n_forecast, len(ts)//4))
            results[region] = result
        except Exception as e:
            logger.error(f"  Erreur SARIMA {region}: {e}")

    return results


# ─── Détection de pics de charge ──────────────────────────────────────────────
def detect_load_peaks(series: pd.Series, threshold_quantile: float = 0.95) -> pd.Series:
    """Identifie les pics de charge au-dessus d'un seuil percentile."""
    threshold = series.quantile(threshold_quantile)
    peaks = series[series > threshold]
    logger.info(f"Pics de charge (>{threshold_quantile*100:.0f}e percentile = {threshold:.1f} MW) : {len(peaks)}")
    return peaks


# ─── Analyse de tendance ──────────────────────────────────────────────────────
def compute_trend_statistics(ts: pd.Series) -> dict:
    """Calcule des statistiques descriptives de la tendance."""
    decomp = decompose_series(ts)
    trend = decomp["trend"].dropna()
    return {
        "mean":         round(trend.mean(), 2),
        "std":          round(trend.std(), 2),
        "min":          round(trend.min(), 2),
        "max":          round(trend.max(), 2),
        "slope_mw_day": round((trend.iloc[-1] - trend.iloc[0]) / (len(trend) / 24), 4),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    proc = DATA_PROC / "ceet_processed.csv"
    if proc.exists():
        df = pd.read_csv(proc, parse_dates=["datetime"])
        ts  = prepare_timeseries(df)
        stat = test_stationarity(ts, "Charge totale")
        decomp = decompose_series(ts)
        print("Décomposition effectuée")

        # LSTM simplifié
        train, test = ts[:-48], ts[-48:]
        lstm = SimpleLSTMNumpy(window=24, horizon=6)
        lstm.fit(train)
        metrics = lstm.evaluate(test, train)
        print(f"SimpleLSTM : MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
