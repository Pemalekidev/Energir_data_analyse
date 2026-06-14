"""
anomaly_detection.py — Détection d'anomalies réseau électrique
CEET Smart Grid · Energy Blackout Prediction Project
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils import get_logger, save_model, MODELS_DIR

logger = get_logger("anomaly_detection")

ANOMALY_FEATURES = [
    "temperature", "total_load_mw", "available_power_mw",
    "transformer_temp", "voltage", "frequency",
    "outage_risk", "load_ratio", "grid_stress_index",
    "voltage_deviation", "frequency_deviation",
    "power_margin", "renewable_share",
]


def prepare_anomaly_data(df: pd.DataFrame, features: list = None) -> tuple:
    """Prépare et scale les données pour la détection d'anomalies."""
    if features is None:
        features = ANOMALY_FEATURES
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X, scaler, available


# ─── Isolation Forest ─────────────────────────────────────────────────────────
def run_isolation_forest(X_scaled: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """Détection par Isolation Forest.
    Retourne un vecteur : 1=normal, -1=anomalie.
    """
    logger.info("Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    preds = model.fit_predict(X_scaled)
    scores = model.score_samples(X_scaled)
    n_anomalies = (preds == -1).sum()
    logger.info(f"  {n_anomalies} anomalies détectées ({contamination*100:.0f}% seuil)")
    save_model(model, "isolation_forest")
    return preds, scores, model


# ─── One-Class SVM ────────────────────────────────────────────────────────────
def run_one_class_svm(X_scaled: np.ndarray, nu: float = 0.05) -> np.ndarray:
    """Détection par One-Class SVM (données réduites pour performance)."""
    logger.info("One-Class SVM...")
    # Sous-échantillonnage pour la scalabilité
    n = min(5000, X_scaled.shape[0])
    idx = np.random.choice(X_scaled.shape[0], n, replace=False)
    X_sample = X_scaled[idx]

    model = OneClassSVM(kernel="rbf", gamma="auto", nu=nu)
    model.fit(X_sample)
    preds = model.predict(X_scaled)
    n_anomalies = (preds == -1).sum()
    logger.info(f"  {n_anomalies} anomalies détectées")
    return preds, model


# ─── DBSCAN ───────────────────────────────────────────────────────────────────
def run_dbscan(X_scaled: np.ndarray, eps: float = 0.5, min_samples: int = 10) -> np.ndarray:
    """Clustering DBSCAN : points de bruit = anomalies (-1)."""
    logger.info(f"DBSCAN (eps={eps}, min_samples={min_samples})...")
    # Réduction PCA pour performance
    pca = PCA(n_components=5, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    logger.info(f"  {n_clusters} clusters, {n_noise} points de bruit (anomalies)")
    return labels, model


# ─── AutoEncoder (NumPy pur – sans TensorFlow) ────────────────────────────────
class AutoEncoderNumpy:
    """AutoEncoder léger basé sur PCA comme proxy de reconstruction."""

    def __init__(self, n_components: int = 5):
        self.pca = PCA(n_components=n_components, random_state=42)
        self.threshold = None

    def fit(self, X: np.ndarray, percentile: float = 95):
        self.pca.fit(X)
        X_rec = self.reconstruct(X)
        errors = self._reconstruction_error(X, X_rec)
        self.threshold = np.percentile(errors, percentile)
        logger.info(f"  AutoEncoder PCA : variance expliquée = {self.pca.explained_variance_ratio_.sum():.2%}")
        logger.info(f"  Seuil reconstruction (p{percentile:.0f}) = {self.threshold:.4f}")
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.pca.inverse_transform(self.pca.transform(X))

    def _reconstruction_error(self, X: np.ndarray, X_rec: np.ndarray) -> np.ndarray:
        return np.mean((X - X_rec) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne 1=normal, -1=anomalie."""
        X_rec = self.reconstruct(X)
        errors = self._reconstruction_error(X, X_rec)
        return np.where(errors > self.threshold, -1, 1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        X_rec = self.reconstruct(X)
        return self._reconstruction_error(X, X_rec)


def run_autoencoder(X_scaled: np.ndarray, n_components: int = 8) -> tuple:
    """Détection par AutoEncoder (PCA-based)."""
    logger.info("AutoEncoder (PCA-based)...")
    ae = AutoEncoderNumpy(n_components=n_components)
    ae.fit(X_scaled)
    preds = ae.predict(X_scaled)
    scores = ae.anomaly_score(X_scaled)
    n_anomalies = (preds == -1).sum()
    logger.info(f"  {n_anomalies} anomalies détectées par AutoEncoder")
    return preds, scores, ae


# ─── Agrégation des résultats ─────────────────────────────────────────────────
def combine_anomaly_detectors(df: pd.DataFrame) -> pd.DataFrame:
    """Lance tous les détecteurs et combine les résultats."""
    logger.info("=" * 60)
    logger.info("DÉTECTION D'ANOMALIES MULTI-MÉTHODES")
    logger.info("=" * 60)

    X_scaled, X_raw, scaler, features = prepare_anomaly_data(df)

    # Isolation Forest
    if_preds, if_scores, _ = run_isolation_forest(X_scaled, contamination=0.05)

    # AutoEncoder
    ae_preds, ae_scores, ae = run_autoencoder(X_scaled, n_components=8)

    # DBSCAN
    db_labels, _ = run_dbscan(X_scaled, eps=0.8, min_samples=15)
    db_preds = np.where(db_labels == -1, -1, 1)

    # Résultats combinés
    df_out = df.copy()
    df_out["if_anomaly"]    = (if_preds == -1).astype(int)
    df_out["if_score"]      = if_scores
    df_out["ae_anomaly"]    = (ae_preds == -1).astype(int)
    df_out["ae_score"]      = ae_scores
    df_out["dbscan_anomaly"]= (db_preds == -1).astype(int)

    # Vote majoritaire (≥ 2/3 détecteurs)
    df_out["anomaly_vote"]  = (
        df_out["if_anomaly"] + df_out["ae_anomaly"] + df_out["dbscan_anomaly"]
    )
    df_out["is_anomaly"]    = (df_out["anomaly_vote"] >= 2).astype(int)

    # Score de risque normalisé (0-100)
    raw_score = (
        df_out["anomaly_vote"] / 3 * 0.4 +
        (df_out["if_score"] - df_out["if_score"].min()) /
        (df_out["if_score"].max() - df_out["if_score"].min() + 1e-9) * 0.3 +
        (df_out["ae_score"] - df_out["ae_score"].min()) /
        (df_out["ae_score"].max() - df_out["ae_score"].min() + 1e-9) * 0.3
    )
    df_out["anomaly_risk_score"] = (raw_score * 100).round(2)

    n_final = df_out["is_anomaly"].sum()
    pct = n_final / len(df_out) * 100
    logger.info(f"\nAnomalies finales (vote ≥ 2/3) : {n_final} ({pct:.2f}%)")

    # Résumé par région
    summary = df_out.groupby("region")["is_anomaly"].agg(
        n_anomalies="sum",
        total="count",
        rate=lambda x: (x.sum() / len(x) * 100).round(2)
    )
    logger.info(f"\nAnomalies par région :\n{summary}")

    return df_out, ae


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import DATA_PROC

    proc = DATA_PROC / "ceet_processed.csv"
    if proc.exists():
        df = pd.read_csv(proc)
        df_anomalies, _ = combine_anomaly_detectors(df)
        print(f"\nAnomalies détectées : {df_anomalies['is_anomaly'].sum()}")
        print(df_anomalies[["datetime","region","city","is_anomaly","anomaly_risk_score"]].
              query("is_anomaly == 1").head(20).to_string(index=False))
