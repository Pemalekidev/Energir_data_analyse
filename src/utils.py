"""
utils.py — Fonctions utilitaires partagées
CEET Smart Grid · Energy Blackout Prediction Project
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ─── Chemins du projet ──────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT_DIR / "data" / "raw"
DATA_PROC  = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS    = ROOT_DIR / "reports"
FIGURES    = REPORTS / "figures"
TABLES     = REPORTS / "tables"
DOCS       = REPORTS / "docs"

for d in [DATA_RAW, DATA_PROC, MODELS_DIR, FIGURES, TABLES, DOCS]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Logging ────────────────────────────────────────────────────────────────
def get_logger(name: str = "energy_grid", level: int = logging.INFO) -> logging.Logger:
    """Retourne un logger configuré avec handler console + fichier."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # Fichier
    log_path = ROOT_DIR / "reports" / "docs" / "pipeline.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ─── Sauvegarde / chargement de modèles ─────────────────────────────────────
def save_model(model, name: str, metadata: dict = None) -> Path:
    """Sauvegarde un modèle sklearn/compatible avec joblib."""
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    if metadata:
        meta_path = MODELS_DIR / f"{name}_meta.json"
        metadata["saved_at"] = datetime.now().isoformat()
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    return path


def load_model(name: str):
    """Charge un modèle sauvegardé."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)


# ─── Utilitaires DataFrame ───────────────────────────────────────────────────
def summarize_df(df: pd.DataFrame) -> dict:
    """Retourne un résumé rapide d'un DataFrame."""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Retourne un masque booléen des outliers IQR."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - factor * iqr) | (series > q3 + factor * iqr)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Retourne un masque booléen des outliers Z-score."""
    z = (series - series.mean()) / series.std()
    return z.abs() > threshold


# ─── Métriques de classification ─────────────────────────────────────────────
def classification_report_dict(y_true, y_pred, y_proba=None) -> dict:
    """Compile un dictionnaire complet de métriques de classification."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
    )
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc":       round(matthews_corrcoef(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
    return metrics


# ─── Métriques de régression ─────────────────────────────────────────────────
def regression_report_dict(y_true, y_pred) -> dict:
    """Compile un dictionnaire de métriques de régression."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "mse":  round(mse, 4),
        "rmse": round(np.sqrt(mse), 4),
        "r2":   round(r2_score(y_true, y_pred), 4),
        "mape": round(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100, 4),
    }


# ─── Configuration des couleurs projet ───────────────────────────────────────
COLORS = {
    "primary":   "#E63946",
    "secondary": "#457B9D",
    "accent":    "#F4A261",
    "success":   "#2A9D8F",
    "warning":   "#E9C46A",
    "dark":      "#1D3557",
    "light":     "#F1FAEE",
    "regions": {
        "Lome":     "#E63946",
        "Maritime": "#457B9D",
        "Plateaux": "#2A9D8F",
        "Centrale": "#E9C46A",
        "Kara":     "#F4A261",
        "Savanes":  "#6D6875",
    }
}
