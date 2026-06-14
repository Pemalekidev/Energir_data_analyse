"""
feature_engineering.py — Construction des features pour le ML
CEET Smart Grid · Energy Blackout Prediction Project
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

from utils import get_logger

logger = get_logger("feature_engineering")

# ─── Features par tâche ───────────────────────────────────────────────────────

BLACKOUT_FEATURES = [
    "temperature", "humidity", "total_load_mw", "available_power_mw",
    "renewable_power_mw", "transformer_temp", "voltage", "frequency",
    "outage_risk", "load_ratio", "power_margin", "power_margin_pct",
    "grid_stress_index", "voltage_deviation", "frequency_deviation",
    "industrial_share", "residential_share", "commercial_share",
    "renewable_share", "is_peak_hour", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "load_lag_1h", "load_lag_3h", "load_lag_6h", "load_lag_24h",
    "outage_risk_lag_1h", "outage_risk_lag_6h",
    "load_roll_mean_3h", "load_roll_mean_24h",
    "load_roll_std_6h", "risk_roll_max_6h", "risk_roll_max_24h",
    "region_enc", "city_enc", "season_enc", "event_enc",
    "population_density", "smart_meter_count",
]

OVERLOAD_FEATURES = [
    "temperature", "humidity", "total_load_mw", "available_power_mw",
    "industrial_load", "residential_load", "commercial_load",
    "transformer_temp", "voltage", "frequency",
    "load_ratio", "power_margin", "grid_stress_index",
    "is_peak_hour", "is_weekend", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "load_lag_1h", "load_lag_6h", "load_roll_mean_6h", "load_roll_std_6h",
    "region_enc", "city_enc", "season_enc", "event_enc",
    "population_density",
]

RISK_FEATURES = [
    "temperature", "humidity", "total_load_mw", "available_power_mw",
    "transformer_temp", "voltage", "frequency",
    "load_ratio", "power_margin_pct", "grid_stress_index",
    "voltage_deviation", "frequency_deviation",
    "is_peak_hour", "is_weekend", "hour_sin", "hour_cos",
    "load_lag_1h", "load_lag_6h", "outage_risk_lag_1h",
    "load_roll_mean_3h", "risk_roll_max_6h",
    "region_enc", "city_enc", "season_enc", "event_enc",
]

DEMAND_FEATURES = [
    "temperature", "humidity",
    "hour", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "load_lag_1h", "load_lag_3h", "load_lag_6h", "load_lag_24h",
    "load_roll_mean_3h", "load_roll_mean_6h", "load_roll_mean_24h",
    "load_roll_std_6h",
    "region_enc", "city_enc", "season_enc", "event_enc",
    "population_density", "renewable_share",
]


# ─── Sélection des features ───────────────────────────────────────────────────
def get_feature_matrix(
    df: pd.DataFrame,
    feature_list: list,
    target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Extrait X et y en vérifiant la disponibilité des colonnes."""
    available = [f for f in feature_list if f in df.columns]
    missing = set(feature_list) - set(available)
    if missing:
        logger.warning(f"Colonnes absentes (ignorées) : {missing}")
    X = df[available].copy()
    y = df[target_col].copy()
    return X, y


def select_k_best(X: pd.DataFrame, y: pd.Series, k: int = 20, task: str = "classification") -> list:
    """Sélectionne les k meilleures features via F-test ou info mutuelle."""
    score_func = f_classif if task == "classification" else mutual_info_classif
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    selector.fit(X.fillna(0), y)
    selected = X.columns[selector.get_support()].tolist()
    scores = dict(zip(X.columns, selector.scores_))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"Top {k} features sélectionnées :")
    for name, score in ranked[:10]:
        logger.info(f"  {name:40s} score={score:.2f}")
    return selected


def risk_classification_label(outage_risk: pd.Series) -> pd.Series:
    """Convertit outage_risk (0-100) en label de risque catégoriel."""
    return pd.cut(
        outage_risk,
        bins=[-1, 30, 60, 80, 100],
        labels=["Faible", "Modéré", "Élevé", "Critique"]
    )


# ─── PCA pour exploration ─────────────────────────────────────────────────────
def apply_pca(X: pd.DataFrame, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    """Applique PCA pour réduction de dimension (visualisation)."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X.fillna(0))
    logger.info(f"PCA : variance expliquée = {pca.explained_variance_ratio_.sum():.2%}")
    return X_pca, pca


# ─── Résumé des features par tâche ────────────────────────────────────────────
def features_summary() -> dict:
    return {
        "blackout_prediction": {"features": BLACKOUT_FEATURES, "target": "blackout", "task": "binary_classification"},
        "overload_prediction":  {"features": OVERLOAD_FEATURES,  "target": "overload", "task": "binary_classification"},
        "risk_classification":  {"features": RISK_FEATURES,      "target": "risk_level", "task": "multiclass_classification"},
        "demand_forecasting":   {"features": DEMAND_FEATURES,    "target": "total_load_mw", "task": "regression"},
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    print("Features par tâche :")
    for task, info in features_summary().items():
        print(f"\n  {task}: {len(info['features'])} features → {info['target']} ({info['task']})")
