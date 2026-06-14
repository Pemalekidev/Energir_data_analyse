"""
data_preprocessing.py — Pipeline de nettoyage et prétraitement des données
CEET Smart Grid · Energy Blackout Prediction Project
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from utils import get_logger, DATA_RAW, DATA_PROC, detect_outliers_iqr, summarize_df

logger = get_logger("preprocessing")

RAW_FILE  = DATA_RAW  / "ceet_togo_smartgrid_dataset.csv"
PROC_FILE = DATA_PROC / "ceet_processed.csv"


# ─── 1. Chargement ──────────────────────────────────────────────────────────
def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Charge le dataset brut et parse les dates."""
    logger.info(f"Chargement depuis {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    logger.info(f"Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


# ─── 2. Nettoyage de base ───────────────────────────────────────────────────
def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons, corrige les types, standardise les colonnes."""
    logger.info("Nettoyage de base...")

    # Doublons
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        logger.warning(f"  {n_dup} doublons supprimés")
        df = df.drop_duplicates()

    # Noms de colonnes : minuscules, underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Tri temporel
    df = df.sort_values("datetime").reset_index(drop=True)

    # Colonnes catégorielles : strip + title case
    for col in ["region", "city", "season", "event"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    logger.info("  Nettoyage de base terminé")
    return df


# ─── 3. Traitement des valeurs manquantes ────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes selon la nature de chaque colonne."""
    logger.info("Traitement des valeurs manquantes...")

    # 'event' : NaN = pas d'événement
    if "event" in df.columns:
        df["event"] = df["event"].fillna("No Event")

    # Colonnes numériques continues : interpolation temporelle
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    miss_num = [c for c in num_cols if df[c].isnull().any()]
    if miss_num:
        df[miss_num] = df[miss_num].interpolate(method="time", limit_direction="both")
        logger.info(f"  Interpolées : {miss_num}")

    # Colonnes catégorielles restantes : mode
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    miss_cat = [c for c in cat_cols if df[c].isnull().any()]
    for col in miss_cat:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        logger.info(f"  '{col}' imputé par mode = {mode_val}")

    logger.info(f"  Valeurs nulles restantes : {df.isnull().sum().sum()}")
    return df


# ─── 4. Détection et traitement des outliers ─────────────────────────────────
def handle_outliers(df: pd.DataFrame, strategy: str = "clip") -> pd.DataFrame:
    """Détecte et traite les outliers sur les colonnes critiques."""
    logger.info(f"Traitement des outliers (stratégie: {strategy})...")

    critical_cols = [
        "temperature", "humidity", "total_load_mw", "available_power_mw",
        "voltage", "frequency", "transformer_temp", "outage_risk"
    ]
    bounds = {
        "temperature":        (15, 50),
        "humidity":           (0, 100),
        "voltage":            (180, 260),
        "frequency":          (47, 53),
        "transformer_temp":   (20, 120),
        "total_load_mw":      (0, 1000),
        "available_power_mw": (0, 1000),
        "outage_risk":        (0, 100),
    }

    for col in critical_cols:
        if col not in df.columns:
            continue
        mask = detect_outliers_iqr(df[col])
        n_out = mask.sum()
        if n_out > 0:
            if strategy == "clip":
                lo, hi = bounds.get(col, (df[col].quantile(0.01), df[col].quantile(0.99)))
                df[col] = df[col].clip(lo, hi)
            elif strategy == "remove":
                df = df[~mask]
            logger.info(f"  {col}: {n_out} outliers traités")

    return df


# ─── 5. Feature Engineering temporel ────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des variables temporelles dérivées."""
    logger.info("Ajout des features temporelles...")
    dt = df["datetime"]

    df["year"]       = dt.dt.year
    df["month"]      = dt.dt.month
    df["day"]        = dt.dt.day
    df["day_of_week"]= dt.dt.dayofweek          # 0=Lundi
    df["day_of_year"]= dt.dt.dayofyear
    df["week"]       = dt.dt.isocalendar().week.astype(int)
    df["quarter"]    = dt.dt.quarter
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Encodage cyclique (sin/cos) pour heure, mois, jour de semaine
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Plages horaires
    df["time_of_day"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 20, 23],
        labels=["Nuit", "Matin", "Après-midi", "Soir", "Nuit tardive"]
    ).astype(str)

    # Heure de pointe (peak hours)
    df["is_peak_hour"] = df["hour"].isin([6, 7, 8, 17, 18, 19, 20, 21]).astype(int)

    logger.info("  Features temporelles ajoutées")
    return df


# ─── 6. Features énergétiques dérivées ───────────────────────────────────────
def add_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule des indicateurs énergétiques supplémentaires."""
    logger.info("Ajout des features énergétiques...")

    # Marge de puissance
    df["power_margin"]     = df["available_power_mw"] - df["total_load_mw"]
    df["power_margin_pct"] = (df["power_margin"] / df["available_power_mw"].replace(0, np.nan)) * 100

    # Taux de pénétration renouvelable
    df["renewable_share"]  = (df["renewable_power_mw"] / df["available_power_mw"].replace(0, np.nan)) * 100

    # Intensité de charge par secteur
    df["industrial_share"] = df["industrial_load"] / df["total_load_mw"].replace(0, np.nan)
    df["residential_share"]= df["residential_load"] / df["total_load_mw"].replace(0, np.nan)
    df["commercial_share"] = df["commercial_load"]  / df["total_load_mw"].replace(0, np.nan)

    # Ratio charge/disponibilité
    df["load_ratio"]       = df["total_load_mw"] / df["available_power_mw"].replace(0, np.nan)

    # Anomalie de tension
    df["voltage_deviation"]    = (df["voltage"] - 220).abs()
    df["frequency_deviation"]  = (df["frequency"] - 50).abs()

    # Indicateur stress réseau
    df["grid_stress_index"] = (
        0.4 * df["load_ratio"].clip(0, 2) +
        0.3 * (df["transformer_temp"] / 100) +
        0.15 * (df["voltage_deviation"] / 20) +
        0.15 * (df["frequency_deviation"] / 3)
    )

    # Lag features (valeurs précédentes)
    for lag in [1, 3, 6, 12, 24]:
        df[f"load_lag_{lag}h"]        = df["total_load_mw"].shift(lag)
        df[f"outage_risk_lag_{lag}h"] = df["outage_risk"].shift(lag)

    # Rolling statistics (fenêtres glissantes)
    for window in [3, 6, 24]:
        df[f"load_roll_mean_{window}h"] = df["total_load_mw"].rolling(window, min_periods=1).mean()
        df[f"load_roll_std_{window}h"]  = df["total_load_mw"].rolling(window, min_periods=1).std().fillna(0)
        df[f"risk_roll_max_{window}h"]  = df["outage_risk"].rolling(window, min_periods=1).max()

    logger.info("  Features énergétiques ajoutées")
    return df


# ─── 7. Encodage des variables catégorielles ──────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode les variables catégorielles avec LabelEncoder."""
    logger.info("Encodage des variables catégorielles...")
    encoders = {}
    cat_cols = ["region", "city", "season", "event", "time_of_day"]

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"  {col}: {len(le.classes_)} classes")

    return df, encoders


# ─── 8. Normalisation ─────────────────────────────────────────────────────────
def scale_features(df: pd.DataFrame, exclude_cols: list = None) -> tuple[pd.DataFrame, StandardScaler]:
    """Applique StandardScaler sur les colonnes numériques."""
    exclude = set(exclude_cols or []) | {
        "blackout", "overload", "is_weekend", "is_peak_hour",
        "year", "month", "day", "hour", "day_of_week", "day_of_year", "week", "quarter"
    }
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
    return df_scaled, scaler


# ─── 9. Pipeline principal ────────────────────────────────────────────────────
def run_preprocessing_pipeline(
    raw_path: Path = RAW_FILE,
    output_path: Path = PROC_FILE,
    save: bool = True
) -> pd.DataFrame:
    """Exécute le pipeline complet de prétraitement."""
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DU PIPELINE DE PRÉTRAITEMENT")
    logger.info("=" * 60)

    df = load_raw(raw_path)
    df = clean_basic(df)
    df = handle_missing(df)
    df = handle_outliers(df, strategy="clip")
    df = add_temporal_features(df)
    df = add_energy_features(df)
    df, encoders = encode_categoricals(df)

    # Remplir les NaN créés par les lag/rolling
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].bfill().fillna(0)

    summary = summarize_df(df)
    logger.info(f"Dataset final : {summary['shape']}")
    logger.info(f"Colonnes : {len(summary['columns'])}")
    logger.info(f"Valeurs nulles : {sum(summary['missing'].values())}")

    if save:
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset sauvegardé → {output_path}")

    logger.info("PIPELINE TERMINÉ ✓")
    return df


if __name__ == "__main__":
    df = run_preprocessing_pipeline()
    print(df.shape)
    print(df.columns.tolist())
