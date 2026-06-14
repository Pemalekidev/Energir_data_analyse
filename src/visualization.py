"""
visualization.py — Visualisations EDA et reporting
CEET Smart Grid · Energy Blackout Prediction Project
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from utils import get_logger, FIGURES, COLORS

logger = get_logger("visualization")

# Style global
plt.rcParams.update({
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

PALETTE = list(COLORS["regions"].values())


# ─── Helpers ──────────────────────────────────────────────────────────────────
def save_fig(name: str, fig=None, dpi: int = 150):
    path = FIGURES / f"{name}.png"
    (fig or plt).savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info(f"  Figure sauvegardée → {path.name}")
    return path


# ─── 1. Aperçu général ────────────────────────────────────────────────────────
def plot_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CEET Smart Grid – Aperçu Général", fontsize=16, fontweight="bold")

    # Distribution de la charge
    axes[0, 0].hist(df["total_load_mw"], bins=50, color=COLORS["primary"], edgecolor="white", alpha=0.8)
    axes[0, 0].set_title("Distribution de la Charge Totale")
    axes[0, 0].set_xlabel("Charge (MW)")

    # Blackouts par région
    if "region" in df.columns:
        region_bl = df.groupby("region")["blackout"].mean() * 100
        region_bl.sort_values().plot.barh(ax=axes[0, 1], color=COLORS["primary"], alpha=0.8)
        axes[0, 1].set_title("Taux de Blackout par Région (%)")
        axes[0, 1].set_xlabel("%")

    # Charge par heure
    if "hour" in df.columns:
        hourly = df.groupby("hour")["total_load_mw"].mean()
        axes[0, 2].plot(hourly.index, hourly.values, color=COLORS["secondary"], linewidth=2)
        axes[0, 2].fill_between(hourly.index, hourly.values, alpha=0.3, color=COLORS["secondary"])
        axes[0, 2].set_title("Charge Moyenne par Heure")
        axes[0, 2].set_xlabel("Heure")
        axes[0, 2].set_ylabel("MW")

    # Température vs Charge
    axes[1, 0].scatter(df["temperature"], df["total_load_mw"],
                       c=df["blackout"], cmap="RdYlGn_r", alpha=0.3, s=5)
    axes[1, 0].set_title("Température vs Charge")
    axes[1, 0].set_xlabel("Température (°C)")
    axes[1, 0].set_ylabel("Charge (MW)")

    # Risque de panne par saison
    if "season" in df.columns:
        season_risk = df.groupby("season")["outage_risk"].mean()
        season_risk.plot.bar(ax=axes[1, 1], color=COLORS["accent"], edgecolor="white", alpha=0.8)
        axes[1, 1].set_title("Risque Moyen par Saison")
        axes[1, 1].set_ylabel("Risque (%)")
        axes[1, 1].tick_params(axis="x", rotation=0)

    # Série temporelle charge (sous-échantillonnée)
    if "datetime" in df.columns:
        ts = df.set_index("datetime")["total_load_mw"].resample("D").mean()
        axes[1, 2].plot(ts.index, ts.values, color=COLORS["dark"], linewidth=1)
        axes[1, 2].set_title("Charge Journalière Moyenne (série)")
        axes[1, 2].set_xlabel("Date")
        axes[1, 2].set_ylabel("MW")

    plt.tight_layout()
    return save_fig("01_overview", fig)


# ─── 2. Heatmap de corrélation ────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = [
        "temperature", "humidity", "total_load_mw", "available_power_mw",
        "renewable_power_mw", "transformer_temp", "voltage", "frequency",
        "outage_risk", "load_shedding_mw", "energy_price",
        "blackout", "overload",
    ]
    avail = [c for c in num_cols if c in df.columns]
    corr = df[avail].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Matrice de Corrélation – Variables Énergétiques", fontsize=14, pad=15)
    plt.tight_layout()
    return save_fig("02_correlation_heatmap", fig)


# ─── 3. Analyse temporelle ────────────────────────────────────────────────────
def plot_temporal_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Analyse Temporelle de la Charge Électrique", fontsize=15, fontweight="bold")

    df2 = df.copy()
    if "datetime" in df2.columns:
        df2["datetime"] = pd.to_datetime(df2["datetime"])
        df2["month"] = df2["datetime"].dt.month

    # Charge par heure et jour de semaine (heatmap)
    if all(c in df2.columns for c in ["hour", "day_of_week", "total_load_mw"]):
        pivot = df2.pivot_table("total_load_mw", "hour", "day_of_week", aggfunc="mean")
        pivot.columns = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        sns.heatmap(pivot, ax=axes[0, 0], cmap="YlOrRd", annot=False,
                    cbar_kws={"label": "MW"})
        axes[0, 0].set_title("Charge Moyenne : Heure × Jour de la Semaine")
        axes[0, 0].set_ylabel("Heure")

    # Risque de panne par heure
    if all(c in df2.columns for c in ["hour", "outage_risk", "blackout"]):
        hourly = df2.groupby("hour").agg(
            risk=("outage_risk", "mean"),
            blackouts=("blackout", "sum")
        )
        ax2 = axes[0, 1].twinx()
        axes[0, 1].bar(hourly.index, hourly["blackouts"], color=COLORS["primary"], alpha=0.5, label="Blackouts")
        ax2.plot(hourly.index, hourly["risk"], color=COLORS["dark"], linewidth=2, label="Risque moyen")
        axes[0, 1].set_title("Blackouts et Risque par Heure")
        axes[0, 1].set_xlabel("Heure")
        axes[0, 1].set_ylabel("Nb Blackouts", color=COLORS["primary"])
        ax2.set_ylabel("Risque Moyen (%)", color=COLORS["dark"])

    # Charge mensuelle
    if "month" in df2.columns:
        monthly = df2.groupby("month")["total_load_mw"].agg(["mean", "max", "min"])
        axes[1, 0].fill_between(monthly.index, monthly["min"], monthly["max"],
                                alpha=0.3, color=COLORS["secondary"])
        axes[1, 0].plot(monthly.index, monthly["mean"], color=COLORS["secondary"],
                        linewidth=2, marker="o")
        axes[1, 0].set_title("Charge par Mois (min/moy/max)")
        axes[1, 0].set_xlabel("Mois")
        axes[1, 0].set_ylabel("MW")
        axes[1, 0].set_xticks(range(1, 13))

    # Pic vs hors-pic
    if "is_peak_hour" in df2.columns:
        peak_data    = [df2[df2["is_peak_hour"] == 1]["total_load_mw"],
                        df2[df2["is_peak_hour"] == 0]["total_load_mw"]]
        axes[1, 1].boxplot(peak_data, labels=["Heures de Pointe", "Hors Pointe"], patch_artist=True,
                           boxprops=dict(facecolor=COLORS["accent"], alpha=0.7))
        axes[1, 1].set_title("Charge : Pointe vs Hors-Pointe")
        axes[1, 1].set_ylabel("MW")

    # Délestage par région
    if all(c in df2.columns for c in ["region", "load_shedding_mw"]):
        region_shed = df2.groupby("region")["load_shedding_mw"].sum().sort_values(ascending=True)
        region_shed.plot.barh(ax=axes[2, 0], color=PALETTE[:len(region_shed)])
        axes[2, 0].set_title("Délestage Total par Région (MW)")
        axes[2, 0].set_xlabel("MW délestés")

    # Évolution du prix de l'énergie
    if all(c in df2.columns for c in ["datetime", "energy_price"]):
        price_ts = df2.set_index("datetime")["energy_price"].resample("W").mean()
        axes[2, 1].plot(price_ts.index, price_ts.values, color=COLORS["accent"], linewidth=1.5)
        axes[2, 1].set_title("Prix de l'Énergie (moyenne hebdomadaire)")
        axes[2, 1].set_xlabel("Date")
        axes[2, 1].set_ylabel("Prix (FCFA/kWh)")

    plt.tight_layout()
    return save_fig("03_temporal_analysis", fig)


# ─── 4. Analyse des événements ───────────────────────────────────────────────
def plot_event_analysis(df: pd.DataFrame):
    if "event" not in df.columns:
        logger.warning("Colonne 'event' absente")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Impact des Événements sur le Réseau Électrique", fontsize=14, fontweight="bold")

    event_stats = df.groupby("event").agg(
        avg_load=("total_load_mw", "mean"),
        avg_risk=("outage_risk", "mean"),
        blackout_rate=("blackout", "mean"),
    ).sort_values("avg_risk", ascending=False)

    event_stats["avg_load"].plot.bar(ax=axes[0], color=COLORS["secondary"], alpha=0.8)
    axes[0].set_title("Charge Moyenne par Événement")
    axes[0].set_ylabel("MW")
    axes[0].tick_params(axis="x", rotation=20)

    event_stats["avg_risk"].plot.bar(ax=axes[1], color=COLORS["warning"], alpha=0.8)
    axes[1].set_title("Risque Moyen par Événement")
    axes[1].set_ylabel("Risque (%)")
    axes[1].tick_params(axis="x", rotation=20)

    (event_stats["blackout_rate"] * 100).plot.bar(ax=axes[2], color=COLORS["primary"], alpha=0.8)
    axes[2].set_title("Taux de Blackout par Événement (%)")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    return save_fig("04_event_analysis", fig)


# ─── 5. Feature Importance ───────────────────────────────────────────────────
def plot_feature_importance(fi_df: pd.DataFrame, title: str = "Feature Importance"):
    fig, ax = plt.subplots(figsize=(10, 8))
    fi_sorted = fi_df.sort_values("importance")
    bars = ax.barh(fi_sorted["feature"], fi_sorted["importance"],
                   color=COLORS["secondary"], alpha=0.8, edgecolor="white")
    # Colorier le top 3
    top3 = fi_sorted.tail(3).index
    for i, bar in enumerate(bars):
        if fi_sorted.index[i] in top3:
            bar.set_color(COLORS["primary"])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return save_fig("05_feature_importance", fig)


# ─── 6. Matrice de confusion ──────────────────────────────────────────────────
def plot_confusion_matrix(cm: list, labels: list = None, title: str = "Matrice de Confusion"):
    import numpy as np
    cm_arr = np.array(cm)
    labels = labels or [str(i) for i in range(cm_arr.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, cbar=False)
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return save_fig("06_confusion_matrix", fig)


# ─── 7. Prévisions vs réel ────────────────────────────────────────────────────
def plot_forecast_vs_actual(actual: pd.Series, forecast: pd.Series,
                            train: pd.Series = None, title: str = "Prévision vs Réel"):
    fig, ax = plt.subplots(figsize=(14, 5))
    if train is not None:
        ax.plot(train.index, train.values, color="gray", linewidth=1, label="Historique", alpha=0.6)
    ax.plot(actual.index,   actual.values,   color=COLORS["dark"],     linewidth=2, label="Réel")
    ax.plot(forecast.index, forecast.values, color=COLORS["primary"],  linewidth=2,
            linestyle="--", label="Prévision")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Charge (MW)")
    ax.legend()
    plt.tight_layout()
    return save_fig("07_forecast_vs_actual", fig)


# ─── 8. Carte de risque par région ───────────────────────────────────────────
def plot_region_risk_map(df: pd.DataFrame):
    """Heatmap géographique simplifiée par région."""
    if "region" not in df.columns:
        return
    region_stats = df.groupby("region").agg(
        avg_risk=("outage_risk", "mean"),
        blackouts=("blackout", "sum"),
        overloads=("overload", "sum"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Analyse du Risque par Région – Togo", fontsize=14, fontweight="bold")

    for ax, col, label, color in zip(
        axes,
        ["avg_risk", "blackouts", "overloads"],
        ["Risque Moyen (%)", "Nb Blackouts", "Nb Surcharges"],
        [COLORS["warning"], COLORS["primary"], COLORS["accent"]]
    ):
        bars = ax.bar(region_stats["region"], region_stats[col], color=color, alpha=0.85, edgecolor="white")
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
        # Annoter les barres
        for bar, val in zip(bars, region_stats[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return save_fig("08_region_risk_map", fig)


# ─── 9. Distribution des anomalies ────────────────────────────────────────────
def plot_anomaly_distribution(df: pd.DataFrame):
    if "anomaly_risk_score" not in df.columns:
        logger.warning("Colonne 'anomaly_risk_score' absente — lancez anomaly_detection.py")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribution des Anomalies Réseau", fontsize=13, fontweight="bold")

    axes[0].hist(df["anomaly_risk_score"], bins=60,
                 color=COLORS["secondary"], edgecolor="white", alpha=0.8)
    axes[0].axvline(df["anomaly_risk_score"].quantile(0.95),
                    color=COLORS["primary"], linestyle="--", label="Seuil 95e")
    axes[0].set_title("Distribution du Score d'Anomalie")
    axes[0].set_xlabel("Score (0-100)")
    axes[0].legend()

    if "region" in df.columns:
        region_anom = df.groupby("region")["is_anomaly"].mean() * 100
        region_anom.sort_values().plot.barh(ax=axes[1], color=COLORS["warning"], alpha=0.85)
        axes[1].set_title("Taux d'Anomalies par Région (%)")
        axes[1].set_xlabel("%")

    plt.tight_layout()
    return save_fig("09_anomaly_distribution", fig)


# ─── Pipeline EDA complet ─────────────────────────────────────────────────────
def run_full_eda(df: pd.DataFrame) -> list:
    """Génère toutes les visualisations EDA."""
    logger.info("=" * 60)
    logger.info("GÉNÉRATION DES VISUALISATIONS EDA")
    logger.info("=" * 60)
    paths = []
    for func, name in [
        (plot_overview,          "overview"),
        (plot_correlation_heatmap, "correlation"),
        (plot_temporal_analysis, "temporal"),
        (plot_event_analysis,    "events"),
        (plot_region_risk_map,   "region_risk"),
    ]:
        try:
            path = func(df)
            if path:
                paths.append(path)
        except Exception as e:
            logger.error(f"Erreur {name}: {e}")
    logger.info(f"EDA terminé : {len(paths)} figures générées")
    return paths


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import DATA_PROC

    proc = DATA_PROC / "ceet_processed.csv"
    if proc.exists():
        df = pd.read_csv(proc, parse_dates=["datetime"])
        paths = run_full_eda(df)
        print(f"\n{len(paths)} figures générées dans {FIGURES}")
    else:
        print("Lancez data_preprocessing.py d'abord")
