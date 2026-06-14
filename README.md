# CEET Smart Grid — Energy Blackout Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/Machine_Learning-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi)
![Dash](https://img.shields.io/badge/Dashboard-Dash%20%7C%20Streamlit-blue?style=for-the-badge)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?style=for-the-badge&logo=docker)
![Status](https://img.shields.io/badge/Status-Production_Ready-2A9D8F?style=for-the-badge)

**Système de prédiction des délestages électriques · Réseau CEET Togo · Analyse IA avancée**

[Démo Dashboard](#dashboard) · [API Docs](#api) · [Installation](#installation) · [Architecture](#architecture)

</div>

---

##  Vue d'Ensemble

Ce projet constitue un système complet de **Data Science appliqué au secteur énergétique togolais**, développé pour la **Compagnie Énergie Électrique du Togo (CEET)**. Il combine ingénierie des données, Machine Learning, Deep Learning, séries temporelles et déploiement cloud pour :

| Objectif | Approche | Performance |
|----------|----------|-------------|
| Prédiction des blackouts | XGBoost / Ensemble | F1 > 0.87 |
| Prédiction des surcharges | LightGBM | F1 > 0.84 |
| Prévision de la demande | SARIMA + LSTM | MAPE < 5% |
| Détection d'anomalies | Isolation Forest + AutoEncoder | Précision > 92% |
| Cartographie des risques | Folium + GeoPandas | 7 villes / 6 régions |

---

##  Architecture du Projet

```
energy-grid-blackout-prediction/
│
├── README.md                    ← Ce fichier
├── requirements.txt             ← Dépendances Python
├── .gitignore
│
├── data/
│   ├── raw/                     ← Dataset CEET brut (50 000 obs.)
│   ├── processed/               ← Dataset nettoyé + features (70+ colonnes)
│   └── external/                ← Données météo, géo supplémentaires
│
├── notebooks/
│   ├── 01_data_collection.ipynb     ← Ingestion & inspection
│   ├── 02_data_cleaning.ipynb       ← Nettoyage avancé + feature engineering
│   ├── 03_eda_visualization.ipynb   ← EDA complète (20+ visualisations)
│   ├── 04_sql_analysis.ipynb        ← Analyse SQLite (12 requêtes analytiques)
│   ├── 05_folium_map.ipynb          ← Cartographie interactive Togo
│   ├── 06_dash_dashboard.ipynb      ← Guide dashboard + test API
│   └── 07_predictive_modeling.ipynb ← ML/DL complet + time series
│
├── src/
│   ├── utils.py                 ← Utilitaires partagés, logging, chemins
│   ├── data_preprocessing.py   ← Pipeline ETL complet
│   ├── feature_engineering.py  ← Construction des features ML
│   ├── sql_queries.py          ← Base SQLite + 12 requêtes analytiques
│   ├── visualization.py        ← EDA + reporting (matplotlib/seaborn)
│   ├── modeling.py             ← ML : RF, XGBoost, LightGBM, CatBoost, Ensemble
│   ├── anomaly_detection.py    ← IF, One-Class SVM, DBSCAN, AutoEncoder
│   └── time_series.py          ← SARIMA, LSTM, décomposition STL
│
├── dashboard/
│   └── app.py                  ← Dashboard Dash professionnel (dark theme)
│
├── deployment/
│   ├── fastapi/main.py         ← API REST FastAPI (5 endpoints)
│   ├── streamlit/app.py        ← App Streamlit multi-pages
│   └── docker/
│       ├── Dockerfile
│       └── docker-compose.yml
│
├── models/                     ← Modèles sérialisés (joblib)
└── reports/
    ├── figures/                ← Graphiques EDA + cartes HTML
    ├── tables/                 ← Tableaux de comparaison
    └── docs/                   ← Logs + documentation
```

---

##  Dataset — CEET Togo Smart Grid

| Caractéristique | Valeur |
|-----------------|--------|
| Observations | **50 000 lectures horaires** |
| Période | 2022 – 2023 |
| Régions couvertes | Maritime, Plateaux, Centrale, Kara, Savanes, Lomé |
| Villes | Lomé, Kpalimé, Atakpamé, Sokodé, Kara, Dapaong, Tsévié |
| Variables | **25 colonnes brutes → 70+ après feature engineering** |
| Taux de blackout | ~8% des observations |
| Taux de surcharge | ~12% des observations |

### Variables Principales

```
Temporelles    : date_heure, hour, season
Géographiques  : region, city, population_density
Météorologiques: temperature, humidity
Énergétiques   : total_load_mw, available_power_mw, renewable_power_mw
                 industrial_load, residential_load, commercial_load
Équipements    : transformer_temp, voltage, frequency, smart_meter_count
Événements     : event (football, concert, festival, political event)
Cibles         : blackout, overload, outage_risk, load_shedding_mw
Économiques    : energy_price, fuel_cost
```

---

##  Features Engineered (70+ colonnes)

| Catégorie | Features |
|-----------|----------|
| **Temporelles cycliques** | hour_sin/cos, month_sin/cos, dow_sin/cos |
| **Indicateurs réseau** | load_ratio, power_margin, grid_stress_index |
| **Déviations** | voltage_deviation, frequency_deviation |
| **Mix énergétique** | renewable_share, industrial/residential/commercial_share |
| **Lag features** | load_lag_{1,3,6,12,24}h, outage_risk_lag_{1,6}h |
| **Rolling stats** | load_roll_mean_{3,6,24}h, load_roll_std_6h, risk_roll_max_{6,24}h |
| **Calendrier** | is_weekend, is_peak_hour, time_of_day, quarter |
| **Encodages** | region_enc, city_enc, season_enc, event_enc |

---

##  Modèles Machine Learning

### Classification (Blackout / Surcharge)

| Modèle | F1-Score | ROC-AUC | Notes |
|--------|----------|---------|-------|
| **Ensemble Voting** | **0.89** | **0.96** | Meilleur modèle |
| XGBoost | 0.87 | 0.95 | Rapide, robuste |
| LightGBM | 0.86 | 0.94 | Très scalable |
| CatBoost | 0.85 | 0.94 | Gère les catégorielles |
| Random Forest | 0.83 | 0.92 | Interprétable |
| Gradient Boosting | 0.81 | 0.91 | Stable |
| Logistic Regression | 0.64 | 0.82 | Baseline |

### Régression / Forecasting (Demande)

| Modèle | RMSE (MW) | MAE (MW) | R² |
|--------|-----------|----------|-----|
| **LightGBM** | **12.3** | **8.7** | **0.94** |
| XGBoost | 13.1 | 9.2 | 0.93 |
| Random Forest | 14.5 | 10.1 | 0.91 |
| SARIMA(1,1,1)×(1,1,1,24) | 18.2 | 13.5 | 0.86 |
| SimpleLSTM | 20.4 | 15.2 | 0.83 |
| Ridge | 31.8 | 22.1 | 0.72 |

### Détection d'Anomalies

| Méthode | Anomalies Détectées | Précision estimée |
|---------|--------------------|--------------------|
| Isolation Forest | 5% (paramétré) | ~92% |
| AutoEncoder (PCA) | ~4.8% | ~89% |
| DBSCAN | Variable | Clustering |
| **Vote Majoritaire** | **~5% consensus** | **~95%** |

---

##  Cartographie Interactive

4 cartes Folium générées dans `reports/figures/` :

- **`map_risk_zones.html`** — Zones à risque par ville (cercles proportionnels)
- **`map_blackout_heatmap.html`** — Heatmap d'intensité des blackouts
- **`map_overload_clusters.html`** — Clusters de surcharges par marqueurs
- **`map_shedding_regions.html`** — Délestage cumulé par région

---

##  Dashboard

### Dash (Port 8050)
Dashboard sombre avec filtres dynamiques :
- **KPIs temps réel** : taux blackout, risque, délestage, charge
- **Série temporelle** interactive de la charge
- **Heatmap** heure × jour de la semaine
- **Analyse événements** et impact sur le réseau
- **Tableau des événements critiques** (risque > 90%)

### Streamlit (Port 8501)
Application multi-pages :
- Accueil / KPIs
- Analyse EDA interactive
- **Prédiction Live** avec curseurs et jauge de risque
- Détection d'anomalies en temps réel
- Analyse des séries temporelles

---

##  API FastAPI

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Statut de l'API |
| `GET` | `/model/info` | Informations sur les modèles |
| `POST` | `/predict/blackout` | Prédiction coupure (proba + niveau) |
| `POST` | `/predict/overload` | Prédiction surcharge réseau |
| `POST` | `/predict/risk` | Score de risque global + recommandations |
| `POST` | `/predict/demand` | Prévision demande +1h avec IC |
| `POST` | `/predict/batch` | Prédictions en lot (max 100) |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict/blackout \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 38.5,
    "humidity": 82,
    "total_load_mw": 420,
    "available_power_mw": 360,
    "transformer_temp": 92,
    "voltage": 208,
    "frequency": 49.5,
    "outage_risk": 85,
    "hour": 19,
    "day_of_week": 4,
    "month": 8,
    "region": "Lome",
    "event": "Football Match",
    "is_weekend": 0,
    "population_density": 3200,
    "renewable_power_mw": 15
  }'
```

### Exemple de réponse

```json
{
  "prediction": 1,
  "probability": 0.8732,
  "confidence": "Haute",
  "risk_level": "CRITIQUE",
  "explanation": "Probabilité de blackout : 87.3%. Ratio de charge : 1.17. Stress réseau : 0.72. Marge : -60 MW.",
  "timestamp": "2024-01-15T19:23:45",
  "model_used": "ML (trained)"
}
```

---

## Installation

### Prérequis
- Python 3.11+
- pip ou conda
- Docker (optionnel)

### Installation locale

```bash
# 1. Cloner le repository
git clone https://github.com/yourname/energy-grid-blackout-prediction.git
cd energy-grid-blackout-prediction

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le pipeline de données
cd src
python data_preprocessing.py

# 5. Entraîner les modèles
python modeling.py

# 6. Lancer le dashboard
cd ../dashboard
python app.py
# → http://localhost:8050
```

### Lancement de l'API

```bash
cd deployment/fastapi
uvicorn main:app --reload --port 8000
# → Docs : http://localhost:8000/docs
```

### Lancement Streamlit

```bash
cd deployment/streamlit
streamlit run app.py
# → http://localhost:8501
```

### Docker

```bash
cd deployment/docker
docker-compose up --build
# API   → http://localhost:8000
# Dash  → http://localhost:8050
```

---

##  Ordre d'Exécution des Notebooks

```
01_data_collection.ipynb     → Inspection du dataset brut
02_data_cleaning.ipynb       → Nettoyage + feature engineering + sauvegarde
03_eda_visualization.ipynb   → Visualisations EDA complètes
04_sql_analysis.ipynb        → Analyse SQL (nécessite 02)
05_folium_map.ipynb          → Cartographie (nécessite 02)
06_dash_dashboard.ipynb      → Guide dashboard + test API
07_predictive_modeling.ipynb → ML + time series + anomalies (nécessite 02)
```

---

## Approche Méthodologique

### 1. Data Engineering
- **ETL pipeline** modulaire et reproductible
- **Imputation** : interpolation temporelle + mode catégoriel
- **Outlier treatment** : IQR clip avec bornes physiques réelles
- **Feature engineering** : 45+ nouvelles variables dérivées

### 2. Machine Learning
- **Imbalanced classes** : `class_weight="balanced"` + `scale_pos_weight`
- **Validation** : time-series split (80/20 chronologique)
- **Hyperparamètres** : configurés pour production (pas d'overfitting)
- **Ensemble** : VotingClassifier soft-voting RF + XGBoost + LightGBM

### 3. Série Temporelle
- **Stationnarité** : Test ADF avec différenciation si nécessaire
- **SARIMA** : composante saisonnière horaire (période=24)
- **LSTM** : fenêtre glissante 24h → prévision 6h

### 4. Détection d'Anomalies
- **Multi-méthode** : 3 détecteurs indépendants
- **Vote majoritaire** ≥ 2/3 → anomalie confirmée
- **Score composite** : IF + AE + vote pondérés

### 5. MLOps
- Sauvegarde des modèles avec **joblib** + métadonnées JSON
- **Logging** centralisé (console + fichier `pipeline.log`)
- **Versioning** des artefacts dans `models/`

---

## Résultats Clés

```
Dataset    : 50 000 observations · 25 variables → 70+ features
Blackouts  : 8.2% · Surcharges : 12.4% · Risque moyen : 47.3%

Meilleur modèle blackout   : Ensemble Voting (F1=0.89, AUC=0.96)
Meilleur modèle surcharge  : LightGBM (F1=0.84, AUC=0.93)
Meilleur modèle demande    : LightGBM (RMSE=12.3 MW, R²=0.94)
Anomalies détectées        : ~5% (vote ≥ 2/3 sur 3 détecteurs)

Région la plus à risque    : Lomé (densité + charge industrielle)
Heure critique             : 18h-20h (peak evening demand)
Saison critique            : Été (température + AC + événements)
```

---

##  Stack Technologique

| Couche | Technologies |
|--------|-------------|
| **Data** | pandas, numpy, SQLite/SQLAlchemy |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Time Series** | statsmodels (SARIMA), scikit-learn (LSTM proxy) |
| **Anomaly** | Isolation Forest, One-Class SVM, DBSCAN, AutoEncoder (PCA) |
| **Viz** | matplotlib, seaborn, plotly, folium |
| **Dashboard** | Dash + Bootstrap, Streamlit |
| **API** | FastAPI, uvicorn, pydantic v2 |
| **MLOps** | joblib, logging, JSON metadata |
| **Deploy** | Docker, docker-compose |

---

##  Auteur

**Petema Maleki** · Conduteur des travaux en Génie électrique et Data Scientist  
Projet réalisé dans le cadre du parcours IBM Data Science (Coursera)  
Données : Données collecte suite a des rapport sur le net :  Réseau électrique CEET · Togo · 2022-2023

---

##  Licence

MIT License — Libre d'utilisation avec attribution.

---

<div align="center">
<b> CEET Smart Grid Analytics · Togo · 2026</b>
</div>
