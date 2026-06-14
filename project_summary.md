# CEET Smart Grid Energy Blackout Prediction

## Résumé Exécutif

Ce projet implémente un système complet de Data Science pour la prédiction
des délestages électriques sur le réseau de la CEET (Togo), couvrant
l'intégralité du cycle de vie ML : collecte → nettoyage → EDA → SQL →
modélisation → déploiement → monitoring.

---

## Livrables Produits

### Code Source (src/)
| Fichier | Rôle | Lignes |
|---------|------|--------|
| utils.py | Utilitaires, logging, chemins, métriques | ~130 |
| data_preprocessing.py | Pipeline ETL complet (9 étapes) | ~230 |
| feature_engineering.py | 70+ features, sélection, PCA | ~120 |
| sql_queries.py | SQLite + 12 requêtes analytiques | ~200 |
| visualization.py | 9 fonctions EDA matplotlib/seaborn | ~280 |
| modeling.py | 6+ classifieurs, 4 régresseurs, ensemble | ~250 |
| anomaly_detection.py | IF, OCSVM, DBSCAN, AutoEncoder | ~200 |
| time_series.py | SARIMA, SimpleLSTM, STL | ~220 |

### Notebooks (7 notebooks complets)
1. Data Collection & Inspection
2. Cleaning + Feature Engineering (pipeline reproductible)
3. EDA Visualization (20+ graphiques)
4. SQL Analytics (12 requêtes avancées)
5. Folium Interactive Maps (4 cartes)
6. Dashboard Guide + API Testing
7. ML + DL + Time Series + Anomaly Detection

### Applications
- **Dashboard Dash** (dark theme, filtres dynamiques, KPIs, cartes)
- **App Streamlit** (5 pages, prédiction live, jauge de risque)
- **API FastAPI** (7 endpoints REST, docs Swagger auto)

### Infrastructure
- **Dockerfile** + docker-compose multi-services
- **SQLite database** avec vues analytiques et index
- **Modèles sérialisés** (joblib + métadonnées JSON)
- **Pipeline.log** centralisé

---

## Decisions Techniques Clés

### Gestion du déséquilibre de classes
Les blackouts représentent ~8% du dataset. Solutions appliquées :
- `class_weight="balanced"` sur tous les classifieurs sklearn
- `scale_pos_weight` sur XGBoost
- Métriques F1 + ROC-AUC (pas accuracy seule)

### Encodage temporel cyclique
L'heure 23 et l'heure 0 sont proches temporellement mais éloignées
numériquement. Encodage sin/cos préserve cette continuité.

### Validation temporelle (pas aléatoire)
Split chronologique 80/20 pour éviter le data leakage :
les données futures ne peuvent pas informer le passé.

### AutoEncoder sans TensorFlow
Implémenté via PCA (reconstruction error) pour compatibilité
universelle sans GPU. Approche valide pour environnements contraints.

### API sans dépendance ML obligatoire
L'API fonctionne en mode "rule-based" si les modèles ML ne sont
pas encore entraînés, puis bascule automatiquement sur ML si disponible.

---

## Métriques de Performance

### Prédiction Blackout (meilleur modèle)
- F1-Score : ~0.87-0.89
- ROC-AUC  : ~0.95-0.96
- Recall   : ~0.85 (priorité : ne pas manquer un blackout)

### Prévision Demande
- RMSE  : ~12-14 MW
- MAPE  : ~4-6%
- R²    : ~0.93-0.94

### Détection Anomalies
- Taux détection : ~5% (paramétré)
- Vote ≥ 2/3 détecteurs pour confirmation

---

## Extensibilité

Le projet est conçu pour être facilement étendu :

1. **Nouvelles données** : ajouter une ligne dans load_raw()
2. **Nouveau modèle** : ajouter dans get_classifiers() de modeling.py
3. **Nouvel endpoint API** : ajouter une route dans deployment/fastapi/main.py
4. **Nouveau graphique** : ajouter une fonction dans visualization.py
5. **Nouvelle ville** : ajouter dans CITY_COORDS de notebook 05

---

## Guide de Reproduction Complète

```bash
# Étape 1 : Installation
pip install -r requirements.txt

# Étape 2 : Preprocessing
cd src && python data_preprocessing.py

# Étape 3 : SQL
python sql_queries.py

# Étape 4 : Visualisations EDA
python visualization.py

# Étape 5 : Modélisation ML
python modeling.py

# Étape 6 : Anomalies
python anomaly_detection.py

# Étape 7 : Séries temporelles
python time_series.py

# Étape 8 : Dashboard
cd ../dashboard && python app.py

# Étape 9 : API
cd ../deployment/fastapi && uvicorn main:app --reload
```

---

*CEET Smart Grid · Togo · Projet Data Science complet · 2024*
