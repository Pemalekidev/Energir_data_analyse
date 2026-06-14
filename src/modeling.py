"""
modeling.py — Machine Learning : entraînement, évaluation, sauvegarde
CEET Smart Grid · Energy Blackout Prediction Project
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report
import xgboost  as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from utils import get_logger, save_model, load_model, classification_report_dict, regression_report_dict, MODELS_DIR
from feature_engineering import get_feature_matrix, BLACKOUT_FEATURES, OVERLOAD_FEATURES, DEMAND_FEATURES

logger = get_logger("modeling")


# ─── Définition des modèles ───────────────────────────────────────────────────
def get_classifiers() -> dict:
    """Retourne un dictionnaire de classifieurs configurés."""
    clfs = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, C=0.1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=5,  # gestion du déséquilibre
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1
        ),
    }
    if CATBOOST_AVAILABLE:
        clfs["catboost"] = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced", random_seed=42,
            verbose=False
        )
    return clfs


def get_regressors() -> dict:
    """Retourne des régresseurs pour la prédiction de charge."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest_reg": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "xgboost_reg": xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1
        ),
        "lightgbm_reg": lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1, verbose=-1
        ),
    }


# ─── Préparation des données ───────────────────────────────────────────────────
def prepare_data(df: pd.DataFrame, feature_list: list, target: str,
                 test_size: float = 0.2) -> tuple:
    """Prépare train/test split avec gestion des NaN."""
    X, y = get_feature_matrix(df, feature_list, target)
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() <= 10 else None
    )
    logger.info(f"Train : {X_train.shape}, Test : {X_test.shape}")
    logger.info(f"Distribution target (test) :\n{y_test.value_counts(normalize=True).round(3)}")
    return X_train, X_test, y_train, y_test


# ─── Entraînement et évaluation ────────────────────────────────────────────────
def train_evaluate_classifier(
    model, model_name: str,
    X_train, X_test, y_train, y_test,
    use_scaler: bool = False
) -> dict:
    """Entraîne un classifieur et retourne ses métriques."""
    if use_scaler:
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    else:
        pipeline = model

    logger.info(f"Entraînement : {model_name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = None

    metrics = classification_report_dict(y_test, y_pred, y_proba)
    metrics["model"] = model_name
    logger.info(
        f"  F1={metrics['f1_score']:.4f} | "
        f"ROC-AUC={metrics.get('roc_auc', 'N/A')} | "
        f"Recall={metrics['recall']:.4f}"
    )
    return metrics, pipeline


def train_evaluate_regressor(
    model, model_name: str,
    X_train, X_test, y_train, y_test
) -> dict:
    """Entraîne un régresseur et retourne ses métriques."""
    logger.info(f"Entraînement : {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = regression_report_dict(y_test, y_pred)
    metrics["model"] = model_name
    logger.info(f"  RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R²={metrics['r2']:.4f}")
    return metrics, model


# ─── Modèle d'ensemble ─────────────────────────────────────────────────────────
def build_ensemble(X_train, y_train) -> VotingClassifier:
    """Construit un VotingClassifier avec les meilleurs modèles."""
    estimators = [
        ("rf",  RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=100, scale_pos_weight=5, use_label_encoder=False,
                                   eval_metric="logloss", random_state=42, n_jobs=-1)),
        ("lgb", lgb.LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42, verbose=-1)),
    ]
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train, y_train)
    logger.info("Ensemble VotingClassifier entraîné")
    return ensemble


# ─── Feature importance ────────────────────────────────────────────────────────
def get_feature_importance(model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """Extrait l'importance des features d'un modèle arbre."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        logger.warning("Ce modèle ne fournit pas d'importance de features")
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_names[:len(imp)],
        "importance": imp
    }).sort_values("importance", ascending=False).head(top_n)
    return fi


# ─── Pipeline principal ────────────────────────────────────────────────────────
def run_blackout_modeling(df: pd.DataFrame) -> dict:
    """Pipeline complet de modélisation de la prédiction de blackout."""
    logger.info("=" * 60)
    logger.info("MODÉLISATION : PRÉDICTION BLACKOUT")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = prepare_data(df, BLACKOUT_FEATURES, "blackout")
    classifiers = get_classifiers()

    results = []
    best_f1 = 0
    best_model_name = None
    best_pipeline = None

    for name, clf in classifiers.items():
        use_scaler = name in ["logistic_regression", "svm"]
        metrics, pipeline = train_evaluate_classifier(
            clf, name, X_train, X_test, y_train, y_test, use_scaler=use_scaler
        )
        results.append(metrics)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = name
            best_pipeline = pipeline

    # Modèle d'ensemble
    ensemble = build_ensemble(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    y_proba_ens = ensemble.predict_proba(X_test)[:, 1]
    ens_metrics = classification_report_dict(y_test, y_pred_ens, y_proba_ens)
    ens_metrics["model"] = "ensemble_voting"
    results.append(ens_metrics)

    if ens_metrics["f1_score"] > best_f1:
        best_model_name = "ensemble_voting"
        best_pipeline = ensemble

    # Sauvegarde du meilleur modèle
    save_model(best_pipeline, "blackout_best_model", {
        "model_name": best_model_name,
        "f1_score": best_f1,
        "features": BLACKOUT_FEATURES
    })
    logger.info(f"\nMeilleur modèle : {best_model_name} (F1={best_f1:.4f})")

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    results_df.to_csv(MODELS_DIR.parent / "reports" / "tables" / "blackout_model_comparison.csv", index=False)
    return {"results": results_df, "best_model": best_pipeline, "best_name": best_model_name}


def run_overload_modeling(df: pd.DataFrame) -> dict:
    """Pipeline complet pour la prédiction de surcharge."""
    logger.info("=" * 60)
    logger.info("MODÉLISATION : PRÉDICTION SURCHARGE")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = prepare_data(df, OVERLOAD_FEATURES, "overload")
    classifiers = get_classifiers()

    results = []
    best_f1 = 0
    best_pipeline = None
    best_name = None

    for name, clf in classifiers.items():
        use_scaler = name == "logistic_regression"
        metrics, pipeline = train_evaluate_classifier(
            clf, name, X_train, X_test, y_train, y_test, use_scaler=use_scaler
        )
        results.append(metrics)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_pipeline = pipeline
            best_name = name

    save_model(best_pipeline, "overload_best_model", {
        "model_name": best_name,
        "f1_score": best_f1,
        "features": OVERLOAD_FEATURES
    })
    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    return {"results": results_df, "best_model": best_pipeline, "best_name": best_name}


def run_demand_forecasting(df: pd.DataFrame) -> dict:
    """Pipeline de forecasting de la demande énergétique."""
    logger.info("=" * 60)
    logger.info("MODÉLISATION : FORECASTING DE LA DEMANDE")
    logger.info("=" * 60)

    X, y = get_feature_matrix(df, DEMAND_FEATURES, "total_load_mw")
    X = X.fillna(0)
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    regressors = get_regressors()
    results = []
    best_rmse = float("inf")
    best_reg = None
    best_name = None

    for name, reg in regressors.items():
        metrics, model = train_evaluate_regressor(reg, name, X_train, X_test, y_train, y_test)
        results.append(metrics)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_reg = model
            best_name = name

    save_model(best_reg, "demand_forecast_model", {
        "model_name": best_name,
        "rmse": best_rmse,
        "features": DEMAND_FEATURES
    })
    results_df = pd.DataFrame(results).sort_values("rmse")
    return {"results": results_df, "best_model": best_reg, "best_name": best_name}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import DATA_PROC

    proc = DATA_PROC / "ceet_processed.csv"
    if not proc.exists():
        print("Lancez d'abord data_preprocessing.py")
    else:
        df = pd.read_csv(proc)
        res_blackout  = run_blackout_modeling(df)
        res_overload  = run_overload_modeling(df)
        res_demand    = run_demand_forecasting(df)
        print("\n=== RÉSULTATS BLACKOUT ===")
        print(res_blackout["results"][["model","f1_score","roc_auc","recall","precision"]].to_string(index=False))
