"""
Entraîner des modèles de base et un modèle optimisé avec GridSearchCV, suivi avec MLflow.

Usage :
python src/models/train.py --data data/processed/final.csv --target status
"""

# Compatibilité avec les annotations de type Python 3.7+
from __future__ import annotations

import argparse  # gestion des arguments de ligne de commande
from pathlib import Path  # manipulation des chemins de fichiers
import pandas as pd  # manipulation des données
import numpy as np
import joblib  # pour sauvegarder et charger les modèles

import mlflow  # suivi des expériences
import mlflow.sklearn  # intégration sklearn avec MLflow

# Librairies scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------
# FONCTION : construction du pipeline de prétraitement
# ---------------------------------------------------
def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    """
    Crée un pipeline de prétraitement :
    - Numériques : imputation par la médiane
    - Catégorielles : imputation par la valeur la plus fréquente + one-hot encoding
    """
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop"
    )
    return pre

# ---------------------------------------------------
# FONCTION : évaluation des modèles binaires
# ---------------------------------------------------
def eval_binary(y_true, y_pred, y_proba=None) -> dict:
    """
    Calcule les métriques classiques :
    accuracy, precision, recall, f1, roc_auc (si probabilités),
    matrice de confusion et score métier (moyenne de precision et recall)
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    # Score métier : moyenne de recall et precision (priorité à recall)
    out["metier_score"] = float((out["recall"] + out["precision"]) / 2.0)
    return out

# ---------------------------------------------------
# MAIN : exécution via la ligne de commande
# ---------------------------------------------------
def main() -> None:
    # Gestion des arguments
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Chemin vers le CSV prétraité (final.csv)")
    p.add_argument("--target", required=True, help="Nom de la colonne cible")
    p.add_argument("--experiment", default="medical-appointments", help="Nom de l'expérience MLflow")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--outdir", default="outputs", help="Dossier pour sauvegarder les modèles")
    args = p.parse_args()

    # Charger le CSV
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Columns: {list(df.columns)}")

    # Séparer features et cible
    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Si la cible n'est pas numérique, la convertir en binaire
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.lower()
        positive_like = {"1","true","yes","cancelled","did not attend"}
        y = y.map(lambda v: 1 if v in positive_like else 0)

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Pipeline de prétraitement
    preprocess = build_preprocess(X_train)

    # ---------------------------------------------------
    # Modèles de base (baseline)
    # ---------------------------------------------------
    baselines = {
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(random_state=args.random_state),
    }

    mlflow.set_experiment(args.experiment)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for name, clf in baselines.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])

        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.log_param("model", name)
            pipe.fit(X_train, y_train)

            # Prédictions
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

            # Calcul des métriques
            metrics = eval_binary(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            # Sauvegarde du modèle
            model_path = Path(args.outdir) / f"{name}.joblib"
            joblib.dump(pipe, model_path)
            mlflow.log_artifact(str(model_path))

    # ---------------------------------------------------
    # Modèle XGBoost optimisé avec GridSearchCV
    # ---------------------------------------------------
    xgb = XGBClassifier(
        random_state=args.random_state,
        eval_metric="logloss",
        n_estimators=300,
        tree_method="hist"
    )
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", xgb)])

    # Grille de paramètres pour GridSearch
    param_grid = {
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    # Score métier : moyenne precision + recall
    from sklearn.metrics import make_scorer
    def metier_scorer(y_true, y_pred):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        return (prec + rec) / 2.0

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=make_scorer(metier_scorer),
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    with mlflow.start_run(run_name="xgb_gridsearch"):
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("cv", "StratifiedKFold(n_splits=5)")

        # Entraînement GridSearch
        grid.fit(X_train, y_train)
        mlflow.log_params(grid.best_params_)

        # Meilleur modèle
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = eval_binary(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        # Sauvegarde du modèle
        model_path = Path(args.outdir) / "best_model.joblib"
        joblib.dump(best, model_path)
        mlflow.log_artifact(str(model_path))

        # Sauvegarde également dans MLflow
        mlflow.sklearn.log_model(best, artifact_path="model")

        print("Best params:", grid.best_params_)
        print("Test metrics:", metrics)

# Point d'entrée si le script est exécuté directement
if __name__ == "__main__":
    main()
