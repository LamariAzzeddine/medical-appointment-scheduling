"""
Expliquer un modèle entraîné en utilisant SHAP.

Usage :
    python -m src.models.explain --data data/processed/final.csv --target status --run-id <MLFLOW_RUN_ID>
    
<MLFLOW_RUN_ID> : ID du BEST_MODEL

"""

# Compatibilité avec les annotations de type Python 3.7+
from __future__ import annotations

import argparse  # pour gérer les arguments de la ligne de commande
from pathlib import Path  # pour gérer les chemins de fichiers
import pandas as pd  # pour manipuler les données
import numpy as np  # pour les calculs numériques
import joblib  # pour charger un modèle sauvegardé
import shap  # pour l'explicabilité des modèles
import matplotlib.pyplot as plt  # pour tracer les graphiques

import mlflow  # pour récupérer les modèles sauvegardés via MLflow

# ---------------------------------------------------
# MAIN : exécution via la ligne de commande
# ---------------------------------------------------

def main() -> None:
    # Gestion des arguments de la ligne de commande
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Chemin vers le CSV prétraité")
    p.add_argument("--target", required=True, help="Nom de la colonne cible")
    p.add_argument("--run-id", required=True, help="ID de la run MLflow contenant le modèle best_model.joblib ou artifact")
    p.add_argument("--outdir", default="reports", help="Dossier pour sauvegarder les graphiques SHAP")
    args = p.parse_args()

    # Créer le dossier de sortie si nécessaire
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Charger le CSV
    df = pd.read_csv(args.data)
    
    # Séparer la cible et les features
    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Si la cible n'est pas numérique, la convertir en binaire
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.lower()
        positive_like = {"1","true","yes","cancelled","did not attend"}  # valeurs considérées comme positives
        y = y.map(lambda v: 1 if v in positive_like else 0)

    # Télécharger le modèle sauvegardé depuis MLflow
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=args.run_id, artifact_path="best_model.joblib"
    )
    model = joblib.load(local_dir)

    # Pour la rapidité, expliquer sur un échantillon maximum de 1000 lignes
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)

    # Récupérer la matrice transformée et les noms des features
    preprocess = model.named_steps["preprocess"]  # pipeline de prétraitement
    X_trans = preprocess.transform(X_sample)
    try:
        feature_names = preprocess.get_feature_names_out()  # noms des features après transformation
    except Exception:
        # Si impossible, créer des noms génériques
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    # Explicabilité avec SHAP
    estimator = model.named_steps["model"]  # modèle entraîné
    try:
        # Si c'est un arbre (RandomForest, XGBoost...), utiliser TreeExplainer
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans)
    except Exception:
        # Sinon fallback sur KernelExplainer
        background = shap.sample(X_trans, 100)
        explainer = shap.KernelExplainer(estimator.predict_proba, background)
        shap_values = explainer.shap_values(X_trans, nsamples=200)

    # -----------------------
    # Importance globale SHAP
    # -----------------------
    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    out1 = Path(args.outdir) / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)  # sauvegarde du graphique global
    plt.close()

    # -----------------------
    # Explication locale pour un exemple
    # -----------------------
    idx = 0  # index de l'exemple à expliquer
    try:
        plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X_trans[idx],
                feature_names=feature_names
            ),
            show=False
        )
        out2 = Path(args.outdir) / "shap_local_waterfall.png"
        plt.tight_layout()
        plt.savefig(out2, dpi=200)  # sauvegarde du graphique local
        plt.close()
    except Exception:
        out2 = None  # si échec, ne rien sauvegarder

    # Affichage des fichiers générés
    print(f"Saved global SHAP plot to {out1}")
    if out2:
        print(f"Saved local SHAP plot to {out2}")


# Point d'entrée si le script est exécuté directement
if __name__ == "__main__":
    main()
