"""
Prétraitement pour le jeu de données de planification de rendez-vous médicaux.

Usage :
python src/data/make_dataset.py --input data/raw/appointments_filtre.csv --output data/processed/final.csv --target status
"""

# Compatibilité avec Python 3.7+ pour les annotations de type
from __future__ import annotations

import argparse  # pour gérer les arguments de ligne de commande
from pathlib import Path  # pour gérer les chemins de fichiers
import pandas as pd  # pour manipuler les données
import numpy as np  # pour les calculs numériques

from sklearn.model_selection import train_test_split  # non utilisé ici mais souvent utile pour ML

# ---------------------------------------------------
# FONCTIONS UTILES POUR LE PRÉTRAITEMENT
# ---------------------------------------------------

def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les noms de colonnes :
    - supprime les espaces avant/après
    - met en minuscules
    - remplace les espaces par des underscores
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tente de convertir en datetime les colonnes contenant 'date', 'time' ou se terminant par '_at'.
    Ignore les erreurs si la conversion échoue.
    """
    df = df.copy()
    for c in df.columns:
        if "date" in c or "time" in c or c.endswith("_at"):
            try:
                df[c] = pd.to_datetime(df[c], errors="ignore")
            except Exception:
                pass
    return df


def _remove_high_leakage_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Supprime les colonnes susceptibles de créer un leakage (fuites de données).
    Ne supprime pas la colonne cible.
    """
    leakage_like = {
        "appointment_outcome",
        "final_status",
        "status_after",
        "was_no_show",
        "attended",
        "attendance",
        "checkin_time",
        "checkout_time",
    }
    # Garder seulement les colonnes présentes dans le DataFrame et exclure la cible
    cols = [c for c in df.columns if c in leakage_like and c != target]
    return df.drop(columns=cols, errors="ignore")


def preprocess(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Fonction principale de prétraitement :
    - standardisation des noms de colonnes
    - conversion des dates
    - suppression des doublons
    - suppression des colonnes 100% manquantes
    - suppression des colonnes de leakage
    - nettoyage et encodage de la colonne cible
    - conversion des colonnes booléennes
    - extraction de features à partir des dates
    """
    # Standardiser les noms de colonnes
    df = _standardize_column_names(df)
    
    # Convertir les colonnes dates
    df = _try_parse_dates(df)

    # Supprimer les doublons
    df = df.drop_duplicates()

    # Supprimer les colonnes avec 100% de valeurs manquantes
    df = df.dropna(axis=1, how="all")

    # Supprimer les colonnes susceptibles de créer un leakage
    df = _remove_high_leakage_columns(df, target)

    # Vérification de la présence de la colonne cible
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    # Conserver seulement les lignes où la cible n'est pas manquante
    df = df[df[target].notna()].copy()

    # Conversion des colonnes "booléennes-like" (true/false, yes/no, 1/0)
    for c in df.columns:
        if df[c].dtype == object:
            vals = df[c].dropna().astype(str).str.lower().unique()[:50]
            if set(vals).issubset({"true","false","0","1","yes","no"}):
                df[c] = df[c].astype(str).str.lower().map({"true":1,"false":0,"1":1,"0":0,"yes":1,"no":0})

    # Feature engineering pour les colonnes datetime
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df[f"{c}_year"] = df[c].dt.year       # année
            df[f"{c}_month"] = df[c].dt.month     # mois
            df[f"{c}_day"] = df[c].dt.day         # jour
            df[f"{c}_dow"] = df[c].dt.dayofweek   # jour de la semaine
            df[f"{c}_hour"] = df[c].dt.hour       # heure
            # Supprimer la colonne datetime brute (souvent inutile pour sklearn)
            df = df.drop(columns=[c])

    return df

# ---------------------------------------------------
# MAIN : exécution via la ligne de commande
# ---------------------------------------------------

def main() -> None:
    # Gestion des arguments de la ligne de commande
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Chemin vers le CSV brut")
    p.add_argument("--output", required=True, help="Chemin pour sauvegarder final.csv")
    p.add_argument("--target", default="target", help="Nom de la colonne cible (défaut: target)")
    args = p.parse_args()

    # Conversion en objets Path
    in_path = Path(args.input)
    out_path = Path(args.output)
    # Création du dossier de sortie si nécessaire
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Chargement du CSV
    df = pd.read_csv(in_path)

    # Prétraitement
    df = preprocess(df, target=args.target)

    # Sauvegarde du CSV final
    df.to_csv(out_path, index=False)
    print(f"Saved processed dataset to: {out_path} (shape={df.shape})")


# Point d'entrée si le script est exécuté directement
if __name__ == "__main__":
    main()
