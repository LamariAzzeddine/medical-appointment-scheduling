# 📅 Medical Appointment Scheduling
## EDA, Machine Learning & MLflow

## 📌 Description

Ce projet est une preuve de concept complète en machine learning appliquée à un jeu de données de planification de rendez-vous médicaux.  
Il couvre l’ensemble du pipeline data science : analyse exploratoire, prétraitement, modélisation, optimisation des hyperparamètres, suivi des expériences avec MLflow et interprétabilité des modèles avec SHAP.

L’objectif principal est de prédire le statut d’un rendez-vous médical (honoré ou non) et d’identifier les facteurs influençant l’absence des patients.

---

## 🎯 Objectifs

- Analyser les données de rendez-vous médicaux
- Prédire le statut des rendez-vous
- Comparer plusieurs modèles de machine learning
- Suivre et reproduire les expériences avec MLflow
- Expliquer les prédictions du modèle avec SHAP

---

## 📊 Jeu de données

- Source : Kaggle  
  https://www.kaggle.com/datasets/carogonzalezgaltier/medical-appointment-scheduling-system

- Variable cible :
  - `status` : statut du rendez-vous 

📁 Le fichier principal doit être placé dans :
    - data/raw/appointments.csv

Avant de commencer, assurez-vous de **télécharger le fichier appointments.csv** depuis Kaggle et de le placer dans ce dossier.

---

## 🛠️ Technologies utilisées

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- MLflow
- SHAP
- Jupyter Notebook

---

## 🚀 Installation

```bash
git clone https://github.com/LamariAzzeddine/medical-appointment-scheduling.git
cd medical-appointment-scheduling


## Créer un environnement virtuel et l’activer
python -m venv .venv
.venv\Scripts\activate

## Installer les dépendances :
pip install -r requirements.txt



##📈 Analyse exploratoire (EDA)

    - Le notebook "notebooks/01_eda.ipynb" permet de :

        - Explorer la structure des données
        - Analyser la distribution de la variable cible
        - Identifier les valeurs manquantes
        - Détecter d’éventuelles fuites de données

Note : Pour de meilleurs résultats, il est recommandé d’exécuter les cellules du notebook pas à pas, dans l’ordre, afin de suivre correctement le flux d’analyse et les visualisations.


## 🔄 Prétraitement des données
python src/data/make_dataset.py --input data/raw/appointments_filtre.csv --output data/processed/final.csv --target status


Le fichier prétraité est généré dans :
  - data/processed/final.csv




## 🧠 Entraînement des modèles

### 1️⃣ Modeles de base (baselines)

| Nom | Type | Description |
|-----|------|-------------|
| `logreg` | Logistic Regression | Régression logistique simple, référence initiale. |
| `rf` | Random Forest | Modèle d’ensemble robuste, capture les interactions entre variables. |

- Entraînés **sans optimisation des hyperparamètres**  
- Fournissent un **benchmark initial**  
- Évalués avec : `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, matrice de confusion et **score métier** (moyenne précision + rappel)  

### 2️⃣ Modèle optimisé : XGBoost

| Nom | Type | Optimisation |
|-----|------|-------------|
| `xgb` | XGBoost (`XGBClassifier`) | Hyperparamètres optimisés avec `GridSearchCV` : `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` |

- Sélection du meilleur modèle via **score métier** `(precision + recall)/2`  
- Modèle final sauvegardé dans `outputs/best_model.joblib`  
- Suivi des métriques et du modèle avec **MLflow**  


Entraîner les modèles :
python src/models/train.py --data data/processed/final.csv --target status


Lancer l’interface MLflow :
    - mlflow ui


Les métriques, paramètres et modèles sont enregistrés dans MLflow:
    - http://localhost:5000/


## 🔍 Interprétabilité avec SHAP
    python -m src.models.explain --data data/processed/final.csv --target status --run-id <MLFLOW_RUN_ID>
    
<MLFLOW_RUN_ID> : ID du BEST_MODEL

SHAP global : importance des variables
SHAP local : explication des prédictions individuelles

- Les resultats de shap seront stockés dans le dossier "reports"

## 📁 Structure du projet
.
├── data
│   ├── raw/
│   └── processed/
├── notebooks
│   └── 01_eda.ipynb
├── src
│   ├── data
│   │   └── make_dataset.py
│   └── models
│       ├── train.py
│       └── explain.py
├── requirements.txt
└── README.md



