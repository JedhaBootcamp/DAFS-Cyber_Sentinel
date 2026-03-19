import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer # Ajout de l'imputeur
from sklearn.pipeline import Pipeline # Pour enchaîner les étapes

CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET = "class"

def preprocess(df: pd.DataFrame) -> Tuple:
    # 1. Séparation Cible/Features
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # Détection des colonnes
    num_cols = [c for c in X.columns if c not in CATEGORICAL]
    cat_cols = [c for c in CATEGORICAL if c in X.columns]

    # 2. Split avec stratification pour garder l'équilibre des classes 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Création de sous-pipelines pour plus de rigueur 
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), # Gère les NA 
        ("scaler", StandardScaler()) # Normalise les données 
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)) # Gère les catégories rares 
    ])

    # 4. Assemblage final avec ColumnTransformer
    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop"
    )

    return X_train, X_test, y_train, y_test, preproc