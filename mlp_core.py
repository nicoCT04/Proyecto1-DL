"""
mlp_core.py — Núcleo reutilizable del Proyecto 1 (MLP para House Prices / Ames).

Contiene la definición del modelo, la construcción del preprocesamiento y las
rutinas de entrenamiento/predicción. Tanto el notebook como `predict.py` importan
de aquí para garantizar que la arquitectura y el pipeline sean IDÉNTICOS en
entrenamiento y en inferencia (requisito explícito del proyecto).

Autor: Nicolás Concuá
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Columnas cuyo NA significa "no aplica / ausencia de la característica" y NO un
# valor faltante (según la documentación del dataset Ames). Se imputan con la
# categoría explícita "None" antes del OneHotEncoder.
# ---------------------------------------------------------------------------
NA_MEANS_NONE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]

TARGET = "SalePrice"
ID_COL = "Id"


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """MLP para regresión con BatchNorm + Dropout.

    La configuración (dims de entrada/ocultas, dropout) se serializa dentro del
    checkpoint, de modo que `predict.py` puede reconstruir la red sin conocer de
    antemano los hiperparámetros elegidos durante la búsqueda.
    """

    def __init__(self, input_dim: int, hidden_dims=(256, 128), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
def prefill_na_none(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena las categóricas 'NA = no aplica' con la categoría 'None'.

    Se aplica ANTES del ColumnTransformer, tanto en train como en test.
    """
    df = df.copy()
    for col in NA_MEANS_NONE:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("None")
    return df


def build_preprocessor(df: pd.DataFrame):
    """Construye (sin fittear) el ColumnTransformer para las features.

    - Numéricas: imputación por mediana + StandardScaler.
    - Categóricas: imputación por 'None' + OneHotEncoder(handle_unknown='ignore').

    `handle_unknown='ignore'` es clave: el dataset de prueba del día de la
    presentación puede traer categorías no vistas en entrenamiento.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    feats = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])
    num_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feats.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre, num_cols, cat_cols


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------
def predict_df(df_raw: pd.DataFrame, preprocessor, model, device="cpu") -> np.ndarray:
    """Aplica el pipeline completo a un DataFrame crudo y devuelve SalePrice.

    El modelo se entrena sobre log(SalePrice); aquí se invierte con expm1.
    """
    df = prefill_na_none(df_raw)
    feats = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])
    X = preprocessor.transform(feats)
    X = torch.tensor(np.asarray(X, dtype=np.float32), device=device)

    model.eval()
    with torch.no_grad():
        y_log = model(X).cpu().numpy()
    return np.expm1(y_log)


def load_model_from_checkpoint(path: str, device="cpu") -> MLP:
    """Reconstruye el MLP desde un checkpoint que incluye su propia config."""
    ckpt = torch.load(path, map_location=device)
    model = MLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=tuple(ckpt["hidden_dims"]),
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
