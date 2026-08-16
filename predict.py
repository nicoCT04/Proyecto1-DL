#!/usr/bin/env python3
"""
predict.py — Genera predicciones de SalePrice para un dataset de prueba.

Carga el preprocesamiento y el modelo ya entrenados desde `artifacts/`, consume
un CSV con la misma estructura que `data/train.csv` (pero sin la columna
`SalePrice`), y escribe un CSV con el formato exigido: dos columnas `Id,Prediction`.

Uso:
    python predict.py --input archivo_prueba/pipeline_test.csv --output output.csv

El pipeline aplicado es EXACTAMENTE el mismo que en entrenamiento (mismo
ColumnTransformer serializado, misma arquitectura), tal como pide el enunciado.

Autor: Nicolás Concuá
"""
from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd

import mlp_core

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRE = os.path.join(HERE, "artifacts", "preprocessor.joblib")
DEFAULT_MODEL = os.path.join(HERE, "artifacts", "model.pt")


def main():
    ap = argparse.ArgumentParser(description="Predicción de SalePrice (MLP).")
    ap.add_argument("--input", required=True, help="CSV de prueba (con columna Id, sin SalePrice).")
    ap.add_argument("--output", default="output.csv", help="Ruta del CSV de salida (Id,Prediction).")
    ap.add_argument("--preprocessor", default=DEFAULT_PRE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    for path in (args.preprocessor, args.model):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró {path}. Entrena primero el modelo en el notebook "
                "para generar los artefactos en artifacts/."
            )

    df = pd.read_csv(args.input)
    if mlp_core.ID_COL not in df.columns:
        raise ValueError(f"El CSV de entrada debe contener la columna '{mlp_core.ID_COL}'.")
    ids = df[mlp_core.ID_COL].values

    preprocessor = joblib.load(args.preprocessor)
    model = mlp_core.load_model_from_checkpoint(args.model)

    preds = mlp_core.predict_df(df, preprocessor, model)

    out = pd.DataFrame({"Id": ids, "Prediction": preds})
    out.to_csv(args.output, index=False)
    print(f"[ok] {len(out)} predicciones escritas en {args.output}")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
