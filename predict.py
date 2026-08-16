#!/usr/bin/env python3

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

    # Parámetros de la transformación del target (log1p + estandarización)
    import torch
    ckpt = torch.load(args.model, map_location="cpu")
    y_log_mean = ckpt.get("y_log_mean", 0.0)
    y_log_std = ckpt.get("y_log_std", 1.0)

    preds = mlp_core.predict_df(df, preprocessor, model,
                                y_log_mean=y_log_mean, y_log_std=y_log_std)

    out = pd.DataFrame({"Id": ids, "Prediction": preds})
    out.to_csv(args.output, index=False)
    print(f"[ok] {len(out)} predicciones escritas en {args.output}")
    print(out.head().to_string(index=False))



main()
