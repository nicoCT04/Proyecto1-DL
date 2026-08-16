# Proyecto 1 — Competencia de Modelación (MLP)

**CC3092 · Deep Learning y Sistemas Inteligentes** · Universidad del Valle de Guatemala

Predicción del precio de venta de viviendas (**`SalePrice`**) sobre el dataset
**House Prices / Ames** mediante un **Multi-Layer Perceptron (MLP)** en PyTorch.
La métrica objetivo de la competencia es el **RMSE** sobre un dataset de prueba
*held-out*.

## Contenido

| Ruta | Descripción |
|------|-------------|
| `notebook/Proyecto1_MLP_HousePrices.ipynb` | Notebook completo: EDA, preprocesamiento, búsqueda de hiperparámetros (iteraciones documentadas), entrenamiento y evaluación del MLP. |
| `mlp_core.py` | Núcleo reutilizable: definición del MLP, construcción del preprocesamiento e inferencia. Importado por el notebook y por `predict.py` para garantizar que train = inferencia. |
| `predict.py` | Script de inferencia: consume un CSV de prueba y escribe `output.csv` con formato `Id,Prediction`. |
| `data/train.csv` | Dataset de entrenamiento entregado. |
| `artifacts/` | Modelo entrenado (`model.pt`) y pipeline de preprocesamiento (`preprocessor.joblib`). |
| `archivo_prueba/` | Archivo de muestra del profesor (`pipeline_test.csv`) y formato de salida esperado (`expected_output.csv`). |
| `reports/` | Informe escrito (no versionado; lo sube el autor). |

## Reproducir

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab   # abrir notebook/Proyecto1_MLP_HousePrices.ipynb y ejecutar de arriba a abajo
```

Ejecutar el notebook completo entrena el modelo y **regenera los artefactos** en
`artifacts/`.

## Generar predicciones sobre un dataset de prueba

Con los artefactos ya presentes en `artifacts/`:

```bash
python predict.py --input archivo_prueba/pipeline_test.csv --output output.csv
```

El script carga el preprocesamiento y el modelo entrenados, aplica **exactamente
el mismo pipeline** usado en entrenamiento y escribe `output.csv` con las columnas
`Id,Prediction`.

## Detalles del enfoque

- **Objetivo transformado:** el modelo se entrena sobre `log1p(SalePrice)` (la
  variable es sesgada a la derecha) y las predicciones se invierten con `expm1`.
- **Preprocesamiento:** `ColumnTransformer` de scikit-learn — imputación por
  mediana + `StandardScaler` para numéricas; imputación por `"None"` +
  `OneHotEncoder(handle_unknown="ignore")` para categóricas. Se fitea **solo con
  train** y se serializa en `artifacts/preprocessor.joblib`.
- Las categóricas cuyo `NA` significa "no aplica" (piscina, callejón, sótano,
  garaje, etc.) se rellenan con la categoría explícita `"None"` antes de codificar.

## Autor

Nicolás Concuá
