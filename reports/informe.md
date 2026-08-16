# Proyecto 1 — Competencia de Modelación (MLP)
### Predicción de precios de vivienda (House Prices / Ames) con un Multi-Layer Perceptron

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Autor:** Nicolás Concuá

> Borrador de trabajo. Todas las cifras y figuras provienen de la ejecución del
> notebook `notebook/Proyecto1_MLP_HousePrices.ipynb`. Las figuras están en
> `reports/figures/`. Convertir a `.docx`/`.pdf` para la entrega del 21 de agosto.

---

## 1. Análisis exploratorio de datos (EDA)

**Dimensiones y tipos.** El dataset de entrenamiento tiene **1,168 observaciones** y
**81 columnas**: `Id`, la variable objetivo `SalePrice` y **79 features** predictoras
(**36 numéricas** y **43 categóricas**). El objetivo es continuo (precio en dólares) →
**problema de regresión**. La métrica de la competencia es el **RMSE**.

**Variable objetivo.** `SalePrice` va de \$34,900 a \$745,000 (media \$181,442, mediana
\$165,000) y está **fuertemente sesgada a la derecha** (skew = 1.74). Al aplicar `log1p`
la distribución se vuelve casi simétrica (skew = 0.13), lo que motiva **entrenar sobre
`log1p(SalePrice)`** e invertir con `expm1`.
*(Figura: `figures/01_target_distribution.png`)*

**Valores nulos.** 19 columnas presentan nulos. La observación clave del EDA es que en
muchas columnas el `NA` **codifica ausencia de la característica**, no un dato faltante:
`PoolQC` (99.5%), `MiscFeature` (96%), `Alley` (94%), `Fence` (80%), `FireplaceQu`
(47%), y las de garaje/sótano (2–5%). Estas se imputan con la categoría explícita
`"None"`. Solo son nulos "reales" `LotFrontage` (18.6%), `GarageYrBlt`, `MasVnrArea` y
`Electrical`, imputados por mediana.
*(Figura: `figures/02_missing_values.png`)*

**Outliers.** El scatter `GrLivArea` vs `SalePrice` revela unas pocas casas muy grandes
(>4000 pies²) vendidas a precio anormalmente bajo (outliers documentados de Ames). Por
ser muy pocos y usar `log1p` (que comprime la cola), se evaluó su impacto empíricamente
en lugar de eliminarlos a priori.
*(Figura: `figures/03_outliers_scatter.png`)*

**Correlaciones.** Las features más asociadas al precio son `OverallQual` (r = 0.79),
`GrLivArea` (r = 0.70), `GarageCars`/`GarageArea` (r ≈ 0.63) y `TotalBsmtSF` (r = 0.60).
Existe multicolinealidad esperable entre pares como `GarageCars`–`GarageArea`.
*(Figuras: `figures/04_corr_heatmap.png`, `figures/05_corr_target_bars.png`,
`figures/06_features_vs_target.png`)*

**Decisiones de preprocesamiento** (derivadas del EDA): imputar `"None"` en categóricas
de ausencia, imputar mediana en numéricas, `StandardScaler` en numéricas,
`OneHotEncoder(handle_unknown="ignore")` en categóricas, y `log1p` + estandarización en
el objetivo. Todo dentro de un `ColumnTransformer` **fitteado solo con train**.

## 2. Metodología de desarrollo

- **Arquitectura.** MLP configurable (`mlp_core.MLP`): capas lineales con `BatchNorm1d`,
  activación ReLU y `Dropout`, y una salida lineal de regresión. Se exploraron
  arquitecturas de 1 a 3 capas ocultas (de `[64]` a `[256,128,64]`).
- **División de datos.** train / validación / test = **70 / 15 / 15** con semilla fija.
  El *test* interno es un held-out que no participa en la selección de hiperparámetros.
- **Pérdida y optimización.** `MSELoss` sobre el objetivo estandarizado; optimizador
  **Adam** (lr 1e-3 / 5e-4), `weight_decay` (L2) hasta 1e-4.
- **Regularización.** Dropout (0.2–0.3), BatchNorm, weight decay y **early stopping**
  (paciencia 40 epochs sobre la pérdida de validación).

## 3. Resultados de iteraciones

Se documentaron **8 iteraciones**, cambiando un aspecto a la vez. RMSE en dólares:

| Iteración | Arquitectura | Regularización | RMSE_train | RMSE_val |
|---|---|---|---:|---:|
| **it7_deep3** 🏆 | 256-128-64 | dropout 0.3 + BN + wd | 9,512 | **25,414** |
| it4_dropout | 128-64 | dropout 0.3 + BN | 11,400 | 28,030 |
| it1_baseline | 64 | ninguna | 19,072 | 28,281 |
| it8_lowlr | 256-128 | dropout 0.2 + BN + wd | 6,521 | 28,925 |
| it5_wdecay | 128-64 | dropout 0.3 + BN + wd | 16,562 | 29,092 |
| it2_deeper | 128-64 | ninguna | 18,326 | 29,479 |
| it6_wider | 256-128 | dropout 0.3 + BN + wd | 8,621 | 29,577 |
| it3_batchnorm | 128-64 | solo BN | 6,665 | 30,442 |

*(Tabla completa: `iteraciones.csv`. Curvas: `figures/07_learning_curves.png`.)*

**Problemas encontrados.** Las curvas de `it1`–`it3` muestran **overfitting** claro
(error de train cayendo a ~0 mientras el de validación se estanca). Se abordó añadiendo
Dropout + weight decay, que cierran la brecha train/val (visible en `it4`, `it5`, `it7`).
Antes de estandarizar el objetivo, los modelos poco convergidos producían predicciones
extremas al invertir `expm1`; **estandarizar `log1p(SalePrice)`** resolvió la
inestabilidad y volvió comparables las iteraciones.

## 4. Discusión de resultados

- **Mayor impacto.** (1) Estandarización del objetivo (estabilidad + comparabilidad);
  (2) BatchNorm + Dropout + weight decay (control de overfitting). La profundidad ayudó
  **solo acompañada de regularización**.
- **Análisis de errores.** El modelo final acierta en el grueso del mercado
  (\$100k–\$300k) y falla más en las **viviendas de lujo**, poco representadas, a las que
  **subestima**. Los residuos están centrados en cero, con varianza que crece levemente
  con el precio. *(Figura: `figures/08_residuals.png`)*
- **Limitaciones.** Dataset pequeño (~1,168 filas) y sesgado a precios medios;
  multicolinealidad entre features; un MLP tabular compite de cerca con modelos simples.
- **Trade-off.** Redes más grandes memorizaban el train sin mejorar validación. El mejor
  equilibrio fue una red moderada **fuertemente regularizada** con early stopping.

## 5. Conclusiones

- **Desempeño final.** Configuración ganadora `it7` (`[256,128,64]`, dropout 0.3,
  BatchNorm, weight decay 1e-4): **RMSE_val ≈ \$25.4k** y **RMSE en test held-out ≈
  \$22.8k** (~12–14% del precio medio de \$181k) — estimación honesta del desempeño
  esperado. Para la competencia, el modelo se **reentrenó sobre todo `train.csv`** con
  esa misma configuración.
- **Aprendizajes.** (1) Transformar/estandarizar el objetivo es tan decisivo como la
  arquitectura; (2) la regularización supera a la capacidad bruta en datasets pequeños;
  (3) un pipeline de preprocesamiento serializado y reproducible es imprescindible para
  que la inferencia sea idéntica al entrenamiento.
- **Trabajo futuro.** *Feature engineering* (áreas totales, antigüedad, interacciones
  calidad×tamaño), *k-fold* + *ensembling* de semillas, y comparación contra *gradient
  boosting* como referencia.

## 6. Repositorio

GitHub: https://github.com/nicoCT04/Proyecto1-DL

Reproducción: ver `README.md`. El notebook regenera los artefactos; `predict.py` genera
las predicciones sobre un dataset de prueba con formato `Id,Prediction`.
