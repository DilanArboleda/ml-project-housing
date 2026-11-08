# Workshop 2 Part A — evaluación final sobre test

Este documento registra la evaluación técnica solicitada. No es el informe
académico definitivo.

## Decisión congelada antes de test

El manifiesto de experimentación seleccionó `e4_wide_128_64` exclusivamente
por el MAE medio de validación de las semillas 7, 42 y 2026. Ninguna corrida de
búsqueda reporta uso de test. Entre los tres checkpoints de esa configuración,
la semilla 42 tenía el menor MAE de validación y fue el checkpoint especificado
para esta evaluación:

`checkpoints/part_a/model_e4_wide_128_64_seed42.pt`

No se entrenó nuevamente el MLP ni se cambió la selección tras observar test.

## Protocolo

- Se reutilizaron exactamente los 440 índices de test de `data/splits.json`.
- El preprocesador del Workshop 1 se ajustó nuevamente sólo con train y se
  aplicó a test.
- El orden de características y las estadísticas de normalización regeneradas
  se comprobaron contra el checkpoint.
- X se normalizó con `input_mean` e `input_scale` del checkpoint.
- La salida se revirtió como
  `expm1(pred_scaled * target_log_scale + target_log_mean)`.
- MAE, RMSE, R² y RMSE_log usan la función y definiciones de `src/baselines.py`.
- Todos los modelos se comparan sobre los mismos 440 objetivos y en el mismo
  orden. El hash SHA-256 de los índices es
  `90596065fefdcddd60780800d95468600b2c62e6ab0bfd60d9459b528be56e32`.

## Métricas definitivas de test

| Modelo | MAE (USD) | RMSE (USD) | R² | RMSE_log |
|---|---:|---:|---:|---:|
| **Deep Learning MLP 128-64** | **14.666,23** | 34.133,71 | 0,8483 | 0,1254 |
| Ridge Regression | 15.031,51 | 44.511,47 | 0,7421 | 0,1254 |
| Random Forest | 16.700,76 | **31.325,66** | **0,8723** | 0,1325 |
| Gradient Boosting | 14.876,80 | 31.939,29 | 0,8672 | **0,1208** |

Las métricas de los baselines son las métricas históricas inmutables recuperadas
de `workshop1_supervised/results/baseline_metrics.json`.

## Comparación

- El MLP obtuvo el mejor MAE: mejora a Ridge en 365,28 USD (2,43%), a Random
  Forest en 2.034,53 USD (12,18%) y a Gradient Boosting en 210,57 USD (1,42%).
- El MLP no fue mejor en todas las métricas. Random Forest tiene el mejor RMSE
  y R²; Gradient Boosting tiene el mejor RMSE_log y también menor RMSE que el
  MLP.
- Frente a su MAE de validación de 14.051,46 USD, el MLP aumentó 614,77 USD
  (4,38%) en test. La diferencia es moderada y no cambia la decisión previa.
- Un caso extremo explica buena parte del RMSE del MLP: el índice 2181 tiene
  precio real 184.750 USD y predicción 775.256,31 USD, residual -590.506,31 USD.
  No se eliminó ni se usó para modificar el modelo después de test.

En consecuencia, el MLP sí mejora los tres baselines en **MAE**, pero no puede
considerarse superior de forma general: los ensembles presentan menores errores
cuadráticos y mejor comportamiento frente a errores extremos.

## Estabilidad descriptiva de los checkpoints ya seleccionados

Estas métricas se calcularon únicamente para describir estabilidad. No se usaron
para cambiar el modelo o escoger otra semilla.

| Semilla | MAE | RMSE | R² | RMSE_log |
|---:|---:|---:|---:|---:|
| 7 | 16.746,04 | 39.594,06 | 0,7959 | 0,1355 |
| **42 — checkpoint final** | **14.666,23** | 34.133,71 | 0,8483 | 0,1254 |
| 2026 | 15.795,35 | **32.770,38** | **0,8602** | 0,1284 |

El MAE medio descriptivo es 15.735,87 ± 1.041,18 USD. La dispersión en test es
considerablemente mayor que en validación, por lo que las conclusiones sobre
una única semilla deben interpretarse con cautela.

## Reproducibilidad y auditoría de baselines

El checkpoint fue cargado dos veces en modelos independientes y produjo arrays
de predicción exactamente iguales. Las métricas recalculadas coinciden con el
JSON final y el CSV conserva los 440 índices y objetivos en el orden del split.

Los modelos del Workshop 1 se reconstruyeron con sus hiperparámetros congelados
para generar gráficas de predicciones y residuos. En el entorno actual no
reproducen bit a bit el JSON histórico: las diferencias de MAE fueron 0,72 USD
para Ridge, 14,75 USD para Random Forest y 145,34 USD para Gradient Boosting.
Estas diferencias, compatibles con variaciones numéricas o de versión de
scikit-learn, se conservan explícitamente en `final_test_metrics.json`. La tabla
definitiva usa las métricas históricas, no las reconstruidas.

## Limitaciones

- Test se observó una única vez después de congelar configuración y semilla; no
  debe reutilizarse para nuevas decisiones de hiperparámetros.
- La partición contiene sólo 440 viviendas y algunos casos extremos dominan
  RMSE y R².
- Las tres semillas comparten una única partición; no miden variación entre
  diferentes splits.
- Los checkpoints no constituyen un ensemble. Promediar sus predicciones sería
  un modelo nuevo y no se evaluó.
- Las predicciones reconstruidas de los baselines son adecuadas para diagnóstico
  visual, pero presentan la pequeña deriva numérica descrita respecto al JSON
  original.

## Archivos generados

- `final_test_metrics.json`: métricas, hashes, auditoría y estabilidad.
- `model_comparison_test.csv`: tabla definitiva de los cuatro modelos.
- `test_predictions.csv`: observaciones, predicciones y residuos.
- `selected_model_seed_stability_test.json`: resultados descriptivos por semilla.
- `results/figures/final_test/actual_vs_predicted_test.png`.
- `results/figures/final_test/residual_analysis_test.png`.
- `results/figures/final_test/error_comparison_test.png`.
