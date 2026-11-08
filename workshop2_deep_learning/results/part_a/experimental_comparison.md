# Workshop 2 — comparación experimental de MLP (Part A)

Este documento registra la comparación ejecutada. No es el informe académico
definitivo y no contiene evaluación sobre test.

## Protocolo y criterio de selección

- Se reutilizó exactamente `data/splits.json`: 2.050 índices de train, 440 de
  validación y 440 de test. Tras retirar únicamente los dos outliers de train
  ya definidos por el preprocesador del Workshop 1, se entrenó con 2.048 filas.
- El preprocesador del Workshop 1 se ajustó sólo con train. La estandarización
  adicional de X y del objetivo también se ajustó sólo con train.
- Test no fue transformado, evaluado ni utilizado para decisiones.
- Todas las comparaciones iniciales usaron semilla 42, batch size 32, máximo
  250 épocas, paciencia 30 y selección por el menor MAE de validación.
- La referencia y las dos mejores configuraciones nuevas no ablativas en
  semilla 42 se repitieron con semillas 7 y 2026.
- La configuración final se seleccionó por el menor **MAE medio de validación
  en las tres semillas**, no por la mejor corrida individual.

## Configuraciones controladas — semilla 42

Cada experimento cambia una variable respecto al experimento 2, excepto las
dos ablaciones indicadas.

| Configuración | Cambio | Parámetros | Época | MAE val. | RMSE val. | R² val. |
|---|---|---:|---:|---:|---:|---:|
| Experimento 2 | Referencia: 64-32, dropout 0,05, wd 1e-5, lr 1e-3 | 18.881 | 75 | 15.837,95 | 24.233,63 | 0,9243 |
| e3 | Dropout 0,00 | 18.881 | 66 | 19.330,28 | 28.932,34 | 0,8921 |
| e4 | Capas 128-64 | 41.857 | 83 | **14.051,46** | **20.626,19** | **0,9452** |
| e5 | Weight decay 1e-4 | 18.881 | 81 | 15.534,41 | 23.628,12 | 0,9280 |
| e6 | Learning rate 5e-4 | 18.881 | 61 | 16.341,55 | 25.099,41 | 0,9188 |
| a1 | Sin estandarización adicional de X | 18.881 | 75 | 15.696,59 | 26.301,79 | 0,9108 |
| a2 | Sin estandarización de `y_log` | 18.881 | 152 | 34.600,95 | 56.754,42 | 0,5847 |

## Estabilidad

| Configuración | Semillas | MAE medio ± DE | Rango MAE | RMSE medio | R² medio |
|---|---|---:|---:|---:|---:|
| **128-64 (e4)** | 7, 42, 2026 | **14.238,79 ± 183,95** | 14.051,46–14.419,17 | 20.779,54 | 0,9443 |
| Weight decay 1e-4 (e5) | 7, 42, 2026 | 14.379,15 ± 1.037,55 | 13.526,67–15.534,41 | 21.415,30 | 0,9405 |
| Experimento 2 | 7, 42, 2026 | 14.695,31 ± 1.021,94 | 13.868,75–15.837,95 | 21.669,05 | 0,9390 |

La arquitectura 128-64 fue seleccionada porque tuvo el menor MAE medio y una
desviación entre semillas mucho menor. `weight_decay=1e-4` produjo la mejor
corrida individual (13.526,67 con semilla 7), pero fue considerablemente más
variable; escoger sólo esa semilla sería una selección optimista.

## Interpretación de cambios y ablaciones

- **Capacidad:** ampliar de 64-32 a 128-64 aportó la mejora más consistente.
  En semilla 42 redujo el MAE 1.786,48 USD (11,28%) respecto al experimento 2.
  En promedio de tres semillas redujo 456,52 USD (3,11%) respecto al promedio
  multisemilla de la referencia.
- **Dropout:** eliminarlo empeoró el MAE en 3.492,33 USD en semilla 42. La
  separación de las curvas train/validación indica mayor sobreajuste; mantener
  dropout 0,05 está justificado.
- **Weight decay:** elevarlo a 1e-4 mejoró 303,53 USD en semilla 42 y produjo
  una corrida excelente con semilla 7, pero no fue estable.
- **Learning rate:** reducirlo a 5e-4 empeoró 503,60 USD; no hay evidencia para
  cambiar el learning rate inicial de 1e-3.
- **Estandarización de X:** quitarla hizo que el MAE de la primera época pasara
  de 31.590,70 a 74.448,73 USD y empeoró RMSE/R² finales, confirmando peor
  condicionamiento. Su mejor MAE aislado fue 141,35 USD menor que la referencia
  en esta única semilla, por lo que no puede afirmarse que el escalado de X
  mejore el mínimo MAE; sí mejora claramente estabilidad inicial y errores de
  cola. Esta ablación requeriría más semillas para una conclusión sobre MAE.
- **Estandarización del objetivo:** quitarla empeoró el MAE en 18.763,01 USD
  (118,47%) y necesitó 152 épocas para su mejor punto. Es la intervención más
  importante para la convergencia.

## Comparación con Ridge y conclusión

La configuración seleccionada es **128-64, dropout 0,05, weight decay 1e-5 y
learning rate 1e-3**, con estandarización train-only de entradas y objetivo.

- Comparada con el experimento 2 reportado, bajo la misma semilla 42, mejora
  de 15.837,95 a 14.051,46 USD: **1.786,48 USD o 11,28%**.
- Esa corrida supera el MAE de Ridge (14.101,86 USD) por **50,40 USD (0,36%)**.
- La conclusión multisemilla es más conservadora: MAE medio 14.238,79 USD,
  **136,92 USD (0,97%) peor que Ridge**. Por ello no hay evidencia robusta de
  que el MLP supere a Ridge, aunque una corrida controlada sí lo hizo.

## Artefactos y reproducibilidad

- `comparison_all_runs.csv`: tabla de las 13 corridas.
- `experiment_manifest.json`: protocolo, configuraciones y selección.
- `stability_summary.json`: agregación multisemilla.
- `history_*.json`: métricas y pérdidas por época.
- `summary_*.json`: mejor época y métricas de cada corrida.
- `checkpoints/part_a/model_*.pt`: checkpoints independientes con escaladores,
  semilla, configuración y metadatos del split.
- `results/figures/part_a/`: pérdidas, comparación de MAE y estabilidad.

El runner permite reanudación sin sobrescribir corridas terminadas:

```powershell
.\.venv\Scripts\python.exe -u `
  workshop2_deep_learning/src/run_experiments.py
```

## Limitaciones

- Sólo tres configuraciones se repitieron con varias semillas; las otras y las
  ablaciones tienen una única semilla.
- Se utilizó una sola partición fija. Las desviaciones reportadas miden la
  aleatoriedad de inicialización/mini-batches, no variación entre particiones.
- La búsqueda es deliberadamente pequeña y controlada, no una optimización
  exhaustiva de hiperparámetros.
- Las diferencias frente a Ridge son menores que la variabilidad entre
  semillas. La evaluación final sobre test queda pendiente y no debe realizarse
  hasta congelar formalmente la configuración.
