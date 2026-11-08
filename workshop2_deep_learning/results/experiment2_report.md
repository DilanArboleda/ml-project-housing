# Experimento 2: diagnóstico y corrección del MLP

## Protocolo

- Se conservaron sin cambios los índices 70/15/15 de `data/splits.json`.
- Todas las estadísticas nuevas se ajustaron exclusivamente con train.
- Validación se usó para early stopping y selección del checkpoint.
- Test no se usó para ajustar hiperparámetros, detener el entrenamiento ni
  seleccionar el modelo, y no se reportan métricas de test.
- `training_history.json` y `checkpoints/model_best.pt` conservan el
  experimento 1. El experimento 2 usa nombres de artefacto distintos.

## Diagnóstico

El preprocesamiento del Taller 1 usa `RobustScaler`, pero las variables muy
dispersas y con IQR cero no quedaron en una escala apropiada para el MLP. En
`X_train` había 741 valores con magnitud mayor que 100. Algunos máximos eran:

| Variable | Máximo absoluto |
|---|---:|
| Misc Val | 15.500 |
| BsmtFin SF 2 | 1.526 |
| Low Qual Fin SF | 1.064 |
| Enclosed Porch | 1.012 |
| Pool Area | 800 |

La matriz completa tenía desviación global 36,09. Después del escalado
adicional entrenado sólo con train tiene media aproximadamente cero y
desviación 1. Esto evita que unas pocas columnas dominen los gradientes.

`y_train_log` tenía media 12,013993 y desviación 0,395353. Entrenar una salida
inicial cercana a cero contra un objetivo centrado alrededor de 12 produjo
gradientes iniciales innecesariamente grandes. El experimento 2 entrena sobre
`(y_log - media_train) / std_train`. El checkpoint conserva ambas estadísticas
y la predicción se revierte mediante:

`precio = expm1(prediccion_estandarizada * std_train + media_train)`

El historial original confirma la mala convergencia: durante 36 épocas la
pérdida de validación en escala log osciló entre 1,093 y 16,563, y el mejor R²
en el punto de menor MAE fue negativo (-2,927).

## Cambios de entrenamiento

- Estandarización adicional de las 261 entradas, ajustada sólo con train.
- Estandarización de `log1p(SalePrice)`, ajustada sólo con train, con inversión
  explícita antes de calcular métricas en dólares.
- Red reducida de 128-64-32 (43.905 parámetros) a 64-32 (18.881 parámetros).
  Para 2.048 observaciones efectivas, la red original tenía regularización y
  capacidad innecesarias.
- Dropout reducido de 0,20 a 0,05 para evitar subajuste tras reducir la red.
- Inicialización Kaiming para capas ReLU, Xavier para la salida y sesgos cero.
- AdamW con learning rate inicial 0,001 y `weight_decay=1e-5`.
- `ReduceLROnPlateau` sobre pérdida de validación y clipping de gradiente a 5.
- Early stopping por MAE de validación con paciencia de 30 épocas.

## Resultados reales

| Métrica de validación | Experimento 1 | Experimento 2 |
|---|---:|---:|
| Mejor época | 16 | 75 |
| MAE | 77.990,05 USD | 15.837,95 USD |
| RMSE | 174.526,51 USD | 24.233,63 USD |
| R² | -2,9268 | 0,9243 |
| Parámetros | 43.905 | 18.881 |

La reducción de MAE es 79,69%. El experimento 2 ejecutó 105 épocas y terminó
por early stopping. En sus últimas 30 épocas la pérdida estandarizada de
validación permaneció entre 0,1292 y 0,1344, sin las oscilaciones explosivas
del experimento 1. El checkpoint recargado reprodujo exactamente MAE
15.837,95, RMSE 24.233,63 y R² 0,9243.

Como referencia, Ridge del Taller 1 obtuvo MAE de validación 14.101,86 USD.
Por tanto, el nuevo MLP ya es competitivo y está correctamente condicionado,
pero no supera todavía el mejor baseline lineal; no se afirma lo contrario.

## Verificación

Comando ejecutado:

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s workshop2_deep_learning/tests -v
```

Resultado: 3 pruebas ejecutadas, 3 correctas. Cubren estadísticas de train,
reversión del objetivo y forma/valores finitos de la salida. También se ejecutó
`py_compile` sobre `train.py` y `model.py` sin errores. `pytest` no está
instalado en el entorno, por lo que se usó `unittest` sin añadir dependencias.
