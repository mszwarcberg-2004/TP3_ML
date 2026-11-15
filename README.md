# TP3 - Grupo 2

Predicción de cancelación de reservas de GPU (`canceled_job`).

## Archivos

| Archivo | Contenido |
|---|---|
| `main.py` | Función predictora `predictor_v1_Grupo_2` y el código que entrena y evalúa el modelo final |
| `modelo_final_grupo_2.joblib` | Modelo final ya entrenado, que usa la función predictora |
| `TP3_Grupo.ipynb` | Notebook con todo el trabajo: análisis, baseline, análisis de errores y optimización |
| `requirements.txt` | Versiones de las librerías |

## Requisitos

Python 3.12 y las librerías de `requirements.txt`. Para instalarlas, en la terminal:

```
pip install -r requirements.txt
```

El modelo se guardó con scikit-learn 1.9.1. Con otra versión puede no cargar.

## Cómo usar la función predictora

`main.py` y `modelo_final_grupo_2.joblib` tienen que estar en la misma carpeta. Desde un notebook o un script de Python ubicado en esa carpeta:

```python
from main import predictor_v1_Grupo_2

predicciones = predictor_v1_Grupo_2(X)
```

- Entrada: un DataFrame de pandas con las columnas del dataset original. Si incluye `canceled_job` u otras columnas que el modelo no usa, se ignoran.
- Salida: un array de numpy con un valor por reserva, en el mismo orden que `X`: 1 si se predice que la reserva se cancela y 0 si no.

La función hace sola todo el preprocesamiento. No hace falta entrenar nada ni correr el notebook.

## Modelo

Ensemble (votación suave) de un Random Forest y un Gradient Boosting, con umbral de decisión 0,42 elegido para maximizar la balanced accuracy. La elección y su evaluación están en las secciones 11 y 12 del notebook.
