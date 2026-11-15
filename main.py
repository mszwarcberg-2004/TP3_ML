
# Modelo final y función predictora.

# Qué hace este archivo (punto 5 de la consigna):
#   1. Repite el split train/test del notebook (sección 3).
#   2. Entrena el baseline (Random Forest) y el modelo final (ensemble Random Forest +
#      Gradient Boosting, sección 11.5) con train.
#   3. Evalúa los dos sobre test, una sola vez: global y por cohorte.
#   4. Guarda el modelo final para que lo use la función predictora.

# Uso:
#   python main.py                              -> entrena, evalúa y guarda el modelo
#   from main import predictor_v1_Grupo_2
#   predicciones = predictor_v1_Grupo_2(X)      -> X: DataFrame con las columnas del dataset


from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# CONFIGURACIÓN
# Valores elegidos en el notebook (TP3_Grupo.ipynb). Si cambian en la corrida
# final del notebook, se actualizan solo acá.

CARPETA = Path(__file__).resolve().parent
RUTA_DATOS = CARPETA / "data" / "gpu_resource_reservations.csv"
RUTA_MODELO = CARPETA / "modelo_final_grupo_2.joblib"

# Baseline: Random Forest (sección 6.4) y su umbral (sección 6.5)
HIPERPARAMETROS_RF = {"max_features": 0.5, "max_depth": None, "min_samples_leaf": 1, "n_estimators": 200}
UMBRAL_BASELINE = 0.43

# Gradient Boosting optimizado (sección 11.4)
HIPERPARAMETROS_GB = {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 63,
                      "min_samples_leaf": 20, "l2_regularization": 0.1}

# Modelo final: ensemble Random Forest + Gradient Boosting y su umbral (sección 11.5)
UMBRAL_FINAL = 0.42

# Después de evaluar en test, el modelo que se entrega se reentrena con train + test,
# para aprovechar todos los datos (la curva de aprendizaje de la sección 8 muestra que
# más datos todavía ayudan un poco). Con False se entrega el modelo entrenado solo con train.
REENTRENAR_CON_TODOS_LOS_DATOS = True


# COLUMNAS (secciones 3, 4.1, 4.4 y 5.1)


TARGET = "canceled_job"
COLS_LEAKAGE = ["execution_status", "execution_status_date"]               # sección 3.1
COLS_REDUNDANTES = ["total_gpu_hours", "total_processes", "execution_week"]  # sección 4.1

COLS_NUMERICAS = [
    "hours_before_execution", "execution_year", "execution_month", "execution_day",
    "peak_gpu_hours", "offpeak_gpu_hours", "num_primary_processes", "num_auxiliary_processes",
    "num_background_processes", "returning_client", "previous_job_cancellations",
    "successful_previous_jobs", "resource_modifications", "queue_wait_time",
    "estimated_cost_per_hour", "reserved_storage_units", "special_resource_requests",
    "high_value_client",
]
COLS_NOMINALES = [
    "service_tier", "client_type", "request_channel", "reserved_gpu_type",
    "assigned_gpu_type", "reservation_commitment", "workload_profile", "client_region",
]
COLS_IDENTIFICADORES = ["orchestration_agent_id", "organization_id"]
VARIABLES_NUEVAS = ["costo_total", "proporcion_horas_pico", "dia_semana"]  # sección 11.3

# Valores por encima de estos umbrales son errores de carga (sección 4.4)
UMBRALES_ERROR = {"num_background_processes": 2, "estimated_cost_per_hour": 50}

COHORTES = ["client_type", "service_tier", "reserved_gpu_type"]

# PREPARACIÓN DE LOS DATOS
# Se hace fuera del modelo guardado, así el archivo .joblib solo contiene objetos
# de scikit-learn y se puede cargar desde cualquier lugar.

def preparar_datos(X): #Deja los datos listos para el modelo: mismas columnas, errores corregidos y variables nuevas.
    X = X.copy()

    # Columnas que el modelo no usa (target, leakage y redundantes), si vinieran en los datos
    X = X.drop(columns=[TARGET] + COLS_LEAKAGE + COLS_REDUNDANTES, errors="ignore")

    # Si falta alguna columna, se agrega vacía y el imputador la completa
    for col in COLS_NUMERICAS + COLS_NOMINALES + COLS_IDENTIFICADORES:
        if col not in X.columns:
            X[col] = np.nan
    for col in COLS_NUMERICAS + COLS_IDENTIFICADORES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Errores de carga -> faltantes (sección 4.4)
    for col, umbral in UMBRALES_ERROR.items():
        X.loc[X[col] > umbral, col] = np.nan

    # Variables nuevas (sección 11.3)
    horas_totales = X["peak_gpu_hours"] + X["offpeak_gpu_hours"]
    X["costo_total"] = horas_totales * X["estimated_cost_per_hour"]
    X["proporcion_horas_pico"] = X["peak_gpu_hours"] / horas_totales.replace(0, np.nan)
    fecha = pd.to_datetime(pd.DataFrame({"year": X["execution_year"],
                                         "month": X["execution_month"],
                                         "day": X["execution_day"]}), errors="coerce")
    X["dia_semana"] = fecha.dt.day_of_week  # 0 = lunes, 6 = domingo

    return X


def armar_columnas(con_variables_nuevas): #Completar valores faltantes, escalado y one-hot por grupo de columnas (sección 5.2)."
    numericas = COLS_NUMERICAS + (VARIABLES_NUEVAS if con_variables_nuevas else [])
    return ColumnTransformer([
        ("numericas", Pipeline([("imputar", SimpleImputer(strategy="median")),
                                ("escalar", StandardScaler())]), numericas),
        ("nominales", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                                ("codificar", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                            min_frequency=0.01,
                                                            sparse_output=False))]), COLS_NOMINALES),
        ("identificadores", Pipeline([("imputar", SimpleImputer(strategy="constant", fill_value=-1)),
                                      ("codificar", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                                  min_frequency=0.01,
                                                                  sparse_output=False))]),
         COLS_IDENTIFICADORES),
    ], remainder="drop")


# MODELOS

def armar_baseline(): #Random Forest de la sección 6 (sin variables nuevas).
    return Pipeline([("columnas", armar_columnas(con_variables_nuevas=False)),
                     ("modelo", RandomForestClassifier(random_state=42, n_jobs=-1, **HIPERPARAMETROS_RF))])


def armar_modelo_final(): #Ensemble de la sección 11.5: promedia las probabilidades del Random Forest y del Gradient Boosting.
    gradient_boosting = Pipeline([
        ("columnas", armar_columnas(con_variables_nuevas=True)),
        ("modelo", HistGradientBoostingClassifier(early_stopping=False, random_state=42, **HIPERPARAMETROS_GB)),
    ])
    return VotingClassifier([("random_forest", armar_baseline()),
                             ("gradient_boosting", gradient_boosting)], voting="soft")


# DATOS Y SPLIT (igual que la sección 3 del notebook)

def cargar_y_separar():
    df = pd.read_csv(RUTA_DATOS).drop(columns=COLS_LEAKAGE)

    # Las filas idénticas forman un grupo y quedan todas del mismo lado del split (sección 3.2)
    cols_features = df.columns.drop(TARGET)
    grupos = df.groupby(list(cols_features), dropna=False).ngroup()

    X, y = df.drop(columns=TARGET), df[TARGET]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    idx_train, idx_test = next(sgkf.split(X, y, groups=grupos))  # el primer fold es el test (20%)

    return X.iloc[idx_train], X.iloc[idx_test], y.iloc[idx_train], y.iloc[idx_test]


# EVALUACIÓN EN TEST

def ba_por_cohorte(X_test, y_test, pred, minimo=30): #Balanced accuracy dentro de cada cohorte con al menos minimo reservas y las dos clases.
    filas = []
    for col in COHORTES:
        for valor, idx in X_test.groupby(col).groups.items():
            y_c = y_test.loc[idx]
            if len(y_c) >= minimo and y_c.nunique() == 2:
                filas.append({"variable": col, "cohorte": valor, "reservas": len(y_c),
                              "BA": balanced_accuracy_score(y_c, pred.loc[idx])})
    return pd.DataFrame(filas).set_index(["variable", "cohorte"])


def evaluar_en_test(modelos, X_test, y_test): #Métricas globales y por cohorte de cada modelo, con su umbral.
    X_prep = preparar_datos(X_test)
    globales, por_cohorte = {}, {}

    for nombre, (modelo, umbral) in modelos.items():
        proba = modelo.predict_proba(X_prep)[:, 1]
        pred = pd.Series((proba >= umbral).astype(int), index=y_test.index)
        globales[nombre] = {"umbral": umbral,
                            "BA": balanced_accuracy_score(y_test, pred),
                            "ROC_AUC": roc_auc_score(y_test, proba)}
        por_cohorte[nombre] = ba_por_cohorte(X_test, y_test, pred)["BA"]

        if nombre == "Modelo final":
            matriz = pd.DataFrame(confusion_matrix(y_test, pred),
                                  index=["Real: no cancelada", "Real: cancelada"],
                                  columns=["Pred: no cancelada", "Pred: cancelada"])

    tabla_global = pd.DataFrame(globales).T
    tabla_cohortes = pd.DataFrame(por_cohorte)
    tabla_cohortes["mejora"] = tabla_cohortes["Modelo final"] - tabla_cohortes["Baseline"]
    return tabla_global, tabla_cohortes, matriz


# FUNCIÓN PREDICTORA

_modelo_guardado = None  # se carga una sola vez, la primera vez que se llama a la función


# Recibe un DataFrame con las columnas del dataset (con o sin canceled_job) y devuelve
# un array con 1 (se cancela) o 0 (no se cancela) para cada reserva.
# Necesita el archivo modelo_final_grupo_2.joblib en la misma carpeta que este archivo.
def predictor_v1_Grupo_2(X):
    
    global _modelo_guardado
    if _modelo_guardado is None:
        _modelo_guardado = joblib.load(RUTA_MODELO)

    proba = _modelo_guardado["modelo"].predict_proba(preparar_datos(X))[:, 1]
    return (proba >= _modelo_guardado["umbral"]).astype(int)


# PROGRAMA PRINCIPAL


def main():
    X_train, X_test, y_train, y_test = cargar_y_separar()
    print(f"Train: {len(X_train)} reservas | Test: {len(X_test)} reservas\n")

    # 1. Entrenamiento con train
    print("Entrenando baseline y modelo final...")
    baseline = armar_baseline().fit(preparar_datos(X_train), y_train)
    modelo_final = armar_modelo_final().fit(preparar_datos(X_train), y_train)

    # 2. Evaluación en test (única vez que se usa el test)
    tabla_global, tabla_cohortes, matriz = evaluar_en_test(
        {"Baseline": (baseline, UMBRAL_BASELINE), "Modelo final": (modelo_final, UMBRAL_FINAL)},
        X_test, y_test,
    )
    print("\n=== Rendimiento en test ===")
    print(tabla_global.round(4).to_string())
    print("\n=== Matriz de confusión del modelo final en test ===")
    print(matriz.to_string())
    print("\n=== Balanced accuracy por cohorte en test ===")
    print(tabla_cohortes.round(4).to_string())

    # 3. Modelo que se entrega
    if REENTRENAR_CON_TODOS_LOS_DATOS:
        print("\nReentrenando el modelo final con train + test...")
        X_todo, y_todo = pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
        modelo_final = armar_modelo_final().fit(preparar_datos(X_todo), y_todo)

    # Compresión máxima para que el archivo pese menos de 25 MB y se pueda mandar por mail
    joblib.dump({"modelo": modelo_final, "umbral": UMBRAL_FINAL}, RUTA_MODELO, compress=9)
    print(f"Modelo guardado en {RUTA_MODELO.name}")

    # 4. Prueba rápida de la función predictora
    print("Prueba del predictor sobre 5 reservas:", predictor_v1_Grupo_2(X_test.head()))


if __name__ == "__main__":
    main()
