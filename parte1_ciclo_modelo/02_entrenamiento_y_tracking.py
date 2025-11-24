# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 1: Ciclo de vida de un modelo - Entrenamiento y Tracking
# MAGIC *   **Autor:** Carolina Torres Zapata
# MAGIC *   **Fecha:** 2025-11-24
# MAGIC *   **Propósito:** Entrenar modelos de clasificación supervisada (Regresión Logística y Random Forest) utilizando los datos procesados. Se implementa **MLflow** para registrar experimentos, métricas y versionar los artefactos del modelo.
# MAGIC *   **Flujo de Trabajo:**
# MAGIC      1.  Carga de datos transformados desde la capa Silver.
# MAGIC      2.  Separación de conjuntos de entrenamiento y prueba (Split).
# MAGIC      3.  Entrenamiento iterativo con registro de experimentos (Tracking).
# MAGIC      4.  Evaluación de desempeño (Métricas).
# MAGIC      

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importar Librerías

# COMMAND ----------

import pandas as pd
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Carga de Datos (Capa Silver)
# MAGIC Leemos la tabla `dev.silver.churn_data` generada en la etapa anterior. Esta tabla ya contiene las características codificadas y escaladas.

# COMMAND ----------

df = spark.table("dev.silver.churn_data").toPandas()
print(f"Dimensiones del dataset: {df.shape}")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Separación de Datos (Train/Test Split)
# MAGIC Preparamos los conjuntos de datos para el modelado:
# MAGIC 1.  **Definición de Features (X):** Eliminamos la variable objetivo (`Churn`) y el identificador (`customerID`) para evitar que el modelo memorice identidades únicas.
# MAGIC 2.  **Estratificación:** Utilizamos `stratify=y` al dividir los datos (80% entrenamiento / 20% prueba). Esto es crucial en problemas de clasificación desbalanceada para garantizar que la proporción de casos de fuga (Churn=1) sea la misma en ambos conjuntos.

# COMMAND ----------

# Definición de X e y
target_col = "Churn"
id_col = "customerID"

X = df.drop(columns=[target_col, id_col])
y = df[target_col]

# Split 80/20 con estratificación (importante porque Churn suele estar desbalanceado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Registros de Entrenamiento: {X_train.shape[0]}")
print(f"Registros de Prueba: {X_test.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configuración del Experimento MLflow
# MAGIC Definimos y configuramos el experimento en el que se registrarán todas las ejecuciones (runs). Establecer una ruta explícita (`set_experiment`) garantiza que los logs, métricas y artefactos queden centralizados y organizados, facilitando la trazabilidad y comparación de modelos.

# COMMAND ----------

experiment_path = "/Users/carolina.torresz@udea.edu.co/churn_experiment_ops"
mlflow.set_experiment(experiment_path)

print(f"Experimento configurado en: {experiment_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##5. Función de Entrenamiento Estandarizada
# MAGIC Implementamos una función reutilizable que encapsula el ciclo de vida del entrenamiento para garantizar consistencia en todos los experimentos.
# MAGIC
# MAGIC **Flujo de la función:**
# MAGIC 1.  **Entrenamiento:** Ajuste del modelo con los datos de entrenamiento.
# MAGIC 2.  **Evaluación:** Cálculo de métricas clave (*Accuracy, F1-Score, AUC-ROC*) sobre el set de prueba.
# MAGIC 3.  **Trazabilidad (MLflow):** Registro automático de:
# MAGIC     *   **Hiperparámetros:** Configuración del modelo.
# MAGIC     *   **Métricas:** Resultados de desempeño.
# MAGIC     *   **Artefacto del Modelo:** Serialización y guardado del modelo junto con un `input_example`. Esto último es crítico para que MLflow registre automáticamente la "firma" (esquema) de los datos de entrada.

# COMMAND ----------

def entrenar_y_registrar(modelo, nombre_run, params):
    """
    Entrena un modelo, calcula métricas y lo registra en MLflow.
    """
    with mlflow.start_run(run_name=nombre_run) as run:
        print(f"🚀 Iniciando entrenamiento: {nombre_run}...")

        # 1. Entrenar
        modelo.fit(X_train, y_train)
        
        # 2. Predecir (Clases y Probabilidades)
        y_pred = modelo.predict(X_test)
        # Para AUC necesitamos probabilidades. Algunos modelos usan predict_proba
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred # Fallback si no soporta proba
            
        # 3. Calcular Métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"   📊 Accuracy: {acc:.4f}")
        print(f"   📊 AUC:      {auc:.4f}")
        
        # 4. Logging en MLflow
        # A. Parámetros
        mlflow.log_params(params)
        
        # B. Métricas
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "auc": auc})
        
        # C. Etiquetas (Tags) para operación
        mlflow.set_tag("env", "dev")
        mlflow.set_tag("algorithm", nombre_run.split("_")[0])
        
        # D. Guardar Modelo (Artefacto)
        input_example = X_train.iloc[:5]
        
        mlflow.sklearn.log_model(
            sk_model=modelo, 
            artifact_path="model",
            input_example=input_example
        )
        
        print(f"✅ Run ID: {run.info.run_id}")
        return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ejecución de Experimentos y Manejo de Desbalance
# MAGIC Procedemos a entrenar dos tipos de algoritmos para comparar su desempeño y establecer una línea base.
# MAGIC
# MAGIC 1.  **Regresión Logística:** Modelo lineal, simple e interpretable.
# MAGIC 2.  **Random Forest:** Modelo de ensamble no lineal, robusto ante ruido.
# MAGIC
# MAGIC **Decisión Técnica (`class_weight='balanced'`):**
# MAGIC Dado el desbalance detectado en el EDA (~26% Churn), se configuran ambos algoritmos con pesos de clase balanceados. Esto ajusta la función de costo para penalizar más severamente los errores en la clase minoritaria, mejorando la capacidad del modelo para detectar fugas reales sin añadir complejidad al pipeline de datos (como SMOTE).

# COMMAND ----------

# --- MODELO A: Regresión Logística ---
params_lr = {
    "C": 1.0, 
    "solver": "liblinear", 
    "class_weight": "balanced",  
    "random_state": 42
}
model_lr = LogisticRegression(**params_lr)

run_id_lr = entrenar_y_registrar(model_lr, "Logistic_Regression_Balanced", params_lr)

print("-" * 30)

# --- MODELO B: Random Forest ---
params_rf = {
    "n_estimators": 100, 
    "max_depth": 10, 
    "min_samples_split": 5,
    "class_weight": "balanced", 
    "random_state": 42
}
model_rf = RandomForestClassifier(**params_rf)

run_id_rf = entrenar_y_registrar(model_rf, "Random_Forest_Balanced", params_rf)

# COMMAND ----------

# Listar las últimas corridas programáticamente para confirmar éxito
runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_path).experiment_id])
display(runs[["run_id", "tags.mlflow.runName", "metrics.auc","metrics.f1_score" ,"metrics.accuracy", "status"]].head(2))