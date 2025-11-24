# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 1: Ciclo de vida de un modelo - Registro / Almacenamiento del Modelo
# MAGIC *   **Autor:** Carolina Torres Zapata
# MAGIC *   **Fecha:** 2025-11-24
# MAGIC *   **Contexto:** En escenarios donde no se dispone de un **Model Registry** centralizado (o como mecanismo de contingencia/fallback), la operación debe ser capaz de recuperar modelos directamente desde el historial de experimentos (Tracking Server).
# MAGIC
# MAGIC **Objetivo de este notebook:**
# MAGIC Implementar un flujo de inferencia automatizado que:
# MAGIC 1.  **Identifique dinámicamente** el mejor modelo entrenado (Champion) basándose en métricas objetivas (AUC).
# MAGIC 2.  Cargue el modelo a memoria directamente desde los **artefactos de MLflow**.
# MAGIC 3.  Simule un pipeline de inferencia por lotes (*Batch Inference*) sobre nuevos datos.
# MAGIC 4.  Genere un **reporte operativo** enriquecido para la toma de decisiones de negocio.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importar Librerías

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Búsqueda del "Mejor Modelo" (Simulación de Registry)
# MAGIC En lugar de copiar y pegar manualmente el `run_id` (lo cual es propenso a errores humanos), consultamos programáticamente el MLflow Tracking Server.
# MAGIC Este bloque busca en el experimento `churn_experiment_ops`, ordena los modelos por la métrica `AUC` descendente y selecciona el ganador ("Champion") para esta ejecución.

# COMMAND ----------

experiment_path = "/Users/carolina.torresz@udea.edu.co/churn_experiment_ops" 
experiment = mlflow.get_experiment_by_name(experiment_path)

if experiment is None:
    print("El experimento no existe. Ejecuta el notebook 02 primero.")
else:
    # Buscar corridas, ordenar por AUC descendente
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.auc DESC"]
    )
    
    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_auc = best_run["metrics.auc"]
    
    print(f"Mejor Run ID recuperado: {best_run_id}")
    print(f"Métrica AUC: {best_auc}")
    print(f"Artifact URI: {best_run.artifact_uri}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Carga del Modelo desde Artefactos
# MAGIC Utilizamos el protocolo `runs:/<id>/model` de MLflow. Esto garantiza que estamos cargando exactamente el binario serializado que generó las métricas en el paso anterior, asegurando la reproducibilidad del entorno productivo.

# COMMAND ----------

model_uri = f"runs:/{best_run_id}/model"

print(f"Cargando modelo desde: {model_uri} ...")
#loaded_model = mlflow.pyfunc.load_model(model_uri)
loaded_model = mlflow.sklearn.load_model(model_uri)

print("Modelo cargado exitosamente en memoria.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inferencia de Prueba (Batch Inference)
# MAGIC Para simular un entorno real de producción:
# MAGIC 1.  Cargamos datos "nuevos" (simulados desde la capa Silver).
# MAGIC 2.  **Saneamiento de Schema:** Separamos los identificadores de cliente (`customerID`) de las variables predictoras (`X`). El modelo solo debe recibir las columnas con las que fue entrenado, sin ruido adicional.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1. Leer tabla Silver (Simulando datos nuevos)

# COMMAND ----------

# 1. Leer tabla Silver (Simulando datos nuevos)
table_name = "dev.silver.churn_data"
df_spark = spark.read.table(table_name)

# Tomamos una muestra de 10 clientes para la demo
df_inference = df_spark.sample(fraction=0.1, seed=42).limit(10).toPandas()

# 2. Separar Metadatos (Lo que el modelo NO debe ver)
# Guardamos ID y Label real en un dataframe aparte para el reporte final
cols_meta = ["customerID", "Churn"]
df_meta = df_inference[cols_meta].copy()

# 3. Crear X para el modelo (Solo las features procesadas)
# Borramos las columnas que no son features
X_new = df_inference.drop(columns=cols_meta)

print("--- Datos de Entrada al Modelo (X_new) ---")
#print(f"Columnas: {X_new.columns.tolist()}")
display(X_new.head(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2. Ejecución de Inferencia
# MAGIC

# COMMAND ----------

# Generamos las predicciones.

print("🤖 Ejecutando predicciones...")

# Predicción de Clase (0 = No se va, 1 = Se va)
preds = loaded_model.predict(X_new)

# Predicción de Probabilidad (0.0 a 1.0)
probs = loaded_model.predict_proba(X_new)[:, 1]

print("✅ Inferencia finalizada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generación de Reporte de Negocio
# MAGIC Un modelo de ML por sí solo devuelve probabilidades crudas (0.0 - 1.0). Para soportar la operación, transformamos estos números en acciones claras:
# MAGIC
# MAGIC *   **Probabilidad de Fuga:** El score crudo del modelo.
# MAGIC *   **Alerta de Gestión:** Regla de negocio aplicada (Threshold > 0.5) para etiquetar visualmente a los clientes de "ALTO RIESGO".

# COMMAND ----------

# Cruzamos las predicciones con el `customerID` original para que el reporte sea accionable.

# 1. Unimos todo en un reporte final
reporte = df_meta.copy() # Empezamos con ID y Valor Real

# 2. Agregamos las predicciones
reporte["Probabilidad_Fuga"] = probs.round(4)
reporte["Prediccion_Modelo"] = preds

# 3. Regla de Negocio: Alerta Visual
# Si la probabilidad es > 50%, marcamos Alerta Roja
reporte["Alerta_Gestion"] = reporte["Prediccion_Modelo"].apply(
    lambda x: "🔴 ALTO RIESGO" if x == 1 else "🟢 Cliente Seguro"
)

# 4. Visualización Final
print("--- REPORTE FINAL PARA SOPORTE A OPERACIÓN ---")
cols_mostrar = ["customerID", "Probabilidad_Fuga", "Prediccion_Modelo", "Alerta_Gestion", "Churn"]
display(reporte[cols_mostrar])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1. Leer la Data de Negocio "Legible" (Capa Silver)

# COMMAND ----------

# Esta tabla tiene 'Contract', 'InternetService', etc. en texto original
print("📥 Leyendo datos maestros de negocio...")
df_negocio = spark.read.table("dev.silver.clean_data").toPandas()
display(df_negocio.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2. Predicciones y Datos de Negocio
# MAGIC Para que el reporte sea útil a los equipos de Marketing/Retención, cruzamos las predicciones con los datos maestros del cliente (Tabla `dev.silver.clean_data`).
# MAGIC Esto permite ver no solo **quién** se va a ir, sino **por qué** (ej. ver su tipo de contrato o antigüedad de forma legible), facilitando la estrategia de retención.
# MAGIC
# MAGIC Este DataFrame final (`df_enriquecido`) representa la tabla que se podría guardarse en la capa **Gold** o que alimentaría un dashboard de PowerBI/Tableau para el equipo de retención de clientes.

# COMMAND ----------

df_enriquecido = pd.merge(
    reporte,
    df_negocio,
    on="customerID",
    how="left"
)

# 4. Selección de Columnas para el Reporte Final
# Seleccionamos una mezcla de métricas del modelo + datos de contexto
df_enriquecido = df_enriquecido[[
    "customerID",
    "Contract",
    "MonthlyCharges",
    "InternetService",
    "tenure",
    "Churn_x",
    "Prediccion_Modelo",
    "Probabilidad_Fuga",
    "Alerta_Gestion"
]].rename(columns={'Churn_x': 'Churn_Real'})

display(df_enriquecido)