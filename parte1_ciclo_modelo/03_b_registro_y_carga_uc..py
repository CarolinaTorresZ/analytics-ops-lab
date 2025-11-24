# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 1: Ciclo de vida de un modelo - Registro / Almacenamiento del Modelo con Unity Catalog 
# MAGIC *   **Autor:** Carolina Torres Zapata
# MAGIC *   **Fecha:** 2025-11-24
# MAGIC *   **Contexto:**
# MAGIC Este notebook implementa el **escenario ideal** de operación productiva solicitado en la prueba (Punto 4.a). Utiliza **Unity Catalog** como *Model Registry* centralizado.
# MAGIC
# MAGIC **Ventajas de este enfoque:**
# MAGIC 1.  **Nombre Estable:** El modelo se invoca por un nombre semántico (`dev.ml_models.churn_model`) en lugar de un hash aleatorio.
# MAGIC 2.  **Gobierno de Datos:** El modelo reside en el mismo catálogo que los datos, permitiendo control de accesos unificado.
# MAGIC 3.  **Gestión de Versiones:** Permite cargar versiones específicas (ej. `Version 1`) o Alias (ej. `@Champion`, `@Production`).

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
# MAGIC ## 2. Configuración del Entorno de Gobierno
# MAGIC Aseguramos que exista el esquema (base de datos) dentro del Unity Catalog donde residirán nuestros modelos registrados.
# MAGIC *   **Catalog:** `dev`
# MAGIC *   **Schema:** `ml_models` (espacio dedicado para artefactos de ML)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS dev.ml_models;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Definición del URI del Modelo Gobernado
# MAGIC  
# MAGIC En lugar de buscar dinámicamente el "mejor run" (como en el método de fallback), aquí ingresamos los valores que el equipo de Data Science ha registrado la versión estable en Unity Catalog.
# MAGIC
# MAGIC Utilizamos la estructura de tres niveles de UC: `catalogo.esquema.nombre_modelo`.

# COMMAND ----------

# URI del modelo registrado en Unity Catalog
model_name = "dev.ml_models.churn_model"
model_version = 1

model_uri_uc = f"models:/{model_name}/{model_version}"

print("📌 Model URI:", model_uri_uc)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Carga del Modelo para Inferencia
# MAGIC Usamos el sabor `pyfunc` de MLflow. Esto es una buena práctica operativa porque abstrae la librería subyacente (Scikit-Learn).

# COMMAND ----------

loaded_model_uc = mlflow.pyfunc.load_model(model_uri_uc)
print("✔ Modelo UC cargado exitosamente como PyFunc MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1. Leer tabla Silver (Simulando datos nuevos)
# MAGIC **Simulación de Pipeline de Inferencia (Batch)**
# MAGIC Leemos los datos más recientes de la capa **Silver**.
# MAGIC *   **Nota:** Al igual que en el entrenamiento, separamos estrictamente los identificadores (`customerID`) de las variables predictivas (`X_new`) para evitar sesgos o errores de esquema.

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
# MAGIC ### 4.2. Ejecución de Inferencia
# MAGIC El modelo registrado en Unity Catalog recibe el DataFrame de Pandas (o Spark) y devuelve las predicciones.

# COMMAND ----------

preds = loaded_model_uc.predict(X_new)
print("✅ Inferencia finalizada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generación de Reporte de Negocio
# MAGIC Convertimos las predicciones técnicas (arrays de 0s y 1s) en información útil para el negocio, re-asociando los resultados con los `customerID` originales.

# COMMAND ----------

# Cruzamos las predicciones con el `customerID` original para que el reporte sea accionable.

# 1. Unimos todo en un reporte final
reporte = df_meta.copy() # Empezamos con ID y Valor Real

# 2. Agregamos las predicciones
#reporte["Probabilidad_Fuga"] = probs.round(4)
reporte["Prediccion_Modelo"] = preds

# 4. Visualización Final
print("--- REPORTE FINAL PARA SOPORTE A OPERACIÓN ---")
#cols_mostrar = ["customerID", "Probabilidad_Fuga", "Prediccion_Modelo", "Alerta_Gestion", "Churn"]
cols_mostrar = ["customerID",  "Prediccion_Modelo", "Churn"]
display(reporte[cols_mostrar])