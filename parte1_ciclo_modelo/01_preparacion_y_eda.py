# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 1: Ciclo de vida de un modelo - Preparación y EDA
# MAGIC *   **Autor:** Carolina Torres Zapata
# MAGIC *   **Fecha:** 2025-11-24
# MAGIC *   **Propósito:** Cargar el dataset Telco Customer Churn, realizar un EDA mínimo y preparar los datos para mod]elado.
# MAGIC *   **Resumen del Flujo:**
# MAGIC      1. Descargar datos crudos desde fuente externa y persistirlos en la capa **Bronze** (Raw).
# MAGIC      2. Realizar perfilamiento básico (EDA) para identificar nulos, tipos de datos incorrectos y desbalance de clases.
# MAGIC      3. Limpiar y estructurar los datos para la capa **Silver**, dejándolos listos para el pipeline de entrenamiento.
# MAGIC
# MAGIC **Arquitectura de Datos (Medallion)**
# MAGIC Se utiliza una estructura de capas lógica para garantizar trazabilidad:
# MAGIC *   `dev.bronze`: Datos crudos inmutables.
# MAGIC *   `devo.silver`: Datos limpios y estructurados (Input para modelos).
# MAGIC    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creación Catálogo y Tablas
# MAGIC Creación de los esquemas (bases de datos) para organizar las tablas
# MAGIC Arquitectura y convenciones<br>
# MAGIC • Crear catalogo llamado dev<br>
# MAGIC • Crear dos esquemas llamados bronce y silver<br>
# MAGIC • **Capa Bronce** (Raw/Ingesta): Datos exactamente como llegan de la fuente, sin cambiarles nada.<br>
# MAGIC • **Capa Silver** (Clean/Refined): Datos limpios y estructurados.<br>

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Celda 1: Creación de la estructura
# MAGIC CREATE CATALOG IF NOT EXISTS dev;
# MAGIC USE CATALOG dev;
# MAGIC CREATE SCHEMA IF NOT EXISTS bronce;
# MAGIC CREATE SCHEMA IF NOT EXISTS silver;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importar Librerías

# COMMAND ----------

#!pip install ydata-profiling

# COMMAND ----------

import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import re

from pyspark.sql import SparkSession

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md ## 2. Carga y Exploración Inicial de los Datos

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Ingesta y Capa Bronce
# MAGIC Descargamos el dataset "Telco Customer Churn" directamente de la fuente raw para garantizar reproducibilidad. Los datos se guardan inmediatamente en la capa **Bronze** para tener un respaldo histórico inmutable.

# COMMAND ----------


# URL del dataset (Fuente: IBM/Kaggle repository)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

try:
    # 1. Lectura en memoria (Pandas)
    print(f"⬇️ Descargando datos desde: {url}")
    df = pd.read_csv(url)
    
    # 2. Persistencia en BRONCE (Spark Delta Table)
    # Convertimos a Spark para aprovechar el almacenamiento optimizado de Databricks
    df_raw_spark = spark.createDataFrame(df)
    
    df_raw_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev.bronce.telco_customer_churn")
    
    print(f"   Registros: {df.shape[0]} | Columnas: {df.shape[1]}")

except Exception as e:
    print(f"❌ Error crítico en la ingesta: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Análisis Exploratorio de Datos (EDA) Mínimo
# MAGIC Realizamos validaciones clave para determinar la estrategia de limpieza.
# MAGIC *   Tipos de datos.
# MAGIC *   Valores nulos o inconsistentes.
# MAGIC *   Distribución de la variable objetivo (`Churn`).

# COMMAND ----------

def info_completa(df, nombre_df):
    print(f"\n📊 --- INFORME: {nombre_df} ---")
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\nTipos de datos:")
    print(df.dtypes)
    
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    print("\nValores únicos por columna:")
    print(df.nunique())

    print("\n=== DUPLICADOS ===")
    print(f"Total duplicados: {df.duplicated().sum()}")

    print("\nPorcentaje de valores faltantes por columna:")
    print((df.isnull().mean() * 100).round(2))

    print("\nPrimeras 5 filas:")
    display(df.head(5))


info_completa(df, "Telco Customer Churn")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Análisis Variable Objetivo

# COMMAND ----------

# Distribución de la variable objetivo
print("\n📌 Distribución de Churn:")
print(df["Churn"].value_counts())
print("\nPorcentaje:")
print(df["Churn"].value_counts(normalize=True) * 100)

# COMMAND ----------

def plot_categoricas_vs_target(df, target="Churn"):
    """
    Grafica variables categóricas vs la variable objetivo.
    
    Parámetros:
    df (DataFrame): dataframe original
    target (str): nombre de la variable objetivo
    """
    
    # columnas categorías
    categoricas = df.select_dtypes(include=["object"]).columns.tolist()
    
    # excluir ID y target (si es object)
    for col in ["customerID", target]:
        if col in categoricas:
            categoricas.remove(col)
    
    print(f"Variables categóricas detectadas: {categoricas}")
    
    for col in categoricas:
        fig = px.histogram(
            df,
            x=col,
            color=target,
            barmode="group",
            title=f"{col} vs {target}",
            template="plotly_white"
        )
        fig.update_layout(xaxis_title=col, yaxis_title="Count")
        fig.show()


plot_categoricas_vs_target(df)


# COMMAND ----------

# Convertir TotalCharges a numérico
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

num_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

for target in ['Churn']:
    print(f"\n=== Variables numéricas vs {target.upper()} ===\n")
    for col in num_features:
        if col in df.columns:
            plt.figure(figsize=(6,3))
            sns.boxplot(x=target, y=col, data=df, palette='pastel')
            plt.title(f"{col} vs {target.upper()}")
            plt.show()
            
            stats = df.groupby(target)[col].agg(['mean','median','std']).round(2)
            print(stats, "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Transformación y Capa Silver
# MAGIC Aplicamos reglas de negocio y limpieza técnica:
# MAGIC 1.  **Corrección de Tipos:** `TotalCharges` se convierte a numérico.
# MAGIC 2.  **Imputación:** Los valores vacíos se rellenan con la media.
# MAGIC
# MAGIC **Estrategia de Transformación:**
# MAGIC
# MAGIC 1.  **Gestión de Identificadores:** Se aísla el `customerID` antes de las transformaciones para preservarlo. Esto es vital para la **trazabilidad del negocio** en la etapa de inferencia (saber qué cliente específico tiene riesgo de fuga).
# MAGIC 2.  **Variables Numéricas (`StandardScaler`):** Se estandarizan columnas como `tenure`, `MonthlyCharges` y `TotalCharges` para que tengan media 0 y desviación estándar 1, optimizando la convergencia de algoritmos lineales.
# MAGIC 3.  **Variables Categóricas (`OneHotEncoder`):** Se convierten las variables nominales (ej. `InternetService`, `PaymentMethod`) en variables dummy numéricas.
# MAGIC     *   *Configuración:* Se utiliza `drop='first'` para evitar la multicolinealidad (trampa de las variables ficticias).
# MAGIC 4.  **Reconstrucción del Dataset:** Finalmente, se concatenan las características transformadas con el `Target` y el `customerID`, reordenando las columnas para dejar el identificador al inicio por facilidad de lectura.

# COMMAND ----------

# 5.1 Hacer copia de trabajo
df_clean = df.copy()

# 5.2 TotalCharges -> numérico y imputación
df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].mean())

# 5.3 Tratar SeniorCitizen como categórica (0/1 => categoría)
df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype("object")

# Guardamos el ID en una variable separada para pegarlo al final
ids = df_clean["customerID"].values 

# 5.4 Codificar target (No en el transformer)
df_clean["Churn"] = df_clean["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# 5.5 Separar X e y (y fuera del pipeline)
y = df_clean["Churn"]
X = df_clean.drop(columns=["Churn", "customerID"])  # customerID lo excluimos del modelado

# 5.6 Identificar columnas categóricas y numéricas
categoricas = X.select_dtypes(include=["object"]).columns.tolist()
numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\n📌 Categóricas detectadas:", categoricas)
print("📌 Numéricas detectadas:", numericas)

# 5.7 Definir ColumnTransformer con OHE (drop='first') y StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categoricas),
        ("num", StandardScaler(), numericas)
    ],
    remainder="drop"  # solo dejamos las columnas procesadas
)

# 5.8 Fit + transform (sobre X)
X_encoded_np = preprocessor.fit_transform(X)

# 5.9 Obtener nombres de columnas RESULTANTES de forma robusta
# Este método devuelve los nombres en el mismo orden que las columnas del output
feature_names = preprocessor.get_feature_names_out()

# 5.10 Construir DataFrame codificado de forma segura
X_encoded = pd.DataFrame(X_encoded_np, columns=feature_names)

# 5.11 Volver a añadir el target al final
df_encoded = X_encoded.copy()
df_encoded["Churn"] = y.values
df_encoded["customerID"] = ids 

# Obtener nombre de la última columna
ultima_columna = df_encoded.columns[-1]

# Crear nueva lista de columnas con la última al principio
columnas_reordenadas = [ultima_columna] + df_encoded.columns[:-1].tolist()

# Reindexar el DataFrame
df_encoded = df_encoded.reindex(columns=columnas_reordenadas)

print("\n✔ Preprocesamiento completado sin desalineación")
print("Shape df_encoded:", df_encoded.shape)
display(df_encoded.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Normalización de Nombres (Compatibilidad Delta Lake)
# MAGIC Se aplica una función de limpieza (`sanitización`) a los nombres de las columnas para eliminar espacios, paréntesis y caracteres especiales. Este paso es **obligatorio** para evitar errores de sintaxis al guardar el DataFrame como tabla en Spark/Delta.

# COMMAND ----------

def clean_column_name(col):
    col = col.strip()

    # Eliminar prefijos del ColumnTransformer
    col = col.replace("cat__", "")
    col = col.replace("num__", "")

    # Reemplazar espacios por _ 
    col = col.replace(" ", "_")
    col = col.replace("_Yes", "")
    col = col.replace("_1", "")

    # Eliminar paréntesis
    col = col.replace("(", "").replace(")", "")

    # Reemplazar guiones por _
    col = col.replace("-", "_")

    # Eliminar caracteres especiales NO permitidos
    col = re.sub(r"[^0-9a-zA-Z_]+", "", col)

    # Reducir múltiples ___ -> _
    col = re.sub("_+", "_", col)

    return col

# Aplicar limpieza
df_encoded.columns = [clean_column_name(c) for c in df_encoded.columns]

print("\n✔ Nombres de columnas normalizados para Spark/Delta:")
print(df_encoded.columns.tolist()[:40])  # mostrar las primeras 40 columnas
display(df_encoded.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Guadar Datos Transformados
# MAGIC Convertimos el DataFrame procesado a formato Spark y lo almacenamos en la tabla Delta `dev.silver.churn_data`. Esto asegura que los datos limpios y transformados estén disponibles y centralizados para la fase de entrenamiento.

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

print("\nGuardando tabla limpias en dev.silver.churn_data ...")
spark_df = spark.createDataFrame(df_encoded)  # pandas -> spark

# Si quieres validar antes:
print("Primeras filas Spark:")
display(spark_df.limit(5).toPandas())

# Guardar como tabla Delta en esquema 'dev.silver'
spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev.silver.churn_data")
print("✔ Tabla dev.silver.churn_data creada/reemplazada correctamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ##5. Guadar la data limpia (Negocio)
# MAGIC Guardamos una copia de los datos limpios (sin nulos ni errores) pero **sin transformar** en la tabla `dev.silver.clean_data`. Esta versión conserva los valores originales (texto legible) para garantizar el contexto de negocio, trazabilidad y facilitar el análisis posterior o el enriquecimiento de reportes.

# COMMAND ----------

print("\nGuardando tabla limpias en dev.silver.clean_data ...")
spark_df_clean = spark.createDataFrame(df_clean)  # pandas -> spark

# Si quieres validar antes:
print("Primeras filas Spark:")
display(spark_df_clean.limit(5).toPandas())

# Guardar como tabla Delta en esquema 'dev.silver'
spark_df_clean.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("dev.silver.clean_data")
print("✔ Tabla dev.silver.clean_data creada/reemplazada correctamente.")