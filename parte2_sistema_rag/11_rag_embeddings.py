# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 2:  Sistema RAG - Generación de Embeddings
# MAGIC *   **Autor:** Carolina Torres Zapata
# MAGIC *   **Fecha:** 2025-11-24
# MAGIC *   **Contexto:** Para que un sistema pueda "buscar por significado" y no solo por palabras clave exactas, necesitamos convertir el texto en representaciones numéricas (vectores).
# MAGIC *   **Objetivo del Notebook:**
# MAGIC      1.  Cargar los fragmentos de texto procesados (Chunks) desde la capa Silver.
# MAGIC      2.  **Vectorización:** Utilizar un modelo de lenguaje pre-entrenado para transformar cada chunk en un vector de n dimensiones.
# MAGIC      3.  **Persistencia:** Guardar la tabla enriquecida con embeddings, que servirá como nuestra "Base de Conocimiento Vectorial".
# MAGIC
# MAGIC **Decisión de Arquitectura:**
# MAGIC Se utiliza un modelo **Open Source (HuggingFace)** ejecutado localmente en el driver. Esto elimina la dependencia de APIs externas (como OpenAI o Azure OpenAI) para este paso, reduciendo costos y latencia de red para volúmenes moderados de datos.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importar Librerías

# COMMAND ----------

import pandas as pd
import numpy as np
#!pip install sentence_transformers
from sentence_transformers import SentenceTransformer
from pyspark.sql.types import ArrayType, FloatType

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Carga de Datos Procesados
# MAGIC Recuperamos la tabla `dev.silver.rag_chunks`. Esto garantiza la trazabilidad del dato: solo vectorizamos lo que ya pasó por la limpieza y segmentación aprobada.

# COMMAND ----------

df_chunks = spark.read.table("dev.silver.rag_chunks").toPandas()

print(f"📚 Cargados {len(df_chunks)} fragmentos de conocimiento.")
display(df_chunks.head(2))

# COMMAND ----------

# MAGIC %md
# MAGIC #  3. Generación de Embeddings (Modelo Local)
# MAGIC
# MAGIC **Selección del Modelo: `all-MiniLM-L6-v2`**
# MAGIC Cumpliendo con el requisito de usar un servicio gratuito/Open Source, elegimos este modelo de `sentence-transformers` por su excelente balance operativo:
# MAGIC *   **Velocidad:** Es muy rápido (ideal para CPUs estándar de clusters pequeños).
# MAGIC *   **Tamaño:** Genera vectores compactos (384 dimensiones), lo que optimiza el almacenamiento y la velocidad de búsqueda posterior (similitud de coseno).
# MAGIC
# MAGIC El resultado es una nueva columna `embedding` que contiene la "huella digital semántica" de cada párrafo.

# COMMAND ----------


#  Utilizamos el modelo `all-MiniLM-L6-v2`. Este modelo transforma el texto en un vector de 384 dimensiones que captura el significado.

print("⏳ Descargando y cargando modelo de embeddings en el driver...")

# Descarga automática del modelo
# Este modelo es gratuito y Open Source (HuggingFace)
model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ Modelo cargado. Iniciando vectorización...")

def generar_embedding_real(texto):
    # Genera el vector
    embedding = model.encode(texto)
    # Lo convertimos a lista de Python para que Spark lo entienda
    return embedding.tolist()

# Aplicamos el modelo a nuestros chunks
df_chunks["embedding"] = df_chunks["chunk_text"].apply(generar_embedding_real)

print("✅ Embeddings Semánticos Generados.")
print(f"Dimensión del vector: {len(df_chunks['embedding'].iloc[0])}")

# Convertir a Spark
df_spark_embeddings = spark.createDataFrame(df_chunks)

display(df_spark_embeddings)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Almacenamiento de la Base Vectorial
# MAGIC Guardamos el resultado en `dev.silver.rag_embeddings`.
# MAGIC Esta tabla Delta es el activo más valioso del sistema RAG, ya que combina:
# MAGIC 1.  El ID del chunk (Trazabilidad).
# MAGIC 2.  El texto original (Contexto para el LLM).
# MAGIC 3.  El vector matemático (Índice de búsqueda).

# COMMAND ----------

target_table = "dev.silver.rag_embeddings"

print(f"💾 Guardando tabla Delta en: {target_table} ...")

df_spark_embeddings.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

print("✅ Proceso finalizado exitosamente.")