# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 2:  Sistema RAG - Procesamiento y Chunking
# MAGIC *   **Autor:** Carolina Torres Zapata
# MAGIC *   **Fecha:** 2025-11-24
# MAGIC *   **Contexto:** Una vez adquirido el dato crudo (Capa Bronce), el paso crítico en un sistema RAG es la **segmentación (Chunking)**. Si cortamos el texto arbitrariamente, rompemos las oraciones y el LLM pierde contexto.
# MAGIC *   **Objetivo del Notebook:**
# MAGIC      1.  **Lectura Raw:** Cargar el archivo de texto desde el Volumen de Unity Catalog.
# MAGIC      2.  **Chunking Semántico:** Implementar una lógica que respete los párrafos y oraciones, agrupándolos en bloques de tamaño óptimo (ej. ~1000 caracteres) para la ventana de contexto del LLM.
# MAGIC      3.  **Estructuración:** Convertir la lista de textos en una Tabla Delta (Capa Silver) con identificadores únicos (`chunk_id`) para trazabilidad futura.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importar Librerías
# MAGIC

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Lectura desde Volumen (Capa Bronce/Raw)
# MAGIC Leemos el archivo directamente desde el almacenamiento gestionado de Databricks
# MAGIC
# MAGIC A diferencia de los procesos ETL tradicionales que leen línea por línea (como CSVs), aquí necesitamos el documento entero como una sola unidad para poder analizar sus párrafos.
# MAGIC Usamos `.option("wholetext", True)` para cargar el contenido completo en una sola fila, preservando los saltos de línea (`\n`) que son vitales para identificar la estructura del documento.

# COMMAND ----------

ruta_volumen = "/Volumes/dev/bronce/azure_databricks_docs/azure_databricks_intro.txt" 

# 'wholetext' lee todo el archivo en una sola fila de un DataFrame de Spark
df_spark_raw = spark.read.option("wholetext", True).text(ruta_volumen)

print("✅ Texto cargado en Spark DataFrame.")
df_spark_raw.printSchema()

# Extraemos el texto del DataFrame para partirlo
full_text = df_spark_raw.first()[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Estrategia de Chunking (Preservación de Contexto)
# MAGIC Implementamos una estrategia de **"Ventana Deslizante basada en Párrafos"**.
# MAGIC En lugar de cortar ciegamente cada 1000 caracteres (lo que podría dejar una frase como *"La clave es..."* en un chunk y *"...la seguridad"* en otro), el algoritmo:
# MAGIC
# MAGIC 1.  Divide el texto por párrafos naturales (`\n\n`).
# MAGIC 2.  Agrupa párrafos completos hasta acercarse al límite de tokens/caracteres (1000).
# MAGIC 3.  **Regla de Calidad:** Filtra líneas de "navegación web" o pies de página cortos (< 50 chars) que son ruido para el modelo.

# COMMAND ----------


# ==========================================
# 1. ESTRATEGIA DE CHUNKING (Lógica Híbrida)
# ==========================================
# Objetivo: Agrupar párrafos completos hasta llegar a un límite de caracteres.
# Beneficio: Evita cortar frases a la mitad y agrupa títulos con su contenido.

# Separación natural por párrafos
paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]

# Inicializar variables
chunks = []
current_chunk = ""
LIMIT_CHARS = 1000  # Límite máximo por chunk

# Iteración para empaquetar párrafos hasta límite
for para in paragraphs:
    # Filtrar párrafos cortos o de sistema
    if len(para) < 50 or "acceso a esta página" in para.lower():
        continue
    
    if len(current_chunk) + len(para) + 2 <= LIMIT_CHARS:
        # Agregar párrafo al chunk actual
        if current_chunk:
            current_chunk += "\n\n" + para
        else:
            current_chunk = para
    else:
        # Guardar chunk actual y empezar uno nuevo
        chunks.append(current_chunk)
        current_chunk = para

# Guardar el último chunk
if current_chunk:
    chunks.append(current_chunk)

print(f"🧩 Se generaron {len(chunks)} chunks optimizados.")
print(f"   Promedio de caracteres por chunk: {sum(len(c) for c in chunks)/len(chunks):.0f}\n")

# Muestra de primeros 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:200]}...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Estructuración y Trazabilidad (Data Modeling)
# MAGIC Para que el sistema RAG funcione, cada fragmento de texto necesita una "cédula de identidad".
# MAGIC Generamos un `chunk_id` único usando `monotonically_increasing_id()`.
# MAGIC *   **Uso posterior:** Cuando el usuario haga una pregunta, el sistema recuperará el ID del chunk más relevante.

# COMMAND ----------

# Creamos un DataFrame Spark a partir de la lista de chunks
# Cada chunk será una fila en la columna "chunk_text"
df_chunks = spark.createDataFrame([(c,) for c in chunks], ["chunk_text"])

# Agregamos una columna "chunk_id" con un identificador único para cada fila
# `monotonically_increasing_id()` genera un ID único creciente para cada chunk
df_chunks = df_chunks.withColumn("chunk_id", monotonically_increasing_id())

# Confirmamos que el DataFrame se creó correctamente mostrando las primeras filas
print("✅ DataFrame Spark de chunks creado:")
display(df_chunks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Guardar tabla Delta en Capa Silver
# MAGIC Guardamos el resultado procesado en `dev.silver.rag_chunks` en formato **Delta**.
# MAGIC Esto permite:
# MAGIC 1.  **Reutilización:** Si falla el proceso de generación de embeddings, no hay que volver a leer ni procesar el texto.
# MAGIC 2.  **Schema Enforcement:** Aseguramos que siempre tengamos las columnas `chunk_text` y `chunk_id`.

# COMMAND ----------

target_table = "dev.silver.rag_chunks"

print(f"💾 Guardando tabla Delta en: {target_table} ...")

df_chunks.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

print("✅ Proceso finalizado exitosamente.")