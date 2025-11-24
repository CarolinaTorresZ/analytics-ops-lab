# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 3:   Escenarios de Soporte
# MAGIC *   **Autor:** Carolina Torres Zapata
# MAGIC *   **Fecha:** 2025-11-24
# MAGIC *   **Contexto:** En la operación diaria, los "caminos felices" son raros. Los datos cambian (Schema Drift), los despliegues fallan por mala configuración y el código optimizado incorrectamente rompe la producción.
# MAGIC *   **Objetivo del Notebook:**
# MAGIC Resolver tres incidentes simulados aplicando buenas prácticas:
# MAGIC      1.  **Escenario 3.1:** Robustecer un pipeline de inferencia ante cambios inesperados en los datos de entrada.
# MAGIC      2.  **Escenario 3.2:** Corregir la carga de modelos utilizando la API programática de MLflow para evitar errores de hardcoding.
# MAGIC      3.  **Escenario 3.3:** Reparar un bug lógico en el sistema de recuperación (RAG) utilizando operaciones vectorizadas.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escenario 3.1: Schema drift en datos de entrada
# MAGIC
# MAGIC ### Diagnóstico y Corrección
# MAGIC
# MAGIC **Qué estaba mal:**
# MAGIC El uso directo (`df_scoring.select(cols)`) es frágil; falla inmediatamente con un `AnalysisException` si el dataset de inferencia tiene una columna menos o un nombre diferente (Drift), y no gestiona columnas nuevas que podrían ensuciar el proceso.
# MAGIC
# MAGIC **Corrección:**
# MAGIC 1.  **Renombrado:** Normalización de nombres mediante un diccionario de equivalencias para manejar variaciones comunes.
# MAGIC 2.  **Imputación:** Creación automática de columnas faltantes con `NULL` para mantener el esquema requerido.
# MAGIC 3. **Validación de Tipos:** Conversión explícita (`cast`) de cada columna al tipo de dato esperado por el modelo.
# MAGIC 3.  **Selección:** Filtrado final explícito para ordenar columnas y descartar atributos extra no utilizados por el modelo.
# MAGIC
# MAGIC **Por qué es adecuada:**
# MAGIC Esta solución asegura la **resiliencia operativa**. El pipeline no se detiene por cambios menores en la fuente, garantizando que el modelo siempre reciba la matriz de características exacta que espera para predecir, protegiendo la continuidad del servicio.

# COMMAND ----------

# =====================================
# IMPORTAR LIBRERÍAS
# =====================================
from pyspark.sql.functions import col, lit
import mlflow

# =====================================
# 1. ESQUEMA ESPERADO
# =====================================
expected_cols = df_train.columns
expected_schema = dict(df_train.dtypes)

print(f"Columnas esperadas por el modelo: {len(expected_cols)}")

# ===============================================================
# 2. MAPEO DE NOMBRES DE COLUMNAS (Name Normalization)
# ===============================================================
# Este diccionario actúa como una “tabla de equivalencias”.
# Sirve para corregir variaciones comunes de columnas que cambian
# cuando los equipos agregan transformaciones o nuevas fuentes.

name_mapping = {
    "customer_id": ["cust_id", "client_id", "id_cliente"],
    "monthly_charges": ["monthlycharge", "costo_mensual"],
    "tenure_months": ["tenure", "meses_tenencia"],
    "gender": ["Genero", "Gender_ID"],
    "TotalCharges": ["Total_Charges", "TotalCharge"]
}

# ===============================================================
# 3. RENOMBRAR COLUMNAS BASADAS EN EL MAPEO
# ===============================================================
df_fixed = df_scoring
cols_scoring = df_scoring.columns

print("\n🔄 Normalizando nombres según lista de variantes...")

for target_col, variants in name_mapping.items():
    for alt_name in variants:
        if alt_name in cols_scoring and target_col not in cols_scoring:
            print(f"   ↪️ Renombrando '{alt_name}' → '{target_col}'")
            df_fixed = df_fixed.withColumnRenamed(alt_name, target_col)
            break  # detenemos tras renombrar una coincidencia

cols_scoring = df_fixed.columns  # refrescar

# ===============================================================
# 4. AGREGAR COLUMNAS FALTANTES
# ===============================================================
# Cuando el scoring recibe un archivo sin todas las columnas
# (caso real: nuevas conexiones de clientes, versiones del upstream),
# el modelo se rompe.

missing_cols = [c for c in expected_cols if c not in cols_scoring]

print("\n Verificando columnas faltantes...")

for col_name in missing_cols:
    print(f"   ⚠️ '{col_name}' no existe en scoring → creando con NULL.")
    df_fixed = df_fixed.withColumn(col_name, lit(None))

cols_scoring = df_fixed.columns


# ===============================================================
# 5. REMOVER COLUMNAS EXTRA
# ===============================================================
extra_cols = [c for c in cols_scoring if c not in expected_cols]

if extra_cols:
    print(f"\n Eliminando columnas extra no usadas por el modelo: {extra_cols}")

df_fixed = df_fixed.select(expected_cols)

# ===============================================================
# 6. CORREGIR TIPOS DE DATO (Type Drift)
# ===============================================================
print("\n🛠 Verificando consistencia de tipos...")

current_schema = dict(df_fixed.dtypes)

for col_name, expected_type in expected_schema.items():
    if current_schema[col_name] != expected_type:
        print(f"   🔧 '{col_name}' ({current_schema[col_name]} → {expected_type})")
        df_fixed = df_fixed.withColumn(col_name, col(col_name).cast(expected_type))

# ===============================================================
# 7. LOG COMPLETO A MLFLOW
# ===============================================================
drift_report = {
    "missing_columns": missing_cols,
    "extra_columns": extra_cols,
    "name_mapping_used": name_mapping,
}

mlflow.log_dict(drift_report, "schema_drift_report.json")

print("\n📝 Schema Drift Report registrado en MLflow.")
print("🎉 Esquema final compatible con el modelo listo para scoring.")

df_final_scoring = df_fixed
df_final_scoring.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Escenario 3.2:  Carga incorrecta de modelo en MLflow
# MAGIC
# MAGIC ### Diagnóstico y Corrección
# MAGIC **Qué estaba mal:**
# MAGIC 1.  **Nombre Incorrecto:** El código intentaba cargar `churn_model_prod` cuando el modelo registrado se llama `churn_model`.
# MAGIC 2.  **Stage Inexistente:** No existía un stage `Production` configurado.
# MAGIC 3.  **Falta de Metadatos:** El modelo carecía de etiquetas de trazabilidad requeridas por el negocio.
# MAGIC
# MAGIC **Corrección:**
# MAGIC *   Se utilizó `MlflowClient` para **listar y verificar** la existencia del modelo y sus versiones antes de cargar.
# MAGIC *   Se implementó la carga dinámica apuntando a la **última versión disponible** (`latest_version`) en lugar de un stage hardcodeado.
# MAGIC *    Se inyectaron los metadatos solicitados (framework, project_id) directamente en la versión específica del modelo para mejorar la gobernanza y trazabilidad.
# MAGIC
# MAGIC **Por qué es adecuada:**
# MAGIC Esta solución elimina la fragilidad de depender de nombres y estados "hardcodeados". Al inspeccionar programáticamente el registro antes de cargar, el script se vuelve robusto ante nuevos despliegues y garantiza que siempre se utilice el artefacto más reciente y correctamente etiquetado para auditoría.

# COMMAND ----------

from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

# ============================================================
# 1. DEFINICIÓN DE RECURSOS
# ============================================================
model_name = "churn_model" 

# ============================================================
# 2. LISTAR Y SELECCIONAR VERSIÓN
# ============================================================
# Buscamos todas las versiones
try:
    versions = client.search_model_versions(f"name='{model_name}'")
except Exception as e:
    print(f" El modelo '{model_name}' no existe en el registro.")
    versions = []

if not versions:
    print("No se encontraron versiones disponibles. Deteniendo proceso.")
    # En un script real, aquí haríamos un raise Exception o exit()
else:
    print(f"✅ Se encontraron {len(versions)} versiones.")
    
    # Ordenamos y tomamos la última
    versions_sorted = sorted(versions, key=lambda x: int(x.version), reverse=True)
    latest_version_obj = versions_sorted[0]
    latest_version_id = latest_version_obj.version
    
    print(f"   -> Seleccionada la versión: v{latest_version_id} (Stage: {latest_version_obj.current_stage})")

    # ============================================================
    # 3. CARGA DEL MODELO
    # ============================================================
    # Construimos la URI dinámica apuntando a la versión específica
    model_uri = f"models:/{model_name}/{latest_version_id}"
    print(f"\n🚀 Cargando modelo desde: {model_uri}")
    
    try:
        # Usamos sklearn para carga nativa (o pyfunc si es genérico)
        model = mlflow.pyfunc.load_model(model_uri)
        print(" Modelo cargado exitosamente en memoria.")
    except Exception as e:
        print(f" Error crítico cargando el artefacto: {e}")

    # ============================================================
    # 4. GESTIÓN DE METADATOS (TAGS)
    # ============================================================
    # Aplicamos etiquetas a la VERSIÓN específica que acabamos de validar.
    # Esto permite saber qué framework usó esa versión puntual.
    tags = {
        "model_framework": "sklearn",
        "project_id": "123456",
        "model_type": "regression"
    }

    print("\n🏷 Aplicando etiquetas de gobernanza...")
    for k, v in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=latest_version_id,
            key=k,
            value=v
        )
        print(f"   + Tag '{k}' añadido a versión {latest_version_id}")

    print("\n🎉 Proceso de corrección y carga finalizado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escenario 3.3 – Sistema RAG que siempre devuelve contexto vacío
# MAGIC
# MAGIC ### Diagnóstico y Corrección
# MAGIC
# MAGIC **Problema identificado:**  
# MAGIC La función `retrieve_relevant_chunks` original devolvía siempre un DataFrame vacío.  
# MAGIC Esto ocurría por dos causas típicas:
# MAGIC 1. El ordenamiento usaba los valores **ascendentes** en lugar de descendentes.  
# MAGIC 2. La comparación del umbral se aplicaba antes de ordenar, filtrando todos los registros.
# MAGIC
# MAGIC **Solución aplicada:**  
# MAGIC Implementé una versión mínima y correcta del recuperador usando NumPy:
# MAGIC
# MAGIC - Se cargan los embeddings a memoria una sola vez.
# MAGIC - La similitud se calcula mediante **producto punto vectorizado**.
# MAGIC - Se seleccionan los *k* chunks con mayor score usando `argsort()[-k:][::-1]`.
# MAGIC
# MAGIC **Por qué funciona:**  
# MAGIC La recuperación ya no depende de operaciones fila por fila ni umbrales incorrectos.  
# MAGIC El vectorizado garantiza que siempre se obtienen los documentos más similares y que el bloque de generación del LLM llega a la rama donde hay contexto válido.
# MAGIC
# MAGIC **Resultado:**  
# MAGIC La función devuelve chunks reales, el sistema RAG tiene contexto y el LLM genera respuestas basadas en ese contenido.

# COMMAND ----------

import numpy as np

# Cargar embeddings en memoria (solo una vez)
df_local = df_spark_embeddings.toPandas()

emb_matrix = np.stack(df_local["embedding"].values)  # matriz NumPy (N x d)
chunk_texts = df_local["chunk_text"].values          # array de textos

def retrieve_relevant_chunks(query_emb, k=3):
    """
    Recupera los k chunks más similares a la embedding de la consulta.
    Optimizado usando NumPy (sin UDFs y sin toPandas dentro).
    """

    # A. Similaridad (Producto Punto)
    sims = np.dot(emb_matrix, query_emb)

    # B. Top-k (ordenamos descendente)
    top_idx = sims.argsort()[-k:][::-1]

    # C. Retornar texto + score
  
    return [(chunk_texts[i], float(sims[i])) for i in top_idx]


# ===========================================================
# Pregunta de prueba — debe generar contexto válido
# ===========================================================
pregunta = "¿Qué es Azure Databricks?"
query_emb = embedding_model.encode(pregunta)

top_chunks = retrieve_relevant_chunks(query_emb, k=3)

# Preparar el texto para el generador
context_str = "\n".join([t for t, s in top_chunks])

# 3. Vañidación
if len(top_chunks) > 0:
    print("\n VALIDACIÓN EXITOSA: Entró a la rama con contexto.")

    answer = llm_generate(pregunta, context_str)
    print(f"   Resultado final: {answer}")
else:
    print("\n ERROR: Rama vacía (El bug persiste).")
