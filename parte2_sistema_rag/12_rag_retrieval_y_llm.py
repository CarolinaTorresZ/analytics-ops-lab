# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 2:  Sistema RAG - Recuperación y Generación (Inferencia)
# MAGIC *   **Autor:** Carolina Torres Zapata
# MAGIC *   **Fecha:** 2025-11-24
# MAGIC *   **Contexto:** Este es el componente final del sistema ("The App"). Aquí integramos la **Base de Conocimiento** construida previamente con un **LLM Generativo** (Llama 3 o similar) para responder preguntas de usuario.
# MAGIC *   **Objetivos del Notebook:**
# MAGIC      1.  **Recuperación (Retrieval):** Implementar un motor de búsqueda vectorial en memoria (rápido y eficiente) usando similitud de coseno.
# MAGIC      2.  **Generación (Generation):** Conectar con un LLM mediante `Databricks Serving Endpoints`.
# MAGIC      3.  **Grounding (Seguridad):** Diseñar un *System Prompt* estricto que obligue al modelo a responder **solo** con la información suministrada, mitigando alucinaciones.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración del Entorno y Dependencias
# MAGIC
# MAGIC Antes de iniciar el flujo de inferencia, aseguramos que el cluster tenga las herramientas necesarias para la orquestación del RAG.
# MAGIC
# MAGIC *   **`sentence-transformers`**: Motor local para vectorizar la pregunta del usuario (debe coincidir con la versión usada en la ingesta).
# MAGIC *   **`databricks-sdk[openai]`**: Cliente oficial para interactuar con los **Serving Endpoints** (LLMs) de Databricks de forma segura.
# MAGIC
# MAGIC **Nota Operativa:** Se ejecuta `dbutils.library.restartPython()` para reiniciar el proceso de Python y forzar la carga de las nuevas librerías instaladas sin necesidad de reiniciar todo el cluster.

# COMMAND ----------

# INSTALACIÓN DE LIBRERÍAS
# sentence-transformers: Para vectorizar la pregunta.
# databricks-sdk[openai]: Cliente necesario para hablar con Llama 3.
# mlflow: Para registro de experimentos.
%pip install -U -q sentence-transformers "databricks-sdk[openai]" mlflow databricks-agents

# REINICIO DEL KERNEL
# Obligatorio para aplicar cambios. Al terminar esta celda, la memoria se limpia.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importar Librerías

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
from sentence_transformers import SentenceTransformer
from databricks.sdk import WorkspaceClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Conexión con el LLM
# MAGIC En un entorno operativo, los endpoints pueden cambiar de nombre o estar inactivos.
# MAGIC Implementamos una lógica de **"Health Check"**: iteramos sobre una lista de modelos aprobados (priorizando Llama 3-70B) y nos conectamos al primero que responda exitosamente. Esto evita que el pipeline falle por un error de configuración estática.

# COMMAND ----------

# CONFIGURACIÓN DEL MODELO LLM
LLM_ENDPOINT_NAME = None

def is_endpoint_available(endpoint_name):
    """Verifica si un endpoint responde."""
    try:
        client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        client.chat.completions.create(
            model=endpoint_name, 
            messages=[{"role": "user", "content": "Test"}]
        )
        return True
    except Exception:
        return False

print("🔄 Buscando endpoint activo...")

# Lista de candidatos (Llama 3 es la prioridad)
candidates = [
    "databricks-meta-llama-3-3-70b-instruct", 
    "databricks-meta-llama-3-1-70b-instruct",
    "databricks-claude-3-7-sonnet"
]

for candidate in candidates:
    if is_endpoint_available(candidate):
        LLM_ENDPOINT_NAME = candidate
        break

# Validación estricta: Si no hay modelo, detenemos el notebook
assert LLM_ENDPOINT_NAME is not None, "❌ No se encontró ningún modelo activo."

print(f"🚀 Conectado exitosamente a: {LLM_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Carga de Recursos (Base de Conocimiento)
# MAGIC Inicialización del Motor de Búsqueda (In-Memory)
# MAGIC Cargamos los dos componentes críticos para la fase de recuperación:
# MAGIC 1.  **Base de Conocimiento (Vectores):** La tabla Silver `rag_embeddings` convertida a matrices de NumPy para cálculos matemáticos ultrarrápidos.
# MAGIC 2.  **Encoder (Modelo de Embeddings):** El mismo modelo `all-MiniLM-L6-v2` usado en la ingesta. *Nota: Es vital usar exactamente el mismo modelo para que los vectores sean comparables.*

# COMMAND ----------

#  CARGA DE MOTOR DE BÚSQUEDA

print("⏳ Cargando recursos en memoria...")

# A. Cargar Tabla Silver (Knowledge Base)
TABLA_VECTORES = "dev.silver.rag_embeddings" 

try:
    df_kb = spark.read.table(TABLA_VECTORES).toPandas()
    
    # Convertir a matriz NumPy para velocidad
    kb_matrix = np.stack(df_kb["embedding"].values)
    kb_texts = df_kb["chunk_text"].values
    
    print(f"   ✅ Base de datos cargada: {len(kb_texts)} documentos.")

    # B. Cargar Modelo de Embeddings (Local)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("   ✅ Modelo de vectorización (all-MiniLM-L6-v2) listo.")

except Exception as e:
    print(f"❌ Error cargando recursos: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##  4. Lógica del Sistema RAG (Funciones)
# MAGIC Implementamos la búsqueda semántica mediante **Producto Punto (Similitud de Coseno)**.
# MAGIC El flujo es:
# MAGIC 1.  El usuario hace una pregunta -> Se convierte en vector.
# MAGIC 2.  Comparamos ese vector contra los 12 vectores de nuestra base de datos.
# MAGIC 3.  Seleccionamos los `k=3` fragmentos más similares (con mayor puntaje).
# MAGIC
# MAGIC Aplicamos técnicas de **Prompt Engineering** para soporte operativo:
# MAGIC *   **Rol:** "Asistente Técnico experto en Databricks".
# MAGIC *   **Restricción Negativa:** Si el contexto no tiene la respuesta, el modelo debe admitirlo explícitamente (*"La información disponible no menciona..."*). Esto es crucial para evitar engañar al usuario.

# COMMAND ----------

# DEFINICIÓN DE FUNCIONES (CORE)

def recuperar_contexto(pregunta, k=3):
    """Vectoriza la pregunta y busca los 3 fragmentos más similares."""
    # 1. Vectorizar
    query_vector = embedding_model.encode(pregunta)
    
    # 2. Similitud (Producto Punto)
    scores = np.dot(kb_matrix, query_vector)
    
    # 3. Ranking
    top_indices = np.argsort(scores)[-k:][::-1]
    
    return [kb_texts[i] for i in top_indices]


def sistema_rag(pregunta):
    """Orquestador: Pregunta -> Contexto -> LLM -> Respuesta"""
    print(f"🔎 Analizando: '{pregunta}'")
    
    # PASO 1: RETRIEVAL
    chunks = recuperar_contexto(pregunta, k=3)
    contexto_str = "\n\n".join(chunks)
    
    print(f"   📄 Contexto encontrado: {len(chunks)} fragmentos.")
    
    # PASO 2: GENERATION
    try:
        w = WorkspaceClient()
        client = w.serving_endpoints.get_open_ai_client()
        
        # Prompt del Sistema (Reglas para el LLM)
        system_instructions = f"""
        Eres un Asistente Técnico experto en Databricks.
        Responde a la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado abajo.
        
        Reglas:
        1. Si la respuesta está en el contexto, explícala claramente en español.
        2. Si la respuesta NO está en el contexto, di textualmente: "La información disponible no menciona este tema".
        3. No inventes información.
        
        CONTEXTO:
        {contexto_str}
        """
        
        response = client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": pregunta}
            ],
            temperature=0.1, 
            max_tokens=500
        )
        
        respuesta_final = response.choices[0].message.content
        
        # Salida visual
        print("\n" + "="*60)
        print("🤖 RESPUESTA GENERADA:")
        print("="*60)
        print(respuesta_final)
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error en la generación: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##5. Pruebas (Interacción)
# MAGIC Ejecutamos escenarios de prueba para validar el comportamiento del sistema:
# MAGIC 1.  **Caso Positivo:** Pregunta sobre "Unity Catalog" (información presente en el documento). Se espera una respuesta técnica y precisa.
# MAGIC 2.  **Caso Negativo (Control):** Pregunta sobre "DBUs" (concepto de facturación no presente en el texto introductorio). Se espera que el sistema active la cláusula de seguridad y **no** invente una respuesta.

# COMMAND ----------

# Pregunta 1: Sobre Gobernanza
sistema_rag("¿Para qué sirve Unity Catalog?")

# Pregunta 2: Sobre Infraestructura
sistema_rag("¿Qué son las DBUs?")

sistema_rag("¿Qué servicios ofrece Azure Databricks?")