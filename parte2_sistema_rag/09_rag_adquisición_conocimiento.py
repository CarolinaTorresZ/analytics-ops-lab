# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 2:  Sistema RAG - Ingesta y Adquisición de Conocimiento
# MAGIC *   **Autor:** Carolina Torres Zapata
# MAGIC *   **Fecha:** 2025-11-24
# MAGIC *   **Contexto:**
# MAGIC La base de un sistema RAG (Retrieval-Augmented Generation) es la calidad de sus datos. En este notebook implementamos la primera fase del pipeline ETL para LLMs: la adquisición de datos no estructurados.
# MAGIC *  **Objetivo:**
# MAGIC      1.  Conectarse a una fuente externa (Documentación oficial de Microsoft Learn).
# MAGIC      2.  Extraer el contenido HTML y aplicar una limpieza estructural (eliminar ruido HTML).
# MAGIC      3.  Persistir la información cruda en la **Capa Bronce** utilizando **Unity Catalog Volumes**, asegurando la trazabilidad del dato fuente antes del procesamiento (chunking).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importar Librerías

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
import requests
from bs4 import BeautifulSoup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Extracción y Limpieza (Web Scraping)
# MAGIC Simulamos la ingesta de un documento técnico. Para garantizar la calidad de los embeddings posteriores, aplicamos una estrategia de limpieza:
# MAGIC
# MAGIC 1.  **Request Robusto:** Verificación de estado HTTP (200 OK) y forzado de encoding UTF-8 para manejo correcto de tildes/eñes.
# MAGIC 2.  **Eliminación de Ruido:** Removemos etiquetas HTML irrelevantes para el contenido (`<script>`, `<style>`, `<nav>`, `<footer>`) que solo añadirían tokens basura al contexto del LLM.
# MAGIC 3.  **Extracción Selectiva:** Solo conservamos encabezados (`h1`-`h3`), párrafos (`p`) y listas (`li`) dentro del contenedor principal.
# MAGIC

# COMMAND ----------


# URL oficial de Microsoft Learn
url = "https://learn.microsoft.com/es-es/azure/databricks/introduction/"

# 1. Descargar HTML
response = requests.get(url)
response.raise_for_status()


# Codificación a UTF-8 para que lea bien las tildes y ñ
response.encoding = 'utf-8' 

# 2. Parsear HTML
soup = BeautifulSoup(response.text, "html.parser")

# 3. Limpieza de elementos irrelevantes
for tag in soup(["script", "style", "nav", "footer", "header", "aside", "meta", "link"]): 
    tag.decompose() 

# 4. Extraer contenido principal
main = soup.find(id="main") or soup 

# 5. Extraer texto limpio
texts = [] 
for tag in main.find_all(["h1", "h2", "h3", "p", "li"]): 
    # get_text strip=True quita espacios extra
    # replace('\xa0', ' ') quita los "espacios de no separación" que ensucian el texto
    content = tag.get_text(" ", strip=True).replace('\xa0', ' ')
    if content: 
        texts.append(content) 

# Unir todo
full_text = "\n\n".join(texts)

# Verificamos
print(full_text[:1000]) # Imprimimos los primeros 500 caracteres para validar

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Persistencia en Capa Bronce (Unity Catalog Volumes)
# MAGIC En lugar de procesar y "chunkear" inmediatamente en memoria, guardamos el texto limpio como un archivo físico en un **Volumen de Unity Catalog**.
# MAGIC
# MAGIC **Por qué hacemos esto (Soporte Operacional):**
# MAGIC *   **Auditabilidad:** Tenemos una copia fiel de lo que se descargó en ese momento.
# MAGIC *   **Desacoplamiento:** Si cambiamos la estrategia de *chunking* (ej. de 500 a 1000 caracteres), no necesitamos volver a hacer scraping de la web, solo releemos este archivo Bronce.

# COMMAND ----------

# Ruta donde guardar el archivo en la capa Bronze
output_path = "/Volumes/dev/bronce/azure_databricks_docs/azure_databricks_intro.txt"

# Guardar el texto como archivo .txt
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print("Archivo guardado correctamente en:", output_path)
