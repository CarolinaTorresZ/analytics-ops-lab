# Prueba Técnica – Soporte a la Operación de Modelos Analíticos (IT)

**Candidata:** Carolina Torres Zapata  
**Fecha:** 24 de Noviembre 2025  
**Plataforma:** Databricks (Community/Free Edition)

## 📋 Resumen Ejecutivo
Este repositorio contiene la solución técnica enfocada en la **operación, diagnóstico y mantenimiento** de flujos analíticos, priorizando la robustez y la trazabilidad (MLOps) sobre la complejidad algorítmica, en cumplimiento con los criterios de evaluación.

El proyecto demuestra:
1.  Flujo End-to-End de ML con trazabilidad **MLflow**.
2.  Implementación de un sistema **RAG** (Retrieval-Augmented Generation) funcional y seguro.
3.  Capacidad de diagnóstico y **corrección de código (Debugging)** en escenarios de soporte críticos.

---

## 📂 Estructura del Proyecto

El repositorio está organizado siguiendo el flujo lógico de la prueba y una arquitectura de datos ordenada:

```text
.
├── parte1_ciclo_modelo/          # Flujo MLOps End-to-End (Churn)
│   ├── 01_preparacion_y_eda.ipynb
│   ├── 02_entrenamiento_y_tracking.ipynb
│   ├── 03_a_registro_y_carga_artefactos.ipynb  (Enfoque Resiliente/Fallback)
│   └── 03_b_registro_y_carga_uc.ipynb          (Enfoque Enterprise/Unity Catalog)
│
├── parte2_sistema_rag/           # Sistema de Preguntas y Respuestas
│   ├── 09_rag_ingesta.ipynb
│   ├── 10_rag_chunking.ipynb
│   ├── 11_rag_embeddings.ipynb
│   └── 12_rag_retrieval_y_llm.ipynb
│
├── parte3_escenarios_soporte/    # Resolución de Incidentes
│   └── 20_escenarios_soporte.ipynb
│
├── data/                         # Simulación Data Lake (Medallion Architecture)
│   ├── bronze/                   # Datos crudos (csv, txt)
│   ├── silver/                   # Datos procesados y tablas delta
│   └── ml_models/                
│
└── README.md                     # Documentación técnica
```

## ⚙️ Guía de Ejecución (Cómo ejecutar los notebooks)

Los notebooks están numerados secuencialmente dentro de sus carpetas para facilitar la ejecución. Se recomienda seguir este orden:

### 1. 📂 `parte1_ciclo_modelo/`
*   **01_preparacion_y_eda:** Ejecutar primero para limpiar los datos y generar el dataset base.
*   **02_entrenamiento_y_tracking:** Entrena los modelos y genera los artefactos en MLflow.
*   **03_registro_y_carga:** *Nota:* Se incluyen dos versiones para cubrir los puntos 4.a y 4.b de las instrucciones:
    *   `03_a_registro_y_carga_artefactos`: Ejecutar si no se tiene acceso a Unity Catalog (Uso de `runs:/`).
    *   `03_b_registro_y_carga_uc`: Ejecutar para validar el flujo ideal con Unity Catalog (`models:/`).

### 2. 📂 `parte2_sistema_rag/`
*   **09_rag_ingesta:** Descarga y limpieza del HTML.
*   **10_rag_chunking:** Segmentación semántica.
*   **11_rag_embeddings:** Generación de vectores.
*   **12_rag_retrieval_y_llm:** Orquestación final y chat con el LLM.

### 3. 📂 `parte3_escenarios_soporte/`
*   **20_escenarios_soporte:** Contiene la resolución de los tres incidentes en un solo notebook autocontenido.

---

## 🛠 Parte 1: Ciclo de vida de un modelo (Enfoque Operacional)

Se implementó un pipeline completo de MLOps para predecir la fuga de clientes.

### 1. Información del Dataset
*   **Nombre:** Telco Customer Churn (IBM/Kaggle).
*   **URL de Origen:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
*   **Variable Objetivo:** `Churn` (Binaria: Yes/No).

### 2. Ejecución y Estrategia
*   **Preparación:** Limpieza de datos, conversión de tipos numéricos y manejo de nulos.
*   **Entrenamiento:** Se entrenaron múltiples modelos (Logistic Regression, Random Forest) registrando métricas (AUC, Accuracy) y parámetros en **MLflow Tracking** para garantizar la auditabilidad del experimento.
*   **Despliegue Híbrido (Decisión Técnica):**
    *   Se incluyen **dos estrategias** de carga en el paso 3:
    *   `03_a`: Carga basada en **Artefactos (Run ID)** Demuestra cómo cargar el modelo productivo directamente desde los **Artefactos de MLflow** usando el `Run ID`, asegurando que la operación no se detenga por fallos en el catálogo central.
    *   `03_b`: Carga basada en **Unity Catalog**. Estándar de gobierno para producción ("Model Registry").

---

## 🤖 Parte 2: Sistema RAG Mínimo (Montaje y Operación)

Se construyó un sistema de *Retrieval-Augmented Generation* modularizado para consultar documentación técnica.

### 1. Fuente del Documento
*   **Tema Principal:** Introducción a Azure Databricks (Plataforma unificada de análisis).
*   **URL:** [Documentación Oficial Microsoft Learn](https://learn.microsoft.com/es-es/azure/databricks/introduction/)

### 2. Arquitectura Técnica
*   **Ingesta y Chunking:** Extracción limpia de HTML y segmentación semántica  por **párrafos completos** con un límite de 1000 caracteres. Esto preserva el contexto semántico mejor que el corte arbitrario por longitud fija.
*   **Embeddings (Decisión de Costo/Eficiencia):**
    *   Se utilizó el modelo Open Source **`sentence-transformers/all-MiniLM-L6-v2`** ejecutado localmente.
    *   *Justificación:* Dado que el entorno gratuito no posee endpoints de embeddings de pago (Azure OpenAI / Databricks Foundation Models provisionados). Permite generar embeddings de alta calidad localmente en el driver del cluster sin costos adicionales ni dependencias de API externas.
*   **Retrieval:** Búsqueda vectorial mediante similitud de coseno (Producto punto).
*   **LLM & Grounding:**
    *   Se utilizó una función de "Health Check" que busca dinámicamente endpoints activos (Llama 3) para garantizar la resiliencia del notebook.
    *   Se implementó un *System Prompt* estricto para evitar alucinaciones: si la respuesta no está en el contexto, el modelo responde: *"La información disponible no menciona este tema"*.

---

## 🚨 Parte 3: Escenarios de Soporte (Diagnóstico y Corrección)

Resolución de bugs críticos y mejora de código para producción.

| Escenario | Problema Detectado | Solución Implementada |
| :--- | :--- | :--- |
| **3.1 Schema Drift** | El pipeline fallaba ante cambios de nombres o nuevas columnas. | Se implementó un esquema defensivo: Mapeo de sinónimos, imputación de nulos para columnas faltantes y casting explícito de tipos. |
| **3.2 Carga de Modelo** | Uso de nombres y stages "hardcodeados" que no existían. | Uso de `MlflowClient` para búsqueda dinámica de la última versión disponible e inyección de etiquetas de gobierno (`project_id`, `framework`). |
| **3.3 RAG Bug** | Recuperación vacía por error lógico en ordenamiento. | Reescritura usando **NumPy vectorizado**: cálculo de producto punto y uso de `argsort` descendente para garantizar la recuperación de los Top-K documentos. |

---

## 💾 Acceso a Datos (Simulación Medallion)

Para facilitar la validación de la prueba sin acceso directo al Workspace de Databricks, se adjunta en este repositorio la carpeta **/data** simulando la arquitectura Medallion con los resultados reales de la ejecución:

*   `data/bronze/`: Archivos crudos (CSV Kaggle, TXT Azure Docs).
*   `data/silver/`: Datos procesados (`churn_data`,`clean_data` ,`rag_chunks`, `rag_embeddings`).
*   `ml_models/` :  Contiene el **artefacto serializado**  (`churn_model_prod.pkl`).
    *   *Propósito:* Este archivo permite validar el modelo entrenado localmente (usando `joblib.load`) sin necesidad de conectarse al servidor de MLflow o Unity Catalog de Databricks.

*Nota: Los notebooks están configurados para leer tablas (`spark.read.table`), representando el entorno productivo real.*

---

## ⚠️ Limitaciones y Workarounds

1.  **Entorno Gratuito (Databricks Community):**
    *   *Limitación:* No hay acceso completo a ciertas funcionalidades empresariales de Unity Catalog (Governance avanzado) ni GPUs para entrenamiento pesado.
    *   *Workaround:* Se utilizaron **Unity Catalog Volumes** para la gestión de archivos y modelos ligeros (Scikit-Learn / Sentence-Transformers) que corren eficientemente en CPU.

2.  **API de LLM:**
    *   El notebook 12 asume la existencia de un Endpoint de Databricks (`databricks-meta-llama-3...`). Se incluyó una lógica de **"Health Check"** que busca dinámicamente endpoints disponibles para evitar fallos si el nombre del modelo cambia.

---
