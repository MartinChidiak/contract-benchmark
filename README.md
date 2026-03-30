# Contract Information Extraction Benchmark

Framework de tesis para evaluar modelos de lenguaje (LLMs) en la tarea de extracción estructurada de información de contratos legales, usando vLLM con Guided Decoding y el dataset CUAD como ground truth.

---

## Descripción general

El sistema ejecuta una matriz de experimentos donde cada experimento combina un **modelo** con una **configuración de prompt**, extrae campos clave de contratos legales en formato JSON estructurado, y evalúa los resultados contra el ground truth de CUAD. Una UI web (Streamlit) permite lanzar corridas, visualizar resultados y comparar experimentos.

---

## Arquitectura

```
filtro_contratos.py     ← Filtra contratos de CUAD por tipo y los copia a Dataset_Filtrado_Tesis/
filtro_labels.py        ← Genera ground_truth.csv filtrando master_clauses.csv por los contratos seleccionados
run_all.py              ← Orquestador principal / fuente de la matriz de experimentos
run_experiment.py       ← Lógica de inferencia con vLLM (un experimento a la vez)
benchmark.py            ← Evaluación: compara JSONs extraídos vs ground_truth.csv
experiment_config.py    ← Definición de RunConfig, ModelConfig, MODELS, versiones de prompts
schema.py               ← Schema Pydantic (RealEstateContract) para Guided Decoding
app.py                  ← UI Streamlit para lanzar corridas y comparar resultados
Dockerfile              ← Imagen Docker con CUDA 12.6 + vLLM + dependencias UI
```

---

## Campos extraídos

El modelo debe extraer 12 campos de cada contrato, mapeados a columnas del dataset CUAD:

| Campo | Tipo | Descripción |
|---|---|---|
| `parties` | Texto | Nombres legales de las partes firmantes |
| `agreement_date` | Fecha (YYYY-MM-DD) | Fecha de firma del contrato |
| `effective_date` | Fecha (YYYY-MM-DD) | Fecha de inicio de obligaciones |
| `expiration_date` | Fecha (YYYY-MM-DD) | Fecha de vencimiento natural |
| `governing_law` | Texto | Jurisdicción aplicable |
| `anti_assignment` | Prohibited / Allowed with consent / Not Mentioned | Restricciones de cesión |
| `renewal_term` | Duración | Período de cada renovación automática |
| `notice_period_to_terminate_renewal` | Duración | Aviso previo para evitar renovación |
| `audit_rights` | Yes / No | ¿Hay derecho de auditoría? |
| `cap_on_liability` | Yes / No | ¿Hay tope de responsabilidad? |
| `termination_for_convenience` | Yes / No | ¿Se puede rescindir sin causa? |
| `liquidated_damages` | Yes / No | ¿Hay daños y perjuicios pactados? |

---

## Modelos soportados

Configurados en `experiment_config.py` (dataclass `ModelConfig` + dict `MODELS`):

| Tag | Modelo HuggingFace | VRAM aprox. |
|---|---|---|
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` | ~16 GB |
| `qwen25_3b` | `Qwen/Qwen2.5-3B-Instruct` | ~6 GB |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` | ~15 GB |
| `qwen25_14b_awq` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | ~9 GB (INT4) |
| `qwen25_32b_awq` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | ~18 GB (INT4) |
| `qwen3_8b` | `Qwen/Qwen3-8B` | ~16 GB |

> Los modelos `*_awq` usan cuantización AWQ (INT4) y requieren la librería `autoawq`.
> Qwen3 usa una variante de prompt con `/no_think` al final para suprimir el modo chain-of-thought y devolver JSON directamente.

---

## Versiones de prompt

Definidas en `experiment_config.py` → `PROMPT_VERSIONS`:

| Versión | Descripción |
|---|---|
| `v1_baseline` | Prompt mínimo, sin reglas específicas |
| `v2_with_date_rules` | Agrega reglas explícitas de extracción de fechas en ISO 8601 |
| `v3_full` | Reglas completas: fechas, anti-assignment, Yes/No, renovación |

Las configuraciones de experimento en `BASE_CONFIGS` combinan versiones de prompt con parámetros adicionales:

| Config | Prompt | Few-shot | Temperatura | Overlap |
|---|---|---|---|---|
| `v1_no_fewshot` | v1_baseline | No | 0.0 | 800 |
| `v2_date_rules` | v2_with_date_rules | No | 0.0 | 800 |
| `v3_full_fewshot` | v3_full | Sí | 0.0 | 800 |
| `v3_overlap1600` | v3_full | Sí | 0.0 | 1600 |
| `v3_temp02` | v3_full | Sí | 0.2 | 800 |

---

## Métricas de evaluación

Calculadas en `run_all.py` → `load_benchmark_summary()` a partir de `benchmark_detailed.csv`:

| Métrica | Definición |
|---|---|
| **macro_acc** | Promedio de accuracy por campo (incluye aciertos absent-absent) |
| **micro_acc** | Accuracy solo sobre filas donde el GT está presente (excluye aciertos "fáciles" de ausencia) |
| **macro_precision** | Promedio de precisión por campo: cuando el modelo extrae, ¿acierta? |
| **macro_recall** | Promedio de recall por campo: cuando el campo existe, ¿lo encuentra? |
| **macro_f1** | Media armónica de macro_precision y macro_recall |

Definiciones de TP/FP/FN:
- **TP**: GT presente **y** predicción correcta
- **FP**: GT ausente **y** modelo extrajo un valor (alucinación)
- **FN**: GT presente **y** predicción incorrecta o ausente
- **TN**: ambos ausentes (no cuenta en P/R/F1)

> `micro_acc` es el indicador más honesto de capacidad de extracción real. Una brecha grande entre `macro_acc` y `micro_acc` indica que el modelo se beneficia de predecir "Not Mentioned" correctamente en campos escasos.

---

## Ejecución

### Con Docker (recomendado)

```bash
# Construir imagen
docker build -t contract-benchmark .

# Lanzar UI Streamlit

# --rm elimina el contenedor automáticamente al cerrarlo (evita acumulación)
docker run --rm --gpus all -p 8501:8501 \
  -v "$(pwd):/app" \
  -v "$HOME/.cache/huggingface:/hf_cache" \
  -e HF_HOME=/hf_cache \
  contract-benchmark \
  streamlit run /app/app.py --server.port 8501 --server.address 0.0.0.0

# Lanzar todos los experimentos por CLI
docker run --rm --gpus all \
  -v $(pwd):/app \
  -v "$HOME/.cache/huggingface:/hf_cache" \
  -e HF_HOME=/hf_cache \
  contract-benchmark \
  python /app/run_all.py

# Solo re-correr benchmark (sin inferencia)
docker run --rm --gpus all \
  -v $(pwd):/app \
  -v "$HOME/.cache/huggingface:/hf_cache" \
  -e HF_HOME=/hf_cache \
  contract-benchmark \
  python /app/run_all.py --only-benchmark

# Solo un subconjunto de modelos
docker run --rm --gpus all \
  -v $(pwd):/app \
  -v "$HOME/.cache/huggingface:/hf_cache" \
  -e HF_HOME=/hf_cache \
  contract-benchmark \
  python /app/run_all.py --models llama31_8b,qwen3_8b
```

### Sin Docker

```bash
pip install vllm outlines pydantic openai accelerate streamlit pandas matplotlib plotly
streamlit run app.py
```

---

## UI Streamlit

La aplicación tiene 5 solapas:

| Solapa | Contenido |
|---|---|
| **Configure & Run** | Selección de modelos, configs, upload de contratos, botón de corrida con UI bloqueada durante el proceso |
| **Results** | Resultados detallados del último experimento corrido |
| **Comparison** | Tabla comparativa entre experimentos con filtro por métrica (Accuracy / Precision / Recall / F1) y selector de campos |
| **Overview** | Gráficos de alto nivel: barras agrupadas, radar chart y heatmap de F1 por campo |
| **Manage Experiments** | Listado de todos los experimentos con filtros por modelo y config, checkboxes para seleccionar y botón de borrado masivo |

Un sidebar compartido entre Comparison y Overview permite filtrar por modelos y configuraciones. Al final del sidebar hay un expander con las definiciones de cada métrica.

---

## Estructura de salida

Cada experimento genera su directorio en `experiments/`:

```
experiments/
  llama31_8b__v3_full_fewshot/
    results/
      <contrato>.txt.json        ← extracción estructurada (un JSON por contrato)
      benchmark_detailed.csv     ← fila por (contrato, campo): pred vs GT
      benchmark_summary.json     ← métricas agregadas del experimento
      inference_summary.json     ← tiempo total de inferencia
  comparison.csv                 ← tabla comparativa de todos los experimentos
```

---

## Dataset

CUAD (Contract Understanding Atticus Dataset) está disponible en [atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad).

- **Ground truth**: `ground_truth.csv` (subconjunto de CUAD generado con `filtro_labels.py`)
- **Contratos**: `Dataset_Filtrado_Tesis/` (subconjunto de CUAD generado con `filtro_contratos.py`)
- La clave de matching se normaliza: minúsculas, sin extensión, espacios colapsados

### Preparación del dataset (ejecutar una sola vez)

Los scripts de preparación toman los archivos originales de CUAD y generan el subconjunto usado en los experimentos.

**Archivos necesarios (no incluidos en el repo):**

| Archivo / Carpeta | Descripción |
|---|---|
| `Contratos_txt/` | Todos los contratos de CUAD en formato `.txt` |
| `master_clauses.csv` | Labels completos de CUAD (todas las cláusulas) |

**Paso 1 — Filtrar contratos por tipo**

`filtro_contratos.py` copia de `Contratos_txt/` solo los contratos cuyos nombres contienen los tipos seleccionados para la tesis (Service Agreement, Affiliate Agreement, Maintenance Agreement, Non-Compete, Management Agreement):

```bash
python filtro_contratos.py
# Genera: Dataset_Filtrado_Tesis/
```

**Paso 2 — Generar ground truth**

`filtro_labels.py` cruza los contratos seleccionados contra `master_clauses.csv` y extrae solo las columnas relevantes al schema de extracción:

```bash
python filtro_labels.py
# Genera: ground_truth.csv
```

Después de estos dos pasos el proyecto está listo para correr experimentos.

---

## Autenticación HuggingFace

Los modelos Llama requieren aceptar los términos de uso en HuggingFace y proveer un token de acceso. Guardar el token en `token.txt` (en el root del proyecto). El script lo lee automáticamente y lo exporta como `HUGGING_FACE_HUB_TOKEN`.

---

## Dependencias principales

| Paquete | Uso |
|---|---|
| `vllm >= 0.7.0` | Motor de inferencia con Guided Decoding (xgrammar backend) |
| `outlines` | Soporte de structured generation |
| `pydantic` | Definición del schema de extracción |
| `streamlit` | UI web |
| `pandas` | Manipulación de datos y tablas |
| `plotly` | Gráficos interactivos (barras, radar, heatmap) |
| `matplotlib` | Requerido por `pandas.Styler.background_gradient()` |
| `transformers` | Tokenizador para aplicar el chat template correcto por modelo |
