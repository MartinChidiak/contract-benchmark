# Flujo del proyecto — paso a paso

Este documento explica cómo se conectan los scripts, en qué orden se ejecutan y qué hace cada uno internamente.

---

## Visión general

```
[CUAD original]
      │
      ▼
filtro_contratos.py  ──→  Dataset_Filtrado_Tesis/   (contratos .txt)
filtro_labels.py     ──→  ground_truth.csv           (labels de referencia)
      │
      ▼
experiment_config.py  ──→  define modelos, prompts y parámetros
schema.py             ──→  define la estructura JSON que el modelo debe producir
      │
      ▼
run_all.py
  └─→ run_experiment.py  ──→  experiments/<modelo>__<config>/results/*.json
  └─→ benchmark.py       ──→  benchmark_detailed.csv + benchmark_summary.json
      │
      ▼
app.py  (UI Streamlit — hace todo lo anterior desde el browser)
```

---

## Fase 1 — Preparación de datos (una sola vez)

### `filtro_contratos.py`

**Entrada:** `Contratos_txt/` — todos los contratos de CUAD en formato `.txt`

**Qué hace:** recorre la carpeta y copia solo los archivos cuyos nombres contienen los tipos de contrato seleccionados para la tesis:
- Service Agreement
- Affiliate Agreement
- Maintenance Agreement
- Non-Compete
- Management Agreement

**Salida:** `Dataset_Filtrado_Tesis/` — subconjunto de contratos a evaluar

```bash
python filtro_contratos.py
```

---

### `filtro_labels.py`

**Entrada:**
- `master_clauses.csv` — archivo maestro de CUAD con labels de todos los contratos
- `Dataset_Filtrado_Tesis/` — para saber qué contratos fueron seleccionados

**Qué hace:** filtra el CSV maestro para quedarse solo con las filas que corresponden a los contratos del paso anterior, y extrae únicamente las columnas que mapean a los 12 campos del schema.

**Salida:** `ground_truth.csv` — tabla de referencia usada en el benchmark

```bash
python filtro_labels.py
```

---

## Fase 2 — Configuración de experimentos

### `schema.py`

Define la clase `RealEstateContract` usando Pydantic. Es el **contrato de datos** entre el modelo y el sistema:
- Especifica los 12 campos a extraer con sus tipos y descripciones
- Genera el JSON Schema que se le pasa a vLLM para el Guided Decoding
- Fuerza que todos los campos sean requeridos en el schema para que xgrammar no cierre el JSON antes de generarlos todos

No se ejecuta directamente — es importado por `run_experiment.py`.

---

### `experiment_config.py`

Define dos cosas:

1. **`PROMPT_VERSIONS`** — diccionario con las versiones de prompt disponibles:
   - `v1_baseline`: prompt mínimo
   - `v2_with_date_rules`: agrega reglas de extracción de fechas
   - `v3_full`: reglas completas (fechas, anti-assignment, Yes/No, renovación)

2. **`RunConfig`** — dataclass con todos los parámetros de un experimento:
   - modelo, directorio de entrada/salida, versión de prompt, few-shot, temperatura, overlap, context window

   Cada `RunConfig` es completamente serializable a JSON (`config.json`), lo que garantiza reproducibilidad total.

No se ejecuta directamente — es importado por `run_all.py` y `run_experiment.py`.

---

## Fase 3 — Ejecución de experimentos

### `run_all.py`

Es el **orquestador**. Define la matriz completa de experimentos cruzando modelos × configuraciones y los ejecuta en secuencia.

**Qué contiene:**
- `MODEL_IDS` — mapeo de tag corto a nombre HuggingFace (ej. `llama31_8b` → `meta-llama/Llama-3.1-8B-Instruct`)
- `BASE_CONFIGS` — lista de configuraciones de prompt/parámetros
- `EXPERIMENTS` — producto cartesiano de modelos × configs, genera un `RunConfig` por combinación
- `load_benchmark_summary()` — lee `benchmark_detailed.csv` y calcula todas las métricas (macro_acc, micro_acc, precision, recall, F1)

**Flujo interno:**
```
Para cada experimento:
  1. run_experiment.run(config)   ← inferencia
  2. benchmark.benchmark(...)    ← evaluación
  3. load_benchmark_summary()    ← métricas
  4. guarda benchmark_summary.json
Al final:
  5. imprime tabla comparativa
  6. guarda experiments/comparison.csv
```

**Optimización importante:** los experimentos del mismo modelo se agrupan consecutivamente para minimizar las recargas del modelo en VRAM (cargar un modelo de 8B toma varios minutos).

```bash
python run_all.py                        # todos los modelos y configs
python run_all.py --models llama31_8b    # solo un modelo
python run_all.py --only-benchmark       # re-evaluar sin re-inferir
```

---

### `run_experiment.py`

Ejecuta **un solo experimento** definido por un `RunConfig`. Es llamado por `run_all.py` pero también puede ejecutarse standalone.

**Flujo interno por contrato:**

```
1. Leer el .txt del contrato
2. ¿Entra en el context window?
   ├── Sí → procesar como un solo bloque (short contract)
   └── No → dividir en chunks con overlap (long contract)
         overlap_chars evita perder información en los cortes

3. Para cada bloque:
   a. build_messages()        ← arma el prompt (system + user + few-shot opcional)
   b. apply_chat_template()   ← convierte mensajes al formato del modelo (Llama/Qwen/etc.)
   c. vLLM.generate()         ← inferencia con Guided Decoding
                                 (el modelo SOLO puede producir JSON válido según el schema)

4. Para contratos largos (múltiples chunks):
   merge_results()  ← combina los JSONs de cada chunk
                      prioriza valores no-vacíos, concatena textos si difieren

5. normalize_extracted_data()  ← estandariza formatos (Yes/No, Not Mentioned, null)
6. RealEstateContract(**data)  ← valida con Pydantic
7. Guarda <contrato>.txt.json  ← resultado final
```

**Manejo de contratos ya procesados:** si el `.json` ya existe para un contrato, lo saltea. Esto permite reanudar una corrida interrumpida sin rehacer trabajo.

**Salida por experimento:**
```
experiments/<modelo>__<config>/results/
  <contrato>.txt.json      ← extracción estructurada
  config.json              ← RunConfig serializado (reproducibilidad)
  inference_summary.json   ← tiempo total, conteos de éxito/fallo
```

---

### `benchmark.py`

Evalúa los resultados comparando cada campo extraído contra el ground truth.

**Flujo interno:**
```
Para cada contrato en ground_truth.csv:
  Para cada campo (12 campos):
    1. Leer predicción del .json correspondiente
    2. Normalizar predicción y ground truth con la función apropiada:
       - fechas     → parse_date()          (convierte a YYYY-MM-DD)
       - duraciones → normalize_duration()  (convierte a "N unit(s)")
       - Yes/No     → normalize_yes_no()
       - parties    → parties_match()       (comparación por tokens, tolerante a aliases)
       - governing_law → normalize_governing_law()
    3. Comparar y registrar match=1 o match=0
```

**Salida:** `benchmark_detailed.csv` — una fila por (contrato, campo) con predicción, ground truth normalizado y resultado del match.

Este CSV es la fuente de todas las métricas calculadas luego en `load_benchmark_summary()`.

---

## Fase 4 — UI (opcional, alternativa a CLI)

### `app.py`

Interfaz web que envuelve todo el flujo anterior. Internamente hace exactamente lo mismo que `run_all.py` + `benchmark.py`, pero desde el browser.

**Solapas:**

| Solapa | Qué hace |
|---|---|
| **Configure & Run** | Seleccionás modelos y configs, subís contratos, ejecutás. La UI se bloquea mientras corre para evitar modificaciones durante la inferencia. Los parámetros se guardan en `session_state` antes del rerun para no perderlos. |
| **Results** | Muestra el `benchmark_detailed.csv` y el `inference_summary.json` del último experimento corrido. |
| **Comparison** | Lee `experiments/comparison.csv` y muestra una tabla comparativa con filtro por métrica (Accuracy / Precision / Recall / F1) y por campo. |
| **Overview** | Tres gráficos de alto nivel filtrados por modelo/config: barras agrupadas, radar chart y heatmap de F1 por campo. |

```bash
streamlit run app.py
# o desde Docker:
docker run --gpus all -v $(pwd):/app -p 8501:8501 contract-benchmark \
  streamlit run /app/app.py --server.address 0.0.0.0
```

---

## Dónde vive cada resultado

```
experiments/
  comparison.csv                          ← tabla final de todos los experimentos

  llama31_8b__v3_full_fewshot/
    results/
      NNN_contrato.txt.json               ← extracción del modelo
      config.json                         ← parámetros exactos del experimento
      inference_summary.json              ← tiempo y conteos
      benchmark_detailed.csv             ← match por (contrato, campo)
      benchmark_summary.json             ← métricas agregadas

  qwen25_7b__v3_full_fewshot/
    results/
      ...
```

---

## Flujo de datos resumido

```
Contrato .txt
    │
    ▼
Chunks (si es largo)
    │
    ▼
Prompt (system + reglas + few-shot + texto)
    │
    ▼
vLLM + Guided Decoding  ──→  JSON garantizado por el schema
    │
    ▼
normalize + validate (Pydantic)
    │
    ▼
<contrato>.txt.json
    │
    ▼
benchmark.py compara vs ground_truth.csv
    │
    ▼
benchmark_detailed.csv
    │
    ▼
load_benchmark_summary()  ──→  macro_acc, micro_acc, precision, recall, F1
    │
    ▼
comparison.csv  /  app.py Overview
```
