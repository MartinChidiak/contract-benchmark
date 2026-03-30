# Conceptos clave

Glosario de los conceptos técnicos necesarios para entender el proyecto. No es necesario conocerlos en profundidad para ejecutarlo, pero ayuda a interpretar las decisiones de diseño y los resultados.

---

### CUAD (Contract Understanding Atticus Dataset)
Dataset público de 510 contratos legales reales anotados manualmente por abogados, con 41 tipos de cláusulas etiquetadas por contrato. Es el estándar de facto para benchmarking de extracción de información legal en inglés. En este proyecto se usa un subconjunto filtrado por tipo de contrato, y `ground_truth.csv` contiene las respuestas correctas contra las que se evalúa el modelo. Disponible en [atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad).

---

### LLM instruction-tuned
Un modelo de lenguaje grande entrenado en dos etapas: primero aprende a predecir texto en general (pretraining), luego es ajustado para seguir instrucciones (fine-tuning con RLHF o similar). Los modelos usados (`Llama-3.1-8B-Instruct`, `Qwen2.5-3B/7B/14B/32B-Instruct`, `Qwen3-8B`) son todos instruction-tuned, lo que les permite recibir un prompt con reglas y devolver una extracción estructurada en lugar de continuar el texto libremente.

---

### vLLM
Motor de inferencia de alta performance para LLMs. Optimiza el uso de VRAM mediante PagedAttention (manejo eficiente de la KV cache) y permite procesar múltiples contratos en paralelo en una sola GPU. En este proyecto reemplaza llamadas directas a HuggingFace Transformers para acelerar la inferencia sobre cientos de contratos.

---

### Guided Decoding (Structured Outputs)
Técnica que restringe los tokens que el modelo puede generar en cada paso usando una gramática formal derivada del JSON Schema del modelo Pydantic. El resultado es que el modelo **no puede producir JSON inválido**: la estructura, los tipos y los campos están garantizados antes de que empiece a generar. En vLLM esto se implementa con el backend xgrammar a través de `StructuredOutputsParams`.

---

### Pydantic
Librería de validación de datos en Python. En este proyecto define el schema de extracción (`RealEstateContract` en `schema.py`) con tipos, valores permitidos y descripciones de cada campo. Cumple dos roles: genera el JSON Schema para el Guided Decoding y valida la salida del modelo antes de guardarla.

---

### Chat template
Formato de tokens especiales que cada modelo espera para distinguir los roles del prompt (sistema, usuario, asistente). Por ejemplo, Llama-3 usa `<|begin_of_text|><|start_header_id|>system<|end_header_id|>...` mientras que Qwen usa `<|im_start|>system\n...`. `run_experiment.py` aplica automáticamente el template nativo de cada modelo usando su tokenizer, lo que hace el código compatible con cualquier modelo instruction-tuned sin modificaciones.

---

### AWQ (Activation-aware Weight Quantization)
Técnica de cuantización que reduce los pesos del modelo de FP16 (16 bits por peso) a INT4 (4 bits), disminuyendo el tamaño en memoria ~4x con pérdida de calidad mínima. A diferencia de cuantización uniforme, AWQ identifica qué pesos son más importantes para la activación del modelo y los preserva con mayor precisión. En el proyecto se usan `Qwen2.5-14B-AWQ` (~9 GB) y `Qwen2.5-32B-AWQ` (~18 GB), modelos que sin cuantizar requerirían ~28 GB y ~64 GB respectivamente — fuera del rango de una RTX 5090 (24 GB).

---

### Chunking con overlap
Los contratos legales suelen ser más largos que el context window del modelo. La solución es dividirlos en fragmentos (chunks) con una zona de superposición (`overlap_chars`) entre fragmentos consecutivos. El overlap evita que información importante quede cortada justo en el borde de un chunk. Los resultados de cada chunk se combinan luego con `merge_results()`, que prioriza valores no-vacíos y concatena si dos chunks extraen valores distintos para el mismo campo.

---

### Macro accuracy vs Micro accuracy
- **Macro accuracy**: promedio de la accuracy por campo. Incluye los casos donde tanto el modelo como el ground truth coinciden en que el campo está ausente ("Not Mentioned"). Se infla en campos poco frecuentes donde acertar la ausencia es trivial.
- **Micro accuracy**: accuracy calculada solo sobre las filas donde el ground truth contiene un valor real. Mide la capacidad de extracción efectiva del modelo, ignorando los aciertos fáciles de ausencia. Es el indicador más honesto para comparar modelos.

---

### Precision, Recall y F1 en extracción
Adaptados de clasificación binaria al contexto de extracción de información:
- **Precision**: de todo lo que el modelo extrajo, ¿qué porcentaje era correcto? Penaliza alucinaciones.
- **Recall**: de todo lo que existía en el contrato, ¿qué porcentaje encontró el modelo? Penaliza omisiones.
- **F1**: media armónica de ambas. Útil cuando precision y recall se comportan de forma dispar entre modelos.

---

### Streamlit
Framework de Python para construir interfaces web sin escribir HTML/CSS/JavaScript. El script `app.py` define la UI completa en Python puro. Streamlit re-ejecuta el script completo ante cualquier interacción del usuario, lo que requiere usar `session_state` para preservar resultados entre re-renders (especialmente los logs de una corrida en curso).

---

### Docker + CUDA
El proyecto corre dentro de un contenedor Docker basado en `nvidia/cuda:12.6.0-devel-ubuntu24.04`. Esto garantiza reproducibilidad del entorno (versiones de CUDA, Python y dependencias) independientemente del sistema operativo del host. La flag `--gpus all` en `docker run` expone las GPUs físicas al contenedor. El volumen `-v $(pwd):/app` monta el directorio del proyecto para que los resultados persistan fuera del contenedor.
