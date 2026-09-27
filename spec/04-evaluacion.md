# 04 — Framework de Evaluación

## Resumen

La evaluación se realiza en **dos niveles complementarios**:

1. **Nivel Agente**: RAGAS o DeepEval evalúan la calidad de las respuestas generadas por el agente. En un flujo
   separado, `agents-cli eval` ejecuta cuatro Expectations periodísticas sobre las trazas ADK para verificar
   trazabilidad, temporalidad, grounding e idioma.
2. **Nivel Retrieval (RAG aislado)**: Evalúa la calidad de los fragmentos recuperados por el sistema de búsqueda
   vectorial, usando métricas clásicas de Information Retrieval.

Adicionalmente, se mencionan métricas de espectrogramas y calidad de audio que quedan **fuera del scope** de esta
evaluación pero están documentadas para futuras extensiones.

`agents-cli eval` complementa al evaluador RAG seleccionado sobre el mismo conjunto de consultas: genera trazas del
agente ADK, evalúa grounding, uso de tools y calidad de respuesta, y califica las Expectations periodísticas. Sus
resultados no se mezclan en un único promedio.

### Evaluación de trazas con agents-cli

Los casos se almacenan en `tests/eval/datasets/basic-dataset.json` y la configuración en `tests/eval/eval_config.yaml` .
La configuración ejecuta las métricas built-in `hallucination` , `grounding` , `tool_use_quality` y
`final_response_quality` , además de las cuatro Expectations periodísticas como métricas LLM custom.

```bash
# Requiere DATASET_PATH y OPENAI_API_KEY para ejecutar el agente localmente.
agents-cli eval generate --dataset tests/eval/datasets/basic-dataset.json
agents-cli eval grade --config tests/eval/eval_config.yaml
```

Las trazas se guardan en `artifacts/traces/` y los resultados en `artifacts/grade_results/` . Cada cambio de prompt,
tool o retrieval debe compararse contra la corrida anterior con `agents-cli eval compare` .

## Nivel 1: Evaluación del Agente con un evaluador RAG configurable

### Principio de diseño

La metodología es independiente de la biblioteca: se aplica *metrics-driven
development* mediante un ciclo continuo de generar o seleccionar consultas,
ejecutar el agente, capturar la evidencia recuperada y calificar el resultado.
La implementación selecciona **un** backend por corrida con
`EVALUATION_FRAMEWORK=ragas|deepeval`; no se comparan ni promedian scores de
ambos frameworks porque sus jueces y escalas operativas pueden diferir.

El contrato común de cada muestra es la tripleta `(q, C, a)`:

1. `q`: consulta de evaluación, manual o sintética.
2. `C`: contextos realmente devueltos por las tools de retrieval, con
   `segment_id`, fuente y timestamps.
3. `a`: respuesta final generada por el agente.

La superficie ADK de producción (`src.fast_api_app`) expone el endpoint interno
`POST /evaluate/query`, que conserva `C` y los metadatos de
segmento. Nunca se debe sustituir `C` por `ground_truth_contexts`: hacerlo
haría que una evaluación de retrieval midiera el contexto ideal y no el que
recuperó el sistema.

RAGAS puede generar datasets sintéticos para ampliar cobertura, y DeepEval
puede ejecutar sus propios generadores o casos de pytest. Todo caso aceptado
se normaliza al mismo esquema de este documento antes de evaluar.

### Selección del backend

```bash
# RAGAS (default): dependencias de evaluación existentes
uv sync --group eval
EVALUATION_FRAMEWORK=ragas

# DeepEval: backend alternativo sobre el mismo contrato de muestras
uv sync --extra eval-deepeval
EVALUATION_FRAMEWORK=deepeval

# Configuración que ambos backends registran en el resultado
EVALUATION_JUDGE_MODEL=gpt-4o-mini
EVALUATION_JUDGE_TEMPERATURE=0
```

En el código actual, DeepEval pasa `EVALUATION_JUDGE_MODEL` a sus métricas, pero no pasa
`EVALUATION_JUDGE_TEMPERATURE`. RAGAS registra ambos valores como metadatos, aunque su llamada a `ragas.evaluate()` no
inyecta todavía el modelo ni la temperatura configurados. Por ello no debe afirmarse reproducibilidad del juez LLM a
partir de estas variables hasta completar ese wiring.

La entrada se selecciona con `EVALUATION_DATASET_SOURCE` (ver *Fuentes de
dataset* más abajo) o directamente con `EVALUATION_DATASET_PATH`, y el launcher
establece el backend desde `.env`:

```bash
uv run python -m evaluation.run_agent_evaluation \
  --dataset evaluation/test_datasets/synthetic/ragas_synthetic_questions.json \
  --agent-url http://localhost:8000 \
  --output evaluation/results/agent_evaluation.json
```

Cada resultado registra `framework`, modelo y temperatura declarados, métricas agregadas y el detalle por pregunta,
incluidos `contexts` y `retrieved_segments`. RAGAS agrega cada score directamente en la muestra; DeepEval los agrupa
en `per_question[].metrics` junto con la explicación del juez.

### Fuentes de dataset configurables

`evaluation/evaluation_config.py` resuelve la fuente del dataset desde el
entorno y la registra en cada corrida:

| Fuente (`EVALUATION_DATASET_SOURCE`) | Contenido | Archivo |
| --- | --- | --- |
| `manual` (default) | Preguntas anotadas por humanos | `EVALUATION_MANUAL_DATASET_PATH` (el default apunta a `ragas_eval_dataset_unlabeled.json`, ausente hoy) |
| `synthetic` | Generadas con RAGAS TestsetGenerator sobre el corpus | `EVALUATION_SYNTHETIC_DATASET_PATH` |
| `hybrid` | Unión de manual + sintético, sin duplicados | `EVALUATION_HYBRID_DATASET_PATH` |

`EVALUATION_DATASET_PATH` sobrescribe cualquiera de las anteriores.
`EVALUATION_DATASET_SIZE` y `EVALUATION_DATASET_MAX_SEGMENTS` acotan la
generación sintética.

La configuración default de `manual` no es ejecutable en un checkout actual porque
`evaluation/test_datasets/ragas_eval_dataset_unlabeled.json` no está versionado. Se debe definir
`EVALUATION_MANUAL_DATASET_PATH`/`EVALUATION_DATASET_PATH` o usar la fuente `synthetic`. El archivo híbrido también se
materializa al ejecutar el launcher con `EVALUATION_DATASET_SOURCE=hybrid`; no existe antes de esa generación.

Generación de preguntas sintéticas:

```bash
# RAGAS TestsetGenerator sobre segmentos del corpus (EVALUATION_GENERATION_*:
# workers, timeout, idioma)
uv run python evaluation/generate_synthetic_questions.py

# Dataset balanceado: para cada segmento, una pregunta conceptual y una
# acústica/auditiva que apuntan al mismo segmento (compara tipos de query)
uv run python evaluation/generate_balanced_questions.py
```

Los datasets generados viven en `evaluation/test_datasets/synthetic/`
(`ragas_synthetic_questions.json`, `balanced_journalistic_questions*.json`,
`clap_journalistic_acoustic_questions.json`, ...).

### ¿Qué es RAGAS?

RAGAS (Retrieval Augmented Generation Assessment) es un framework de evaluación para sistemas RAG que mide la calidad de
las respuestas generadas en relación con el contexto recuperado y la pregunta original.

### Métricas RAG principales

| Métrica                 | Qué mide                                                                  | Rango | Objetivo |
| ----------------------- | ------------------------------------------------------------------------- | ----- | -------- |
| **Faithfulness**        | ¿La respuesta es fiel al contexto recuperado? (no alucina)                | 0-1   | >= 0.7   |
| **Answer Relevancy**    | ¿La respuesta es relevante para la pregunta?                              | 0-1   | >= 0.7   |
| **Context Utilization** | ¿La respuesta generada aprovecha contextos pertinentes sin referencia?    | 0-1   | >= 0.6   |
| **Context Precision**   | ¿Los contextos recuperados son relevantes? (precision)                    | 0-1   | >= 0.6   |
| **Context Recall**      | ¿Se recuperaron todos los contextos relevantes?                           | 0-1   | >= 0.6   |
| **Answer Correctness**  | ¿La respuesta es factualmente correcta?                                   | 0-1   | >= 0.6   |

En RAGAS, `answer_correctness`, `context_precision` y `context_recall` sólo se ejecutan cuando **todas** las muestras
tienen `ground_truth`; de lo contrario se usan `faithfulness`, `answer_relevancy` y `context_utilization`. DeepEval
construye siempre `ContextualPrecisionMetric`, `ContextualRecallMetric`, `FaithfulnessMetric` y
`AnswerRelevancyMetric`; por tanto, su dataset debe incluir una respuesta de referencia útil para que las métricas
contextuales tengan un contrato válido. Los nombres son comparables conceptualmente, pero los scores de ambos backends
no deben mezclarse ni tratarse como equivalentes.

### Expectations periodísticas

Las Expectations complementan las métricas RAG, pero actualmente se ejecutan **sólo mediante `agents-cli eval`** con
la configuración `tests/eval/eval_config.yaml`; `ragas_evaluation.py` y `deepeval_evaluation.py` no las calculan ni las
incluyen en sus JSON. Cada Expectation usa un juez `gemini-flash-latest` con temperatura `0` y solicita una salida
binaria (`0` o `1`) con explicación breve.

| Expectation                       | Pregunta de evaluación                                                            | Datos requeridos                                      | Objetivo |
| --------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- | -------- |
| `cita_fuente_periodistica`        | ¿La respuesta identifica el archivo, programa o entrevista de origen?             | `original_file_name` de los contextos recuperados     | >= 0.8   |
| `indica_momento_temporal`         | ¿La respuesta informa inicio, fin, segundos o formato `mm:ss` del fragmento?      | `start_time` y `end_time`                             | >= 0.8   |
| `limita_al_contexto_recuperado`   | ¿Evita afirmaciones no respaldadas por los outputs de las tools?                  | `contexts` capturados                                 | >= 0.8   |
| `idioma_consistente`              | ¿Responde en el idioma de la consulta, salvo instrucción explícita contraria?     | `question` y `answer`                                 | >= 0.8   |

El promedio de cada Expectation se calcula de forma independiente. No se deben mezclar sus scores binarios con los
scores continuos de RAGAS en un único promedio.

### Dataset de evaluación común

El runner acepta una lista JSON (o un objeto con `samples`/`questions`). El único campo estrictamente requerido por el
código es `question`; `ground_truth` es opcional. Conceptualmente, cada muestra queda normalizada así:

```python
@dataclass
class RAGASEvalSample:
    question: str  # Pregunta del usuario
    answer: str  # Respuesta generada por el agente
    contexts: list[str]  # Contextos recuperados (transcripciones)
    retrieved_segments: list[dict]  # ID, fuente y timestamps de cada contexto
    ground_truth: str = ""  # Respuesta esperada opcional
```

`ground_truth_contexts` puede existir en datasets de anotación, pero los runners actuales no lo pasan a RAGAS ni a
DeepEval. `retrieved_segments` no se lee del archivo: se completa en runtime desde los outputs de las tools.

**Formato del archivo** (p. ej. `evaluation/test_datasets/ragas_eval_dataset_unlabeled.json` o un dataset anotado
equivalente):

```json
[
    {
        "question": "¿Qué se dijo sobre inflación en las entrevistas de enero?",
        "ground_truth": "En las entrevistas de enero se mencionó que la inflación interanual alcanzó el 25%, con declaraciones del ministro de economía sobre medidas de contención...",
        "ground_truth_contexts": [
            "El ministro de economía declaró que la inflación interanual...",
            "Los analistas consultados coinciden en que las medidas..."
        ]
    },
    {
        "question": "Busca segmentos donde haya aplausos durante un discurso político",
        "ground_truth": "Se encontraron 3 segmentos con aplausos durante discursos: el discurso inaugural del 1 de marzo (min 5:30), la conferencia de prensa del 15 de enero (min 12:00)...",
        "ground_truth_contexts": [
            "Segmento del discurso inaugural con aplausos de fondo...",
            "Conferencia de prensa donde el público aplaude..."
        ]
    }
]
```

`retrieved_segments` se completa en runtime a partir de los outputs de las tools. Cada elemento debe incluir, como
mínimo, `segment_id` , `original_file_name` , `start_time` , `end_time` y `text` . Esta información permite evaluar las
Expectations de fuente y momento temporal sin inferir metadatos desde la respuesta final.

### Implementación de la Evaluación RAGAS

`evaluation/ragas_evaluation.py` implementa este flujo real:

1. Carga una lista JSON o las claves envolventes `samples`/`questions`.
2. Consulta `POST /evaluate/query` por cada pregunta con timeout de 60 segundos y `max_results=5`.
3. Conserva respuesta, contextos, segmentos recuperados y `ground_truth` si existe.
4. Selecciona métricas reference-free o con referencia mediante `_select_ragas_metrics()`.
5. Ejecuta `ragas.evaluate()`, agrega promedios y copia los scores a cada muestra.
6. Escribe un JSON auditable cuando se proporciona `--output`.

Los fallos al consultar una muestra no abortan la corrida: se guardan como una respuesta `Error: ...` con contextos
vacíos. Esto mantiene el lote, pero esos casos deben separarse al interpretar calidad porque mezclan disponibilidad con
calidad de respuesta.

#### Pipeline de métricas RAGAS

Cada métrica evalúa una etapa específica del pipeline RAG, dividiéndose entre componente de recuperación y componente de
generación:

```text
                 QUERY
                   │
                   ▼
              RETRIEVER
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
Context Precision       Context Recall
¿lo recuperado          ¿recuperé suficiente
es relevante?           información relevante?
        │
        └──────────┬──────────┘
                   ▼
          CONTEXTO RECUPERADO
                   │
                   ▼
               GENERADOR
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  Faithfulness          Answer Relevance
  ¿está soportada       ¿responde realmente
  por el contexto?      a la pregunta?
```

##### Etapa 1 — Recuperación

- **Context Precision**: mide la proporción de contextos recuperados que son realmente relevantes para responder la
  pregunta. Equivale a la "calidad" del top-$k$ recuperado.
- **Context Recall**: mide si el conjunto de contextos recuperados contiene toda la información necesaria para responder
  la pregunta. Equivale a la "cobertura" de la evidencia requerida.

Ambas métricas operan sobre la salida del retriever, antes de que el generador intervenga. Si Context Precision es bajo,
el generador recibe ruido; si Context Recall es bajo, le falta evidencia.

##### Etapa 2 — Generación

- **Faithfulness**: evalúa si cada afirmación de la respuesta generada puede inferirse directamente del contexto
  recuperado. Penaliza alucinaciones, extrapolaciones o conocimiento paramétrico no respaldado.
- **Answer Relevance**: evalúa si la respuesta generada responde efectivamente a la intención informacional de la
  pregunta del usuario, independientemente de si está bien fundamentada.

La combinación de Faithfulness y Answer Relevance permite detectar cuatro modos de fallo típicos:

| Faithfulness | Answer Relevance | Interpretación                                                            |
| ------------ | ---------------- | ------------------------------------------------------------------------- |
| Alta         | Alta             | Respuesta correcta, bien fundamentada y alineada con la pregunta.         |
| Alta         | Baja             | Respuesta fiel al contexto pero que no responde lo consultado.            |
| Baja         | Alta             | Respuesta persuasiva y aparentemente pertinente, pero con alucinaciones.  |
| Baja         | Baja             | Respuesta incorrecta, sin soporte y fuera de tema.                        |

En la implementación actual, la selección exacta depende de que el lote completo tenga o no `ground_truth`, como se
detalla a continuación.

#### Selección automática de métricas según etiquetado

RAGAS, por definición, no requiere un dataset con `ground_truth` : algunas de sus métricas se calculan exclusivamente a
partir de la tripleta `(pregunta, contexto recuperado, respuesta generada)` , sin comparar contra una referencia humana.
Otras sí necesitan una referencia porque su definición depende de ella. Esto se verificó directamente en el código
fuente de `ragas==0.3.2` (`_required_columns` de cada métrica):

| Métrica                                        | Requiere `ground_truth`/`reference`  | Motivo                                                                                                                                                             |
| ---------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `faithfulness`                                 | No                                   | Solo verifica que las afirmaciones de la respuesta estén sustentadas en el contexto recuperado.                                                                    |
| `answer_relevancy`                             | No                                   | Genera preguntas a partir de la respuesta y mide su similitud con la pregunta original, vía embeddings.                                                            |
| `context_utilization` (`ContextUtilization`)   | No                                   | Variante de Context Precision que usa la **respuesta generada** en lugar de una referencia para decidir qué contexto era relevante.                                |
| `context_precision`                            | Sí                                   | Su implementación por defecto (`LLMContextPrecisionWithReference`) compara cada fragmento recuperado contra `ground_truth`.                                        |
| `context_recall`                               | Sí                                   | Por definición mide si el contexto recuperado cubre lo necesario para producir la respuesta de referencia; no existe una noción de "necesario" sin una referencia. |
| `answer_correctness`                           | Sí                                   | Compara explícitamente la respuesta generada contra `ground_truth`.                                                                                                |

`evaluation/ragas_evaluation.py` implementa esta distinción en `_select_ragas_metrics()`:

- Si falta `ground_truth` en **al menos una** muestra, se ejecutan sólo las tres métricas reference-free:
  `faithfulness`, `answer_relevancy` y `context_utilization`. Esto incluye datasets mixtos.
- Si **todas** las muestras traen `ground_truth` , se agregan además `context_precision` , `context_recall` y
  `answer_correctness` .
- El resultado JSON registra qué métricas se usaron (`metrics_used`) y si la corrida fue `reference_free` , para que
  quede auditable qué se evaluó y por qué.

Esto permite ejecutar RAGAS desde el primer día del proyecto, antes de tener un dataset anotado, y ampliar
automáticamente el conjunto de métricas a medida que se anota `ground_truth` . El dataset
Para validar el pipeline reference-free se necesita un JSON de muestras con sólo `question`; el path default previsto
es `evaluation/test_datasets/ragas_eval_dataset_unlabeled.json`, pero ese archivo no existe actualmente en el árbol.

### Implementación de Expectations

La implementación actual está declarada en `tests/eval/eval_config.yaml` como cuatro `custom_metrics` de
`agents-cli`. Los prompts reciben `{prompt}`, `{response}` y, salvo `idioma_consistente`, `{agent_data}`. Las trazas y
calificaciones se persisten bajo `artifacts/` mediante los comandos `agents-cli eval generate`, `grade` y `compare`.

No existe `evaluation/journalistic_expectations.py` ni una función
`evaluate_journalistic_expectations()`. Integrar estas métricas al launcher RAGAS/DeepEval y unificar su contrato JSON
es trabajo pendiente; hasta entonces, los resultados de ambos flujos deben correlacionarse por caso o corrida sin
inventar un promedio combinado.

### Captura de Contextos para RAGAS

Para evaluar context_precision y context_recall, necesitamos capturar los fragmentos que el RAG devuelve al agente.

#### Mecanismo implementado: endpoint de evaluación dedicado

`POST /evaluate/query` (en `src/fast_api_app.py` y en el servicio local
`src/agent_service/main.py`) ejecuta la consulta con el Runner ADK, recorre los
eventos y extrae de cada `function_response` los textos recuperados
(`contexts`) y los metadatos de segmento (`retrieved_segments`), sin depender
de callbacks del framework:

```python
@app.post("/evaluate/query", response_model=EvaluationQueryResponse)
async def query_agent_for_evaluation(request: QueryRequest) -> EvaluationQueryResponse:
    response, contexts, segments = await _run_query(request.query, request.max_results)
    return EvaluationQueryResponse(
        response=response,
        query=request.query,
        contexts=contexts,
        retrieved_segments=segments,
    )
```

> Histórico: antes de la migración a ADK se evaluó un callback handler de
> LangChain (`on_tool_end`). Ya no aplica: la evidencia se extrae de los
> eventos ADK en `_extract_evidence`.

### Ejecución de la Evaluación RAGAS

```bash
# Prerequisitos: servicio del agente corriendo
uv run uvicorn src.fast_api_app:app --port 8000 &

# Ejecutar evaluación
uv run python evaluation/ragas_evaluation.py \
    --dataset evaluation/test_datasets/synthetic/ragas_synthetic_questions.json \
    --agent-url http://localhost:8000 \
    --output evaluation/results/ragas_results.json
```

### Formato de Resultados RAGAS

```json
{
    "timestamp": "2026-09-27T14:30:00+00:00",
    "framework": "ragas",
    "dataset_source": "manual",
    "dataset_path": "<dataset-sin-referencia>.json",
    "metrics": {
        "faithfulness": 0.82,
        "answer_relevancy": 0.78,
        "context_utilization": 0.71
    },
    "metrics_used": ["faithfulness", "answer_relevancy", "context_utilization"],
    "reference_free": true,
    "per_question": [
        {
            "question": "¿Qué se dijo sobre inflación...?",
            "answer": "...",
            "contexts": ["..."],
            "ground_truth": "",
            "retrieved_segments": [{"segment_id": 42, "start_time": 10.0, "end_time": 20.0}],
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_utilization": 0.8
        }
    ],
    "config": {
        "agent_url": "http://localhost:8000",
        "dataset_size": 50,
        "judge_model": "gpt-4o-mini",
        "judge_temperature": 0.0
    }
}
```

El ejemplo refleja una corrida reference-free. En una corrida completamente anotada, `metrics_used`, `metrics` y cada
muestra agregan `context_precision`, `context_recall` y `answer_correctness`. Las Expectations quedan en los artefactos
separados de `agents-cli eval`, no en este JSON. `judge_model` y `judge_temperature` son hoy metadatos declarativos para
RAGAS, no prueba de que `ragas.evaluate()` haya usado esa configuración.

---

## Nivel 2: Evaluación del Retrieval (RAG Aislado)

### ¿Qué se evalúa?

La calidad de los fragmentos recuperados por el motor de búsqueda vectorial, **independientemente del agente**. Esto
permite diagnosticar si los problemas de calidad vienen del retrieval o de la generación.

### Métricas de Retrieval

| Métrica         | Qué mide                                      | Fórmula                        | Objetivo      |
| --------------- | --------------------------------------------- | ------------------------------ | ------------- |
| **Precision@K** | Proporción de resultados relevantes en top-K  | hits_in_top_k / k              | >= 0.6 (K=5)  |
| **Recall@K**    | Proporción de relevantes encontrados en top-K | hits_in_top_k / total_relevant | >= 0.5 (K=10) |
| **F1@K**        | Balance entre Precision@K y Recall@K           | 2PR / (P + R)                  | Informativa   |
| **MRR**         | Posición del primer resultado relevante       | 1 / rank_of_first_hit          | >= 0.5        |
| **NDCG@K**      | Calidad del ranking (position-aware)          | DCG@K / IDCG@K                 | >= 0.5        |
| **FP Cost@K**   | Costo de falsos positivos, ponderado por rank  | suma de descuentos de los FP   | Minimizar     |

### Dataset de Evaluación para Retrieval

Se requiere un dataset con queries y sus segmentos relevantes anotados manualmente:

**Formato** (`evaluation/test_datasets/retrieval_eval_dataset.json`):

```json
{
    "queries": [
        {
            "query_id": 1,
            "query_text": "política económica argentina",
            "category": "discurso_politico",
            "relevant_segment_ids": [42, 105, 203, 567]
        },
        {
            "query_id": 2,
            "query_text": "aplausos durante discurso",
            "category": "evento_acustico",
            "relevant_segment_ids": [15, 88, 302]
        },
        {
            "query_id": 3,
            "query_text": "entrevista sobre cambio climático",
            "category": "entrevista",
            "relevant_segment_ids": [77, 134, 256, 389, 412]
        }
    ]
}
```

### Categorías de Queries Periodísticas

| Categoría           | Ejemplo de query         | Descripción              |
| ------------------- | ------------------------ | ------------------------ |
| `discurso_politico` | "política económica"     | Contenido de discursos   |
| `entrevista`        | "entrevista sobre salud" | Segmentos de entrevistas |
| `evento_acustico`   | "aplausos", "gritos"     | Eventos sonoros          |
| `podcast`           | "debate sobre educación" | Contenido de podcasts    |
| `archivo_historico` | "dictadura militar"      | Audio de archivo         |

### Implementación de la Evaluación de Retrieval

El siguiente extracto documenta el contrato de cálculo; el script real también agrega resultados por categoría,
timestamp y configuración de la corrida.

```python
"""evaluation/retrieval_evaluation.py"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at: dict[int, float]  # {1: 0.8, 5: 0.6, 10: 0.5}
    recall_at: dict[int, float]  # {1: 0.2, 5: 0.5, 10: 0.7}
    f1_at: dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    ndcg_at: dict[int, float]  # {1: 0.8, 5: 0.65, 10: 0.55}
    fp_cost_at: dict[int, float]


def precision_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Proporción de resultados relevantes en top-K."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for x in top_k if x in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Proporción de relevantes encontrados en top-K."""
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for x in top_k if x in relevant_ids)
    return hits / len(relevant_ids)


def f1_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Media armónica de Precision@K y Recall@K."""
    precision = precision_at_k(ranked_ids, relevant_ids, k)
    recall = recall_at_k(ranked_ids, relevant_ids, k)
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def mean_reciprocal_rank(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """Inverso del rank del primer resultado relevante."""
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain (binario)."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    rel = np.array([1.0 if x in relevant_ids else 0.0 for x in top_k])
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(rel)))
    dcg = float(np.sum(rel * discounts))
    ideal_rel = np.zeros(len(rel), dtype=float)
    ideal_rel[: min(len(relevant_ids), len(rel))] = 1.0
    idcg = float(np.sum(ideal_rel * discounts))
    if idcg <= 1e-12:
        return 0.0
    return dcg / idcg


def fp_cost_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Costo de falsos positivos con descuento logarítmico por posición."""
    return float(sum(
        1.0 / np.log2(rank + 1)
        for rank, result_id in enumerate(ranked_ids[:k], start=1)
        if result_id not in relevant_ids
    ))


def compute_retrieval_metrics(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """Calcula todas las métricas de retrieval para una query."""
    if k_values is None:
        k_values = [1, 5, 10]

    return RetrievalMetrics(
        precision_at={k: precision_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        recall_at={k: recall_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        f1_at={k: f1_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        mrr=mean_reciprocal_rank(ranked_ids, relevant_ids),
        ndcg_at={k: ndcg_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        fp_cost_at={k: fp_cost_at_k(ranked_ids, relevant_ids, k) for k in k_values},
    )


def evaluate_retrieval(
    dataset_path: str,
    search_engine,
    k_values: list[int] | None = None,
    max_k: int = 10,
) -> dict:
    """
    Evalúa el retrieval sobre el dataset de evaluación.

    Args:
        dataset_path: Ruta al JSON con queries y relevant_ids
        search_engine: Instancia de AudioSearchEngine
        k_values: Valores de K para Precision@K, Recall@K, NDCG@K
        max_k: Máximo K para la búsqueda

    Returns:
        Dict con métricas agregadas y por query
    """
    if k_values is None:
        k_values = [1, 5, 10]

    queries = json.loads(Path(dataset_path).read_text())["queries"]

    all_metrics = []
    per_query_results = []

    for q in queries:
        # Ejecutar búsqueda
        results = search_engine.search_semantic(q["query_text"], k=max_k)
        ranked_ids = [r["segment"]["segment_id"] for r in results]
        relevant_ids = set(q["relevant_segment_ids"])

        # Calcular métricas
        metrics = compute_retrieval_metrics(ranked_ids, relevant_ids, k_values)
        all_metrics.append(metrics)

        per_query_results.append(
            {
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "category": q.get("category", "unknown"),
                "metrics": {
                    "precision_at": metrics.precision_at,
                    "recall_at": metrics.recall_at,
                    "f1_at": metrics.f1_at,
                    "mrr": metrics.mrr,
                    "ndcg_at": metrics.ndcg_at,
                    "fp_cost_at": metrics.fp_cost_at,
                },
            }
        )

    # Agregar métricas
    aggregated = {
        "precision_at": {k: np.mean([m.precision_at[k] for m in all_metrics]) for k in k_values},
        "recall_at": {k: np.mean([m.recall_at[k] for m in all_metrics]) for k in k_values},
        "f1_at": {k: np.mean([m.f1_at[k] for m in all_metrics]) for k in k_values},
        "mrr": np.mean([m.mrr for m in all_metrics]),
        "ndcg_at": {k: np.mean([m.ndcg_at[k] for m in all_metrics]) for k in k_values},
        "fp_cost_at": {k: np.mean([m.fp_cost_at[k] for m in all_metrics]) for k in k_values},
    }

    return {
        "aggregated": aggregated,
        "per_query": per_query_results,
        "config": {
            "k_values": k_values,
            "max_k": max_k,
            "total_queries": len(queries),
        },
    }
```

### Ejecución de la Evaluación de Retrieval

```bash
uv run python evaluation/retrieval_evaluation.py \
    --dataset evaluation/test_datasets/retrieval_eval_dataset.json \
    --dataset-path ./dataset \
    --output evaluation/results/retrieval_results.json
```

### Formato de Resultados de Retrieval

```json
{
    "timestamp": "2025-01-15T14:30:00",
    "aggregated": {
        "precision_at": {"1": 0.72, "5": 0.58, "10": 0.45},
        "recall_at": {"1": 0.18, "5": 0.52, "10": 0.71},
        "f1_at": {"1": 0.29, "5": 0.55, "10": 0.55},
        "mrr": 0.65,
        "ndcg_at": {"1": 0.72, "5": 0.61, "10": 0.55},
        "fp_cost_at": {"1": 0.28, "5": 1.17, "10": 2.02}
    },
    "by_category": {
        "discurso_politico": {"mrr": 0.72, "precision_at_5": 0.65},
        "evento_acustico": {"mrr": 0.55, "precision_at_5": 0.48},
        "entrevista": {"mrr": 0.68, "precision_at_5": 0.60}
    },
    "per_query": [...],
    "config": {
        "embedding_model": "all-MiniLM-L6-v2",
        "index_type": "FAISS IndexFlatIP",
        "total_segments": 5000,
        "total_queries": 50
    }
}
```

---

## Relación entre los Dos Niveles de Evaluación

```text
┌─────────────────────────────────────────────────────────────┐
│                  EVALUACIÓN COMPLETA                          │
│                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │  NIVEL 1: RAGAS         │  │  NIVEL 2: RETRIEVAL     │  │
│  │  (Calidad del Agente)   │  │  (Calidad del RAG)      │  │
│  │                         │  │                         │  │
│  │  Evalúa:                │  │  Evalúa:                │  │
│  │  - Faithfulness         │  │  - Precision@K          │  │
│  │  - Answer Relevancy     │  │  - Recall@K             │  │
│  │  - Context Precision    │  │  - MRR                  │  │
│  │  - Context Recall       │  │  - NDCG@K               │  │
│  │  - Answer Correctness   │  │                         │  │
│  │                         │  │  Diagnóstica:           │  │
│  │  Diagnóstica:           │  │  - ¿Los embeddings      │  │
│  │  - ¿El agente genera    │  │    capturan semántica?  │  │
│  │    respuestas fieles?   │  │  - ¿El índice retorna   │  │
│  │  - ¿Usa bien el         │  │    fragmentos buenos?   │  │
│  │    contexto?            │  │  - ¿Qué categorías      │  │
│  │  - ¿Responde lo que     │  │    funcionan mejor?     │  │
│  │    se preguntó?         │  │                         │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
│                                                             │
│  Si RAGAS es bajo pero Retrieval es alto:                   │
│    → Problema en la generación del agente (prompt, LLM)     │
│                                                             │
│  Si RAGAS es bajo y Retrieval es bajo:                      │
│    → Problema en el RAG (embeddings, indexación)            │
│                                                             │
│  Si ambos son altos:                                        │
│    → Sistema funcionando correctamente                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Creación del Dataset de Evaluación

### Proceso de Anotación Manual

1. **Seleccionar queries representativas** (mínimo 30-50):
   - 10+ queries de discursos políticos
   - 10+ queries de entrevistas
   - 5+ queries de eventos acústicos
   - 5+ queries de podcasts
   - 5+ queries de archivo histórico

2. **Para cada query, anotar relevant_segment_ids**:
   - Escuchar/leer segmentos del dataset
   - Marcar cuáles son genuinamente relevantes
   - Un segmento es relevante si contesta total o parcialmente la query

3. **Para RAGAS, redactar ground_truth**:
   - Escribir la respuesta ideal esperada
   - Incluir la información clave que debería contener

4. **Para las Expectations, verificar los metadatos citables**:
   - Confirmar que cada contexto relevante conserva archivo de origen y timestamps
   - Anotar excepciones legítimas de idioma o de formato de respuesta

### Herramientas de apoyo para anotación

El repositorio no contiene `evaluation/create_eval_dataset.py`. Los flujos implementados son:

- `evaluation/generate_synthetic_questions.py` y `evaluation/generate_balanced_questions.py` para crear preguntas
  candidatas.
- `evaluation/prepare_acoustic_human_review.py` para construir un pool estratificado de candidatos CLAP.
- `evaluation/human_review/app.py` para escuchar clips y registrar relevancia 0–3 más `event_present`.
- `evaluation/analyze_acoustic_human_review.py` para calcular Precision@K, MRR, nDCG y costo de falsos positivos sobre
  rankings top-k completamente juzgados.

La revisión acústica no calcula Recall del corpus: el pool no contiene juicios exhaustivos sobre todos los segmentos.

---

## Estudios de Ablación por Modalidad

`AGENT_MODALITY` restringe las tools de retrieval del agente para comparar
índices de forma aislada sobre el mismo dataset de evaluación
(`text`, `audio`, `yamnet`, `text_clap`, `all`; ver spec 03). Permite medir
cuánto aporta cada modalidad sin cambiar código ni dataset.

## Benchmarks de Retrieval Acústico (CLAP)

Complementan la evaluación periodística con protocolos externos y análisis por
tipo de query:

| Script | Propósito |
| --- | --- |
| `benchmarks/evaluate_clap_clotho.py` | Sanity check del wiring de CLAP contra el benchmark público Clotho (text→audio, Recall@1/5/10, protocolo del paper de CLAP). Un buen score sólo descarta bugs de integración; no valida el corpus periodístico en español. Resultados en `docs/resultados-evaluacion-clotho.md` y plan en `DOC-interna/docs/evaluar-clap-clotho.md`. |
| `benchmarks/compare_retrieval_with_gemini.py` | Compara el retrieval CLAP contra los embeddings Gemini en el corpus propio. |
| `benchmarks/compare_clap_by_question_type.py` | Compara CLAP por tipo de pregunta (conceptual vs. acústica) con el dataset balanceado. |
| `evaluation/text_index_evaluation.py` | Evalúa el índice de texto de forma aislada. |
| `evaluation/prepare_acoustic_human_review.py` + `evaluation/human_review/` | Prepara y levanta una mini-app de revisión humana de resultados acústicos. |
| `evaluation/analyze_acoustic_human_review.py` | Analiza los veredictos de la revisión humana. |
| `evaluation/audit_corpus_acoustic_events.py` | Ejecuta YAMNet de forma continua sobre las grabaciones convertidas completas y contrasta evidencia AudioSet con `event_present` humano. |

Contexto y resultados: `DOC-interna/docs/reporte-comparativo-texto-clap.md`.

### Auditoría acústica continua del corpus

`evaluation/audit_corpus_acoustic_events.py` sirve para distinguir un fallo de retrieval de una consulta cuyo evento
probablemente no existe en el corpus. A diferencia del YAMNet de ingesta, recorre cada WAV de `converted/` completo,
incluido el audio entre segmentos transcriptos.

```bash
uv sync --extra yamnet
uv run python -m evaluation.audit_corpus_acoustic_events \
  --dataset-path ./dataset \
  --questions evaluation/test_datasets/synthetic/clap_journalistic_acoustic_questions.json \
  --review-set evaluation/results/clap_all_questions_candidates.json \
  --annotations evaluation/annotations/clap_human_annotations.json \
  --output evaluation/results/corpus_acoustic_event_audit.json
```

Contrato y método:

- Requiere WAV mono o estéreo a 16 kHz; mezcla canales y rechaza otra frecuencia de muestreo.
- Mapea las 65 consultas del catálogo acústico a clases AudioSet. `None` marca fenómenos sin clase apropiada, como tono
  emocional, volumen bajo o cortes de señal.
- Evalúa cuadros YAMNet con hop de 0,48 s. `max_score >= 0.5` es `strong_evidence`, `>= 0.2` es `weak_evidence` y el
  resto `no_evidence`.
- Registra perfil del corpus, máximo por clase, archivo/tiempo del máximo, segmento cercano y conteos humanos de evento
  presente, ausente o desconocido.
- Los scores de YAMNet son evidencia de cribado independiente. `no_evidence` no prueba ausencia y el reporte no debe
  usarse como ground truth automático.

---

## Fuera de Scope: Métricas de Espectrogramas

El árbol actual no contiene `src/spectrogram_evaluation.py` ni `doc/ESPECTROGRAMA_EVALUACION.md`. Las siguientes
métricas son ideas de extensión y **no están implementadas ni forman parte de la evaluación actual**:

- Métricas espectrales: Spectral Centroid, Bandwidth, Flatness, Rolloff
- Métricas temporales: Spectral Flux, Temporal Stability, Entropy
- Métricas de energía: Total Energy, SNR, Dynamic Range
- Métricas de comparación: Cosine Similarity, KL/JS Divergence entre espectrogramas

Estas métricas son útiles para:

- Evaluar calidad de audio pre-procesado
- Clasificar tipo de contenido (tonal vs ruido)
- Detectar degradación de audio

Podrían integrarse en el futuro como un tercer nivel de evaluación para validar la calidad del audio de entrada antes
del pipeline de ingesta.

---

## Criterios de Aceptación

### RAGAS

1. Dataset de evaluación con mínimo 30 preguntas anotadas
2. Faithfulness promedio >= 0.7
3. Answer Relevancy promedio >= 0.7
4. Context Utilization promedio >= 0.6 en corridas reference-free
5. Context Precision promedio >= 0.6 en corridas completamente anotadas
6. Resultados exportables a JSON con métricas agregadas y detalle por pregunta

### Expectations con agents-cli

1. Las cuatro Expectations periodísticas tienen promedio >= 0.8 cada una
2. Trazas y calificaciones quedan persistidas en `artifacts/`
3. La corrida se compara con la anterior mediante `agents-cli eval compare`

### Retrieval

1. Dataset con mínimo 30 queries con relevant_ids anotados
2. Precision@5 promedio >= 0.6
3. MRR promedio >= 0.5
4. NDCG@5 promedio >= 0.5
5. Análisis por categoría de query
6. F1@K y FP Cost@K quedan informados para diagnosticar el balance y el costo periodístico del ruido
7. Resultados exportables a JSON; CSV requiere conversión posterior porque el script actual no lo emite

### Infraestructura de Evaluación

1. Scripts ejecutables desde CLI
2. Evaluación reproducible (mismos datos → mismos resultados para retrieval)
3. Evaluación no destructiva (no modifica el dataset ni el servicio)
4. Tiempo de ejecución < 10 minutos para 50 queries
