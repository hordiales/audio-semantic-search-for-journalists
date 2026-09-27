# 04 — Framework de Evaluación

## Resumen

La evaluación se realiza en **dos niveles complementarios**:

1. **Nivel Agente (evaluador RAG configurable + Expectations periodísticas)**: RAGAS o DeepEval evalúan la calidad de
   las respuestas generadas por el agente; cuatro Expectations verifican requisitos editoriales de trazabilidad,
   temporalidad, grounding e idioma.
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

# DeepEval: backend alternativo, sin cambiar el dataset ni las expectativas
uv sync --extra eval-deepeval
EVALUATION_FRAMEWORK=deepeval

# En ambos casos: juez determinista y resultado auditable
EVALUATION_JUDGE_MODEL=gpt-4o-mini
EVALUATION_JUDGE_TEMPERATURE=0
```

La entrada se selecciona con `EVALUATION_DATASET_SOURCE` (ver *Fuentes de
dataset* más abajo) o directamente con `EVALUATION_DATASET_PATH`, y el launcher
establece el backend desde `.env`:

```bash
uv run python -m evaluation.run_agent_evaluation \
  --dataset evaluation/test_datasets/ragas_eval_dataset_unlabeled.json \
  --agent-url http://localhost:8000 \
  --output evaluation/results/agent_evaluation.json
```

Cada resultado registra `framework`, modelo y temperatura del juez, métricas
agregadas y el detalle por pregunta, incluidos `contexts`,
`retrieved_segments` y los scores por métrica de cada pregunta (no sólo los
promedios), lo que permite auditar qué casos arrastran cada métrica.

### Fuentes de dataset configurables

`evaluation/evaluation_config.py` resuelve la fuente del dataset desde el
entorno y la registra en cada corrida:

| Fuente (`EVALUATION_DATASET_SOURCE`) | Contenido | Archivo |
| --- | --- | --- |
| `manual` (default) | Preguntas anotadas por humanos | `EVALUATION_MANUAL_DATASET_PATH` (default `ragas_eval_dataset_unlabeled.json`) |
| `synthetic` | Generadas con RAGAS TestsetGenerator sobre el corpus | `EVALUATION_SYNTHETIC_DATASET_PATH` |
| `hybrid` | Unión de manual + sintético, sin duplicados | `EVALUATION_HYBRID_DATASET_PATH` |

`EVALUATION_DATASET_PATH` sobrescribe cualquiera de las anteriores.
`EVALUATION_DATASET_SIZE` y `EVALUATION_DATASET_MAX_SEGMENTS` acotan la
generación sintética.

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
| **Context Precision**   | ¿Los contextos recuperados son relevantes? (precision)                    | 0-1   | >= 0.6   |
| **Context Recall**      | ¿Se recuperaron todos los contextos relevantes?                           | 0-1   | >= 0.6   |
| **Answer Correctness**  | ¿La respuesta es factualmente correcta?                                   | 0-1   | >= 0.6   |

`answer_correctness` requiere una respuesta de referencia, por lo que sólo se
ejecuta sobre casos anotados. Las cuatro primeras métricas sostienen el
diagnóstico principal: Context Precision y Context Recall evalúan retrieval;
Faithfulness y Answer Relevancy evalúan generación. RAGAS las implementa con
sus métricas nativas. DeepEval usa `ContextualPrecisionMetric`,
`ContextualRecallMetric`, `FaithfulnessMetric` y `AnswerRelevancyMetric` con
el mismo significado operativo.

### Expectations periodísticas

Las Expectations complementan las métricas del evaluador RAG seleccionado: no sustituyen faithfulness ni las métricas de
contexto. Cada una se evalúa por muestra con una salida binaria (`0` o `1` ) y una explicación breve. Pueden ejecutarse
mediante un juez LLM configurado o mediante una regla determinista cuando el contrato lo permita; el mecanismo elegido
debe quedar registrado en el resultado de la corrida.

| Expectation                       | Pregunta de evaluación                                                            | Datos requeridos                                      | Objetivo |
| --------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- | -------- |
| `cita_fuente_periodistica`        | ¿La respuesta identifica el archivo, programa o entrevista de origen?             | `original_file_name` de los contextos recuperados     | >= 0.8   |
| `indica_momento_temporal`         | ¿La respuesta informa inicio, fin, segundos o formato `mm:ss` del fragmento?      | `start_time` y `end_time`                             | >= 0.8   |
| `limita_al_contexto_recuperado`   | ¿Evita afirmaciones no respaldadas por los outputs de las tools?                  | `contexts` capturados                                 | >= 0.8   |
| `idioma_consistente`              | ¿Responde en el idioma de la consulta, salvo instrucción explícita contraria?     | `question` y `answer`                                 | >= 0.8   |

El promedio de cada Expectation se calcula de forma independiente. No se deben mezclar sus scores binarios con los
scores continuos de RAGAS en un único promedio.

### Dataset de evaluación común

Se requiere un dataset de evaluación con el siguiente esquema:

```python
@dataclass
class RAGASEvalSample:
    question: str  # Pregunta del usuario
    answer: str  # Respuesta generada por el agente
    contexts: list[str]  # Contextos recuperados (transcripciones)
    ground_truth: str  # Respuesta esperada (anotada manualmente)
    ground_truth_contexts: list[str]  # Contextos relevantes (ground truth)
    retrieved_segments: list[dict]  # ID, fuente y timestamps de cada contexto
```

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

```python
"""evaluation/ragas_evaluation.py"""

from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from datasets import Dataset


def run_ragas_evaluation(
    eval_dataset_path: str,
    agent_service_url: str = "http://localhost:8000",
) -> dict:
    """
    Ejecuta evaluación RAGAS contra el servicio del agente.

    Flujo:
    1. Carga dataset de evaluación
    2. Para cada pregunta, consulta al agente y captura:
       - La respuesta generada
       - Los contextos recuperados (interceptados del tool call)
    3. Ejecuta métricas RAGAS
    4. Retorna scores agregados

    Args:
        eval_dataset_path: Ruta al JSON con preguntas y ground truth
        agent_service_url: URL del servicio del agente

    Returns:
        Dict con scores por métrica
    """
    # Cargar dataset
    eval_samples = load_eval_dataset(eval_dataset_path)

    # Generar respuestas del agente
    results = []
    for sample in eval_samples:
        response, contexts, retrieved_segments = query_agent_with_context_capture(
            agent_service_url, sample.question
        )
        results.append(
            {
                "question": sample.question,
                "answer": response,
                "contexts": contexts,
                "ground_truth": sample.ground_truth,
                "retrieved_segments": retrieved_segments,
            }
        )

    # Crear HuggingFace Dataset para RAGAS
    dataset = Dataset.from_list(results)

    # Evaluar
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
    )

    expectation_scores = evaluate_journalistic_expectations(results)

    return {
        "ragas": scores,
        "expectations": expectation_scores,
    }


def query_agent_with_context_capture(
    service_url: str, question: str
) -> tuple[str, list[str], list[dict]]:
    """
    Consulta al agente y captura la respuesta, los contextos y los metadatos
    de los segmentos recuperados por las tools.

    Implementación: consume el endpoint `POST /evaluate/query`, que retorna
    respuesta + contextos + metadatos de segmento extraídos de los eventos ADK.
    """
    ...
```

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

En la implementación actual, `evaluation/ragas_evaluation.py` ejecuta estas cuatro métricas más `answer_correctness` ,
que compara la respuesta generada contra la respuesta de referencia (`ground_truth`) cuando está disponible.

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

- Si **ninguna** muestra del dataset trae `ground_truth` , se ejecutan solo las tres métricas reference-free:
  `faithfulness` , `answer_relevancy` y `context_utilization` .
- Si **todas** las muestras traen `ground_truth` , se agregan además `context_precision` , `context_recall` y
  `answer_correctness` .
- El resultado JSON registra qué métricas se usaron (`metrics_used`) y si la corrida fue `reference_free` , para que
  quede auditable qué se evaluó y por qué.

Esto permite ejecutar RAGAS desde el primer día del proyecto, antes de tener un dataset anotado, y ampliar
automáticamente el conjunto de métricas a medida que se anota `ground_truth` . El dataset
`evaluation/test_datasets/ragas_eval_dataset_unlabeled.json` es un ejemplo mínimo sin etiquetas (solo `question` ), útil
para validar el pipeline reference-free.

### Implementación de Expectations

La implementación vive junto a la evaluación RAGAS, por ejemplo en `evaluation/journalistic_expectations.py` , y recibe
los artefactos ya capturados por la corrida. El contrato de cada evaluador es:

```python
@dataclass
class ExpectationResult:
    name: str
    score: int  # 0 o 1
    explanation: str
    evaluator: str  # regla determinista o modelo/juez configurado


def evaluate_journalistic_expectations(
    samples: list[dict],
) -> dict[str, dict]:
    """Retorna resultados por muestra y promedio para las cuatro Expectations."""
    ...
```

La evaluación debe conservar la consulta, respuesta, contextos y metadatos de segmento usados por el juez. Si se usa un
LLM juez, la configuración debe fijar modelo, temperatura `0` , versión de prompt y fecha de ejecución para que el
resultado sea auditable.

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
    --dataset evaluation/test_datasets/ragas_eval_dataset_unlabeled.json \
    --agent-url http://localhost:8000 \
    --output evaluation/results/ragas_results.json
```

### Formato de Resultados RAGAS

```json
{
    "timestamp": "2025-01-15T14:30:00",
    "metrics": {
        "faithfulness": 0.82,
        "answer_relevancy": 0.78,
        "context_precision": 0.71,
        "context_recall": 0.65,
        "answer_correctness": 0.69
    },
    "expectations": {
        "cita_fuente_periodistica": {"average": 0.88, "target": 0.8},
        "indica_momento_temporal": {"average": 0.83, "target": 0.8},
        "limita_al_contexto_recuperado": {"average": 0.91, "target": 0.8},
        "idioma_consistente": {"average": 0.97, "target": 0.8}
    },
    "per_question": [
        {
            "question": "¿Qué se dijo sobre inflación...?",
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.8,
            "context_recall": 0.7,
            "answer_correctness": 0.75,
            "expectations": {
                "cita_fuente_periodistica": {"score": 1, "explanation": "Cita el archivo de origen."},
                "indica_momento_temporal": {"score": 1, "explanation": "Indica el intervalo recuperado."},
                "limita_al_contexto_recuperado": {"score": 1, "explanation": "No agrega hechos externos."},
                "idioma_consistente": {"score": 1, "explanation": "Consulta y respuesta están en español."}
            }
        }
    ],
    "config": {
        "agent_model": "gpt-4o-mini",
        "embedding_model": "all-MiniLM-L6-v2",
        "dataset_size": 50,
        "k": 5
    }
}
```

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
| **MRR**         | Posición del primer resultado relevante       | 1 / rank_of_first_hit          | >= 0.5        |
| **NDCG@K**      | Calidad del ranking (position-aware)          | DCG@K / IDCG@K                 | >= 0.5        |

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
    mrr: float  # Mean Reciprocal Rank
    ndcg_at: dict[int, float]  # {1: 0.8, 5: 0.65, 10: 0.55}


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
    ideal_rel = np.sort(rel)[::-1]
    idcg = float(np.sum(ideal_rel * discounts))
    if idcg <= 1e-12:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k_values: list[int] = [1, 5, 10],
) -> RetrievalMetrics:
    """Calcula todas las métricas de retrieval para una query."""
    return RetrievalMetrics(
        precision_at={k: precision_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        recall_at={k: recall_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        mrr=mean_reciprocal_rank(ranked_ids, relevant_ids),
        ndcg_at={k: ndcg_at_k(ranked_ids, relevant_ids, k) for k in k_values},
    )


def evaluate_retrieval(
    dataset_path: str,
    search_engine,
    k_values: list[int] = [1, 5, 10],
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
                    "mrr": metrics.mrr,
                    "ndcg_at": metrics.ndcg_at,
                },
            }
        )

    # Agregar métricas
    aggregated = {
        "precision_at": {k: np.mean([m.precision_at[k] for m in all_metrics]) for k in k_values},
        "recall_at": {k: np.mean([m.recall_at[k] for m in all_metrics]) for k in k_values},
        "mrr": np.mean([m.mrr for m in all_metrics]),
        "ndcg_at": {k: np.mean([m.ndcg_at[k] for m in all_metrics]) for k in k_values},
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
        "mrr": 0.65,
        "ndcg_at": {"1": 0.72, "5": 0.61, "10": 0.55}
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

### Script de Apoyo para Anotación

```bash
# Crear estructura inicial del dataset de evaluación
uv run python evaluation/create_eval_dataset.py \
    --dataset-path ./dataset \
    --output evaluation/test_datasets/ \
    --num-queries 50 \
    --categories discurso_politico entrevista evento_acustico podcast archivo
```

Este script:

1. Genera queries candidatas basadas en el contenido del dataset
2. Crea el esqueleto JSON con `relevant_segment_ids: []` vacíos
3. El evaluador humano completa los `relevant_segment_ids` y `ground_truth`

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
| `benchmarks/evaluate_clap_clotho.py` | Sanity check del wiring de CLAP contra el benchmark público Clotho (text→audio, Recall@1/5/10, protocolo del paper de CLAP). Un buen score sólo descarta bugs de integración; no valida el corpus periodístico en español. Resultados en `docs/resultados-evaluacion-clotho.md` y plan en `docs/evaluar-clap-clotho.md`. |
| `benchmarks/compare_retrieval_with_gemini.py` | Compara el retrieval CLAP contra los embeddings Gemini en el corpus propio. |
| `benchmarks/compare_clap_by_question_type.py` | Compara CLAP por tipo de pregunta (conceptual vs. acústica) con el dataset balanceado. |
| `evaluation/text_index_evaluation.py` | Evalúa el índice de texto de forma aislada. |
| `evaluation/prepare_acoustic_human_review.py` + `evaluation/human_review/` | Prepara y levanta una mini-app de revisión humana de resultados acústicos. |
| `evaluation/analyze_acoustic_human_review.py` | Analiza los veredictos de la revisión humana. |

Contexto y resultados: `docs/reporte-comparativo-texto-clap.md`.

---

## Fuera de Scope: Métricas de Espectrogramas

Las siguientes métricas están implementadas en `src/spectrogram_evaluation.py` y documentadas en
`doc/ESPECTROGRAMA_EVALUACION.md` , pero **no forman parte de la evaluación actual**:

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
4. Context Precision promedio >= 0.6
5. Las cuatro Expectations periodísticas tienen promedio >= 0.8 cada una
6. Resultados exportables a JSON con detalle por pregunta, explicación y evaluador de cada Expectation

### Retrieval

1. Dataset con mínimo 30 queries con relevant_ids anotados
2. Precision@5 promedio >= 0.6
3. MRR promedio >= 0.5
4. NDCG@5 promedio >= 0.5
5. Análisis por categoría de query
6. Resultados exportables a JSON/CSV

### Infraestructura de Evaluación

1. Scripts ejecutables desde CLI
2. Evaluación reproducible (mismos datos → mismos resultados para retrieval)
3. Evaluación no destructiva (no modifica el dataset ni el servicio)
4. Tiempo de ejecución < 10 minutos para 50 queries
