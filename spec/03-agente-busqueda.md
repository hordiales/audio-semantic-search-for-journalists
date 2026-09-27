# 03 — Agente de Búsqueda Semántica

## Resumen

El componente central del sistema es un **agente Google ADK** que actúa como intermediario inteligente entre el usuario
(periodista) y el sistema RAG multimodal. El agente interpreta consultas en lenguaje natural, decide qué herramientas
usar, ejecuta búsquedas semánticas y genera respuestas contextualizadas.

## Arquitectura del Agente

```text
┌─────────────────────────────────────────────────────────────┐
│                    AudioAgent                                 │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  LLM: GPT-4o-mini (OpenAI vía LiteLLM)                │  │
│  │  Temperature: 0.0                                     │  │
│  │  Patrón: Function Tools (ADK)                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  ADK Runner                                            │  │
│  │  - sesiones locales en memoria                         │  │
│  │  - trazas para agents-cli eval                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Tools (AGENT_MODALITY filtra las de retrieval):        │ │
│  │  - buscar_audio            (texto → transcripciones)    │ │
│  │  - buscar_evento_acustico  (texto → audio CLAP)         │ │
│  │  - buscar_clase_audio      (texto → clases YAMNet)      │ │
│  │  - obtener_info_segmento   (metadatos de un segmento)   │ │
│  │  - obtener_clases_audio    (clases YAMNet de un seg.)   │ │
│  └──────────────────────────┬──────────────────────────────┘ │
└─────────────────────────────┼─────────────────────────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            ▼                                    ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│   AudioSearchEngine (local)  │  │ SearchServiceClient (deploy) │
│  - FAISS texto 384-dim       │  │  HTTP → Cloud Run search     │
│  - FAISS audio CLAP 512-dim  │  │  service (SEARCH_SERVICE_URL)│
│  - Índice invertido YAMNet   │  │  El agente no importa        │
│  - DataFrame con metadatos   │  │  PyTorch/FAISS               │
└──────────────────────────────┘  └──────────────────────────────┘
```

Ambos backends exponen los mismos métodos de retrieval; `initialize_search_engine`
elige uno u otro al arrancar y las tools no bifurcan.

## System Prompt del Agente

Prompt vigente (`SYSTEM_PROMPT` en `src/agent_service/agent.py`):

```text
Eres un asistente especializado en búsqueda semántica de contenido de audio para periodistas.

Reglas:
1. Usa buscar_audio para contenido dicho o escrito en las transcripciones.
2. Usa buscar_evento_acustico para sonidos como aplausos, música, gritos o risas.
3. Usa buscar_clase_audio cuando convenga buscar explícitamente etiquetas acústicas
   estandarizadas de AudioSet/YAMNet, por ejemplo Applause, Speech o Music. Sus scores
   combinan coincidencia de clase y confianza del clasificador; no son similitud vectorial.
4. Usa obtener_clases_audio para inspeccionar todas las clases AudioSet detectadas por
   YAMNet en un segmento recuperado. Explica que son etiquetas en inglés y scores del
   clasificador.
5. Usa obtener_info_segmento cuando se pidan detalles de un segmento identificado.
6. Responde en español, salvo que el usuario pida otro idioma.
7. No inventes información: limita todas las afirmaciones a los resultados de las tools.
8. Para cada hallazgo, cita archivo de origen, timestamp de inicio y fin, e índice
   consultado, usando exactamente el campo search_index_label devuelto por la tool.
9. Si no hay resultados, dilo claramente y sugiere una reformulación.
10. Si un hallazgo trae clip_url, incluí un enlace Markdown "▶ Reproducir audio" con
    esa URL, sin inventarla ni alterarla.
11. Si el mensaje comienza exactamente con DIRECT_SEARCH_PLANNER_V1, es una solicitud
    interna de planificación para la búsqueda directa: no llames herramientas y devolvé
    únicamente el objeto JSON solicitado (ver spec 08).
```

## Herramientas (Tools)

Las tools son funciones Python planas que ADK convierte en `FunctionTool` a
partir de la signatura y el docstring (`get_all_tools()`). Devuelven dicts con
`status` (`success`/`error`/`not_available`), `modality` y `results`, nunca
lanzan excepciones al LLM.

`get_all_tools()` filtra las tools de retrieval según `AGENT_MODALITY`
(estudios de ablación):

| Valor | Tools de búsqueda habilitadas |
| --- | --- |
| `all` (default), `both` (alias deprecado) | texto + CLAP + YAMNet |
| `text` | sólo `buscar_audio` |
| `audio` | sólo `buscar_evento_acustico` |
| `yamnet` | sólo `buscar_clase_audio` |
| `text_clap` | texto + CLAP (evaluaciones de dos vías) |

`obtener_info_segmento` y `obtener_clases_audio` están siempre disponibles.

### Tool 1: `buscar_audio`

**Propósito**: Búsqueda semántica de texto sobre las transcripciones (MiniLM + FAISS).

```python
def buscar_audio(query: str, k: int = 5, tool_context: ToolContext | None = None) -> dict:
    """Busca texto semánticamente en las transcripciones del corpus de audio.

    Args:
        query: Consulta del periodista en lenguaje natural.
        k: Cantidad máxima de segmentos a devolver, entre 1 y 20.
    """
```

- `k` se sobrescribe con `max_results` del estado de sesión ADK si existe.
- Queries vacías o `k` fuera de rango devuelven `{"status": "error", ...}`.

### Tool 2: `buscar_evento_acustico`

**Propósito**: Búsqueda cross-modal texto→audio con CLAP para sonidos que pueden
no estar transcriptos (aplausos, música, gritos, risas).

```python
def buscar_evento_acustico(query: str, k: int = 5, tool_context: ToolContext | None = None) -> dict:
```

Internamente llama `search_audio_by_text`; la query se traduce al inglés antes
del encoder CLAP cuando `QUERY_LANGUAGE` no es inglés.

### Tool 3: `buscar_clase_audio`

**Propósito**: Búsqueda por clases AudioSet detectadas por YAMNet sobre el índice
invertido (`search_audio_by_classes`). No usa similitud vectorial: traduce la
consulta al inglés, la compara con nombres de clases (`Applause`, `Music`, ...)
y pondera cobertura de tokens × confianza del clasificador.

```python
def buscar_clase_audio(query: str, k: int = 5, tool_context: ToolContext | None = None) -> dict:
```

### Tool 4: `obtener_info_segmento`

**Propósito**: Metadatos completos de un segmento por `segment_id` (incluye
sentimiento y demás campos del dataset).

### Tool 5: `obtener_clases_audio`

**Propósito**: Todas las clases AudioSet/YAMNet de un segmento. Si el dataset se
generó sin YAMNet devuelve `status: not_available` con instrucciones de
reprocesamiento, para que el agente lo explique en lugar de fallar.

**Formato de salida de las tools de búsqueda**:

```json
{
  "status": "success",
  "modality": "text",
  "results": [
    {
      "segment_id": 42,
      "search_index": "text",
      "search_index_label": "Índice de texto (transcripciones)",
      "text": "El presidente anunció nuevas medidas económicas...",
      "similarity": 0.8723,
      "similarity_percent": 87.2,
      "start_time": 125.5,
      "end_time": 142.3,
      "duration": 16.8,
      "original_file_name": "entrevista_2024_01_15.wav",
      "language": "es",
      "confidence": 0.95,
      "clip_url": "https://storage.googleapis.com/.../segment_42.opus?...",
      "clip_start_time": 120.5,
      "clip_end_time": 147.3,
      "clip_expires_at": "2026-08-30T22:15:00Z",
      "yamnet_audio_classes": [{"class_name": "Speech", "score": 0.81}],
      "yamnet_matched_classes": []
    }
  ]
}
```

`search_index`/`search_index_label` identifican el índice de origen de cada
resultado y el prompt obliga a citarlos. `clip_*` y `yamnet_*` sólo aparecen
cuando el dataset los contiene.

### Tools Futuras (extensibilidad)

El diseño permite agregar nuevas herramientas sin modificar el agente:

- `buscar_por_sentimiento`: Filtrar por sentimiento dominante
- `resumir_segmentos`: Generar resumen de múltiples segmentos
- `comparar_segmentos`: Comparar dos segmentos semánticamente

## Backend de Retrieval

Las tools operan contra un backend polimórfico elegido una sola vez por
`initialize_search_engine()`:

- **`SearchServiceClient`** (`src/agent_service/search_client.py`): si
  `SEARCH_SERVICE_URL` está definido, el retrieval se delega por HTTP al search
  service de Cloud Run (`src/search_service/app.py`). El proceso del agente no
  importa PyTorch, FAISS ni el dataset (imports perezosos), por lo que la imagen
  se construye con `--no-group ml` y es desplegable en Agent Runtime.
  `SEARCH_SERVICE_TIMEOUT` (default 180 s) cubre el cold start de Cloud Run.
- **`AudioSearchEngine`** (`src/agent_service/search_engine.py`): modo local
  (desarrollo, playground, ingesta). Carga dataset, embeddings e índices en
  proceso. Si `DATASET_GCS_URI` está definido, `resolve_dataset_path` descarga
  antes el release de GCS a un snapshot local atómico.

### Inicialización (AudioSearchEngine)

```python
class AudioSearchEngine:
    def __init__(self, dataset_path: str):
        """
        Carga dataset, embeddings e índices FAISS.

        Orden de carga de embeddings:
        1. Columna 'text_embedding' del DataFrame (si existe)
        2. Archivos .npy individuales en embeddings/text_embeddings/
        3. Índice FAISS pre-construido en indices/text_index.faiss
        4. Fallback: genera embeddings en tiempo de búsqueda (lento)
        """
```

`MIN_SEGMENT_DURATION_SECONDS` (default 5) filtra de los resultados los
segmentos demasiado cortos para ser citables/escuchables.

### Búsqueda Semántica

```python
def search_semantic(self, query_text: str, k: int = 5) -> list[dict]:
    """
    Flujo de búsqueda:
    1. Genera embedding del query con Sentence Transformers
    2. Normaliza el embedding (L2)
    3. Busca top-K en índice FAISS (producto interno = coseno)
    4. Retorna segmentos con scores de similitud

    Returns:
        Lista de dicts con keys: segment, similarity, distance
    """
```

### Búsqueda Cross-Modal (CLAP)

```python
def search_audio_by_text(self, query_text: str, k: int = 5) -> list[dict]:
    """
    Búsqueda texto→audio usando CLAP.

    1. Genera embedding de texto en espacio CLAP (512-dim)
    2. Busca en índice FAISS de embeddings de audio
    3. Retorna segmentos ordenados por similitud cross-modal

    Útil para queries como "aplausos", "música", "gritos"
    que buscan por contenido acústico, no textual.
    """
```

### Búsqueda por clases acústicas (YAMNet)

```python
def search_audio_by_classes(self, query_text: str, k: int = 5) -> list[dict]:
    """
    Búsqueda texto→clases AudioSet sobre el índice invertido YAMNet.

    1. Traduce la query al inglés si QUERY_LANGUAGE no es inglés
    2. Normaliza tokens y lee los postings del índice invertido
    3. Score = cobertura de tokens × confianza media de las clases coincidentes
    4. Adjunta yamnet_matched_classes (las clases que justifican el resultado)

    No es comparable con la similitud coseno de texto o CLAP.
    """
```

## API REST (FastAPI)

La superficie de producción (`src/fast_api_app.py`) se construye con
`google.adk.cli.fast_api.get_fast_api_app` y agrega rutas propias:

| Método | Ruta | Descripción |
| -------- | ------ | ------------- |
| GET | `/health` | `healthy` si el runner está inicializado, `degraded` si no |
| POST | `/query` | Consulta al agente (principal) |
| POST | `/evaluate/query` | Consulta con evidencia de tools para evaluación (`contexts`, `retrieved_segments`) |
| POST | `/feedback` | Feedback estructurado del usuario a Cloud Logging |
| * | rutas ADK web | UI de desarrollo y API estándar ADK (`web=True`) |
| POST | `/a2a/src` | Endpoint A2A v1.0 (con interfaz de compatibilidad 0.3) |
| POST | `/api/stream_reasoning_engine`, `/api/reasoning_engine` | Adapter para Vertex AI Console Playground |

Además, el lifespan configura **telemetría OpenTelemetry**
(`src/app_utils/telemetry.py`, trazas a Cloud Trace en Agent Engine) y servicios
de sesión/artefactos compartidos entre las rutas ADK, A2A y el adapter.

El search service de Cloud Run expone su propia API de retrieval
(`/search/semantic`, `/search/audio`, `/search/yamnet`, `/segments/{id}`,
`/segments/{id}/audio-classes`, `/corpus`, `/health`, `/readyz`); el proxy
CopilotKit la usa para la búsqueda directa (`/api/search*`, ver spec 08).

### POST `/query`

**Request**:

```json
{
    "query": "Busca segmentos sobre política económica",
    "max_results": 5
}
```

**Response**:

```json
{
    "response": "Encontré 5 segmentos relevantes sobre política económica:\n\n1. **Segmento #42** (87.2% similitud)\n   Texto: \"El presidente anunció nuevas medidas...\"\n   Tiempo: 2:05 - 2:22\n   Archivo: entrevista_2024_01_15.m4a\n   ...",
    "query": "Busca segmentos sobre política económica"
}
```

### GET `/health`

**Response**:

```json
{
    "status": "healthy"
}
```

(`degraded` hasta que el lifespan inicializa el runner.)

## Configuración

### Variables de Entorno

| Variable | Default | Descripción |
| ---------- | --------- | ------------- |
| `DATASET_PATH` | `./dataset` | Ruta al dataset procesado (o destino del snapshot GCS) |
| `DATASET_GCS_URI` | — | Release versionado en GCS a descargar al arrancar |
| `SEARCH_SERVICE_URL` | — | Si está definida, el retrieval se delega al search service Cloud Run |
| `SEARCH_SERVICE_TIMEOUT` | `180` | Timeout (s) de cada llamada al search service (cubre cold start) |
| `MIN_SEGMENT_DURATION_SECONDS` | `5` | Descarta resultados con duración menor o igual a este valor |
| `OPENAI_API_KEY` | (requerido) | API key de OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo LLM a usar |
| `QUERY_LANGUAGE` | `es` | Idioma de las queries; si no es inglés se traducen antes de CLAP/YAMNet |
| `AGENT_MODALITY` | `all` | Filtra tools de retrieval: `text`/`audio`/`yamnet`/`text_clap`/`all` |
| `ALLOW_ORIGINS` | — | Orígenes CORS separados por coma |
| `LOG_LEVEL` | `INFO` | Nivel de logging |

### Arranque del Servicio

```bash
# Variables de entorno
export DATASET_PATH=./dataset
export OPENAI_API_KEY=sk-...

# Iniciar servicio
uv run uvicorn src.fast_api_app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

## Ciclo de Vida de una Consulta

```text
1. [API] Recibe POST /query con {"query": "...", "max_results": 5}
   │
2. [Agent] AudioAgent.query(user_query) invocado
   │
3. [LLM] GPT-4o-mini recibe system prompt + user query
   │
4. [LLM] Decide usar tool 'buscar_audio' con parámetros inferidos
   │
5. [Tool] buscar_audio(query="...", k=5) se ejecuta
   │  ├── Genera embedding del query (Sentence Transformers)
   │  ├── Busca top-5 en FAISS
   │  └── Retorna JSON con resultados
   │
6. [LLM] Recibe resultados como "observation"
   │
7. [LLM] (Opcionalmente) usa otra tool o decide responder
   │
8. [LLM] Genera respuesta final formateada en español
   │
9. [API] Retorna QueryResponse con la respuesta del agente
```

## Patrones de Diseño del Agente

### Patrón Actual: Function Tools (ADK)

- Usa `google.adk.agents.Agent` y funciones Python tipadas
- ADK deriva el schema de cada FunctionTool desde signatura y docstring
- El modelo, accedido vía LiteLLM, infiere cuándo y cómo llamar cada tool
- Eficiente para búsquedas directas

### Extensibilidad futura: orquestación ADK

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

agent = Agent(name="audio_search_agent", model=LiteLlm(model="openai/gpt-4o-mini"), tools=tools)
```

- Para consultas complejas se pueden incorporar `SequentialAgent`, `ParallelAgent` o `LoopAgent`.

### Extensibilidad: Plan-and-Execute

Para consultas que requieren múltiples búsquedas y síntesis:

```text
Query: "Compara cómo se habla de inflación en enero vs marzo"
Plan:
  1. Buscar segmentos sobre inflación en enero
  2. Buscar segmentos sobre inflación en marzo
  3. Comparar tonos y contenidos
  4. Generar respuesta comparativa
```

## Manejo de Errores

| Escenario | Comportamiento |
| ----------- | --------------- |
| Dataset no encontrado | Error 503 en startup, log con instrucciones |
| OpenAI API key inválida | Error 503, log warning |
| Query vacía | Error 400, mensaje descriptivo |
| Sin resultados | Agente sugiere reformulación |
| Timeout de LLM | Retry hasta max_iterations, luego error 500 |
| FAISS index corrupto | Reconstruir desde embeddings en memoria |

## Criterios de Aceptación

1. El agente responde consultas en lenguaje natural en < 5 segundos (p95)
2. Las respuestas incluyen segmentos relevantes con metadata completa
3. El agente maneja consultas ambiguas sugiriendo reformulaciones
4. La API soporta consultas concurrentes (async)
5. El servicio arranca correctamente con un dataset válido
6. Health endpoint responde en < 100ms
7. El agente responde siempre en español
8. Las tools retornan JSON válido incluso en caso de error
