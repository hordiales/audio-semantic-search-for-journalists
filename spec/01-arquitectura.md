# 01 — Arquitectura del Sistema

## Resumen

El sistema implementa una **búsqueda agéntica multimodal** sobre un corpus de audio. Un agente de IA (Google ADK + LLM)
actúa como intermediario entre el usuario y un sistema RAG que combina búsqueda semántica por texto (transcripciones) y
por audio (embeddings acústicos).

## Diagrama de Arquitectura General

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         USUARIO (Periodista)                         │
│                    Consulta en lenguaje natural                       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CAPA DE AGENTE                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  AudioAgent (Google ADK Agent + Runner)                       │  │
│  │  - LLM: GPT-4o-mini (OpenAI vía LiteLLM)                     │  │
│  │  - Patrón: Function Tools (ADK)                               │  │
│  │  - Prompt: Especializado en búsqueda de audio periodístico    │  │
│  │                                                               │  │
│  │  Tools (filtrables con AGENT_MODALITY):                       │  │
│  │  ┌─────────────────┐  ┌──────────────────────────────────┐   │  │
│  │  │ buscar_audio    │  │ buscar_evento_acustico (CLAP)    │   │  │
│  │  │ (texto)         │  │ buscar_clase_audio (YAMNet)      │   │  │
│  │  ├─────────────────┤  ├──────────────────────────────────┤   │  │
│  │  │ obtener_info_   │  │ obtener_clases_audio             │   │  │
│  │  │ segmento        │  │ (etiquetas AudioSet por segmento)│   │  │
│  │  └────────┬────────┘  └──────────────┬───────────────────┘   │  │
│  └───────────┼──────────────────────────┼───────────────────────┘  │
└──────────────┼──────────────────────────┼───────────────────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CAPA DE RETRIEVAL (RAG)                          │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  AudioSearchEngine                                            │  │
│  │                                                               │  │
│  │  ┌──────────────────┐    ┌──────────────────────────────┐    │  │
│  │  │ Búsqueda Texto   │    │ Búsqueda Audio (CLAP)        │    │  │
│  │  │ SentenceTransf.  │    │ Audio↔Texto cross-modal      │    │  │
│  │  │ FAISS IndexFlatIP│    │ FAISS IndexFlatIP            │    │  │
│  │  └────────┬─────────┘    └──────────────┬───────────────┘    │  │
│  │           │                             │                    │  │
│  │           ▼                             ▼                    │  │
│  │  ┌──────────────────────────────────────────────────────┐    │  │
│  │  │              Dataset Indexado                         │    │  │
│  │  │  - Segmentos con transcripciones                     │    │  │
│  │  │  - Embeddings de texto (384 dim)                     │    │  │
│  │  │  - Embeddings de audio CLAP (512 dim)                │    │  │
│  │  │  - Metadatos (timestamps, archivo, idioma)           │    │  │
│  │  │  - Sentimiento del texto                             │    │  │
│  │  └──────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
               ▲
               │ (offline, batch)
               │
┌─────────────────────────────────────────────────────────────────────┐
│                     CAPA DE INGESTA (Pipeline)                       │
│                                                                     │
│  Audio → Conversión → Transcripción → Embeddings → Indexación       │
│  (.m4a)   (ffmpeg)    (Whisper)       (ST + CLAP)   (FAISS)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Componentes Principales

### 1. Capa de Agente

**Responsabilidad**: Interpretar consultas en lenguaje natural y orquestar la búsqueda.

| Componente | Archivo | Descripción |
| --- | --- | --- |
| Entrypoint ADK | `src/agent.py` | `root_agent`/`app` para agents-cli y Agent Runtime |
| AudioAgent | `src/agent_service/agent.py` | Agente ADK (`root_agent`) ejecutado por Runner |
| Tools | `src/agent_service/tools.py` | 5 herramientas de búsqueda, filtrables con `AGENT_MODALITY` |
| Backend de retrieval | `src/agent_service/search_client.py` | Cliente HTTP del search service (si `SEARCH_SERVICE_URL` está definido) |
| API REST | `src/fast_api_app.py` | Superficie ADK de producción: `/query`, `/evaluate/query`, `/feedback`, `/health`, rutas ADK web, A2A (`/a2a/src`) y adapter de Reasoning Engine |

**Decisiones de diseño**:

- El agente usa `google.adk.agents.Agent` con funciones Python tipadas como tools
- El modelo se accede mediante LiteLLM para preservar la configuración OpenAI existente
- El Runner ADK gestiona eventos y sesiones locales en memoria
- El prompt del sistema está en español y orientado al dominio periodístico
- El prompt incluye un modo interno de planificación (`DIRECT_SEARCH_PLANNER_V1`) que la búsqueda directa usa para
  obtener un plan estructurado sin ejecutar el loop conversacional
- Las tools leen `max_results` del estado de sesión ADK (`ToolContext`), no de una instrucción al LLM
- Cada resultado incluye `search_index` y `search_index_label` para que la respuesta cite el índice consultado de forma
  verificable
- Con `SEARCH_SERVICE_URL` definido, las tools delegan el retrieval al servicio Cloud Run y el agente nunca importa
  PyTorch/FAISS (imports perezosos), lo que mantiene la imagen del agente desplegable en Agent Runtime

**Interfaces de usuario**:

| Componente | Directorio | Descripción |
| --- | --- | --- |
| Widget conversacional | `copilotkit-proxy/widget/` | Widget embebible CopilotKit (AG-UI/A2A), publicado en GitHub Pages |
| Búsqueda directa | `frontend-direct-search/` | Consola de archivo React+Vite con reproductor inline (ver spec 08) |
| Proxy | `copilotkit-proxy/` | Expone AG-UI/A2A para el widget y la API estructurada `/api/search*` hacia el search service |

### 2. Capa de Retrieval (RAG)

**Responsabilidad**: Búsqueda vectorial semántica sobre el corpus indexado.

| Componente | Archivo | Descripción |
| --- | --- | --- |
| AudioSearchEngine | `src/agent_service/search_engine.py` | Motor de búsqueda con FAISS |
| Search service | `src/search_service/app.py` | Misma lógica expuesta por HTTP en Cloud Run (imagen con checkpoint CLAP) |
| Configuración | `src/embedding_config.py` | Lee `EMBEDDINGS_ACTIVE`/`CLASSIFIERS_ACTIVE` del entorno y autogenera `config/embeddings.toml` (no editar a mano) |
| CLAP Embeddings | `src/clap_audio_embeddings.py` | Embeddings audio-texto cross-modal |
| Gemini Embeddings | `src/gemini_multimodal_embeddings.py` | Embeddings nativos texto-audio para comparación |
| YAMNet | `src/yamnet_audio_classifier.py` | Clasificación AudioSet opcional por segmento |
| Índice invertido YAMNet | `src/yamnet_inverted_index.py` | Postings token AudioSet → segmentos para `search_audio_by_classes` |
| Traducción de queries | `src/query_translation.py` | Traduce ES→EN antes de los encoders CLAP/YAMNet (entrenados en inglés) |
| Text Embeddings | `src/text_embeddings.py` | Embeddings de transcripciones |

**Modalidades de búsqueda**:

1. **Texto → Texto**: MiniLM contra transcripciones, si `text` está activo (tool `buscar_audio`).
2. **Texto → Audio**: CLAP contra ventanas de audio, si `clap` está activo (tool `buscar_evento_acustico` ). La query se
   traduce al inglés antes del encoder (`QUERY_LANGUAGE`).
3. **Texto → Clases**: búsqueda por clases AudioSet sobre el índice invertido YAMNet (tool `buscar_clase_audio` ), sin
   similitud vectorial: cobertura de tokens × confianza del clasificador.
4. **Gemini audio**: índice separado para benchmark; no se expone como tool del agente.
5. **YAMNet (ingesta)**: clasifica cada segmento en etiquetas AudioSet y publica un índice
   invertido JSON (`indices/yamnet_inverted_index.json`). No es un índice
   vectorial: cada token de una etiqueta apunta a sus segmentos y scores.

**Índices vectoriales**:

- FAISS IndexFlatIP (producto interno, equivalente a coseno con vectores normalizados)
- Dimensiones configurables por modelo: MiniLM 384, CLAP 512 y Gemini 1536 por defecto.

### 3. Capa de Ingesta (Pipeline)

**Responsabilidad**: Procesar audio crudo y generar un dataset indexado listo para búsqueda.

| Componente | Archivo | Descripción |
| --- | --- | --- |
| Pipeline Principal | `src/simple_dataset_pipeline.py` | Orquestador del pipeline completo |
| Conversión Audio | `src/audio_conversion.py` | Conversión a WAV 16kHz mono |
| Transcripción | `src/audio_transcription.py` | Whisper speech-to-text |
| Text Embeddings | `src/text_embeddings.py` | Generación de embeddings de texto |
| Segmentación acústica | `src/audio_segmenting.py` | Ventanas WAV alineadas a timestamps |
| Chunking | `src/audio_transcription.py` | Estrategias whisper, fixed, sentence y paragraph |
| Embeddings | Variables `EMBEDDINGS_*`/`CLASSIFIERS_*` → `config/embeddings.toml` autogenerado | Selección de MiniLM, CLAP, Gemini y YAMNet |
| Indexación | `src/vector_indexing.py` | Construcción de índices FAISS |
| Sentimiento | `src/sentiment_analysis.py` | Análisis de sentimiento del texto |
| Clips de reproducción | `src/segment_clips.py` | Un Opus por segmento (ver spec 02, etapa 6) |
| Publicación | `scripts/publish_dataset_release.py` | Publica el dataset como release versionado en GCS |

### 4. Capa de Infraestructura

**Responsabilidad**: Despliegue, almacenamiento y operaciones.

| Componente | Tecnología | Descripción |
| --- | --- | --- |
| Compute (agente) | GCP Agent Runtime | Agente ADK desplegado con `agents-cli deploy`, sin dependencias ML |
| Compute (retrieval) | GCP Cloud Run | Search service con checkpoint CLAP en la imagen, scale-to-zero |
| Compute (proxy) | GCP Cloud Run | copilotkit-proxy: widget AG-UI/A2A + API `/api/search*` |
| Frontends | GitHub Pages | Widget embebible + front de búsqueda directa (estáticos) |
| Storage | GCS (Google Cloud Storage) | Releases del dataset (índices + clips Opus), URLs firmadas |
| CI/CD | GitHub Actions | `pr_checks` → `staging` (push a main) → `deploy-to-prod` (manual) + `deploy-pages` |
| Container | Docker | Dos imágenes: agente (`Dockerfile`) y search service (`Dockerfile.search-service`) |

La división agente/retrieval existe porque el checkpoint CLAP (~1.7 GB) y las
dependencias ML son una mala combinación con un runtime scale-to-zero: el agente
se despliega sin PyTorch/FAISS y llama al search service por HTTP con
`SEARCH_SERVICE_URL`. Ver `ARCHITECTURE-CLOUD.md`.

## Flujo de Datos

### Flujo de Ingesta (Offline)

```text
1. Audio crudo (.m4a, .mp3, .wav, .opus, .flac, .ogg)
   │
2. ▼ Conversión (ffmpeg) → WAV 16kHz mono
   │
3. ▼ Transcripción (Whisper) → Segmentos con timestamps + texto
   │
4. ▼ Embeddings de texto (Sentence Transformers) → vectors 384-dim
   │
5. ▼ Embeddings de audio (CLAP) → vectors 512-dim
   │
6. ▼ Clasificación opcional YAMNet → clases AudioSet por segmento
   │
   ▼ Índice invertido JSON (token AudioSet → postings de segmento/clase/score)
   │
7. ▼ Análisis de sentimiento → positive/negative/neutral scores
   │
8. ▼ Indexación FAISS → índices vectoriales serializados
   │
9. ▼ Dataset final (pickle/CSV + índices .faiss)
```

### Flujo de Búsqueda (Online)

```text
1. Usuario envía query en lenguaje natural
   │
2. ▼ API REST recibe request POST /query
   │
3. ▼ AudioAgent (ADK) interpreta la consulta
   │
4. ▼ Agent decide qué tool usar (buscar_audio)
   │
5. ▼ Tool ejecuta búsqueda semántica:
   │   a. Genera embedding del query (Sentence Transformers)
   │   b. Busca top-K en índice FAISS (similitud coseno)
   │   c. Retorna segmentos con scores
   │
6. ▼ Agent analiza resultados y genera respuesta en lenguaje natural
   │
7. ▼ Respuesta JSON al usuario con segmentos relevantes
```

## Modelos de Datos

### Segmento de Audio (unidad mínima)

```python
@dataclass
class AudioSegment:
    segment_id: int  # ID único del segmento
    text: str  # Transcripción del segmento
    start_time: float  # Tiempo de inicio (segundos)
    end_time: float  # Tiempo de fin (segundos)
    original_file_name: str  # Archivo de audio original
    language: str  # Idioma detectado (es, en, ...)
    confidence: float  # Confianza de la transcripción
    text_embedding: np.ndarray | None  # MiniLM, si está activo
    clap_embedding: np.ndarray | None  # CLAP, si está activo
    gemini_embedding: np.ndarray | None  # Gemini, si está activo
    audio_window_count: int  # Ventanas WAV asociadas al segmento
    yamnet_top_classes: list[dict] | None  # Etiquetas AudioSet opcionales
    sentiment_positive: float  # Score sentimiento positivo
    sentiment_negative: float  # Score sentimiento negativo
    sentiment_neutral: float  # Score sentimiento neutral
    dominant_sentiment: str  # Sentimiento dominante
```

### Resultado de Búsqueda

```python
@dataclass
class SearchResult:
    segment: AudioSegment  # Segmento encontrado
    similarity: float  # Similitud coseno (0-1)
    distance: float  # Distancia (1 - similarity)
```

### Request/Response de la API

```python
class QueryRequest(BaseModel):
    query: str  # Consulta en lenguaje natural
    max_results: int = 5  # Máximo resultados (1-20)


class QueryResponse(BaseModel):
    response: str  # Respuesta del agente (texto libre)
    query: str  # Consulta original
```

### Índice invertido YAMNet

YAMNet se ejecuta durante la ingesta sobre ventanas de audio; la transcripción
sólo aporta los límites temporales del segmento. El pipeline conserva las
`top_k` clases por segmento en `yamnet_top_classes` y publica además:

```text
dataset/indices/yamnet_inverted_index.json
  "applause" -> [{segment_id, class_id, class_name, score, class_rank}, ...]
  "speech"   -> [{segment_id, class_id, class_name, score, class_rank}, ...]
```

En una consulta, se traduce la descripción al inglés, se normalizan sus tokens
y se leen sólo los postings de esos tokens. Para cada segmento candidato se
conservan las clases coincidentes y se calcula:

```text
score = cobertura de tokens de la consulta × promedio de confianza YAMNet
```

Por lo tanto, el coste online depende de los postings de las clases consultadas,
no de recorrer todos los segmentos. Los scores siguen sin ser comparables con
la similitud coseno de texto o CLAP y se muestran en una columna separada.

## Decisiones de Diseño

### ¿Por qué un agente y no solo retrieval directo?

1. **Interpretación de consultas ambiguas**: El agente reformula y entiende intención
2. **Respuestas contextualizadas**: No devuelve solo una lista, sino una respuesta coherente
3. **Capacidad de multi-step**: Puede buscar, filtrar, refinar en múltiples pasos
4. **Extensibilidad**: Agregar nuevas tools sin cambiar la interfaz

### ¿Por qué FAISS y no una vector DB en la nube?

1. **Latencia**: Búsqueda in-memory < 1ms para corpus de hasta ~100K segmentos
2. **Sin dependencias externas**: No requiere servicio de BD vectorial
3. **Costo**: Incluido en la instancia de Cloud Run
4. **Simplicidad**: Un archivo `.faiss` serializable

### ¿Por qué búsqueda multimodal (texto + audio)?

1. **Complementariedad**: Texto captura "de qué se habla", audio captura "cómo suena"
2. **Casos de uso periodísticos**: Buscar "aplausos durante discurso" requiere ambos
3. **CLAP permite cross-modal**: Un query de texto puede buscar directamente en embeddings de audio

## Dependencias Principales

Las dependencias se declaran en `pyproject.toml` (formato PEP 621) y se bloquean
en `uv.lock`; se instalan con `uv sync`. Grupos y extras relevantes:

- **Núcleo**: `openai-whisper`, `sentence-transformers`, `transformers`, `librosa`,
  `soundfile`, `faiss-cpu`, `pandas`, `numpy`, `laion-clap`.
- **Agente/API**: `google-adk >=2.0,<3.0`, `litellm`, `fastapi`, `uvicorn`,
  `a2a-sdk`, `google-cloud-aiplatform[evaluation,agent-engines]`.
- **Grupo `ml`** (`uv sync --group ml`): PyTorch/FAISS y modelos. La imagen del
  agente se construye con `--no-group ml` y delega el retrieval al search service.
- **Grupo `eval`** (`uv sync --group eval`): RAGAS. **Extra `eval-deepeval`**:
  backend alternativo DeepEval.
- **Extra `yamnet`** (`uv sync --extra yamnet`): TensorFlow/TF-Hub para la
  clasificación AudioSet en la ingesta.
- **Índice `pytorch-cpu`**: builds CPU de PyTorch para las imágenes Docker.
