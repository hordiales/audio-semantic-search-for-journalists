# Búsqueda Agéntica de Audio para Periodistas

Sistema de **búsqueda agéntica multimodal** (texto y audio) diseñado para periodistas. Un agente de IA con acceso a herramientas realiza búsqueda semántica sobre un corpus de audios previamente transcritos e indexados, combinando embeddings de texto y embeddings acústicos.

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Gestión de dependencias | Poetry |
| Transcripción | OpenAI Whisper |
| Embeddings de texto | Sentence Transformers (all-MiniLM-L6-v2) |
| Embeddings de audio | CLAP (LAION) |
| Indexación vectorial | FAISS |
| Agente | LangChain + OpenAI GPT-4o-mini |
| API REST | FastAPI + Uvicorn |
| Evaluación | RAGAS + métricas IR custom |

## Setup

### Prerequisitos

```bash
# Python 3.11
pyenv install 3.11.13
pyenv local 3.11.13

# Poetry
curl -sSL https://install.python-poetry.org | python3 -

# ffmpeg (macOS)
brew install ffmpeg
```

### Instalación

```bash
# Instalar dependencias
poetry install

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu OPENAI_API_KEY

# Instalar pre-commit hooks
poetry run pre-commit install
```

## Uso

### Configurar embeddings activos

El pipeline lee [config/embeddings.toml](config/embeddings.toml). La lista
`[embeddings].active` define qué modelos se procesan y qué índices se crean:

```toml
[embeddings]
active = ["text", "clap", "gemini"]
```

- `text`: MiniLM para consulta↔transcripción (`text_index.faiss`).
- `clap`: CLAP para consulta↔audio (`audio_index.faiss`).
- `gemini`: Gemini Embedding 2 nativo para consulta↔audio
  (`gemini_audio_index.faiss`); requiere `GEMINI_API_KEY`.

Para usar otro archivo, pasa `--embeddings-config ruta/al/archivo.toml`. El
manifiesto final registra los embeddings efectivamente generados, sus modelos,
dimensiones e índices.

### 1. Pipeline de Ingesta

Procesa archivos de audio y genera un dataset indexado:

```bash
poetry run python -m src.simple_dataset_pipeline \
    --input data/ \
    --output ./dataset \
    --whisper-model base \
    --batch-size 8 \
    --verbose
```

Para testing sin CLAP (más rápido):

```bash
poetry run python -m src.simple_dataset_pipeline \
    --input data/ \
    --output ./dataset \
    --mock-audio \
    --verbose
```

### 2. Servicio del Agente

```bash
# Iniciar API
poetry run uvicorn src.agent_service.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

### 3. Consultar al Agente

```bash
# POST request
curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "Busca segmentos sobre política económica", "max_results": 5}'

# Health check
curl http://localhost:8000/health
```

## Evaluación

### Comparativa: esquema actual vs. Gemini Embedding 2

El benchmark no modifica el servicio de producción. Evalúa por separado `minilm_text`
(consulta↔transcripción), `clap_audio` (consulta↔audio) y
`gemini_embedding_2_audio` (consulta↔WAV) sobre los mismos segmentos y queries
anotadas. Requiere ventanas acústicas y `relevant_segment_ids` no vacíos.

```bash
export GEMINI_API_KEY="..."
uv run python -m benchmarks.compare_retrieval_with_gemini \
    --dataset-path ./dataset \
    --queries evaluation/test_datasets/retrieval_eval_dataset.json \
    --output evaluation/results/gemini_comparison.json
```

### Retrieval (RAG aislado)

```bash
poetry run python -m evaluation.retrieval_evaluation \
    --dataset evaluation/test_datasets/retrieval_eval_dataset.json \
    --dataset-path ./dataset \
    --output evaluation/results/retrieval_results.json
```

### RAGAS (calidad del agente)

```bash
# Requiere servicio corriendo
poetry run python -m evaluation.ragas_evaluation \
    --dataset evaluation/test_datasets/ragas_eval_dataset.json \
    --agent-url http://localhost:8000 \
    --output evaluation/results/ragas_results.json
```

## Tests

```bash
poetry run pytest tests/ -v
```

## Estructura del Proyecto

```
├── src/                           # Código fuente
│   ├── agent_service/             # Agente + API
│   │   ├── main.py                # FastAPI app
│   │   ├── agent.py               # AudioAgent (LangChain)
│   │   ├── search_engine.py       # Motor FAISS
│   │   └── tools.py               # Tools del agente
│   ├── audio_conversion.py        # ffmpeg conversion
│   ├── audio_transcription.py     # Whisper
│   ├── text_embeddings.py         # Sentence Transformers
│   ├── clap_audio_embeddings.py   # CLAP
│   ├── sentiment_analysis.py      # Sentiment analysis
│   ├── vector_indexing.py         # FAISS indexing
│   └── simple_dataset_pipeline.py # Pipeline orchestrator
├── evaluation/                    # Framework de evaluación
│   ├── ragas_evaluation.py        # RAGAS evaluation
│   ├── retrieval_evaluation.py    # IR metrics
│   └── test_datasets/             # Evaluation datasets
├── tests/                         # Unit tests
├── data/                          # Audio source (not versioned)
├── dataset/                       # Processed dataset (not versioned)
├── spec/                          # Specification documents
├── pyproject.toml                 # Poetry dependencies
└── .env.example                   # Environment template
```
