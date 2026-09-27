# Especificación: Búsqueda Agéntica de Audio para Periodistas

## Visión General

Sistema de **búsqueda agéntica multimodal** (texto y audio) diseñado para periodistas. Un agente de IA con acceso a
herramientas (tools) realiza búsqueda semántica sobre un corpus de audios previamente transcritos e indexados,
combinando embeddings de texto y embeddings acústicos para ofrecer resultados relevantes en lenguaje natural.

La solución se evalúa en dos niveles:

1. **Calidad del agente** — usando RAGAS y agents-cli eval (grounding, uso de tools y calidad de respuesta), más cuatro
   Expectations periodísticas: cita de fuente, momento temporal, grounding e idioma consistente
2. **Calidad del retrieval** — métricas clásicas de Information Retrieval (Precision@K, Recall@K, MRR, NDCG@K)

## Dominio de Aplicación

Periodistas que necesitan:

- Buscar fragmentos específicos en grandes volúmenes de audio (entrevistas, podcasts, programas de radio/TV, discursos,
  archivo histórico)
- Encontrar segmentos por contenido semántico ("política económica", "declaraciones sobre clima")
- Encontrar segmentos por eventos acústicos ("aplausos", "gritos de protesta", "música de fondo")
- Obtener respuestas contextualizadas en lenguaje natural

## Stack Tecnológico

| Componente | Tecnología | Versión |
| --- | --- | --- |
| Lenguaje | Python | 3.11.13 |
| Gestión de dependencias | uv | lockfile `uv.lock` |
| Transcripción | OpenAI Whisper | base/medium |
| Embeddings de texto | Sentence Transformers | all-MiniLM-L6-v2 |
| Embeddings de audio | CLAP (LAION) | laion/clap-htsat-unfused |
| Embeddings multimodales opcionales | Gemini Embedding 2 | gemini-embedding-2 |
| Clasificación de audio | YAMNet (TensorFlow Hub) | 1.0 |
| Indexación vectorial | FAISS | >= 1.7 |
| Agente | Google ADK + LiteLLM | GPT-4o-mini (OpenAI) |
| Protocolo de agente | A2A (a2a-sdk) | v1.0 (+ compat 0.3) |
| API REST | FastAPI + Uvicorn | >= 0.100 |
| Servicio de retrieval | FastAPI en Cloud Run (contenedor con checkpoint CLAP) | — |
| Frontends | React 18 + TypeScript + Vite | direct-search + widget CopilotKit |
| Proxy | Node.js (Express) | CopilotKit AG-UI + API de búsqueda directa |
| Evaluación agente | RAGAS o DeepEval + agents-cli eval | >= 0.1 / >= 1.1 |
| Evaluación retrieval | Custom (Precision@K, MRR, NDCG) + benchmark Clotho para CLAP | — |
| Infraestructura | GCP Agent Runtime + Cloud Run + GCS + GitHub Pages | — |
| CI/CD | GitHub Actions | — |
| Containerización | Docker | — |

## Documentos de Especificación

| # | Documento | Descripción |
| --- | --- | --- |
| 1 | [01-arquitectura.md](./01-arquitectura.md) | Arquitectura general, componentes, flujo de datos, decisiones de diseño |
| 2 | [02-pipeline-ingesta.md](./02-pipeline-ingesta.md) | Pipeline de ingesta: conversión, transcripción, embeddings, indexación |
| 3 | [03-agente-busqueda.md](./03-agente-busqueda.md) | Agente Google ADK, tools RAG multimodal, API REST |
| 4 | [04-evaluacion.md](./04-evaluacion.md) | Evaluación: RAGAS/DeepEval + agents-cli eval (agente) + métricas retrieval (RAG) + benchmarks CLAP |
| 5 | [05-infraestructura-deploy.md](./05-infraestructura-deploy.md) | Infraestructura GCP, CI/CD, Docker, despliegue |
| 6 | [06-plan-incorporacion-spec-comparar.md](./06-plan-incorporacion-spec-comparar.md) | Plan y estado de incorporación de `spec-comparar` (histórico) |
| 7 | [07-guia-preparar-procesar-dataset.md](./07-guia-preparar-procesar-dataset.md) | Guía operativa: preparar y procesar un dataset de audio |
| 8 | [08-frontend-busqueda-directa.md](./08-frontend-busqueda-directa.md) | Front de búsqueda directa con escucha de segmentos |

## Prerequisitos del Entorno

```bash
# Sistema operativo: macOS o Linux
# Hardware mínimo: 8GB RAM, CPU multi-core
# Hardware recomendado: 16GB+ RAM, GPU opcional (acelera Whisper/CLAP)

# Software requerido
pyenv install 3.11.13
pyenv local 3.11.13
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv (gestión de dependencias)
brew install ffmpeg  # macOS
uv sync              # instala las dependencias bloqueadas de uv.lock
uv tool install google-agents-cli  # agents-cli (evaluación y despliegue)
```

## Convenciones

- **Idioma del código**: inglés (nombres de variables, funciones, clases)
- **Idioma de la documentación**: español
- **Idioma del prompt del agente**: español (sistema y respuestas)
- **Estilo de código**: Ruff (linting + formatting), pre-commit hooks
- **Testing**: pytest
- **Estructura de imports**: absolutos desde raíz del proyecto

## Estructura del Proyecto

```text
audio-semantic-search-for-journalists/
├── src/                        # Código fuente principal
│   ├── agent.py                # Entrypoint ADK (agents-cli / Agent Runtime)
│   ├── agent_service/          # Agente Google ADK y tools de retrieval
│   │   ├── agent.py            # root_agent ADK + AudioAgent compatible
│   │   ├── search_engine.py    # Motor de búsqueda FAISS (in-process)
│   │   ├── search_client.py    # Cliente HTTP del search service (SEARCH_SERVICE_URL)
│   │   ├── tools.py            # 5 tools del agente + filtrado AGENT_MODALITY
│   │   └── main.py             # Servicio local del agente (desarrollo)
│   ├── app_utils/              # Telemetry OTel, servicios, A2A, adapter reasoning engine
│   ├── search_service/         # Servicio de retrieval para Cloud Run
│   │   └── app.py              # /search/semantic, /search/audio, /search/yamnet, /corpus
│   ├── audio_transcription.py  # Transcripción con Whisper
│   ├── fast_api_app.py         # Superficie ADK de producción (A2A, feedback, /query)
│   ├── embedding_config.py     # Config de ingesta desde variables de entorno
│   ├── text_embeddings.py      # Embeddings MiniLM
│   ├── clap_audio_embeddings.py # Embeddings CLAP
│   ├── gemini_multimodal_embeddings.py # Embeddings Gemini opcionales
│   ├── yamnet_audio_classifier.py # Clases AudioSet con YAMNet (opcional)
│   ├── yamnet_inverted_index.py # Índice invertido token→segmentos (YAMNet)
│   ├── query_translation.py    # Traducción ES→EN de queries para CLAP/YAMNet
│   ├── segment_clips.py        # Clips Opus de reproducción por segmento
│   ├── segment_clip_storage.py # URLs firmadas GCS para los clips
│   ├── dataset_storage.py      # Descarga de releases del dataset desde GCS
│   ├── simple_dataset_pipeline.py # Pipeline de ingesta incremental
│   └── ...
├── copilotkit-proxy/           # Proxy Node: widget conversacional + API /api/search
├── frontend-direct-search/     # Front React de búsqueda directa (GitHub Pages)
├── widget/                     # Bundle publicado del widget embebible
├── benchmarks/                 # Evaluación y benchmarks
│   ├── evaluate_clap_clotho.py # Sanity check de CLAP contra Clotho
│   ├── compare_clap_by_question_type.py
│   └── ...
├── evaluation/                 # Evaluación RAGAS/DeepEval y retrieval
│   ├── ragas_evaluation.py     # Evaluación con RAGAS
│   ├── deepeval_evaluation.py  # Backend alternativo DeepEval
│   ├── evaluation_config.py    # Fuentes de dataset manual/synthetic/hybrid
│   ├── retrieval_evaluation.py # Evaluación retrieval aislado
│   ├── generate_synthetic_questions.py # Generación sintética de preguntas
│   ├── test_datasets/          # Datasets de evaluación
│   └── results/                # Resultados de evaluación
├── scripts/                    # Utilidades operativas
│   └── publish_dataset_release.py # Publica releases del dataset en GCS
├── deployment/                 # Configuración de despliegue (agents-cli)
├── data/                       # Audio fuente (no versionado)
├── dataset/                    # Dataset procesado (no versionado)
├── tests/                      # Tests unitarios y de integración
├── docs/                       # Documentación
├── spec/                       # Esta especificación
├── Dockerfile                  # Imagen del agente (sin grupo ml)
├── Dockerfile.search-service   # Imagen del search service (con checkpoint CLAP)
├── cloudbuild.search-service.yaml # Build del search service en Cloud Build
├── pyproject.toml              # Dependencias (uv)
├── uv.lock                     # Lockfile
├── .env                        # Variables de entorno (no versionado)
└── README.md
```

## Criterios de Aceptación Globales

1. El pipeline de ingesta procesa audios y genera un dataset indexado con embeddings de texto/audio y, opcionalmente,
   clases acústicas YAMNet
2. El agente responde consultas en lenguaje natural usando tools de búsqueda semántica
3. La búsqueda es multimodal: puede buscar por contenido textual (transcripción) y por eventos acústicos
4. RAGAS evalúa la calidad de respuestas del agente con Faithfulness >= 0.7 y las cuatro Expectations periodísticas
   alcanzan >= 0.8 promedio cada una
5. agents-cli eval registra trazas ADK y evalúa grounding, uso de tools y calidad de respuesta
6. El retrieval aislado alcanza Precision@5 >= 0.6 y MRR >= 0.5 en el dataset de evaluación periodístico
7. La solución se despliega en Google Agent Platform (Agent Runtime) con CI/CD automatizado
