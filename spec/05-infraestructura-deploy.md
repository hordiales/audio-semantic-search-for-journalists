# 05 — Infraestructura y Despliegue

## Resumen

El sistema se despliega en Google Cloud Platform (GCP) con **cuatro piezas
separadas**, más los frontends estáticos:

1. **Agente ADK → Agent Runtime** (Google Agent Platform), desplegado con
   `agents-cli deploy`. La imagen no contiene PyTorch/FAISS/CLAP.
2. **Search service → Cloud Run**: todo lo que necesita los modelos de ML
   (checkpoint CLAP ~1.7 GB, FAISS, dataset). El agente lo llama por HTTP
   (`SEARCH_SERVICE_URL`).
3. **copilotkit-proxy → Cloud Run**: proxy Node que sirve el widget
   conversacional (AG-UI/A2A) y la API estructurada `/api/search*` del front de
   búsqueda directa.
4. **Frontends → GitHub Pages**: widget embebible y front de búsqueda directa,
   más la documentación de `docs/`.

Cada dataset se publica en GCS como **release versionado**. El release por defecto se descarga como snapshot atómico al
arrancar y los demás se cargan bajo demanda cuando hay catálogo; los clips Opus se sirven on-demand con URLs firmadas.
GitHub Actions automatiza checks, despliegue del agente y GitHub Pages. El despliegue del search service y del proxy no
tiene workflow propio en el árbol actual. No se usa Datastore: los metadatos viajan dentro de cada release.

La división agente/retrieval existe porque el checkpoint CLAP y las
dependencias ML inflaban la imagen del agente a ~8 GB y Agent Runtime nunca
terminaba de descargarla. Detalle en `DOC-interna/ARCHITECTURE-CLOUD.md`.

## Arquitectura de Infraestructura

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Usuarios                                     │
│   Widget embebible (Pages)      Front búsqueda directa (Pages)       │
└───────────────┬───────────────────────────┬─────────────────────────┘
                │ AG-UI/A2A                 │ REST /api/search*
                ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Cloud Run: copilotkit-proxy (contenedor Node, Dockerfile propio)    │
│  - Reenvía chat al agente vía A2A                                    │
│  - Planner: pide el SearchPlan al agente vía A2A                     │
│    (DIRECT_SEARCH_PLANNER_V1)                                        │
│  - Llama al search service con identidad IAM (ID token)              │
└───────────────┬───────────────────────────┬─────────────────────────┘
                │ A2A                       │ HTTP + IAM
                ▼                           ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│  Agent Runtime (GCP)         │  │  Cloud Run: audio-search-service │
│  - Imagen sin ML             │  │  - Imagen con checkpoint CLAP    │
│    (--no-group ml)           │  │  - /search/semantic|audio|yamnet │
│  - /query, /evaluate/query,  │  │  - /segments/{id}, /corpus       │
│    /feedback, A2A, OTel      │  │  - /health + /readyz             │
│  - retrieval vía             │  │  - scale-to-zero; cold start     │
│    SEARCH_SERVICE_URL ───────┼─▶│    carga el checkpoint           │
└───────────────┬──────────────┘  └───────────────┬──────────────────┘
                │                                 │
                ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Cloud Storage (GCS)                                                 │
│  gs://<bucket>/releases/<fecha>/   ← publish_dataset_release.py      │
│    ├── indices/ (FAISS)   ├── final/ (pkl + manifiesto)              │
│    └── segment_clips/ (Opus, servidos con URLs firmadas V4)          │
│  gs://<logs-bucket>/load-test-results/  ← resultados de load tests   │
│  Artifact Registry: imágenes Docker versionadas por commit SHA       │
│  IAM: Workload Identity Federation (GitHub → GCP), SAs por servicio  │
└─────────────────────────────────────────────────────────────────────┘
```

## Docker

Hay **tres imágenes**, una por pieza de cómputo:

### Agente — `Dockerfile` (raíz)

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir uv==0.8.13
WORKDIR /code
COPY ./pyproject.toml ./README.md ./uv.lock ./
COPY ./src ./src
# Sin PyTorch, CLAP, Whisper ni FAISS: con ellos la capa llegaba a ~8 GB y
# Agent Runtime nunca terminaba de descargarla. El retrieval vive en
# Dockerfile.search-service.
RUN uv sync --frozen --no-dev --no-group eval --no-group ml
ARG COMMIT_SHA=""
ENV COMMIT_SHA=${COMMIT_SHA}
EXPOSE 8080
CMD ["uv", "run", "--no-sync", "uvicorn", "src.fast_api_app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Search service — `Dockerfile.search-service`

Mismo patrón (`uv sync` desde el lockfile), pero **con** el grupo `ml`:
PyTorch, CLAP, el encoder de Sentence Transformers y los índices FAISS. El
checkpoint CLAP viaja en la imagen para acotar el cold start. El build corre en
Cloud Build (`cloudbuild.search-service.yaml`, builder Docker, timeout 1800 s).

### Proxy — `copilotkit-proxy/Dockerfile`

Multi-stage Node 22: `npm ci` + `npm run build` (TypeScript) y runtime con
`npm ci --omit=dev`. Expone 8080 y arranca `node dist/server.js`.

## CI/CD con GitHub Actions

Cuatro workflows en `.github/workflows/`:

| Workflow | Disparo | Qué hace |
| --- | --- | --- |
| `pr_checks.yaml` | Pull request a `main` con cambios en `src/**`, `tests/**`, `deployment/**` o `poetry.lock` | `uv sync --locked`, `pytest tests/unit`, tests deterministas |
| `staging.yaml` | Push a `main` con esos mismos filtros | Deploy a **staging** en Agent Runtime (`uvx google-agents-cli@1.1.0 deploy`), actualiza env vars (`COMMIT_SHA`, `LOGS_BUCKET_NAME`), corre load test (`tests/load_test/`) y exporta resultados a GCS; al terminar invoca `deploy-to-prod.yaml` |
| `deploy-to-prod.yaml` | `workflow_dispatch` / `workflow_call` | Deploy a **producción** en Agent Runtime |
| `deploy-pages.yaml` | Push a `main` (`docs/**`, `copilotkit-proxy/widget/**`, `frontend-direct-search/**` o el workflow) o manual | Build del widget y del front direct-search (npm/Vite), copia `docs/index.html` y ambos bundles a `_site/`, y despliega a GitHub Pages |

La autenticación a GCP usa **Workload Identity Federation**
(`google-github-actions/auth@v2` con `WIF_POOL_ID`/`WIF_PROVIDER_ID` y
`GCP_SERVICE_ACCOUNT`), sin claves JSON versionadas.

Límites del CI actual:

- `pr_checks.yaml` y `staging.yaml` filtran por `poetry.lock`, aunque el proyecto usa `uv.lock`; cambios exclusivos en
  `pyproject.toml` o `uv.lock` no disparan esos workflows.
- `staging.yaml` llama siempre a `deploy-to-prod.yaml` después del load test. La aprobación sólo existe si el entorno
  `production` tiene *required reviewers* configurados en GitHub; `workflow_dispatch` no es la única vía de producción.
- No hay jobs para construir y desplegar `Dockerfile.search-service` ni `copilotkit-proxy/Dockerfile`; esos despliegues
  siguen siendo operaciones separadas.

## Despliegue del Agente (Agent Runtime)

- Configuración declarativa: `agents-cli-manifest.yaml`
  (`deployment_target: agent_runtime`, `agent_directory: src`).
- El estado del último despliegue queda en `deployment_metadata.json`
  (incluye `remote_agent_runtime_id`, `is_a2a: true` y timestamp).
- Comando manual (requiere aprobación humana, ver `AGENTS.md`):

```bash
agents-cli deploy  # o: uvx google-agents-cli@1.1.0 deploy
```

- Variables requeridas: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`
  (p. ej. `us-east1`), `LOGS_BUCKET_NAME`, y en runtime `SEARCH_SERVICE_URL`
  para delegar el retrieval.

## Dataset: releases en GCS

`scripts/publish_dataset_release.py` publica el dataset procesado como release
versionado:

```text
gs://<bucket>/releases/<fecha>/
├── indices/            # text_index.faiss, audio_index.faiss, yamnet_inverted_index.json
├── final/              # complete_dataset.pkl, dataset_manifest.json, ...
└── segment_clips/      # segment_{id}.opus (≈14 MB por hora de corpus)
```

- El runtime lo descarga al arrancar con `DATASET_GCS_URI` + `DATASET_PATH`
  (`src/dataset_storage.py`): staging + reemplazo atómico, y salta la descarga
  si el snapshot local ya es el actual.
- `_download_prefix` omite `segment_clips/`: los clips se resuelven on-demand
  con **URLs firmadas V4** (TTL `SEGMENT_CLIP_URL_TTL_SECONDS`, default 900 s)
  desde `src/segment_clip_storage.py`. Requiere
  `roles/iam.serviceAccountTokenCreator` sobre la propia SA del servicio.
- Retención de releases viejas: política de lifecycle pendiente de definir
  (ver decisiones abiertas en spec 08 §12).

### Catálogo multi-dataset

El search service admite varios releases sin duplicar servicios. `DATASET_CATALOG_GCS_URI` apunta a un JSON en GCS:

```json
{
  "version": 1,
  "datasets": [
    {
      "id": "news-2026",
      "label": "Noticias 2026",
      "language": "es",
      "release_uri": "gs://<bucket>/releases/news-2026",
      "enabled": true
    }
  ]
}
```

`DEFAULT_DATASET_ID` selecciona el corpus inicial y `DATASET_CACHE_MAX_ENTRIES`
(default 2) limita los motores simultáneamente cargados. El dataset inicial se
calienta durante el startup; los restantes se descargan y cargan bajo demanda en
`/tmp/audio-search-datasets/`, con expulsión LRU. Los encoders de texto y CLAP se
comparten entre motores. El service account del search service necesita lectura
tanto del catálogo como de todos los releases referenciados.

Estado actual del contrato: `/datasets` obtiene `active_indexes` y `dataset_version` del engine en memoria. Por eso sólo
el dataset calentado tiene esos datos; las demás entradas salen con lista vacía y versión nula hasta su primera carga.
Como el frontend deshabilita las fuentes según esa lista, hoy no puede iniciar la primera búsqueda de un dataset frío.
El catálogo debe declarar metadatos de índices o el endpoint debe leer el manifiesto antes de completar esta función.

Aunque los comentarios de `src/search_service/app.py` describen CLAP como perezoso, `DatasetRegistry._load_engine()`
accede a `engine.clap_model` para guardarlo como modelo compartido. Ese acceso instancia CLAP durante el warm-up del
dataset por defecto. La operación y el dimensionamiento de memoria deben asumir carga de CLAP en startup hasta corregir
esa inicialización.

Si `DATASET_CATALOG_GCS_URI` no está definido, sigue vigente el modo de un solo
release con `DATASET_GCS_URI`/`DATASET_PATH`.

## Configuración de GCP

### Secrets y variables de GitHub

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| `WIF_POOL_ID`, `WIF_PROVIDER_ID` | secret | Workload Identity Federation |
| `GCP_SERVICE_ACCOUNT` | secret | SA que ejecuta los deploys |
| `GCP_PROJECT_NUMBER` | variable | Número del proyecto para el provider WIF |
| `CICD_PROJECT_ID` | variable | Proyecto donde corre la autenticación CI |
| `STAGING_PROJECT_ID` | variable | Proyecto GCP de staging |
| `REGION` | variable | Región de despliegue |
| `APP_SERVICE_ACCOUNT_STAGING` | variable | SA de runtime del agente en staging |
| `LOGS_BUCKET_NAME_STAGING` | variable | Bucket de artefactos y resultados de load tests |

### Service Accounts

1. **GitHub Actions SA** (WIF): permisos de deploy sobre Agent Runtime, Cloud
   Run, Artifact Registry y GCS.
2. **SA del agente (Agent Runtime)**: lectura del bucket de releases
   (`roles/storage.objectViewer`) y escritura de logs/trazas.
3. **SA del search service (Cloud Run)**: lectura del bucket de releases y
   `roles/iam.serviceAccountTokenCreator` sobre sí misma (firmar URLs V4).
4. **SA del proxy (Cloud Run)**: `roles/run.invoker` sobre el search service
   (llamadas con ID token) y acceso A2A al agente.

## Configuración Local (Desarrollo)

### Variables de Entorno (.env)

La referencia completa y comentada está en `.env.example`. Grupos principales:

```bash
# Agente
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
QUERY_LANGUAGE=es
AGENT_MODALITY=all            # text | audio | yamnet | text_clap | all

# Dataset
DATASET_PATH=./dataset
# Modo catálogo (producción con varios corpus)
# DATASET_CATALOG_GCS_URI=gs://<bucket>/catalogs/datasets.json
# DEFAULT_DATASET_ID=news-2026
DATASET_CACHE_MAX_ENTRIES=2
# Fallback de un solo corpus
# DATASET_GCS_URI=gs://<bucket>/releases/<fecha>
AUDIO_INPUT_DIR=./data/audio
DATASET_OUTPUT=./dataset

# Ingesta (autogeneran config/embeddings.toml; no editar el TOML a mano)
EMBEDDINGS_ACTIVE=text,clap
CLASSIFIERS_ACTIVE=yamnet

# Retrieval remoto (producción)
SEARCH_SERVICE_URL=https://...   # si está, el agente no carga ML
SEARCH_SERVICE_TIMEOUT=180
MIN_SEGMENT_DURATION_SECONDS=5

# Clips de reproducción
SEGMENT_CLIP_CONTEXT_SEC=5
SEGMENT_CLIP_BITRATE=32k
SEGMENT_CLIP_URL_TTL_SECONDS=900

# Evaluación
EVALUATION_FRAMEWORK=ragas       # ragas | deepeval
EVALUATION_DATASET_SOURCE=manual # manual | synthetic | hybrid

# GCP / Agent Runtime
GOOGLE_CLOUD_PROJECT=...
GOOGLE_CLOUD_LOCATION=us-east1
```

### Desarrollo Local

```bash
# Instalar dependencias bloqueadas
uv sync                # agregar --extra yamnet / --group eval según la tarea

# Servicio del agente (superficie de producción)
uv run uvicorn src.fast_api_app:app --reload --host 0.0.0.0 --port 8000

# Servicio local del agente (alternativo, desarrollo)
uv run uvicorn src.agent_service.main:app --reload --host 0.0.0.0 --port 8001

# Search service local (requiere grupo ml)
uv run uvicorn src.search_service.app:app --reload --host 0.0.0.0 --port 8080

# Proxy CopilotKit (widget + API de búsqueda directa)
cd copilotkit-proxy && npm install && npm run dev

# Front de búsqueda directa (Vite, proxy /api/* → localhost:8080)
cd frontend-direct-search && npm install && npm run dev

# Playground interactivo del agente
agents-cli playground
```

### Docker Local

```bash
# Search service (incluye CLAP; el build tarda por el checkpoint)
docker build -t audio-search-service -f Dockerfile.search-service .
docker run -p 8080:8080 --env-file .env audio-search-service

# Proxy
docker build -t copilotkit-proxy ./copilotkit-proxy
docker run -p 8080:8080 --env-file copilotkit-proxy/.env copilotkit-proxy
```

## Terraform (Opcional)

Disponible para gestión declarativa de infraestructura vía
`agents-cli infra single-project` (ver `deployment/`). No se ejecuta
automáticamente en CI/CD. Conflicto de recursos existentes (Error 409):
usar `terraform import` en lugar de reintentar la creación.

## Monitoreo y Observabilidad

### Health Checks

- **Agente**: `GET /health` → `healthy` cuando el runner está listo,
  `degraded` durante el arranque.
- **Search service**: `GET /health` (liveness), `GET /readyz` (readiness;
  falso hasta que termina el warm-up del dataset por defecto, sus índices y, en el código actual, ambos encoders) y
  `GET /datasets` (catálogo público sin URIs privadas), para que Cloud Run
  no enrute tráfico a una instancia fría).

### Logging y trazas

- **Cloud Logging**: logging estructurado (incluye `/feedback`).
- **OpenTelemetry → Cloud Trace**: `src/app_utils/telemetry.py`
  (`setup_telemetry` + `setup_agent_engine_telemetry`) instrumenta el agente en
  Agent Engine.
- **agents-cli eval**: trazas ADK y grading de grounding/tools/respuesta
  (ver spec 04).

### Métricas de Runtime

- Request latency (p50, p95, p99) y cold start del search service.
- Error rate (4xx, 5xx) por servicio.
- Instance count y memoria/CPU (Cloud Run), sesiones activas (Agent Runtime).
- Resultados de load tests en `gs://<logs-bucket>/load-test-results/`.

## Criterios de Aceptación

1. Un pull request o push a `main` que coincida con los filtros corre, respectivamente, checks o deploy de staging con
   load test
2. El deploy a producción requiere aprobación mediante la protección del entorno `production`, incluso cuando lo
   invoca automáticamente el workflow de staging
3. Health checks (`/health`, `/readyz`) pasan después de cada deploy
4. El agente responde en < 5s (p95) una vez caliente el search service
5. El search service escala a cero sin tráfico y su cold start queda cubierto por `SEARCH_SERVICE_TIMEOUT`
6. Los frontends se publican en GitHub Pages al modificar `docs/`, `copilotkit-proxy/widget/`,
   `frontend-direct-search/` o el propio workflow
7. Secrets no expuestos en código ni en logs (WIF sin claves JSON)
8. La imagen del agente no contiene el grupo `ml` (descargable por Agent Runtime)
9. Los clips Opus nunca se descargan al runtime: sólo URLs firmadas con TTL acotado
10. Todos los datasets habilitados informan sus índices antes de ser seleccionados y se pueden cargar desde la UI
