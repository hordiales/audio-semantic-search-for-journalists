# Despliegue en Google Cloud Platform (GCP)

Este directorio agrupa la configuración y referencias para el despliegue sobre **Google Cloud Platform**:

1. **Cloud Storage**: almacena las releases inmutables del dataset (`final/`, `indices/` FAISS y `segment_clips/` `.opus`).
2. **Cloud Run (`audio-search-service`)**: servicio pesado (`--group ml`) con PyTorch CPU, MiniLM, CLAP y FAISS. Descarga `final/` e `indices/` a `/tmp` al iniciar y firma URLs V4 para los clips `.opus`.
3. **Vertex AI Agent Runtime (`audio-search-journalists`)**: ejecuta el agente Google ADK (`src/agent_service/agent.py`) en una imagen liviana (`--no-group ml`) y delega la búsqueda a Cloud Run vía HTTP + IAM ID Token.
4. **Cloud Run (`copilotkit-proxy`)**: traduce AG-UI a A2A y expone la API de búsqueda directa (`/api/search`, `/api/datasets`, `/api/corpus`, `/api/segments/:id`).

---

## Archivos en `deploy/gcp/`

- `deploy.env.example`: plantilla de variables de entorno para la sesión de despliegue en GCP.
- `Dockerfile.agent`: copia de referencia del `Dockerfile` raíz (usado por `agents-cli deploy` en la raíz del repo).
- `Dockerfile.search-service`: imagen del servicio de búsqueda para Cloud Run.
- `cloudbuild.search-service.yaml`: configuración de Cloud Build para construir `audio-search-service`.

> **Nota de compatibilidad:** `agents-cli deploy` requiere que `Dockerfile` y `agents-cli-manifest.yaml` existan en la raíz del repositorio. Esos archivos se mantienen en la raíz y los de `deploy/gcp/` sirven como referencia organizada y pueden invocarse explícitamente con `--config deploy/gcp/cloudbuild.search-service.yaml`.

---

## Flujo rápido de despliegue en GCP

Desde la **raíz del repositorio**:

```bash
# 1. Configurar variables de entorno
cp deploy/gcp/deploy.env.example deploy.env
set -a && source deploy.env && set +a
gcloud config set project "$PROJECT_ID"

# 2. Publicar release del dataset en Cloud Storage
uv run python scripts/publish_dataset_release.py \
  --dataset ./dataset \
  --bucket "$DATASET_BUCKET" \
  --release "$DATASET_RELEASE"

# 3. Construir y desplegar el search service en Cloud Run
export SEARCH_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${SEARCH_SERVICE}:latest"
gcloud builds submit --config=deploy/gcp/cloudbuild.search-service.yaml \
  --substitutions="_IMAGE=$SEARCH_IMAGE" .

gcloud run deploy "$SEARCH_SERVICE" \
  --image="$SEARCH_IMAGE" \
  --region="$REGION" --project="$PROJECT_ID" \
  --service-account="$SEARCH_SERVICE_ACCOUNT" \
  --no-allow-unauthenticated \
  --memory=4Gi --cpu=2 --concurrency=4 \
  --min-instances=0 --max-instances=2 \
  --set-secrets="OPENAI_API_KEY=${OPENAI_SECRET}:latest" \
  --update-env-vars="DATASET_GCS_URI=${DATASET_GCS_URI},DATASET_PATH=${DATASET_RUNTIME_PATH},CLAP_DEVICE=cpu,QUERY_LANGUAGE=${QUERY_LANGUAGE:-es},MIN_SEGMENT_DURATION_SECONDS=${MIN_SEGMENT_DURATION_SECONDS:-5}"

# 4. Desplegar el agente ADK en Vertex AI Agent Runtime
export SEARCH_SERVICE_URL="$(gcloud run services describe "$SEARCH_SERVICE" --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)')"
agents-cli deploy \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --service-account="$AGENT_SERVICE_ACCOUNT" \
  --secrets="OPENAI_API_KEY=${OPENAI_SECRET}" \
  --update-env-vars="SEARCH_SERVICE_URL=${SEARCH_SERVICE_URL},SEARCH_SERVICE_AUTH=iam,SEARCH_SERVICE_TIMEOUT=180"

# 5. Desplegar el proxy CopilotKit / Direct Search en Cloud Run
cd copilotkit-proxy
export PROXY_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${PROXY_SERVICE}:latest"
gcloud builds submit --tag "$PROXY_IMAGE" .
gcloud run deploy "$PROXY_SERVICE" \
  --image="$PROXY_IMAGE" \
  --region="$REGION" --project="$PROJECT_ID" \
  --service-account="$PROXY_SERVICE_ACCOUNT" \
  --allow-unauthenticated \
  --update-env-vars="A2A_AGENT_URL=${A2A_AGENT_URL},SEARCH_SERVICE_URL=${SEARCH_SERVICE_URL},ALLOWED_ORIGINS=${WIDGET_ORIGIN}"
```

Para detalles completos de permisos IAM, Secret Manager y solución de problemas, consultá `DOC-interna/ARCHITECTURE-CLOUD.md` y `DOC-interna/INSTALL-AGENT-WIDGET.md`.
