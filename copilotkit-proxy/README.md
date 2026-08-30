# Proxy CopilotKit para búsqueda de audio

Este servicio Cloud Run convierte AG-UI (CopilotKit) a A2A (Agent Runtime) y
expone la API REST de la búsqueda directa bajo `/api`.
La URL `A2A_AGENT_URL` debe terminar en `/api/a2a/audio-search-journalists`.

## Desarrollo

```bash
cp .env.example .env
npm install
npm run dev
```

La service account de Cloud Run debe tener `roles/aiplatform.user` en el
proyecto que contiene el Agent Runtime. Para producción, restringí
`ALLOWED_ORIGINS` al dominio que sirve el widget.

Para búsqueda directa configurá también `SEARCH_SERVICE_URL` con la URL de
`audio-search-service`. El proxy obtiene un ID token de su service account para
invocarlo: el navegador nunca llama al servicio de índices privado. Las rutas
disponibles son `POST /api/search/plan`, `POST /api/search`,
`GET /api/segments/:id` y `GET /api/corpus`.

## Compilar y desplegar

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/ARTIFACT_REPOSITORY/audio-search-copilotkit-proxy:latest .
gcloud run deploy audio-search-copilotkit-proxy \
  --image REGION-docker.pkg.dev/PROJECT/ARTIFACT_REPOSITORY/audio-search-copilotkit-proxy:latest \
  --region REGION --service-account PROXY_SERVICE_ACCOUNT \
  --update-env-vars "A2A_AGENT_URL=AGENT_A2A_URL,SEARCH_SERVICE_URL=SEARCH_SERVICE_URL,ALLOWED_ORIGINS=https://widget.example.com"
```

Construí el widget con:

```bash
cd widget
npm install
VITE_COPILOT_RUNTIME_URL=https://PROXY_URL/api/copilotkit npm run build:widget
```

Publicá `widget/dist-widget/` en un CDN o bucket estático e incluí
`<script type="module" src="https://CDN/audio-search-widget.js"></script>`.

El despliegue en GCP requiere aprobación explícita y no se ejecuta desde este
repositorio automáticamente.
