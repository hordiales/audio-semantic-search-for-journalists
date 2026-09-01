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
`GET /api/segments/:id`, `GET /api/corpus` y `POST /api/feedback`.

## Feedback del widget

Los controles de pulgar arriba/abajo de CopilotKit envían al proxy el mensaje,
la pregunta asociada y el voto. El proxy los escribe en Cloud Logging bajo
`audio_search_journalists_feedback`; los sinks de Terraform los exportan a la
tabla BigQuery particionada
`audio_search_journalists_telemetry.audio_search_journalists_feedback`.

La service account del proxy necesita `roles/logging.logWriter` en el proyecto:

```bash
gcloud projects add-iam-policy-binding PROJECT \
  --member="serviceAccount:PROXY_SERVICE_ACCOUNT" \
  --role="roles/logging.logWriter"
```

La creación del sink y la autorización de su identidad de escritura se aplican
con el Terraform de `deployment/terraform/`; desplegar sólo Cloud Run no crea
la tabla de BigQuery.

Con `rewrite: true`, el proxy solicita por A2A al mismo agente que usa el
widget un plan JSON antes de consultar los índices. El plan puede reformular
las consultas, pero no agregar ni quitar índices elegidos por el cliente. Texto
es el default para clientes compatibles, pero se puede desactivar con
`include_text: false`; CLAP se agrega con `include_clap: true` y la búsqueda por
clases AudioSet/YAMNet con `include_yamnet: true`. Las tres fuentes son
independientes y debe haber al menos una activa. El search service traduce ambas
descripciones acústicas al
inglés y devuelve las queries efectivas usadas. Si A2A no está
disponible o devuelve JSON inválido, la solicitud falla: con `rewrite: true`
no se permite degradar silenciosamente a una búsqueda literal. Para pedir una
búsqueda literal de forma deliberada, enviar `rewrite: false`.

Ejemplo de búsqueda textual con CLAP incluido:

```json
{
  "query": "aplausos al terminar el discurso",
  "include_text": true,
  "include_clap": true,
  "include_yamnet": true,
  "k": 10,
  "rewrite": true
}
```

Si `include_text` se omite, texto se mantiene activo por compatibilidad; si vale
`false`, `/api/search` no consulta transcripciones. Si `include_clap` se omite o vale `false`, `/api/search` no consulta
`audio_index.faiss`. Si `include_yamnet` se omite o vale `false`, no busca sobre
las etiquetas almacenadas en `yamnet_top_classes`. YAMNet no es un índice
vectorial: rankea coincidencias de clases usando sus scores del clasificador.
El campo legado `indexes` continúa disponible para clientes
existentes y planes persistidos.

La service account del proxy necesita `roles/run.invoker` sobre el search
service. Otorgalo antes del deploy:

```bash
gcloud run services add-iam-policy-binding audio-search-service \
  --region REGION --project PROJECT \
  --member="serviceAccount:PROXY_SERVICE_ACCOUNT" \
  --role="roles/run.invoker"
```

## Compilar y desplegar

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/ARTIFACT_REPOSITORY/audio-search-copilotkit-proxy:latest .
gcloud run deploy audio-search-copilotkit-proxy \
  --image REGION-docker.pkg.dev/PROJECT/ARTIFACT_REPOSITORY/audio-search-copilotkit-proxy:latest \
  --region REGION --service-account PROXY_SERVICE_ACCOUNT \
  --update-env-vars "A2A_AGENT_URL=AGENT_A2A_URL,SEARCH_SERVICE_URL=SEARCH_SERVICE_URL,ALLOWED_ORIGINS=https://widget.example.com"
```

`SEARCH_SERVICE_URL` es la URL base de Cloud Run (por ejemplo,
`https://audio-search-service-XXXX.a.run.app`). Si el proxy ya está desplegado,
agregala sin perder las otras variables:

```bash
gcloud run services update audio-search-copilotkit-proxy \
  --region REGION --project PROJECT \
  --update-env-vars "SEARCH_SERVICE_URL=https://audio-search-service-XXXX.a.run.app"
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
