# Frontend de búsqueda en Vercel (`web/`)

Documento de implementación del frontend de búsqueda para el despliegue **Vercel + Neon (pgvector) + Eve Agents**. Es una adaptación del `frontend-direct-search/` existente: mismo contrato de API, misma experiencia, pero orientado a las rutas serverless de este directorio y a los clips públicos de **Vercel Blob**.

## Estado de despliegue

**Desplegado en producción: https://audio-semantic-search.vercel.app** (proyecto Vercel `audio-semantic-search`, equipo `hordiales-projects`, región iad1 — deploy via `vercel deploy --prod` con build en la nube para Linux x64).

Verificado en producción (`vercel curl`, porque el proyecto tiene Deployment Protection):

```text
GET /              → 200 (SPA)
GET /api/health    → 200  {"status":"ok","target":"vercel-neon-eve"}
GET /eve/v1/health → 200  {"ok":true,"status":"ready","workflowId":"workflow//eve//workflowEntry"}
GET /api/datasets  → 502  (esperado: DATABASE_URL aún no configurada en el proyecto)
```

**Deployment Protection está ACTIVADA** (Vercel Authentication/SSO): cualquier visitante debe loguearse con la cuenta de Vercel. Para hacerlo público: Project Settings → Deployment Protection → desactivar "Vercel Authentication". Dejarla activa es válido para uso interno.

**Pendiente para que la búsqueda funcione end-to-end:**

1. Configurar env vars en el proyecto (Production): `DATABASE_URL`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_API_TOKEN` — ver `.env.example`.
2. Aplicar `db/schema.sql` en Neon.
3. Publicar el dataset: `scripts/publish_dataset_neon.py`.
4. Redeploy (o `vercel redeploy`) para que las funciones levanten las vars.

Para re-deployar cambios:

```bash
cd deploy/vercel
vercel deploy --prod        # build en la nube (recomendado: el bundle eve incluye nativos linux)
```

## Arquitectura

```text
Browser (SPA React en dist-web/)
   │
   ├──► POST /api/agent-chat   (drawer del asistente, mismas tools del Eve agent)
   ├──► POST /api/feedback
   ├──► POST /api/search       ┐
   ├──► POST /api/search/plan  │  Direct Search API
   ├──► GET  /api/datasets     │  (servicio "api" — server.ts Express)
   ├──► GET  /api/corpus       │
   ├──► GET  /api/segments/:id │
   ├──► GET  /api/health       ┘
   └──► /eve/v1/*              Eve Agent Runtime (servicio generado por withEve)
            │
            ▼
   Neon Postgres (pgvector) + Vercel Blob (.opus) + APIs de embedding (HF/Gemini/OpenAI)
```

`vercel.ts` compone el grafo con `withEve` — **3 servicios**:

- `web` (framework `vite`, root `.`): `npm run build:web` → `dist-web/`. Solo estáticos.
- `api` (framework `express`, entrypoint `server.ts`): los handlers de `api/` montados en `/api/*`, empaquetados como una sola Vercel Function. **Importante**: en modo `services`, Vercel NO construye el directorio `api/` como functions sueltas ("The api/ directory will not be built because services are configured") — por eso existe `server.ts`.
- `eve` (generado por `withEve`, root aislado `.eve/vercel-services/eve`): runtime completo del agente en `/eve/v1/*` (health, info, session, callbacks, workflows).

Rutas: `/eve/v1/*` → eve, `/api/*` → api, resto → web. Los handlers de `api/` usan paths nativos (sin rewrites):

| Ruta | Archivo |
|---|---|
| `POST /api/search` | `api/search.ts` |
| `POST /api/search/plan` | `api/search/plan.ts` |
| `POST /api/agent-chat` | `api/agent-chat.ts` |
| `GET /api/datasets` | `api/datasets.ts` |
| `GET /api/corpus` | `api/corpus.ts` |
| `GET /api/segments/:segmentId` | `api/segments/[segmentId].ts` |
| `POST /api/feedback` | `api/feedback.ts` |
| `GET /api/health` | `api/health.ts` |

## Archivos del frontend

| Archivo | Rol |
|---|---|
| `web/App.tsx` | UI completa: toggles de índice, datasets, plan editable, columnas por índice, filtros, player de audio, reproducción continua, atajos de teclado, estado en URL |
| `web/AgentDrawer.tsx` | Drawer flotante del asistente: chat contra `/api/agent-chat`, feedback 👍/👎 a `/api/feedback` |
| `web/api.ts` | Cliente fetch (`VITE_API_BASE_URL`, vacío = same-origin) |
| `web/types.ts` | Contrato `SearchRequest`/`SearchResponse`/`SearchPlan`/`IndexResults` |
| `web/i18n.ts` | Textos ES/EN, `detectLocale()` por URL/localStorage/navegador |
| `web/styles.css` | Sistema visual completo + estilos del drawer |
| `web/main.tsx` | Bootstrap React |
| `index.html` | Shell de la SPA (entrada `/web/main.tsx`) |
| `vite.config.ts` | `base: "/"`, outDir `dist-web`, proxy local `/api` y `/eve` → `localhost:3000` (solo en `npm run dev:web`; desactivado bajo `VERCEL=1` porque en `vercel dev` las funciones van al servicio `api`) |
| `server.ts` | Entrypoint Express del servicio `api`: monta los handlers de `api/` en rutas `/api/*`; `app.listen` solo fuera de Vercel |
| `tsconfig.web.json` | Typecheck separado del frontend (DOM, `jsx: react-jsx`, bundler resolution) |

## Comportamiento preservado del frontend original

- Índice de **texto activo por defecto**; CLAP y YAMNet desactivados; al menos un índice debe quedar seleccionado.
- Resultados agrupados por índice (scores nunca se mezclan), con `effective_query` y `translated_query` visibles.
- **Plan de búsqueda** visible, copiable y editable; editar descripciones re-ejecuta el retrieval.
- **Playback**: usa `clip_start_time`/`clip_end_time` y timestamps del segmento; renovación de URLs expiradas repitiendo el plan guardado (`clip_expires_at`).
- Atajos `J`/`K`/`Espacio`; filtros por archivo y rango temporal; copiar cita; descargar `.opus`.
- **Diferencia clave**: los clips vienen de Vercel Blob (URL pública) — no se aplica la validación de URLs firmadas GCS del widget CopilotKit original.

## Eve Agent: dos puntos de entrada

1. **Drawer incluido** (`AgentDrawer` → `POST /api/agent-chat`): endpoint same-origin que ejecuta las mismas cinco tools del agente (`buscar_audio`, `buscar_evento_acustico`, `buscar_clase_audio`, `obtener_info_segmento`, `obtener_clases_audio`) con `generateText` del AI SDK y `stopWhen: stepCountIs(5)`. Funciona sin canal Eve completo.
2. **Canal Eve real** (`/eve/v1/*`): `agent/channels/eve.ts` monta `eveChannel` con auth `vercelOidc()` + `localDev()` + bearer opcional (`EVE_PUBLIC_BEARER_TOKEN`) + fallback anónimo `none()`. Para usarlo desde el frontend se puede reemplazar el drawer por `useEveAgent()` de `eve/react` sin tocar el backend.

## Notas de configuración

- `agent/sandbox.ts` fija `JustBashSandbox` (sandbox JS puro): las tools solo consultan Neon, no necesitan contenedor. Evita la dependencia de microsandbox/Docker en local y de Vercel Sandbox en producción.
- Peer deps de `eve@0.80`: `ai@^7`, `@ai-sdk/openai@^4`, `zod@^3.25`.
- Scripts: `dev` (eve dev), `dev:web` (vite), `build` (eve build + tsc), `build:web`, `typecheck`, `test`.
- Variables: ver `.env.example` (comentado por secciones: obligatorias, providers de embedding, opcionales).
- **CORS**: `src/cors.ts` implementa `withCors` en todos los handlers de `api/` respetando `ALLOWED_ORIGINS` (lista separada por comas o `*`, con preflight `OPTIONS`). Necesario solo si el frontend vive en otro origen.
- CLAP/PyTorch **no** está en el bundle del frontend ni en las Functions — los embeddings de consulta se generan por API (ver `plan.md` y `src/embeddings.ts`).

## Verificación

```text
npm run typecheck   → tsc --noEmit (backend) + tsc -p tsconfig.web.json (web)  ✓
npm test            → 16 tests (search logic + CORS)  ✓
npm run build:web   → vite build: dist-web/ (≈174 kB JS / 56 kB gzip)  ✓
npm run build       → eve build (bundle del agente) + tsc --noEmit  ✓
vercel build        → services/web estáticos + services/api index.func + services/eve protocolo completo  ✓
vercel dev --local  → / 200, /api/health 200, /eve/v1/health 200, /eve/v1/info 200  ✓
vercel deploy --prod → https://audio-semantic-search.vercel.app (3 servicios vivos)  ✓
```

Pendiente: configurar env vars de producción, provisionar Neon + Blob y publicar el dataset con `scripts/publish_dataset_neon.py` (ver "Estado de despliegue"). Opcionalmente migrar el drawer a `useEveAgent()` para streaming/sesiones durables del canal Eve.
