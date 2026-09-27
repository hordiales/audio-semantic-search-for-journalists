# 08 — Product Definition: Front de Búsqueda Directa con Escucha de Segmentos

## 1. Problema

El problema que motivó esta interfaz era que la única superficie disponible era conversacional (agente ADK + widget
CopilotKit) y devolvía resultados como texto plano. La búsqueda directa descrita en este documento ya está implementada
y convive con el chat:

```text
Segmento 1
Texto: "Exacto. Perfecto. Lindo. Gracias..."
Archivo: Ernesto Tenembaum sobre el vínculo entre... [68HlGgpKLRA].wav
Timestamp: 50.0 - 80.0
Índice: Índice de audio (CLAP)
Similitud: 13.9%
```

Esto falla para el caso de uso central del periodista:

1. **No se puede escuchar.** El resultado más importante de una búsqueda acústica ("¿hay música de fondo?", "¿hay
   aplausos?") sólo se valida escuchando. Un timestamp en texto obliga a abrir el archivo en otra herramienta y buscar
   el minuto a mano.
2. **El razonamiento del LLM hoy es invisible y está acoplado a la presentación.** El LLM puede aportar valor al
   reformular la consulta, distinguir contenido hablado de eventos acústicos y producir una consulta efectiva más
   adecuada para cada índice. El problema es que actualmente también selecciona, resume y presenta los resultados, por
   lo que puede omitir evidencia y vuelve difícil reproducir la búsqueda.
3. **No hay control ni trazabilidad del plan de búsqueda.** El periodista no puede fijar contra qué índice buscar ni
   inspeccionar la reformulación utilizada. Para trabajo de archivo se necesita control explícito, mostrar las queries
   efectivas y ejecutar el retrieval de forma determinista.
4. **La "similitud" se presenta de forma engañosa.** "13.9%" sugiere baja confianza, cuando en CLAP los scores coseno
   del top-1 suelen estar en ese rango. El número absoluto no es interpretable por el usuario.

## 2. Solución implementada

Una **aplicación de búsqueda estructurada** (no chat): un buscador tipo "consola de archivo" que expone tres fuentes de
recuperación de forma explícita y hace de la escucha del segmento la acción primaria, a un clic del resultado: texto, el
espacio vectorial CLAP y las clases AudioSet/YAMNet.

El LLM se conserva como una capa de **interpretación y query rewrite**, no como renderer ni filtro de resultados. A
partir de la consulta original propone un plan inspeccionable: índice o índices, query efectiva para texto, descripción
del evento acústico para CLAP y traducción al inglés cuando corresponda. `AudioSearchEngine` ejecuta ese plan y el
frontend muestra el top-K completo sin síntesis intermedia.

El usuario puede editar el plan, activar o desactivar texto, CLAP y YAMNet independientemente, o desactivar la
reformulación con un modo de búsqueda literal. Texto empieza activado y CLAP/YAMNet desactivados por defecto. El front
conversacional existente **no se elimina**: convive como modo alternativo para consultas exploratorias.

## 3. Usuarios y casos de uso

**Usuario primario:** periodista de investigación / productor de archivo con horas de audio ya ingestadas.

| Caso de uso | Índice | Ejemplo |
| --- | --- | --- |
| Encontrar qué se dijo sobre un tema | Texto (transcripciones) | "vínculo entre Tech Security y el Banco Nación" |
| Encontrar un evento acústico no transcripto | Audio (CLAP) | "música de fondo", "aplausos", "gritos" |
| Filtrar por una clase acústica estandarizada | Clases (YAMNet/AudioSet) | "Applause", "Speech", "Music" |
| Verificar y citar un momento exacto | Una o más fuentes | escuchar 50.0–80.0 y confirmar antes de publicar |
| Comparar cobertura | Texto + CLAP + YAMNet | ver si las tres señales coinciden en el mismo segmento |

## 4. Objetivos y no-objetivos

### Objetivos

- O1. Reproducir el audio de cualquier resultado en ≤ 1 clic, sin salir de la página.
- O2. Tres fuentes seleccionables de forma independiente; texto se activa por defecto y CLAP/YAMNet son opt-ins.
- O3. Aprovechar el LLM para interpretar intención y reformular queries sin permitirle ocultar, reordenar ni sintetizar
  los resultados recuperados.
- O4. Hacer visible y editable el plan de búsqueda: consulta original, índices y query efectiva enviada a cada encoder.
- O5. Resultados de retrieval reproducibles al volver a ejecutar un plan guardado con el mismo release y `k`.
- O6. Presentar la similitud de forma honesta e interpretable.
- O7. Que cada resultado sea citable: archivo, timestamp inicio/fin, índice, `segment_id`.

### No-objetivos (v1)

- No hay chat ni memoria de conversación en el buscador; el LLM sólo genera un plan de búsqueda estructurado y no
  sintetiza los resultados.
- No hay edición/recorte de audio ni exportación de clips editados.
- No hay autenticación multiusuario ni colaboración (se hereda la del entorno de deploy).
- No hay re-indexación ni carga de audio desde la UI.

## 5. Datos disponibles hoy

Fuente: `AudioSearchEngine._row_to_segment_dict` (<ref_file
file="src/agent_service/search_engine.py" />) y
`_serialize_results` (<ref_file
file="src/agent_service/tools.py" />).

| Campo | Tipo | Notas |
| --- | --- | --- |
| `segment_id` | int | Identificador estable dentro del release del dataset |
| `text` | str | Transcripción Whisper del segmento |
| `start_time` / `end_time` | float (s) | Offset dentro del archivo original |
| `original_file_name` | str | Nombre del `.wav` convertido (16 kHz mono) |
| `language`, `confidence` | str / float | Metadatos de transcripción |
| `similarity` | float | Coseno (FAISS `IndexFlatIP` sobre vectores normalizados) |
| `sentiment_*`, `dominant_sentiment` | float / str | Sólo vía `get_segment_info` |
| `yamnet_top_classes` | list | Etiquetas AudioSet en inglés + score. Vacío si el dataset se generó sin `CLASSIFIERS_ACTIVE=yamnet` |

Dos índices FAISS en `dataset/indices/` : `text_index.faiss` (MiniLM, 384d) y `audio_index.faiss` (CLAP, 512d). YAMNet
no crea un tercer índice FAISS ni un espacio vectorial: guarda el top de clases y scores en `yamnet_top_classes` , que
se usa para filtrar y rankear segmentos. El manifiesto `dataset/final/dataset_manifest.json` declara por separado
`active_embeddings` y `active_classifiers` ; una fuente puede no estar disponible.

## 6. Disponibilidad del audio — RESUELTO (Fase 0 implementada)

El problema original: `dataset/audio_segments/segment_*_window_*.wav` son artefactos transitorios que el pipeline
elimina al finalizar, y los WAV completos de `converted/` nunca se publicaban en el release. **El sistema no podía
reproducir nada.**

Solución implementada: el pipeline genera **un clip Opus por segmento** en `{DATASET_OUTPUT}/segment_clips/` , que se
publica en el release y se sirve **on-demand con URLs firmadas**.

| Pieza | Dónde |
| --- | --- |
| Generación del clip (bounds + encoder) | <ref_file file="src/segment_clips.py" /> |
| Etapa 6 del pipeline, incremental y con limpieza de huérfanos | <ref_file file="src/simple_dataset_pipeline.py" /> |
| Resolución on-demand y firma V4 | <ref_file file="src/segment_clip_storage.py" /> |
| Publicación del release | <ref_file file="scripts/publish_dataset_release.py" /> |

Decisiones y por qué:

- **Un clip por segmento, no las ventanas de CLAP.** Las ventanas se solapan 2 s por diseño (para que el embedding no
  pierda eventos en los bordes): reproducidas en secuencia repiten audio y suenan entrecortadas. El clip de reproducción
  es un único archivo por `segment_id` .
- **Con contexto configurable** (`--clip-context`, default ±5 s). El clip cubre `[start-ctx, end+ctx]` recortado a los
  límites del archivo, así que el periodista puede oír cómo entra y cómo sigue el fragmento. Por eso el dataset expone
  dos pares de timestamps: `start_time` /`end_time` (el segmento, lo que se cita) y `clip_start_time` /`clip_end_time`
  (lo que contiene el archivo de audio). El player arranca en el segmento y deja el contexto disponible hacia atrás.
- **Opus a 32 kbps mono.** Un fragmento de 30 s pesa ~120 KB contra ~1 MB del WAV 16 kHz: ≈14 MB por hora de corpus.
  Decodifica nativo en todo navegador actual.
- **No se descarga al runtime.** `_download_prefix` omite `segment_clips/` explícitamente, de modo que el arranque en
  frío sigue escalando con el tamaño del índice y no con las horas de audio.
- **Signed URLs V4, TTL 15 min.** El bucket queda privado y los bytes van del bucket al navegador sin pasar por el
  contenedor del agente. Requiere `roles/iam.serviceAccountTokenCreator` sobre la propia SA; las URLs se cachean en
  proceso para no gastar una llamada a `signBlob` por cada resultado.
- **Sin dependencia del nombre de archivo original.** Los clips se direccionan por `segment_id` (`segment_42.opus`), lo
  que evita todo el problema de espacios, acentos y corchetes de los `original_file_name` , y elimina la superficie de
  path traversal.

Consecuencia operativa: **los datasets ya publicados no tienen clips.** Hay que reprocesar la ingesta y publicar un
release nuevo. La ingesta es incremental, pero agregar clips cambia la firma del pipeline, así que la primera corrida
reprocesa todo el corpus.

## 7. API requerida

El backend actual (<ref_file file="src/fast_api_app.py"
/>) sólo expone `/query` , `/evaluate/query` , `/feedback` y las rutas ADK/A2A. Todas pasan por un flujo conversacional.
Se agrega una API estructurada en dos pasos: el LLM construye un plan de búsqueda tipado e inspeccionable y
`AudioSearchEngine` ejecuta ese plan sin dejar que el LLM filtre o reordene resultados.

> **Estado de implementación (2026-08-30): IMPLEMENTADO.** La API vive en el
> proxy `copilotkit-proxy` (`copilotkit-proxy/src/server.ts`: `GET /api/datasets`,
> `POST /api/search/plan`, `POST /api/search`, `GET /api/segments/:segmentId`, `GET /api/corpus`):
>
> - El planner (`copilotkit-proxy/src/a2a-planner.ts`) pide el `SearchPlan` al
>   agente por A2A usando el modo interno `DIRECT_SEARCH_PLANNER_V1` del system
>   prompt, y respeta siempre los índices elegidos por el usuario.
> - El retrieval corre contra el search service de Cloud Run
>   (`src/search_service/app.py`: `/datasets`, `/search/semantic`, `/search/audio`,
>   `/search/yamnet`, `/segments/{id}`, `/corpus`) con autenticación IAM (ID
>   token); el search service conserva IAM y sólo el proxy lo invoca.
> - El front (`frontend-direct-search/`, desplegado en GitHub Pages) apunta con
>   `VITE_API_BASE_URL` al proxy, nunca directo al search service. En
>   desarrollo, Vite redirige `/api/*` a `http://localhost:8080`.

```text
GET  /api/datasets
  resp: { datasets: [{ id, label, language, active_indexes, dataset_version? }] }

POST /api/search/plan
  body: { query: str, indexes?: ["text"|"audio"|"yamnet"] }
  resp: {
    original_query,
    indexes,
    text_query?: str,
    audio_query?: str,
    audio_query_en?: str,
    yamnet_query?: str,
    yamnet_query_en?: str,
    rationale?: str
  }

POST /api/search
  body: {
    query: str,
    include_text: bool = true,
    include_clap: bool = false,
    include_yamnet: bool = false,
    k: int (1..50),
    rewrite: bool = true,
    dataset_id?: str,
    plan?: SearchPlan
  }
  resp: {
    query, dataset_id, dataset_label, dataset_release, plan, took_ms,
    indexes: {
      text:  { available: bool, effective_query: str, results: [SearchResult], error?: str },
      audio: { available: bool, effective_query: str, translated_query?: str, results: [SearchResult], error?: str },
      yamnet: { available: bool, effective_query: str, translated_query?: str, results: [SearchResult], error?: str }
    }
  }

SearchResult = {
  segment_id, text, start_time, end_time, duration,
  original_file_name, language, confidence,
  similarity, rank, search_index, search_index_label,
  clip_url,            # URL firmada del Opus, servida directo desde GCS
  clip_start_time,     # inicio del audio del clip (incluye el contexto)
  clip_end_time,
  clip_expires_at,
  yamnet_audio_classes?,   # todas las clases guardadas; en resultados de texto, CLAP y YAMNet
  yamnet_matched_classes?  # clases que justifican un resultado YAMNet
}

GET  /api/segments/{segment_id}                -> metadatos completos del dataset por defecto
GET  /api/corpus?dataset_id=...                  -> { dataset_id, total_segments, files, active_indexes, dataset_version }
```

La URL del clip sale de `SegmentClipStore.reference(segment_id)`. Con GCS es una URL V4 firmada. Sin bucket, el código
actual devuelve un URI `file://` al clip dentro de `DATASET_PATH`; ni el search service ni el proxy montan una ruta
estática para ese archivo, por lo que el frontend servido por HTTP no puede reproducirlo de forma portable. El modo
local necesita una ruta HTTP de clips o un bucket de desarrollo. Para GCS, el front debe renovar los resultados si el
usuario deja la página abierta más que `SEGMENT_CLIP_URL_TTL_SECONDS`.

Notas de diseño:

- Al montar, el frontend consulta `GET /api/datasets`, selecciona el dataset indicado por `?dataset=<id>` o el primero
  disponible y envía su `dataset_id` en cada búsqueda. El selector sólo habilita índices declarados en `active_indexes`.
  Hoy el endpoint devuelve esa lista vacía para todo engine aún no cargado, por lo que la UI no puede disparar su primera
  búsqueda. La selección multi-dataset está integrada de punta a punta, pero este bootstrap sigue pendiente.
- El search service carga el dataset elegido de forma diferida mediante `DatasetRegistry`; un ID desconocido produce
  404 allí. `/api/search` captura ese error por índice y devuelve una respuesta 200 con las fuentes no disponibles. La
  respuesta exitosa identifica el dataset y release efectivos para hacer reproducible el resultado.
- El search service acepta `dataset_id` al consultar segmentos. La ruta actual del proxy
  `GET /api/segments/:segmentId` no reenvía ese parámetro y resuelve el dataset por defecto; debe corregirse antes de usar
  la vista de detalle con otros corpus.
- Con `rewrite=true` , el backend obtiene primero un `SearchPlan` estructurado. El LLM puede convertir “¿se escucha una
  cortina mientras habla el entrevistado?” en una descripción acústica breve; el search service la traduce al inglés
  antes del encoder CLAP.
- `include_text` , `include_clap` e `include_yamnet` controlan las fuentes de forma independiente y al menos una debe
  quedar activa. Para compatibilidad, si clientes anteriores no envían `include_text` , texto queda activo. El planner
  no puede cambiar esa selección.
- El plan se devuelve siempre y puede reenviarse en `plan` para repetir exactamente el retrieval sin invocar nuevamente
  al LLM. Con `rewrite=false` , las queries efectivas se derivan literalmente de `query` .
- `/api/search` ejecuta las búsquedas seleccionadas **en paralelo** y degrada con gracia si una fuente no está activa en
  el manifiesto.
- Para el índice CLAP se devuelven tanto la descripción acústica generada como `translated_query` (la query final se
  traduce a inglés antes del encoder RoBERTa, ver `src/query_translation.py` ): el usuario debe poder ver qué se buscó
  realmente. Es la causa más común de resultados inesperados.
- Para YAMNet también se devuelve la traducción efectiva. El score mostrado no es coseno: es cobertura de términos
  AudioSet multiplicada por la confianza media de las clases coincidentes. Cada resultado expone esas coincidencias por
  separado.
- La respuesta del LLM se valida contra un schema cerrado; si el planner falla o excede su timeout, la API devuelve un
  error claro. El usuario puede desactivar `rewrite` para ejecutar una búsqueda literal deliberada.
- El diseño buscaba cargar CLAP de forma perezosa, pero `DatasetRegistry._load_engine()` accede hoy a
  `engine.clap_model` al registrar el modelo compartido. Ese acceso carga CLAP durante el warm-up del dataset por defecto.

## 8. UX

### 8.1 Layout

```text
┌───────────────────────────────────────────────────────────────────────┐
│  Dataset: [ Noticias 2026 ▼ ]                                      │
│  [ ¿hay música mientras habla el entrevistado?       ]  [ Buscar ]    │
│  [✓] Texto  [ ] CLAP  [ ] YAMNet         [✓] Reformular con IA │
│  Plan: Texto "música" · CLAP "música de fondo" · YAMNet "música" │
│  CLAP buscó en inglés: "background music under speech"                │
├──────────────────────────────┬────────────────────────────────────────┤
│ ÍNDICE DE TEXTO   3 result.  │ ÍNDICE DE AUDIO (CLAP)   10 result.    │
│                              │                                        │
│ ┌──────────────────────────┐ │ ┌────────────────────────────────────┐ │
│ │ #1  ▓▓▓▓▓▓▓▓░░  0.412    │ │ │ #1  ▓▓▓▓▓▓▓▓░░  0.139              │ │
│ │ Tenembaum... [68Hl].wav  │ │ │ Tenembaum... [68Hl].wav            │ │
│ │ 00:50 → 01:20  (30s)     │ │ │ 00:50 → 01:20  (30s)               │ │
│ │ ▶ ──────●────────  0:12  │ │ │ ▶ ──────●────────  0:12            │ │
│ │ "Exacto. Perfecto..."    │ │ │ "Exacto. Perfecto. Lindo..."       │ │
│ │ [Copiar cita] [Detalle]  │ │ │ music 0.72 · speech 0.55  (YAMNet) │ │
│ └──────────────────────────┘ │ └────────────────────────────────────┘ │
└──────────────────────────────┴────────────────────────────────────────┘
```

Cada fuente activa aparece como una sección vertical, plegable y con color propio; texto puede desactivarse igual que
CLAP o YAMNet. Antes de los resultados se muestra el plan efectivo en una franja compacta y editable. El toggle
"Reformular con IA" permite comparar el plan interpretado con una búsqueda literal sin cambiar de interfaz.

### 8.2 Reproducción (el corazón del producto)

- **Player inline por resultado.** Cada tarjeta tiene su propio botón play + barra de progreso, acotada a
  `[start_time, end_time]` . Un solo audio suena a la vez: iniciar uno pausa el anterior.
- **Barra espaciadora** reproduce/pausa el resultado enfocado; `J` /`K` navegan entre resultados. Recorrer 10 resultados
  acústicos debe tomar segundos, no minutos.
- **Contexto expandible.** Botón "±10s" que extiende la ventana de reproducción más allá del segmento: imprescindible
  para juzgar si la música es cortina, fondo o corte.
- **Reproducción continua opcional** ("reproducir todos"): encadena los resultados del ranking, para triage rápido de
  una búsqueda acústica.
- **Precarga** del audio del primer resultado al recibir la respuesta.
- **Descarga del clip** del segmento (para llevarlo a la edición) y **copia de cita** en texto:
  `archivo · 00:50–01:20 · segment_id 42 · índice CLAP` .

### 8.3 Presentación honesta de la similitud

Regla de producto: **nunca mostrar el coseno crudo como porcentaje de confianza.**

- Mostrar el **rank** (#1, #2, …) como señal primaria.
- Mostrar una barra **normalizada dentro del ranking de ese índice** (relativa al top-1), más el valor numérico crudo
  (`0.139`) en tamaño reducido, con tooltip: *"Similitud coseno. Los valores de CLAP y de texto no son comparables entre
  sí."*
- **Nunca mezclar ni promediar** scores de las tres fuentes en un ranking único: texto y CLAP son espacios vectoriales
  distintos; YAMNet usa otro score de ranking.
- Si el top-1 de un índice cae por debajo del umbral opcional de
  `VITE_WEAK_TEXT_SIMILARITY`, `VITE_WEAK_AUDIO_SIMILARITY` o `VITE_WEAK_YAMNET_SCORE`, se muestra un aviso de
  coincidencias débiles. Además, cada tarjeta por debajo de `VITE_LOW_SCORE_WARNING` (default `0.10`) marca score, barra
  y borde en rojo. Los umbrales por índice quedan sin valor por defecto hasta calibrarlos con el corpus real.

### 8.4 Evidencia acústica adicional

Para cualquier resultado cuyo segmento tenga clases YAMNet, mostrarlas (`Music 0.72`, `Speech 0.55` ) como evidencia
acústica adicional, incluso si llegó por el índice de texto. Son una señal independiente de la recuperación que ayuda a
contextualizar y filtrar falsos positivos. Etiquetar explícitamente que son etiquetas AudioSet en inglés y scores de
clasificador, no similitud.

### 8.5 Filtros y estados

- Filtro por archivo de origen y por rango temporal dentro del archivo.
- Control de `k` (5 / 10 / 20 / 50).
- Selector de dataset cargado desde `/api/datasets`; muestra estados de carga/error y persiste la selección en `?dataset=`.
- Estados explícitos: cargando (con aviso de warm-up de CLAP en la primera query), sin resultados, índice no disponible
  en este dataset, error de backend.
- URL con estado (`?dataset=news-2026&q=...&text=0&clap=1&yamnet=1&k=10`) para compartir corpus y selección; texto se activa si falta `text=0`, y CLAP/YAMNet sólo se activan con sus parámetros.

## 9. Stack propuesto

Reutilizar lo que ya existe: **React 18 + TypeScript + Vite**, igual que `copilotkit-proxy/widget` (<ref_file
file="copilotkit-proxy/widget/package.json" />). Sin
dependencia de CopilotKit ni de AG-UI/A2A: el front habla REST plano contra `/api/*` . El backend puede reutilizar el
modelo y la configuración del agente para generar el `SearchPlan` , pero mediante una salida estructurada dedicada, sin
ejecutar el loop conversacional completo.

Reproducción con el elemento `<audio>` nativo + Media Fragments (`#t=start,end`), sin librería de waveform en v1 (agrega
peso y requiere descargar y decodificar el audio completo). Waveform queda como mejora posterior.

Servido como estáticos desde el mismo FastAPI, o como bundle separado detrás del proxy existente. Reutilizar
`ALLOW_ORIGINS` para CORS.

## 10. Métricas de éxito

| Métrica | Objetivo |
| --- | --- |
| Tiempo desde query hasta primer audio sonando | < 5 s (excluyendo warm-up en frío) |
| Latencia p95 del retrieval (tres fuentes, k=10) | < 1.5 s con CLAP ya cargado, excluyendo planner |
| Latencia p95 del planner | medir por separado y fijar objetivo con tráfico real |
| Clics para escuchar un resultado | 1 |
| Resultados escuchados por búsqueda | ≥ 3 (indica que el triage es efectivamente barato) |
| Trazabilidad del retrieval | 100% de búsquedas muestran y permiten copiar el plan efectivo |
| Fallback ante fallo del planner | búsqueda literal disponible sin error para el usuario |
| Costo de LLM por búsqueda | una llamada estructurada sólo cuando `rewrite=true` |

## 11. Fases

**Fase 0 — Desbloquear el audio (P0). HECHA.** Clips Opus por segmento generados en la ingesta, publicados en el release
y resueltos on-demand por URL firmada. Falta reprocesar el corpus de producción y publicar un release nuevo.

**Fase 1 — API de búsqueda estructurada. HECHA.** Implementada en el proxy
`copilotkit-proxy` sobre el search service Cloud Run: `POST /api/search/plan`,
`POST /api/search`, `GET /api/segments/{id}`, `GET /api/corpus`, validación del
planner y warm-up en el lifespan del servicio. Tests del planner en
`copilotkit-proxy/test/`.

**Fase 2 — Front v1. HECHO.** `frontend-direct-search/`: buscador, toggles
independientes de texto/CLAP/YAMNet (texto activo por defecto), toggle de
reformulación, plan visible/editable/copiable, secciones plegables con color
por índice, player inline, atajos de teclado (`Espacio`, `J`, `K`), copiar
cita. Desplegado en GitHub Pages por `deploy-pages.yaml`.

**Fase 3 — Refinamientos. EN CURSO.** Ya implementados: filtros locales por
archivo y timestamp, clases YAMNet en la tarjeta, reproducción continua del
ranking, estado en URL (`?dataset=...&q=...&text=0&clap=1&yamnet=1&k=10`), descarga de
clip, renovación preventiva de URLs firmadas y avisos configurables de coincidencia débil. El selector multi-dataset y
su propagación en búsquedas están implementados; quedan pendientes el bootstrap de `active_indexes` para datasets fríos
y el reenvío de `dataset_id` desde `/api/segments/:segmentId`.

**Fase 4 — Convivencia.** Toggle "Búsqueda directa" / "Preguntar al agente" que reutiliza el widget conversacional
existente para consultas exploratorias.

## 12. Decisiones

### Cerradas

1. **Servido de audio (§6):** clip por segmento con contexto, Opus 32 kbps, publicación en el release y servido
   on-demand por URL firmada.
2. **Rol del LLM:** se conserva para interpretar intención y hacer query rewrite mediante un plan estructurado, visible
   y editable. No filtra, resume ni reordena los resultados del motor.
3. **Control del usuario:** la selección manual de índices prevalece sobre el planner y siempre existe un modo literal
   sin reformulación.

### Abiertas

1. ~~**Modelo de deploy del front:** estáticos en el FastAPI del agente vs. servicio Cloud Run separado.~~ **Resuelta
   (2026-08-30):** estáticos en GitHub Pages (`deploy-pages.yaml`) + proxy en Cloud Run; el search service no se expone
   al navegador.
2. **Control de acceso:** el corpus puede contener material sensible sin publicar. Una URL firmada es un secreto
   portable: quien la recibe puede compartirla y seguirá funcionando hasta que expire. Definir si `/api/search` requiere
   autenticación, y si el TTL de 15 minutos es aceptable para el material del corpus.
3. **Umbral de "coincidencia débil"** por índice: requiere medirlo sobre el corpus real; hay insumos en
   `DOC-interna/docs/reporte-comparativo-texto-clap.md` y `docs/resultados-evaluacion-clotho.md` . La UI ya acepta los
   tres umbrales por variables Vite; falta fijar valores editoriales.
4. **Retención de releases viejas:** con los clips, cada release pasa a pesar ≈14 MB por hora de audio. Definir política
   de borrado o lifecycle en el bucket.
5. **Metadatos de datasets fríos:** decidir si `active_indexes`/`dataset_version` se declaran en el catálogo o se leen
   del manifiesto sin cargar el engine completo.
