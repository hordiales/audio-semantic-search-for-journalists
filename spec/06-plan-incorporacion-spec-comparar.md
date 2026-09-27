# Plan de incorporación de `spec-comparar`

**Fecha:** 2026-08-09  
**Última actualización de estado:** 2026-09-27

**Objetivo:** incorporar en las especificaciones vigentes los aportes útiles de `spec-comparar` , sin declarar como
implementadas decisiones futuras ni introducir contradicciones con el código actual.

> Nota de revisión: `spec-comparar/` ya no está presente en el árbol actual. Las referencias al final de este documento
> son históricas y no son enlaces operativos. El estado vigente se describe en las specs 01–05, 07 y 08.

## Decisión general

`spec-comparar` no debe copiarse como reemplazo directo de `spec`. Contiene tres clases de material:

1. mejoras y hallazgos aplicables a corto plazo;
2. propuestas que requieren una decisión explícita de arquitectura;
3. documentación histórica o experimental que debe conservarse como referencia, no como requisito de producción.

## Alcance

- Actualizar las specs vigentes y su índice cuando exista una decisión aprobada.
- Mantener la implementación actual como fuente de verdad para describir el estado presente.
- Registrar las desviaciones entre la arquitectura objetivo y la implementación mediante ADRs y backlog verificable.

No incluye implementar DeepEval, FLAM ni aprovisionar recursos GCP. La migración a Google ADK, el scaffolding de Agent
Runtime y GitHub Actions fueron aprobados e iniciados.

## Estado actual de ejecución

### Completado

- [x] Decisión de producción: Google ADK + LiteLLM, Agent Runtime de Google Agent Platform y GitHub Actions.
- [x] Migración de LangChain a `root_agent` ADK, FunctionTools y `Runner` en `src/agent_service`.
- [x] Tool explícita `buscar_evento_acustico` que usa el índice CLAP; se mantienen `buscar_audio` y
  `obtener_info_segmento` .
- [x] Propagación de `max_results` mediante estado de sesión ADK, no por una instrucción al LLM.
- [x] Scaffolding de Agent Runtime, A2A, Terraform, GitHub Actions, manifiesto de agents-cli y configuración de
  evaluación.
- [x] Expectations periodísticas incorporadas a la spec de evaluación y a `tests/eval/eval_config.yaml`.
- [x] Dependencias ADK/LiteLLM declaradas; `uv.lock` generado para el flujo requerido por agents-cli.
- [x] Validación determinista incorporada a la suite de tests y Ruff para los archivos migrados.
- [x] Dataset sintético de prueba generado en `dataset/`, con tres segmentos, manifiesto e índices FAISS de texto/audio.
- [x] Clasificación opcional YAMNet integrada en la ingesta: persiste clases AudioSet por segmento, agrega scores entre
  ventanas y expone `obtener_clases_audio` al agente. Requiere instalar el extra `yamnet` y reprocesar corpus reales.
- [x] Contrato de evaluación RAG independiente de framework: RAGAS o DeepEval se seleccionan con `EVALUATION_FRAMEWORK`
  , comparten la tripleta real `(q, C, a)` y registran la configuración del juez por corrida.
- [x] Ingesta incremental con fingerprinting de contenido (`sha256`) y firma de configuración que invalida artefactos
  obsoletos (`src/simple_dataset_pipeline.py`).
- [x] Configuración de embeddings por variables de entorno (`EMBEDDINGS_ACTIVE`, `CLASSIFIERS_ACTIVE` , ...);
  `config/embeddings.toml` se autogenera.
- [x] División agente/retrieval: search service propio en Cloud Run (`src/search_service/`, `Dockerfile.search-service`
  , `cloudbuild.search-service.yaml` ) y agente sin dependencias ML desplegado en Agent Runtime (2026-08-28,
  `deployment_metadata.json` , A2A habilitado).
- [x] Cliente de retrieval polimórfico (`src/agent_service/search_client.py`) con imports perezosos:
  `SEARCH_SERVICE_URL` delega el retrieval por HTTP.
- [x] Búsqueda por clases YAMNet como tercera modalidad (`buscar_clase_audio`, índice invertido) y etiquetado de índice
  en resultados (`search_index`/`search_index_label`).
- [x] Telemetría OpenTelemetry, rutas A2A v1.0 y adapter de Reasoning Engine en la superficie de producción
  (`src/fast_api_app.py`, `src/app_utils/` ).
- [x] Widget embebible CopilotKit y front de búsqueda directa (`frontend-direct-search/`), publicados en GitHub Pages;
  proxy Node containerizado (`copilotkit-proxy/`).
- [x] Benchmarks CLAP: Clotho (`benchmarks/evaluate_clap_clotho.py`), comparación por tipo de pregunta y revisión humana
  acústica (`evaluation/human_review/`).
- [x] Releases versionados del dataset en GCS (`scripts/publish_dataset_release.py`) con descarga atómica y clips Opus
  servidos por URLs firmadas.

### Pendiente o bloqueado

- [x] Ejecutar `agents-cli run` y evaluaciones del agente: hay corridas en `evaluation/results/agent_evaluation.json` y
  el deploy a Agent Runtime existe.
- [x] ~~Completar `agents-cli install --locked` : uv falla al extraer dependencias en su caché con
  `No space left on device` ~~ Resuelto: `uv sync --locked` corre en CI y local.
- [x] Configurar credenciales `OPENAI_API_KEY` para LiteLLM y las variables/secretos GCP requeridos por Agent Runtime y
  GitHub Actions (WIF configurado; deploys en CI).
- [x] Aprovisionar infraestructura y desplegar: pipelines `staging` (push a `main` ) y `deploy-to-prod` (manual)
  operativos.
- [ ] Umbral de "coincidencia débil" por índice y política de retención de releases viejas (decisiones abiertas de spec
  08 §12).

## Fase 0 — Consolidar la fuente de verdad

### Acciones

- [x] Definir producción en Google Agent Platform (Agent Runtime) con CI/CD de GitHub Actions; el agente conserva
  ejecución local mediante agents-cli.
- [x] Corregir las referencias históricas restantes a Cloud Run/local-only, en particular `spec/memory.md` y las
  secciones no migradas de infraestructura (2026-08-30).
- [x] Documentar la arquitectura de producción, ruta de servicio, contrato de API y estrategia de despliegue:
  `DOC-interna/ARCHITECTURE-CLOUD.md` + spec 05 reescrita (2026-08-30).
- [x] Completar la actualización de `spec/summary.md` , `01-arquitectura.md` , `03-agente-busqueda.md` y
  `05-infraestructura-deploy.md` (2026-08-30).

### Criterio de terminado

Existe una única arquitectura de producción documentada y no hay requisitos incompatibles entre los documentos de `spec`
.

## Fase 1 — Incorporaciones inmediatas a las specs

### 1. Backlog de hallazgos de implementación

Incorporar el contenido de `spec-comparar/PLAN_CORRECCION_FINDINGS.md` como backlog priorizado, ajustándolo a la
evidencia actual del código.

- [x] Documentar y exponer la búsqueda acústica: `buscar_evento_acustico` usa `search_audio_by_text` con CLAP.
- [x] Fusión/ranking entre modalidades: decidido en spec 08 — nunca mezclar scores de espacios distintos; cada índice se
  presenta en su propia sección con su ranking y score.
- [ ] Definir la correspondencia persistida entre vector FAISS, `segment_id` y metadatos.
- [ ] Exigir `segment_id` global, estable y único, más una clave de respaldo `source_file + local_segment_id` (spec 07
  verifica unicidad en la guía operativa).
- [x] Snapshots atómicos para índice, metadatos y manifiesto: `src/dataset_storage.py` descarga a staging y reemplaza el
  snapshot de forma atómica.
- [x] Hacer que `max_results` se propague a las tools por estado de sesión ADK.
- [x] Distinguir liveness de readiness: `/health` (agente) vs. `/health` + `/readyz` (search service, readiness falso
  hasta completar el warm-up).

### 2. Evaluación orientada al uso periodístico

Incorporar cuatro criterios funcionales, independientes del framework de evaluación elegido:

- [x] `cita_fuente_periodistica`: la respuesta identifica archivo, programa o entrevista de origen.
- [x] `indica_momento_temporal`: incluye `start_time`, `end_time` o un formato temporal equivalente.
- [x] `limita_al_contexto_recuperado`: las afirmaciones están respaldadas por los resultados de retrieval.
- [x] `idioma_consistente`: conserva el idioma de la consulta salvo instrucción contraria.

Para cada criterio, definir entrada, salida binaria (`0|1`), explicación del juez y umbral promedio objetivo de `>= 0.8`
sobre un dataset anotado.

### Criterio de terminado

Las specs de arquitectura, agente y evaluación incluyen contratos verificables de retrieval multimodal, identidad de
segmentos, disponibilidad del servicio y calidad editorial.

## Fase 2 — Decisiones de arquitectura antes de actualizar tecnología

### Google ADK + LiteLLM frente a LangChain

La migración propuesta en `spec-comparar/spec/01-arquitectura.md` y `03-agente-busqueda.md` fue aprobada: el servicio de
producción usa Google ADK con LiteLLM y conserva GPT-4o-mini como modelo configurado.

- [x] Elegir Google ADK + LiteLLM como ruta de producción.
- [x] Migrar `agent.py`, `tools.py` y la API compatible a `root_agent`, FunctionTools y Runner.
- [x] Añadir scaffolding de Agent Runtime, A2A, GitHub Actions y agents-cli eval.
- [x] Ejecutar el smoke test y las evaluaciones contra un dataset procesado y credenciales configuradas: resultados en
  `evaluation/results/` (agente, retrieval, Clotho, por tipo de pregunta).

### Evaluación: RAGAS frente a agents-cli eval + DeepEval

- [x] Mantener las métricas de retrieval actuales: Precision@K, Recall@K, MRR y NDCG@K.
- [x] Evaluar la compatibilidad de RAGAS con el lockfile: `ragas<=0.4.x` importa
  `langchain_community.chat_models.vertexai` incondicionalmente; quedó pinneado por debajo de esa versión hasta el fix
  upstream (ragas #2741/#2753, nota en `pyproject.toml` ).
- [x] Completar la adopción de agents-cli eval: la configuración y los casos existen y hay resultados versionados en
  `evaluation/results/`; las Expectations siguen siendo un flujo separado de RAGAS/DeepEval.
- [x] Adoptar DeepEval con versiones, juez y formato fijados: `evaluation/deepeval_evaluation.py` +
  `EVALUATION_FRAMEWORK=deepeval` (extra `eval-deepeval` ).
- [x] Usar una única fuente anotada para datasets, aun si se ejecutan varios evaluadores: `EVALUATION_DATASET_SOURCE`
  (manual/synthetic/hybrid) en `evaluation/evaluation_config.py` .

### Criterio de terminado

La tecnología del agente y del evaluador está decidida mediante ADR; las dependencias, comandos y ejemplos de las specs
coinciden con esa decisión.

## Fase 3 — Documentación operativa

Revisar y trasladar el contenido útil de los siguientes documentos como guías de uso, no como requisitos de
arquitectura:

- [ ] `INSTALL.md`: instalación con Poetry/pip y prerrequisitos.
- [ ] `GUIA_DATASET_Y_BUSQUEDA.md`: creación de datasets y uso de CLI.
- [ ] `DATASET.md` y `SIMPLE_DATASET_PIPELINE.md`: formatos, manifiesto y operación del pipeline.
- [ ] `GPU_CONSIDERATIONS.md` y `TROUBLESHOOTING.md`: MPS, CUDA, Triton, memoria y ffmpeg.
- [ ] `PYTHON_VERSION.md`: requisito y motivo de Python 3.11.
- [ ] `comandos-útiles.md`: referencia breve de comandos validados.

Antes de trasladar contenido:

- [ ] Ejecutar y validar los comandos contra el `pyproject.toml` y el código actual.
- [ ] Eliminar rutas, scripts y servicios inexistentes.
- [ ] Consolidar los documentos para evitar duplicar instrucciones de instalación y troubleshooting.

### Criterio de terminado

Existe un índice de documentación operativa y todos sus comandos, rutas y dependencias son válidos para el repositorio
actual.

## Fase 4 — Elementos que no se incorporan aún

### Chunking

No eliminar el chunking configurable de `spec/02-pipeline-ingesta.md` . La versión comparativa lo elimina sin una
justificación técnica; se mantiene como capacidad opcional hasta contar con evidencia de evaluación que recomiende otra
decisión.

### Dependencias de infraestructura

No copiar el pin `pandas==1.5.3` de la spec comparativa: contradice la dependencia actual `pandas>=2.0` . Cualquier pin
debe justificarse mediante una prueba de compatibilidad y quedar fijado en el lockfile.

### FLAM

Mantener FLAM como investigación o roadmap. Los documentos de `spec-comparar` son contradictorios respecto de
disponibilidad, instalación y checkpoints.

- [ ] Verificar repositorio oficial, licencia, pesos, instalación y compatibilidad con Python 3.11.
- [ ] Definir un dataset con timestamps anotados antes de evaluar localización temporal.
- [ ] Comparar FLAM con CLAP únicamente tras disponer de artefactos reproducibles.

Hasta entonces, CLAP permanece como tecnología de retrieval acústico de producción.

## Orden recomendado

1. Resolver la arquitectura y alcance de producción (Fase 0).
2. Incorporar contratos y backlog de hallazgos (Fase 1).
3. Decidir el framework de agente y evaluación (Fase 2).
4. Consolidar la documentación operativa validada (Fase 3).
5. Mantener FLAM y otras propuestas experimentales fuera de la spec de producción (Fase 4).

## Referencias

- `spec-comparar/PLAN_CORRECCION_FINDINGS.md`
- `spec-comparar/spec/01-arquitectura.md`
- `spec-comparar/spec/03-agente-busqueda.md`
- `spec-comparar/spec/04-evaluacion.md`
- `spec-comparar/FLAM_AVAILABILITY_STATUS.json`
