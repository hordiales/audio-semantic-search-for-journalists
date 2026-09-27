# Memory

Project: audio-semantic-search-for-journalists-v2
Path: /path/to/audio-semantic-search-for-journalists-v2

Stack: Python 3.11, uv (dependency management), Whisper (transcription), Sentence Transformers (text embeddings),
CLAP/LAION (audio embeddings), YAMNet (AudioSet classes, optional), FAISS (vector search), Google ADK + LiteLLM +
GPT-4o-mini (agent), FastAPI (API), RAGAS o DeepEval (agent evaluation), A2A (agent protocol), GCP Agent Runtime + Cloud
Run + GCS + GitHub Pages (deployment), GitHub Actions (CI/CD), Docker.

Key modules:

- src/audio_conversion.py - ffmpeg WAV normalization
- src/audio_transcription.py - Whisper transcription
- src/text_embeddings.py - Sentence Transformers (384-dim)
- src/clap_audio_embeddings.py - CLAP cross-modal (512-dim)
- src/gemini_multimodal_embeddings.py - Gemini Embedding 2 (opcional, benchmark)
- src/yamnet_audio_classifier.py + src/yamnet_inverted_index.py - AudioSet classes + inverted index
- src/sentiment_analysis.py - pysentimiento (Spanish)
- src/vector_indexing.py - FAISS IndexFlatIP
- src/simple_dataset_pipeline.py - Incremental ingestion pipeline (content fingerprinting + pipeline signature)
- src/segment_clips.py + src/segment_clip_storage.py - Opus playback clips + GCS signed URLs
- src/dataset_storage.py - Dataset release download from GCS (DATASET_GCS_URI)
- src/embedding_config.py - Ingestion config from env vars (EMBEDDINGS_ACTIVE/CLASSIFIERS_ACTIVE);
  config/embeddings.toml is auto-generated, never edited by hand
- src/query_translation.py - ES->EN query translation for CLAP/YAMNet encoders
- src/agent.py - ADK entrypoint (agents-cli / Agent Runtime)
- src/agent_service/agent.py - root_agent ADK + AudioAgent wrapper (5 tools, system prompt with DIRECT_SEARCH_PLANNER_V1
  mode)
- src/agent_service/tools.py - buscar_audio, buscar_evento_acustico, buscar_clase_audio, obtener_info_segmento,
  obtener_clases_audio; AGENT_MODALITY filtering; max_results via session state
- src/agent_service/search_engine.py - AudioSearchEngine (FAISS, in-process)
- src/agent_service/search_client.py - SearchServiceClient (HTTP) selected by SEARCH_SERVICE_URL; keeps agent image free
  of PyTorch/FAISS
- src/search_service/app.py - Cloud Run retrieval service (/datasets, /search/semantic, /search/audio, /search/yamnet,
  /segments/{id}, /corpus, /health, /readyz); direct retrieval accepts dataset_id
- src/search_service/dataset_registry.py - GCS catalog, default dataset, lazy staging under /tmp, shared encoders and
  LRU eviction; falls back to the DATASET_GCS_URI/DATASET_PATH single-dataset mode
- src/fast_api_app.py - Production ADK surface (/query, /evaluate/query, /feedback, A2A routes, reasoning engine
  adapter, OTel telemetry)
- src/app_utils/ - telemetry, services, A2A v1.0, reasoning_engine_adapter
- evaluation/retrieval_evaluation.py - IR metrics (Precision/Recall/F1@K, MRR, NDCG, FP Cost)
- evaluation/audit_corpus_acoustic_events.py - continuous YAMNet screening of complete recordings correlated with
  human event-presence judgments; screening evidence, not ground truth
- evaluation/ragas_evaluation.py / deepeval_evaluation.py - selectable via EVALUATION_FRAMEWORK
- evaluation/evaluation_config.py - dataset sources manual/synthetic/hybrid via EVALUATION_DATASET_SOURCE
- benchmarks/evaluate_clap_clotho.py - CLAP sanity check vs Clotho (Recall@1/5/10)
- copilotkit-proxy/ - Node proxy: conversational widget (AG-UI/A2A) + structured direct-search API (/api/datasets,
  /api/search, /api/search/plan, /api/segments/:id, /api/corpus)
- frontend-direct-search/ - React+Vite archive-console UI (GitHub Pages)
- copilotkit-proxy/widget/ - embeddable widget build (GitHub Pages)
- scripts/publish_dataset_release.py - publishes dataset release to GCS

Multi-dataset scope: direct search passes dataset_id through browser -> proxy -> search service. SearchServiceClient,
used by the conversational agent, does not pass dataset_id and therefore uses DEFAULT_DATASET_ID. The current /datasets
response only has active_indexes for engines that have already loaded; non-default datasets are returned with an empty
list, and the frontend consequently disables every source for them. Full selector operation remains pending.

Scope decisions: production on Google Agent Platform (Agent Runtime) for the agent, Cloud Run for the retrieval service
(ships CLAP checkpoint), GitHub Pages for static frontends, CI/CD with GitHub Actions (pr_checks, staging,
deploy-to-prod, deploy-pages). Docker included. Datastore not used.
