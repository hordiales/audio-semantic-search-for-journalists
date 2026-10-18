# Dual Deployment Architecture: GCP vs. Vercel (Neon + Eve Agents)

This repository supports **two independent deployment targets** from the same audio ingestion pipeline (`src/simple_dataset_pipeline.py`), without breaking local development or existing scripts.

```text
                       ┌───────────────────────────────────────────┐
                       │   Local Ingestion Pipeline (Python/uv)    │
                       │     src/simple_dataset_pipeline.py        │
                       │  Produces: dataset/ (pkl, embeddings,     │
                       │            yamnet classes, .opus clips)   │
                       └───────────┬───────────────────┬───────────┘
                                   │                   │
         scripts/publish_dataset_release.py     scripts/publish_dataset_neon.py
                                   │                   │
                                   ▼                   ▼
                    ┌──────────────────────┐   ┌──────────────────────────┐
                    │   TARGET 1: GCP      │   │   TARGET 2: VERCEL       │
                    │  (deploy/gcp/)       │   │  (deploy/vercel/)        │
                    ├──────────────────────┤   ├──────────────────────────┤
                    │ • Storage: GCS       │   │ • Storage: Neon Postgres │
                    │   + FAISS in /tmp    │   │   (pgvector) +           │
                    │ • Clips: GCS V4      │   │   Vercel Blob / S3 (.opus│
                    │   signed URLs        │   │   playback clips)        │
                    │ • Search: Cloud Run  │   │ • Agent: Eve Agents      │
                    │   (PyTorch + CLAP)   │   │   (agent/agent.ts +      │
                    │ • Agent: Vertex AI   │   │   agent/tools/*.ts)      │
                    │   Agent Runtime (ADK)│   │ • API & Direct Search:   │
                    │ • Proxy: Cloud Run   │   │   Vercel Functions /     │
                    │   (AG-UI ↔ A2A)      │   │   Eve HTTP routes        │
                    └──────────────────────┘   └──────────────────────────┘
```

---

## Directory Organization

```text
├── src/                                  # Shared Python core, ingestion & local/GCP runtime
│   ├── simple_dataset_pipeline.py        # Audio → transcription, embeddings, YAMNet, clips
│   ├── neon_exporter.py                  # Exports processed dataset to Neon pgvector + Blob/S3
│   ├── agent_service/                    # Python ADK agent & FAISS search engine
│   └── search_service/                   # Cloud Run retrieval FastAPI service
├── scripts/
│   ├── publish_dataset_release.py        # Publishes dataset release to GCS (GCP target)
│   └── publish_dataset_neon.py           # Publishes dataset release to Neon + Vercel Blob (Vercel target)
├── deploy/
│   ├── README.md                         # This architecture & dual-deploy plan
│   ├── gcp/                              # Target 1: Google Cloud Platform
│   │   ├── README.md                     # Step-by-step GCP deployment guide
│   │   ├── deploy.env.example            # GCP environment template
│   │   ├── Dockerfile.agent              # Lightweight ADK agent image (--no-group ml)
│   │   ├── Dockerfile.search-service     # Heavy Cloud Run search service image (--group ml)
│   │   └── cloudbuild.search-service.yaml
│   └── vercel/                           # Target 2: Vercel + Neon + Eve Agents
│       ├── README.md                     # Step-by-step Vercel + Neon + Eve deployment guide
│       ├── package.json                  # TypeScript dependencies (eve, @neondatabase/serverless, @vercel/blob, ai, zod)
│       ├── tsconfig.json
│       ├── vercel.ts                     # Vercel service composition with withEve()
│       ├── .env.example                  # Neon DATABASE_URL, BLOB_READ_WRITE_TOKEN, API keys
│       ├── db/
│       │   └── schema.sql                # Neon PostgreSQL + pgvector + GIN schema
│       ├── src/
│       │   ├── db.ts                     # Neon serverless client & pgvector search queries
│       │   ├── embeddings.ts             # API-based query embeddings (OpenAI / Gemini / HF)
│       │   └── direct-search.ts          # Direct search & AI query planner over Neon
│       ├── agent/                        # Eve Agents filesystem definition
│       │   ├── agent.ts                  # defineAgent() configuration
│       │   ├── instructions.md           # Journalist assistant system prompt
│       │   ├── channels/
│       │   │   └── eve.ts                # HTTP channel authentication policy
│       │   └── tools/
│       │       ├── buscar_audio.ts           # Semantic transcript search (pgvector)
│       │       ├── buscar_evento_acustico.ts # Acoustic search (pgvector audio embeddings)
│       │       ├── buscar_clase_audio.ts     # YAMNet/AudioSet class search (JSONB/inverted)
│       │       ├── obtener_info_segmento.ts  # Segment metadata & Blob clip URL lookup
│       │       └── obtener_clases_audio.ts   # YAMNet classes for a specific segment
│       └── api/                          # Vercel Serverless API routes (Direct Search contract)
│           ├── health.ts                 # GET /api/health
│           ├── datasets.ts               # GET /api/datasets
│           ├── corpus.ts                 # GET /api/corpus
│           ├── search.ts                 # POST /api/search
│           ├── search-plan.ts            # POST /api/search/plan
│           ├── segments.ts               # GET /api/segments/:segmentId
│           └── feedback.ts               # POST /api/feedback (persisted in Neon)
├── copilotkit-proxy/                     # Cloud Run AG-UI ↔ A2A proxy & widget (GCP target)
└── frontend-direct-search/               # Direct search React/Vite UI (compatible with both targets)
```

> **Backwards Compatibility:** Root files (`Dockerfile`, `Dockerfile.search-service`, `cloudbuild.search-service.yaml`, `agents-cli-manifest.yaml`, `deploy.env.example`) remain in place so `agents-cli` and existing Cloud Build commands continue working without modification.

---

## Comparison of Targets

| Capability | Target 1: GCP (`deploy/gcp/`) | Target 2: Vercel (`deploy/vercel/`) |
|---|---|---|
| **Agent Framework** | Google ADK (`src/agent_service/agent.py`) on Vertex AI Agent Runtime | **Eve Agents** (`deploy/vercel/agent/`) on Vercel Functions + Workflows |
| **Vector Store** | FAISS (`IndexFlatIP`) staged from GCS into `/tmp` | **Neon Postgres (`pgvector`)** with HNSW cosine indexes |
| **Query Embeddings** | Local PyTorch in Cloud Run (`MiniLM`, `CLAP`) | **API-based embeddings** (OpenAI `text-embedding-3-small`, Gemini `gemini-embedding-2`, or HF/custom endpoint for 384d/512d) |
| **YAMNet Search** | In-memory JSON inverted index (`yamnet_inverted_index.json`) | **Neon Postgres** (`segment_yamnet_postings` table + B-tree/GIN index) |
| **Audio Clips (`.opus`)** | Private GCS bucket + V4 signed URLs | **Vercel Blob** (or S3/R2) URLs stored per segment in Neon |
| **Direct Search API** | `copilotkit-proxy` (Cloud Run) → `audio-search-service` | Vercel Functions (`/api/search`, `/api/datasets`, `/api/corpus`, `/api/segments/:id`) querying Neon directly |
| **Feedback Storage** | Cloud Logging → BigQuery sink | `feedback_logs` table in **Neon Postgres** |
