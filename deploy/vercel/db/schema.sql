-- Neon PostgreSQL + pgvector schema for audio-semantic-search-for-journalists
-- Used by scripts/publish_dataset_neon.py and deploy/vercel/lib/db.ts

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'es',
    dataset_version TEXT,
    active_indexes JSONB NOT NULL DEFAULT '["text"]'::jsonb,
    total_segments INTEGER NOT NULL DEFAULT 0,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audio_segments (
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    segment_id INTEGER NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION NOT NULL,
    duration DOUBLE PRECISION GENERATED ALWAYS AS (end_time - start_time) STORED,
    original_file_name TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT 'unknown',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    clip_file_name TEXT,
    clip_start_time DOUBLE PRECISION,
    clip_end_time DOUBLE PRECISION,
    clip_url TEXT,
    sentiment_positive DOUBLE PRECISION,
    sentiment_negative DOUBLE PRECISION,
    sentiment_neutral DOUBLE PRECISION,
    dominant_sentiment TEXT,
    yamnet_top_classes JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Unbounded vector columns allow either local ingestion dimensions (384/512)
    -- or API-generated dimensions (e.g., 1536 for OpenAI / Gemini Embedding 2).
    text_embedding vector,
    audio_embedding vector,
    PRIMARY KEY (dataset_id, segment_id)
);

CREATE INDEX IF NOT EXISTS idx_audio_segments_dataset_duration
    ON audio_segments (dataset_id, duration);

CREATE TABLE IF NOT EXISTS segment_yamnet_postings (
    dataset_id TEXT NOT NULL,
    token TEXT NOT NULL,
    segment_id INTEGER NOT NULL,
    class_id TEXT NOT NULL,
    class_name TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    class_rank INTEGER NOT NULL,
    PRIMARY KEY (dataset_id, token, segment_id, class_id),
    FOREIGN KEY (dataset_id, segment_id)
        REFERENCES audio_segments (dataset_id, segment_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_yamnet_postings_lookup
    ON segment_yamnet_postings (dataset_id, token);

CREATE TABLE IF NOT EXISTS feedback_logs (
    id BIGSERIAL PRIMARY KEY,
    service_name TEXT NOT NULL DEFAULT 'audio-search-journalists-vercel',
    message_id TEXT NOT NULL,
    feedback TEXT NOT NULL CHECK (feedback IN ('thumbsUp', 'thumbsDown')),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
