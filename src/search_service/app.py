"""HTTP surface for the retrieval half of the agent.

This service owns everything that needs PyTorch: the CLAP text encoder, the
sentence-transformers encoder, the FAISS indices and the dataset itself. The ADK
agent runs in a separate container without any of it and calls these endpoints
through ``src.agent_service.search_client``.

The split exists because those models are a poor fit for a scale-to-zero
runtime: the CLAP checkpoint is ~1.7 GB and is fetched at first use rather than
baked by pip, so every cold start paid for it before answering. Here the image
ships the checkpoint, while ``min-instances=0`` permits scale-to-zero. Each
cold start still loads the checkpoint from the image. See ARCHITECTURE-CLOUD.md.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent_service.search_engine import AudioSearchEngine
from src.dataset_storage import resolve_dataset_path

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Loading the encoders costs far more than a request, so it happens at startup
# rather than on the first query. Readiness stays false until it finishes, which
# is what lets Cloud Run hold traffic back instead of serving a slow request.
_engine: AudioSearchEngine | None = None
_startup_error: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Consulta en lenguaje natural")
    k: int = Field(default=5, ge=1, le=20, description="Cantidad de segmentos")


class SearchResponse(BaseModel):
    """Raw engine output: the agent-side serialization stays in tools.py."""

    results: list[dict]


def _warm_up() -> AudioSearchEngine:
    """Load the dataset, the FAISS indices and both text encoders."""
    engine = AudioSearchEngine(resolve_dataset_path())
    # Encoding a throwaway string forces the lazy model properties to resolve,
    # including the CLAP checkpoint read. Failing here is intentional: a service
    # that cannot encode is not ready, and reporting it now beats a timeout on
    # the first real query.
    engine.text_model.generate_embedding("warm up")
    engine.clap_model.generate_text_embedding("warm up")
    logger.info("Search service warm: %d segments indexed", engine.total_segments)
    return engine


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _engine, _startup_error
    try:
        _engine = _warm_up()
    except Exception as error:  # noqa: BLE001 - surfaced through /readyz
        # Swallowed so the container still binds a port and can report why it is
        # unhealthy; crashing would only produce an opaque restart loop.
        _startup_error = str(error)
        logger.exception("Search service failed to warm up")
    yield


app = FastAPI(
    title="audio-search-retrieval",
    description="Índices FAISS y encoders de texto/CLAP del corpus de audio",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_engine() -> AudioSearchEngine:
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Search engine unavailable: {_startup_error or 'still warming up'}",
        )
    return _engine


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up, regardless of whether the models loaded."""
    return {"status": "healthy"}


@app.get("/readyz")
async def readyz() -> dict:
    """Readiness: only true once the dataset and both encoders are loaded."""
    if _engine is None:
        raise HTTPException(status_code=503, detail=_startup_error or "warming up")
    return {"status": "ready", "total_segments": _engine.total_segments}


@app.get("/corpus")
async def corpus() -> dict:
    """Metadata needed by the direct-search UI; no corpus text is exposed here."""
    engine = _require_engine()
    return {
        "total_segments": engine.total_segments,
        "files": [],
        "active_indexes": engine.active_indexes,
        "dataset_version": engine.dataset_version,
    }


@app.post("/search/semantic", response_model=SearchResponse)
async def search_semantic(request: SearchRequest) -> SearchResponse:
    """Text search over the transcriptions."""
    return SearchResponse(results=_require_engine().search_semantic(request.query, k=request.k))


@app.post("/search/audio", response_model=SearchResponse)
async def search_audio(request: SearchRequest) -> SearchResponse:
    """Cross-modal text-to-audio search through CLAP."""
    return SearchResponse(
        results=_require_engine().search_audio_by_text(request.query, k=request.k)
    )


@app.get("/segments/{segment_id}")
async def get_segment(segment_id: int) -> dict:
    """Full metadata for one segment."""
    segment = _require_engine().get_segment_info(segment_id)
    if segment is None:
        raise HTTPException(status_code=404, detail=f"Segmento {segment_id} no encontrado.")
    return {"segment": segment}


@app.get("/segments/{segment_id}/audio-classes")
async def get_audio_classes(segment_id: int) -> dict:
    """YAMNet AudioSet labels stored for one segment.

    An empty list means the dataset was ingested without YAMNet enabled, which
    the caller reports differently from a missing segment.
    """
    classes = _require_engine().get_audio_classes(segment_id)
    if classes is None:
        raise HTTPException(status_code=404, detail=f"Segmento {segment_id} no encontrado.")
    return {"classes": classes}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
