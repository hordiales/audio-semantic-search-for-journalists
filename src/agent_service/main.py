"""FastAPI application for the Audio Search Agent service."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent_service.agent import AudioAgent

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

agent: AudioAgent | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Consulta en lenguaje natural")
    max_results: int = Field(default=5, ge=1, le=20, description="Máximo resultados")


class QueryResponse(BaseModel):
    response: str
    query: str


class HealthResponse(BaseModel):
    status: str
    dataset_path: str
    model_name: str
    agent_initialized: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup."""
    global agent

    dataset_path = os.getenv("DATASET_PATH", "./dataset")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    agent = AudioAgent(dataset_path=dataset_path, model_name=model_name)

    try:
        agent.initialize()
        logger.info("Agent service ready")
    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        logger.error("Run the ingestion pipeline first: poetry run python src/simple_dataset_pipeline.py")
    except ValueError as e:
        logger.error("Configuration error: %s", e)

    yield

    logger.info("Shutting down agent service")


app = FastAPI(
    title="Audio Semantic Search Agent",
    description="Búsqueda agéntica multimodal de audio para periodistas",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Service information."""
    return {
        "service": "Audio Semantic Search Agent",
        "version": "0.1.0",
        "description": "Búsqueda agéntica multimodal de audio para periodistas",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    dataset_path = os.getenv("DATASET_PATH", "./dataset")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    return HealthResponse(
        status="healthy" if agent and agent.is_initialized else "degraded",
        dataset_path=dataset_path,
        model_name=model_name,
        agent_initialized=agent.is_initialized if agent else False,
    )


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Consulta al agente de búsqueda semántica.

    El agente interpreta la consulta en lenguaje natural y busca
    segmentos de audio relevantes usando búsqueda semántica.
    """
    if not agent or not agent.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Check dataset and configuration.",
        )

    response = await agent.query(request.query, max_results=request.max_results)
    return QueryResponse(response=response, query=request.query)


@app.get("/query/sync")
async def query_agent_sync(q: str, max_results: int = 5):
    """Synchronous query endpoint (for compatibility/testing)."""
    if not agent or not agent.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Check dataset and configuration.",
        )

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    response = await agent.query(q, max_results=max_results)
    return {"response": response, "query": q}
