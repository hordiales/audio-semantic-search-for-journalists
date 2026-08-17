"""Compatibility FastAPI service for local agent and RAG-evaluation execution."""

from __future__ import annotations

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


class EvaluationQueryResponse(QueryResponse):
    """Agent answer plus the retrieval evidence used to generate it."""

    contexts: list[str]
    retrieved_segments: list[dict]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the local dataset before serving compatibility endpoints."""
    global agent
    agent = AudioAgent(
        dataset_path=os.getenv("DATASET_PATH", "./dataset"),
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    try:
        agent.initialize()
        logger.info("Agent service ready")
    except (FileNotFoundError, ValueError) as error:
        logger.error("Agent service not initialized: %s", error)
    yield


app = FastAPI(
    title="Audio Semantic Search Agent",
    description="Búsqueda agéntica multimodal de audio para periodistas",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_initialized_agent() -> AudioAgent:
    if not agent or not agent.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Check dataset and configuration.",
        )
    return agent


@app.get("/health")
async def health() -> dict:
    """Expose whether this local compatibility service can answer queries."""
    return {
        "status": "healthy" if agent and agent.is_initialized else "degraded",
        "dataset_path": os.getenv("DATASET_PATH", "./dataset"),
        "agent_initialized": agent.is_initialized if agent else False,
    }


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    """Execute a normal agent query without returning internal evidence."""
    response = await _require_initialized_agent().query(request.query, request.max_results)
    return QueryResponse(response=response, query=request.query)


@app.post("/evaluate/query", response_model=EvaluationQueryResponse)
async def query_agent_for_evaluation(request: QueryRequest) -> EvaluationQueryResponse:
    """Execute the agent while retaining contexts and metadata returned by tools."""
    response, contexts, retrieved_segments = await _require_initialized_agent().query_with_evidence(
        request.query, request.max_results
    )
    return EvaluationQueryResponse(
        response=response,
        query=request.query,
        contexts=contexts,
        retrieved_segments=retrieved_segments,
    )
