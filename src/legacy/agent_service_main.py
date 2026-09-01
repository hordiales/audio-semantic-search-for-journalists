"""Deprecated standalone FastAPI server; use ``src.fast_api_app`` in production."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent_service.agent import AudioAgent

load_dotenv()
logger = logging.getLogger(__name__)
agent: AudioAgent | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    response: str
    query: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agent
    agent = AudioAgent(
        dataset_path=os.getenv("DATASET_PATH", "./dataset"),
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    try:
        agent.initialize()
    except (FileNotFoundError, ValueError) as error:
        logger.error("Legacy server initialization failed: %s", error)
    yield


app = FastAPI(title="Legacy Audio Search Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy" if agent and agent.is_initialized else "degraded"}


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    if not agent or not agent.is_initialized:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    return QueryResponse(
        response=await agent.query(request.query, max_results=request.max_results),
        query=request.query,
    )
