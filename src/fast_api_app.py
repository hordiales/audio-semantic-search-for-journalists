# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging
from google.genai import types
from pydantic import BaseModel, Field

from src.app_utils import services
from src.app_utils.a2a import attach_a2a_routes
from src.app_utils.reasoning_engine_adapter import (
    attach_reasoning_engine_routes,
)
from src.app_utils.telemetry import (
    setup_agent_engine_telemetry,
    setup_telemetry,
)
from src.app_utils.typing import Feedback

load_dotenv()
setup_telemetry()
# Must run before get_fast_api_app to set the tracer provider resource.
setup_agent_engine_telemetry()
_, project_id = google.auth.default()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)
allow_origins = os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Consulta en lenguaje natural")
    max_results: int = Field(default=5, ge=1, le=20, description="Máximo de segmentos")


class QueryResponse(BaseModel):
    response: str
    query: str


class EvaluationQueryResponse(QueryResponse):
    contexts: list[str]
    retrieved_segments: list[dict]


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Runner for the A2A path, sharing the same session/artifact services as the
    # adk_api and reasoning_engine paths (see services.py). Imported here so the
    # agent is built after env/telemetry setup.
    from src.agent import app as adk_app
    from src.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    # Shared by the A2A path and the reasoning_engine adapter routes.
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    app.state.session_service = services.get_session_service()
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=False,
    lifespan=lifespan,
)
app.title = "audio-search-journalists"
app.description = "API for interacting with the Agent audio-search-journalists"


# Proxy routes so the Vertex AI Console Playground (reasoning_engine SDK) can
# talk to this agent alongside the native adk_api routes.
attach_reasoning_engine_routes(app)


def _extract_evidence(payload: object) -> tuple[list[str], list[dict]]:
    """Return RAGAS contexts and source metadata from function-tool events."""
    if not isinstance(payload, dict):
        return [], []
    candidates = payload.get("results", [])
    if "segment" in payload:
        candidates = [payload["segment"]]
    if not isinstance(candidates, list):
        return [], []

    contexts, segments = [], []
    for candidate in candidates:
        segment = candidate.get("segment", candidate) if isinstance(candidate, dict) else None
        if not isinstance(segment, dict):
            continue
        if isinstance(segment.get("text"), str) and segment["text"]:
            contexts.append(segment["text"])
        required = {"segment_id", "original_file_name", "start_time", "end_time"}
        if required <= segment.keys():
            segments.append({key: segment[key] for key in required | {"text"} if key in segment})
    return contexts, segments


async def _run_query(query: str, max_results: int) -> tuple[str, list[str], list[dict]]:
    """Execute the production ADK runner through the stable REST contract."""
    if not hasattr(app.state, "runner"):
        raise HTTPException(status_code=503, detail="Agent runner is not initialized.")
    session_id = str(uuid4())
    user_id = "rest-api-user"
    await app.state.session_service.create_session(
        app_name=app.state.agent_app_name,
        user_id=user_id,
        session_id=session_id,
        state={"max_results": max_results},
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=query)])
    response, contexts, segments = "No pude procesar tu consulta.", [], []
    async for event in app.state.runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            function_response = getattr(part, "function_response", None)
            payload = getattr(function_response, "response", None)
            tool_contexts, tool_segments = _extract_evidence(payload)
            contexts.extend(tool_contexts)
            segments.extend(tool_segments)
        if event.is_final_response():
            text_parts = [part.text for part in event.content.parts if part.text]
            if text_parts:
                response = "\n".join(text_parts)
    return response, contexts, segments


@app.get("/health")
async def health() -> dict[str, str]:
    """Health status for the deployed ADK serving surface."""
    return {"status": "healthy" if hasattr(app.state, "runner") else "degraded"}


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    response, _, _ = await _run_query(request.query, request.max_results)
    return QueryResponse(response=response, query=request.query)


@app.post("/evaluate/query", response_model=EvaluationQueryResponse)
async def query_agent_for_evaluation(request: QueryRequest) -> EvaluationQueryResponse:
    response, contexts, segments = await _run_query(request.query, request.max_results)
    return EvaluationQueryResponse(
        response=response,
        query=request.query,
        contexts=contexts,
        retrieved_segments=segments,
    )


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
