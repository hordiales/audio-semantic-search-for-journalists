"""Google ADK agent for journalist-oriented semantic audio search."""

import logging
import os
from uuid import uuid4

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.agent_service.tools import get_all_tools, initialize_search_engine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un asistente especializado en búsqueda semántica de contenido de audio para periodistas.

Reglas:
1. Usa buscar_audio para contenido dicho o escrito en las transcripciones.
2. Usa buscar_evento_acustico para sonidos como aplausos, música, gritos o risas.
3. Usa obtener_clases_audio para identificar las clases AudioSet detectadas por YAMNet en un segmento recuperado. Explica que son etiquetas en inglés y scores del clasificador.
4. Usa obtener_info_segmento cuando se pidan detalles de un segmento identificado.
5. Responde en español, salvo que el usuario pida otro idioma.
6. No inventes información: limita todas las afirmaciones a los resultados de las tools.
7. Para cada hallazgo, cita archivo de origen, timestamp de inicio y fin, e índice consultado.
   Usa exactamente el campo search_index_label devuelto por la tool (por ejemplo,
   "Índice de texto (transcripciones)" o "Índice de audio (CLAP)").
8. Si no hay resultados, dilo claramente y sugiere una reformulación.
"""


def _configured_model() -> LiteLlm:
    """Keep the existing OpenAI model configuration through LiteLLM."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return LiteLlm(model=f"openai/{model_name}")


root_agent = Agent(
    name="audio_search_agent",
    model=_configured_model(),
    instruction=SYSTEM_PROMPT,
    tools=get_all_tools(),
)

# The name must match the agents-cli configured agent directory (`src`).
app = App(root_agent=root_agent, name="src")


class AudioAgent:
    """Compatibility wrapper that executes the ADK root agent through a Runner."""

    def __init__(self, dataset_path: str, model_name: str = "gpt-4o-mini"):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self._session_service = InMemorySessionService()
        self._runner: Runner | None = None

    def initialize(self) -> None:
        """Load retrieval data and prepare the in-memory local ADK runner."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

        initialize_search_engine(self.dataset_path)
        self._runner = Runner(
            app=app,
            session_service=self._session_service,
            auto_create_session=True,
        )
        logger.info("ADK AudioAgent initialized (dataset=%s)", self.dataset_path)

    @property
    def is_initialized(self) -> bool:
        return self._runner is not None

    @staticmethod
    def _extract_evidence(payload: object) -> tuple[list[str], list[dict]]:
        """Extract retrieved text and segment metadata from FunctionTool output."""
        if not isinstance(payload, dict):
            return [], []
        candidates = payload.get("results", [])
        if "segment" in payload:
            candidates = [payload["segment"]]
        if not isinstance(candidates, list):
            return [], []

        contexts: list[str] = []
        segments: list[dict] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            segment = candidate.get("segment", candidate)
            if not isinstance(segment, dict):
                continue
            text = segment.get("text")
            if isinstance(text, str) and text:
                contexts.append(text)
            if {"segment_id", "original_file_name", "start_time", "end_time"} <= segment.keys():
                segments.append(
                    {
                        key: segment[key]
                        for key in (
                            "segment_id",
                            "original_file_name",
                            "start_time",
                            "end_time",
                            "text",
                        )
                        if key in segment
                    }
                )
        return contexts, segments

    async def query_with_evidence(
        self, user_query: str, max_results: int = 5
    ) -> tuple[str, list[str], list[dict]]:
        """Run ADK and return its answer plus evidence emitted by retrieval tools."""
        if self._runner is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        session_id = str(uuid4())
        await self._session_service.create_session(
            app_name=app.name,
            user_id="api-user",
            session_id=session_id,
            state={"max_results": max_results},
        )
        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
        final_response = "No pude procesar tu consulta."
        contexts: list[str] = []
        retrieved_segments: list[dict] = []
        async for event in self._runner.run_async(
            user_id="api-user", session_id=session_id, new_message=message
        ):
            if not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                function_response = getattr(part, "function_response", None)
                payload = getattr(function_response, "response", None)
                tool_contexts, tool_segments = self._extract_evidence(payload)
                contexts.extend(tool_contexts)
                retrieved_segments.extend(tool_segments)
            if event.is_final_response():
                text_parts = [part.text for part in event.content.parts if part.text]
                if text_parts:
                    final_response = "\n".join(text_parts)
        return final_response, contexts, retrieved_segments

    async def query(self, user_query: str, max_results: int = 5) -> str:
        """Run an ADK session and return its final response text."""
        response, _, _ = await self.query_with_evidence(user_query, max_results)
        return response
