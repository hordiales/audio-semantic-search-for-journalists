"""Function tools exposed by the ADK audio-search agent."""

import logging
import os
from pathlib import Path

from google.adk.tools import ToolContext

from src.agent_service.search_engine import AudioSearchEngine
from src.dataset_storage import resolve_dataset_path

logger = logging.getLogger(__name__)

_search_engine: AudioSearchEngine | None = None


def set_search_engine(engine: AudioSearchEngine) -> None:
    """Set the process-wide search engine used by the agent tools."""
    global _search_engine
    _search_engine = engine


def initialize_search_engine(dataset_path: str | None = None) -> AudioSearchEngine:
    """Initialize the search engine once from the configured processed dataset."""
    global _search_engine
    if _search_engine is None:
        configured_path = resolve_dataset_path(dataset_path)
        _search_engine = AudioSearchEngine(configured_path)
        logger.info("Search engine initialized from %s", Path(configured_path))
    return _search_engine


def get_search_engine() -> AudioSearchEngine:
    """Return the initialized search engine, creating it lazily if needed."""
    return initialize_search_engine()


def _serialize_results(results: list[dict], search_index: str, search_index_label: str) -> list[dict]:
    """Return JSON-compatible search results with journalist-facing metadata."""
    serialized: list[dict] = []
    for result in results:
        segment = result["segment"]
        serialized.append(
            {
                "segment_id": segment["segment_id"],
                "search_index": search_index,
                "search_index_label": search_index_label,
                "text": segment["text"],
                "similarity": round(result["similarity"], 4),
                "similarity_percent": round(result["similarity"] * 100, 1),
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": round(segment["end_time"] - segment["start_time"], 1),
                "original_file_name": segment["original_file_name"],
                "language": segment["language"],
                "confidence": segment["confidence"],
            }
        )
    return serialized


def buscar_audio(query: str, k: int = 5, tool_context: ToolContext | None = None) -> dict:
    """Busca texto semánticamente en las transcripciones del corpus de audio.

    Args:
        query: Consulta del periodista en lenguaje natural.
        k: Cantidad máxima de segmentos a devolver, entre 1 y 20.

    Returns:
        Un objeto con resultados, fuente, timestamps y similitud.
    """
    if tool_context is not None:
        k = int(tool_context.state.get("max_results", k))
    if not query.strip():
        return {"status": "error", "error": "La consulta no puede estar vacía.", "results": []}
    if not 1 <= k <= 20:
        return {"status": "error", "error": "k debe estar entre 1 y 20.", "results": []}

    try:
        return {
            "status": "success",
            "modality": "text",
            "results": _serialize_results(
                get_search_engine().search_semantic(query, k=k),
                search_index="text",
                search_index_label="Índice de texto (transcripciones)",
            ),
        }
    except Exception as error:
        logger.exception("Text search failed")
        return {"status": "error", "error": str(error), "results": []}


def buscar_evento_acustico(
    query: str, k: int = 5, tool_context: ToolContext | None = None
) -> dict:
    """Busca eventos acústicos con CLAP a partir de una descripción textual.

    Úsala para aplausos, música, gritos, risas u otros sonidos que pueden no
    estar presentes en la transcripción.

    Args:
        query: Descripción textual del evento acústico.
        k: Cantidad máxima de segmentos a devolver, entre 1 y 20.

    Returns:
        Un objeto con resultados acústicos, fuente, timestamps y similitud.
    """
    if tool_context is not None:
        k = int(tool_context.state.get("max_results", k))
    if not query.strip():
        return {"status": "error", "error": "La consulta no puede estar vacía.", "results": []}
    if not 1 <= k <= 20:
        return {"status": "error", "error": "k debe estar entre 1 y 20.", "results": []}

    try:
        return {
            "status": "success",
            "modality": "audio",
            "results": _serialize_results(
                get_search_engine().search_audio_by_text(query, k=k),
                search_index="audio",
                search_index_label="Índice de audio (CLAP)",
            ),
        }
    except Exception as error:
        logger.exception("Audio-event search failed")
        return {"status": "error", "error": str(error), "results": []}


def obtener_info_segmento(segment_id: int) -> dict:
    """Obtiene los metadatos completos de un segmento recuperado previamente.

    Args:
        segment_id: Identificador estable del segmento.

    Returns:
        Los metadatos del segmento o un error si no existe.
    """
    try:
        result = get_search_engine().get_segment_info(segment_id)
    except Exception as error:
        logger.exception("Segment lookup failed")
        return {"status": "error", "error": str(error)}

    if result is None:
        return {"status": "error", "error": f"Segmento {segment_id} no encontrado."}
    return {"status": "success", "segment": result}


def obtener_clases_audio(segment_id: int) -> dict:
    """Obtiene las clases de AudioSet detectadas por YAMNet para un segmento.

    Úsala después de recuperar un segmento para identificar eventos acústicos
    estandarizados. Las etiquetas de YAMNet están en inglés y sus scores son
    probabilidades del clasificador, no porcentajes de similitud de CLAP.

    Args:
        segment_id: Identificador estable del segmento.

    Returns:
        Las clases acústicas detectadas, o información para reprocesar el dataset.
    """
    try:
        classes = get_search_engine().get_audio_classes(segment_id)
    except Exception as error:
        logger.exception("YAMNet class lookup failed")
        return {"status": "error", "error": str(error)}

    if classes is None:
        return {"status": "error", "error": f"Segmento {segment_id} no encontrado."}
    if not classes:
        return {
            "status": "not_available",
            "error": "El dataset no contiene clases YAMNet para este segmento. "
            "Reprocésalo habilitando yamnet en config/embeddings.toml.",
            "classes": [],
        }
    return {"status": "success", "classifier": "yamnet", "classes": classes}


def get_all_tools() -> list:
    """Return plain Python functions that ADK converts to FunctionTools.

    Use ``AGENT_MODALITY`` to restrict retrieval for modality ablations:
    - ``text``  → only the text/transcription search tool.
    - ``audio`` → only the CLAP cross-modal audio search tool.
    - ``both`` (default) → all tools.
    """
    modality = os.environ.get("AGENT_MODALITY", "both").lower()
    tools = [obtener_info_segmento, obtener_clases_audio]
    if modality in ("both", "text"):
        tools.append(buscar_audio)
    if modality in ("both", "audio"):
        tools.append(buscar_evento_acustico)
    return tools
