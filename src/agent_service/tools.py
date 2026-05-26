"""Agent tools for audio semantic search."""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool

from src.agent_service.search_engine import AudioSearchEngine

logger = logging.getLogger(__name__)

_search_engine: AudioSearchEngine | None = None


def set_search_engine(engine: AudioSearchEngine):
    """Set the global search engine instance for tools."""
    global _search_engine
    _search_engine = engine


def get_search_engine() -> AudioSearchEngine:
    """Get the global search engine instance."""
    if _search_engine is None:
        raise RuntimeError("Search engine not initialized. Call set_search_engine() first.")
    return _search_engine


@tool
def buscar_audio(
    query: Annotated[str, "Texto de búsqueda en lenguaje natural"],
    k: Annotated[int, "Número de resultados a retornar (default: 5)"] = 5,
) -> str:
    """
    Busca segmentos de audio usando búsqueda semántica.

    Esta herramienta permite buscar contenido en audios transcritos usando
    embeddings semánticos. Retorna los segmentos más relevantes según la consulta.

    Args:
        query: Texto de búsqueda en lenguaje natural (ej: "política económica",
               "entrevista sobre tecnología", "música de fondo")
        k: Número de resultados a retornar (por defecto 5)

    Returns:
        String JSON con los resultados de búsqueda, incluyendo:
        - segment_id: ID del segmento
        - text: Texto transcrito
        - similarity: Similitud con la consulta (0-1)
        - start_time: Tiempo de inicio en segundos
        - end_time: Tiempo de fin en segundos
        - original_file_name: Nombre del archivo de audio original
        - language: Idioma detectado
    """
    engine = get_search_engine()

    try:
        results = engine.search_semantic(query, k=k)
    except Exception as e:
        logger.error("Search failed: %s", e)
        return json.dumps({"error": str(e), "results": []})

    formatted = []
    for r in results:
        seg = r["segment"]
        formatted.append({
            "segment_id": seg["segment_id"],
            "text": seg["text"],
            "similarity": round(r["similarity"], 4),
            "similarity_percent": round(r["similarity"] * 100, 1),
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "duration": round(seg["end_time"] - seg["start_time"], 1),
            "original_file_name": seg["original_file_name"],
            "language": seg["language"],
            "confidence": seg["confidence"],
        })

    return json.dumps(formatted, ensure_ascii=False)


@tool
def obtener_info_segmento(
    segment_id: Annotated[int, "ID del segmento a consultar"],
) -> str:
    """
    Obtiene información detallada de un segmento específico.

    Usa esta herramienta cuando necesites información completa sobre un
    segmento de audio, incluyendo metadatos, transcripción completa y
    características del audio.

    Args:
        segment_id: ID numérico del segmento

    Returns:
        String JSON con toda la información del segmento
    """
    engine = get_search_engine()

    info = engine.get_segment_info(segment_id)
    if info is None:
        return json.dumps({"error": f"Segmento {segment_id} no encontrado"})

    return json.dumps(info, ensure_ascii=False)


def get_all_tools() -> list:
    """Return all available tools for the agent."""
    return [buscar_audio, obtener_info_segmento]
