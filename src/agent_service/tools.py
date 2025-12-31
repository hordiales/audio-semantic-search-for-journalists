"""Herramientas (tools) para el agente LangChain"""

import logging
from typing import Annotated

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


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
    # Esta función será modificada dinámicamente con el motor de búsqueda
    # cuando se inicialice el agente
    raise RuntimeError(
        "Motor de búsqueda no inicializado. "
        "Usa AudioAgent.initialize() primero."
    )


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
        segment_id: ID numérico del segmento (se obtiene de los resultados
                   de búsqueda)

    Returns:
        String JSON con toda la información del segmento
    """
    # Esta función será modificada dinámicamente con el motor de búsqueda
    # cuando se inicialice el agente
    raise RuntimeError(
        "Motor de búsqueda no inicializado. "
        "Usa AudioAgent.initialize() primero."
    )


def get_tools(search_engine) -> list:
    """
    Obtiene las herramientas configuradas con el motor de búsqueda

    Args:
        search_engine: Instancia de AudioSearchEngine

    Returns:
        Lista de herramientas LangChain configuradas
    """
    import json

    @tool
    def buscar_audio_impl(
        query: Annotated[str, "Texto de búsqueda en lenguaje natural"],
        k: Annotated[int, "Número de resultados a retornar (default: 5)"] = 5,
    ) -> str:
        """Busca segmentos de audio usando búsqueda semántica."""
        try:
            results = search_engine.search_semantic(query, k)
            # Formatear resultados para el agente
            formatted_results = []
            for result in results:
                segment = result["segment"]
                formatted_results.append(
                    {
                        "segment_id": segment.get("segment_id", ""),
                        "text": segment.get("text", ""),
                        "similarity": round(result["similarity"], 4),
                        "similarity_percent": round(result["similarity"] * 100, 1),
                        "start_time": segment.get("start_time", 0),
                        "end_time": segment.get("end_time", 0),
                        "duration": segment.get("end_time", 0)
                        - segment.get("start_time", 0),
                        "original_file_name": segment.get(
                            "original_file_name", segment.get("source_file", "N/A")
                        ),
                        "language": segment.get("language", "N/A"),
                        "confidence": segment.get("confidence"),
                    }
                )
            return json.dumps(formatted_results, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def obtener_info_segmento_impl(
        segment_id: Annotated[int, "ID del segmento a consultar"],
    ) -> str:
        """Obtiene información detallada de un segmento específico."""
        try:
            segment_info = search_engine.get_segment_info(segment_id)
            if segment_info is None:
                return json.dumps({"error": f"Segmento {segment_id} no encontrado"})
            return json.dumps(segment_info, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error obteniendo info del segmento: {e}")
            return json.dumps({"error": str(e)})

    return [buscar_audio_impl, obtener_info_segmento_impl]
