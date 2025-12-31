"""Servicio FastAPI con agente LangChain para búsqueda semántica de audio"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import AudioAgent

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Variable global para el agente
audio_agent: AudioAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor de ciclo de vida de la aplicación"""
    global audio_agent

    # Inicialización al arrancar
    logger.info("Iniciando servicio de agente de audio...")

    # Obtener configuración
    dataset_path = os.getenv("DATASET_PATH", "./dataset")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        logger.warning(
            "OPENAI_API_KEY no configurada. "
            "El servicio puede no funcionar correctamente."
        )

    # Verificar que existe el dataset
    if not Path(dataset_path).exists():
        logger.error(f"Dataset no encontrado en: {dataset_path}")
        logger.error(
            "Genera un dataset primero con: "
            "poetry run python src/simple_dataset_pipeline.py --input data/ --output ./dataset"
        )
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")

    try:
        # Inicializar agente
        audio_agent = AudioAgent(
            dataset_path=dataset_path,
            model_name=model_name,
            openai_api_key=openai_api_key,
        )
        audio_agent.initialize()
        logger.info("Servicio de agente iniciado correctamente")
    except Exception as e:
        logger.error(f"Error inicializando agente: {e}")
        raise

    yield

    # Limpieza al detener
    logger.info("Deteniendo servicio de agente...")
    audio_agent = None


# Crear aplicación FastAPI
app = FastAPI(
    title="Audio Semantic Search Agent API",
    description=(
        "API REST con agente LangChain para búsqueda semántica de contenido de audio. "
        "Proporciona búsqueda inteligente usando embeddings semánticos y modelos de lenguaje."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelos Pydantic
class QueryRequest(BaseModel):
    """Request model para consultas"""

    query: str = Field(
        ...,
        description="Consulta en lenguaje natural para buscar en el contenido de audio",
        examples=["Busca segmentos sobre política económica"],
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número máximo de resultados a retornar",
    )


class QueryResponse(BaseModel):
    """Response model para consultas"""

    response: str = Field(..., description="Respuesta del agente")
    query: str = Field(..., description="Consulta original")


class HealthResponse(BaseModel):
    """Response model para health check"""

    status: str
    dataset_path: str
    model_name: str
    agent_initialized: bool


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz"""
    return {
        "service": "Audio Semantic Search Agent API",
        "version": "1.0.0",
        "description": (
            "Servicio con agente LangChain para búsqueda semántica de audio"
        ),
    }


@app.get("/health", tags=["General"], response_model=HealthResponse)
async def health():
    """
    Health check endpoint

    Retorna el estado del servicio y configuración
    """
    global audio_agent

    return HealthResponse(
        status="healthy" if audio_agent is not None else "not_initialized",
        dataset_path=os.getenv("DATASET_PATH", "./dataset"),
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        agent_initialized=audio_agent is not None,
    )


@app.post("/query", tags=["Search"], response_model=QueryResponse)
async def query_audio(request: QueryRequest):
    """
    Ejecuta una consulta usando el agente LangChain

    El agente interpreta la consulta en lenguaje natural y utiliza herramientas
    de búsqueda semántica para encontrar segmentos de audio relevantes.

    Ejemplos de consultas:
    - "Busca segmentos sobre política económica"
    - "Encuentra audio donde se hable de tecnología"
    - "Busca entrevistas relacionadas con ciencia"
    - "¿Qué segmentos mencionan inteligencia artificial?"

    Args:
        request: Request con la consulta y parámetros opcionales

    Returns:
        Respuesta del agente con los resultados de búsqueda formateados

    Raises:
        HTTPException: Si el agente no está inicializado o hay un error
    """
    global audio_agent

    if audio_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agente no inicializado. Verifica la configuración del servicio.",
        )

    try:
        # Construir consulta incluyendo max_results si es necesario
        query = request.query
        if request.max_results != 5:
            query += f" (retorna máximo {request.max_results} resultados)"

        response = await audio_agent.query(query)

        return QueryResponse(response=response, query=request.query)

    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando consulta: {str(e)}",
        )


@app.get("/query/sync", tags=["Search"], response_model=QueryResponse)
async def query_audio_sync(query: str, max_results: int = 5):
    """
    Versión síncrona del endpoint de consulta (para compatibilidad)

    Args:
        query: Consulta en lenguaje natural
        max_results: Número máximo de resultados (1-20)

    Returns:
        Respuesta del agente

    Raises:
        HTTPException: Si el agente no está inicializado o hay un error
    """
    global audio_agent

    if audio_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agente no inicializado. Verifica la configuración del servicio.",
        )

    if not 1 <= max_results <= 20:
        raise HTTPException(
            status_code=400, detail="max_results debe estar entre 1 y 20"
        )

    try:
        if max_results != 5:
            query += f" (retorna máximo {max_results} resultados)"

        response = audio_agent.query_sync(query)

        return QueryResponse(response=response, query=query)

    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando consulta: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
