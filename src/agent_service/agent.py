"""Agente LangChain para búsqueda semántica de audio"""

import logging
from typing import Any

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from .search_engine import AudioSearchEngine
from .tools import get_tools

logger = logging.getLogger(__name__)


class AudioAgent:
    """Agente LangChain especializado en búsqueda semántica de audio"""

    def __init__(
        self,
        dataset_path: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        openai_api_key: str | None = None,
    ):
        """
        Inicializa el agente de audio

        Args:
            dataset_path: Ruta al directorio del dataset
            model_name: Nombre del modelo de OpenAI a usar
            temperature: Temperatura para la generación (0.0 = determinista)
            openai_api_key: API key de OpenAI (opcional, puede usar variable de entorno)
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key

        self.search_engine: AudioSearchEngine | None = None
        self.agent_executor: AgentExecutor | None = None

    def initialize(self) -> None:
        """Inicializa el motor de búsqueda y el agente"""
        logger.info("Inicializando motor de búsqueda...")
        self.search_engine = AudioSearchEngine(self.dataset_path)

        logger.info("Inicializando agente LangChain...")
        tools = get_tools(self.search_engine)

        # Configurar modelo LLM
        llm_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if self.openai_api_key:
            llm_kwargs["api_key"] = self.openai_api_key

        llm = ChatOpenAI(**llm_kwargs)

        # Crear prompt del agente
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Eres un asistente especializado en búsqueda semántica de contenido de audio.

Tu función es ayudar a los usuarios a encontrar segmentos de audio relevantes basándote en sus consultas en lenguaje natural.

INSTRUCCIONES:
1. Usa la herramienta 'buscar_audio' para realizar búsquedas semánticas cuando el usuario solicite buscar contenido
2. Analiza los resultados y presenta la información de manera clara y organizada
3. Si el usuario pregunta sobre detalles específicos de un segmento, usa 'obtener_info_segmento'
4. Responde siempre en español
5. Sé conciso pero informativo
6. Si no encuentras resultados, sugiere alternativas o reformulación de la consulta

FORMATO DE RESPUESTAS:
- Menciona el número de resultados encontrados
- Para cada resultado relevante, muestra:
  * ID del segmento
  * Texto transcrito (resumen si es muy largo)
  * Similitud porcentual
  * Tiempo (inicio - fin)
  * Archivo de origen
  * Idioma

EJEMPLOS DE CONSULTAS:
- "Busca segmentos sobre política económica"
- "Encuentra audio donde se hable de tecnología"
- "Busca entrevistas relacionadas con ciencia"
""",
                ),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Crear agente
        agent = create_openai_functions_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        logger.info("Agente inicializado correctamente")

    async def query(self, user_query: str) -> str:
        """
        Ejecuta una consulta usando el agente

        Args:
            user_query: Consulta del usuario en lenguaje natural

        Returns:
            Respuesta del agente

        Raises:
            RuntimeError: Si el agente no ha sido inicializado
        """
        if self.agent_executor is None:
            raise RuntimeError(
                "Agente no inicializado. Llama a initialize() primero."
            )

        logger.info(f"Procesando consulta: {user_query}")
        try:
            result = await self.agent_executor.ainvoke({"input": user_query})
            return result["output"]
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            raise

    def query_sync(self, user_query: str) -> str:
        """
        Ejecuta una consulta usando el agente (versión síncrona)

        Args:
            user_query: Consulta del usuario en lenguaje natural

        Returns:
            Respuesta del agente

        Raises:
            RuntimeError: Si el agente no ha sido inicializado
        """
        if self.agent_executor is None:
            raise RuntimeError(
                "Agente no inicializado. Llama a initialize() primero."
            )

        logger.info(f"Procesando consulta: {user_query}")
        try:
            result = self.agent_executor.invoke({"input": user_query})
            return result["output"]
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            raise
