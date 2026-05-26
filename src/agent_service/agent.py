"""AudioAgent: LangChain agent for semantic audio search."""

import logging
import os

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.agent_service.search_engine import AudioSearchEngine
from src.agent_service.tools import get_all_tools, set_search_engine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un asistente especializado en búsqueda semántica de contenido de audio.

Tu función es ayudar a los usuarios a encontrar segmentos de audio relevantes
basándote en sus consultas en lenguaje natural.

INSTRUCCIONES:
1. Usa la herramienta 'buscar_audio' para realizar búsquedas semánticas cuando
   el usuario solicite buscar contenido
2. Analiza los resultados y presenta la información de manera clara y organizada
3. Si el usuario pregunta sobre detalles específicos de un segmento, usa
   'obtener_info_segmento'
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
  * Idioma"""


class AudioAgent:
    """Agente LangChain para búsqueda semántica de audio."""

    def __init__(
        self,
        dataset_path: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_iterations: int = 5,
    ):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations

        self._search_engine: AudioSearchEngine | None = None
        self._agent_executor: AgentExecutor | None = None

    def initialize(self):
        """Initialize the agent with search engine and LLM."""
        logger.info("Initializing AudioAgent...")

        # Initialize search engine
        self._search_engine = AudioSearchEngine(self.dataset_path)
        set_search_engine(self._search_engine)

        # Initialize LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent
        tools = get_all_tools()
        agent = create_openai_functions_agent(llm, tools, prompt)

        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            verbose=True,
        )

        logger.info(
            "AudioAgent initialized (model=%s, dataset=%s, segments=%d)",
            self.model_name,
            self.dataset_path,
            self._search_engine.total_segments,
        )

    @property
    def is_initialized(self) -> bool:
        return self._agent_executor is not None

    async def query(self, user_query: str, callbacks=None) -> str:
        """
        Process a user query through the agent.

        Args:
            user_query: Natural language query from the user
            callbacks: Optional LangChain callbacks

        Returns:
            Agent's response string
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        logger.info("Processing query: %s", user_query[:100])

        try:
            result = await self._agent_executor.ainvoke(
                {"input": user_query},
                config={"callbacks": callbacks} if callbacks else None,
            )
            response = result.get("output", "No pude procesar tu consulta.")
        except Exception as e:
            logger.error("Agent query failed: %s", e)
            response = f"Error procesando la consulta: {str(e)}"

        return response

    def query_sync(self, user_query: str, callbacks=None) -> str:
        """Synchronous version of query."""
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        logger.info("Processing query (sync): %s", user_query[:100])

        try:
            result = self._agent_executor.invoke(
                {"input": user_query},
                config={"callbacks": callbacks} if callbacks else None,
            )
            response = result.get("output", "No pude procesar tu consulta.")
        except Exception as e:
            logger.error("Agent query failed: %s", e)
            response = f"Error procesando la consulta: {str(e)}"

        return response
