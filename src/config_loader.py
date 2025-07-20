"""
Sistema de carga de configuración desde variables de entorno y archivos .env
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("⚠️  python-dotenv no instalado. Instala con: pip install python-dotenv")


@dataclass
class SystemConfig:
    """Configuración del sistema cargada desde variables de entorno"""
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_models: list = None
    
    # Sistema
    default_llm_backend: str = "auto"
    default_whisper_model: str = "base"
    default_text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Base de datos
    database_path: str = ":memory:"
    
    # Streamlit
    streamlit_port: int = 8501
    streamlit_theme: str = "light"
    
    # Configuración avanzada
    api_timeout: int = 30
    max_results_default: int = 10
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000
    
    # Segmentación
    segmentation_method: str = "silence"
    min_silence_len: int = 500
    silence_thresh: int = -40
    segment_duration: float = 10.0
    
    # Búsqueda
    index_type: str = "cosine"
    text_weight: float = 0.7
    audio_weight: float = 0.3
    
    # Desarrollo
    auto_reload: bool = False
    verbose_logging: bool = False
    sample_data_dir: str = "./sample_data"
    indices_dir: str = "./indices"
    
    def __post_init__(self):
        """Procesa configuraciones después de la inicialización"""
        if self.ollama_models is None:
            self.ollama_models = ["llama3", "mistral:7b"]
        
        # Calcular audio_weight automáticamente
        self.audio_weight = 1.0 - self.text_weight


class ConfigLoader:
    """Cargador de configuración desde múltiples fuentes"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Inicializa el cargador de configuración
        
        Args:
            env_file: Ruta al archivo .env (por defecto busca automáticamente)
        """
        self.config = None
        self.env_file = env_file
        self._load_env_file()
    
    def _load_env_file(self):
        """Carga archivo .env si está disponible"""
        if not DOTENV_AVAILABLE:
            logging.warning("🔧 Usando solo variables de entorno del sistema")
            return
        
        # Buscar archivo .env
        if self.env_file:
            env_path = Path(self.env_file)
        else:
            # Buscar en el directorio actual y directorios padre
            current_dir = Path.cwd()
            env_path = None
            
            for path in [current_dir] + list(current_dir.parents):
                potential_env = path / ".env"
                if potential_env.exists():
                    env_path = potential_env
                    break
        
        if env_path and env_path.exists():
            load_dotenv(env_path)
            logging.info(f"✅ Archivo .env cargado desde: {env_path}")
        else:
            logging.info("ℹ️  No se encontró archivo .env, usando variables del sistema")
    
    def load_config(self) -> SystemConfig:
        """
        Carga la configuración completa del sistema
        
        Returns:
            SystemConfig con toda la configuración
        """
        if self.config is None:
            self.config = self._build_config()
        return self.config
    
    def _build_config(self) -> SystemConfig:
        """Construye la configuración desde variables de entorno"""
        return SystemConfig(
            # API Keys
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            
            # Ollama
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_models=self._parse_list(os.getenv("OLLAMA_MODELS", "llama3,mistral:7b")),
            
            # Sistema
            default_llm_backend=os.getenv("DEFAULT_LLM_BACKEND", "auto"),
            default_whisper_model=os.getenv("DEFAULT_WHISPER_MODEL", "base"),
            default_text_model=os.getenv("DEFAULT_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            
            # API
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=self._parse_int(os.getenv("API_PORT", "8000")),
            debug_mode=self._parse_bool(os.getenv("DEBUG_MODE", "false")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            
            # Base de datos
            database_path=os.getenv("DATABASE_PATH", ":memory:"),
            
            # Streamlit
            streamlit_port=self._parse_int(os.getenv("STREAMLIT_PORT", "8501")),
            streamlit_theme=os.getenv("STREAMLIT_THEME", "light"),
            
            # Configuración avanzada
            api_timeout=self._parse_int(os.getenv("API_TIMEOUT", "30")),
            max_results_default=self._parse_int(os.getenv("MAX_RESULTS_DEFAULT", "10")),
            llm_temperature=self._parse_float(os.getenv("LLM_TEMPERATURE", "0.1")),
            llm_max_tokens=self._parse_int(os.getenv("LLM_MAX_TOKENS", "1000")),
            
            # Segmentación
            segmentation_method=os.getenv("SEGMENTATION_METHOD", "silence"),
            min_silence_len=self._parse_int(os.getenv("MIN_SILENCE_LEN", "500")),
            silence_thresh=self._parse_int(os.getenv("SILENCE_THRESH", "-40")),
            segment_duration=self._parse_float(os.getenv("SEGMENT_DURATION", "10.0")),
            
            # Búsqueda
            index_type=os.getenv("INDEX_TYPE", "cosine"),
            text_weight=self._parse_float(os.getenv("TEXT_WEIGHT", "0.7")),
            
            # Desarrollo
            auto_reload=self._parse_bool(os.getenv("AUTO_RELOAD", "false")),
            verbose_logging=self._parse_bool(os.getenv("VERBOSE_LOGGING", "false")),
            sample_data_dir=os.getenv("SAMPLE_DATA_DIR", "./sample_data"),
            indices_dir=os.getenv("INDICES_DIR", "./indices")
        )
    
    def _parse_bool(self, value: str) -> bool:
        """Convierte string a boolean"""
        return value.lower() in ("true", "1", "yes", "on")
    
    def _parse_int(self, value: str) -> int:
        """Convierte string a integer"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def _parse_float(self, value: str) -> float:
        """Convierte string a float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _parse_list(self, value: str) -> list:
        """Convierte string separado por comas a lista"""
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    
    def get_llm_config_dict(self) -> Dict[str, Any]:
        """
        Obtiene configuración específica para LLM en formato diccionario
        
        Returns:
            Diccionario con configuración para LLM
        """
        config = self.load_config()
        return {
            'whisper_model': config.default_whisper_model,
            'text_embedding_model': config.default_text_model,
            'index_type': config.index_type,
            'segmentation_method': config.segmentation_method,
            'min_silence_len': config.min_silence_len,
            'silence_thresh': config.silence_thresh,
            'segment_duration': config.segment_duration,
            'top_k_results': config.max_results_default,
            'text_weight': config.text_weight,
            'audio_weight': config.audio_weight
        }
    
    def print_config_summary(self):
        """Imprime un resumen de la configuración cargada"""
        config = self.load_config()
        
        logging.info("🔧 Configuración del Sistema")
        logging.info("=" * 40)
        
        # API Keys disponibles
        logging.info("\n📡 API Keys:")
        logging.info(f"  OpenAI: {'✅' if config.openai_api_key else '❌'}")
        logging.info(f"  Anthropic: {'✅' if config.anthropic_api_key else '❌'}")
        logging.info(f"  Google: {'✅' if config.google_api_key else '❌'}")
        
        # Configuración principal
        logging.info(f"\n🤖 LLM Backend: {config.default_llm_backend}")
        logging.info(f"🎤 Whisper Model: {config.default_whisper_model}")
        logging.info(f"📝 Text Model: {config.default_text_model}")
        
        # API
        logging.info(f"\n🌐 API: {config.api_host}:{config.api_port}")
        logging.info(f"🐛 Debug: {config.debug_mode}")
        logging.info(f"📊 Log Level: {config.log_level}")
        
        # Ollama
        logging.info(f"\n🦙 Ollama: {config.ollama_base_url}")
        logging.info(f"📦 Modelos: {', '.join(config.ollama_models)}")
    
    def create_env_file(self, output_path: str = ".env"):
        """
        Crea un archivo .env con valores por defecto
        
        Args:
            output_path: Ruta donde crear el archivo .env
        """
        env_content = """# Configuración del Sistema de Búsqueda Semántica
# Generado automáticamente - Edita según tus necesidades

# ================================
# API Keys (¡IMPORTANTE!)
# ================================
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# ================================
# Configuración Principal
# ================================
DEFAULT_LLM_BACKEND=auto
DEFAULT_WHISPER_MODEL=base

# ================================
# API REST
# ================================
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=false

# ================================
# Configuración Avanzada
# ================================
LLM_TEMPERATURE=0.1
MAX_RESULTS_DEFAULT=10
TEXT_WEIGHT=0.7
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        logging.info(f"✅ Archivo .env creado en: {output_path}")
        logging.info("🔑 ¡No olvides agregar tus API keys!")


# Instancia global del cargador de configuración
config_loader = ConfigLoader()


def get_config() -> SystemConfig:
    """
    Función conveniente para obtener la configuración del sistema
    
    Returns:
        SystemConfig con toda la configuración
    """
    return config_loader.load_config()


def setup_logging():
    """Configura el sistema de logging basado en la configuración"""
    config = get_config()
    
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Configurar el handler para que escriba en stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Configurar el logger raíz
    logging.basicConfig(level=level, handlers=[handler])
    
    if config.verbose_logging:
        # Habilitar logs detallados para librerías específicas
        logging.getLogger("sentence_transformers").setLevel(logging.INFO)
        logging.getLogger("transformers").setLevel(logging.INFO)


# Auto-configurar logging al importar
setup_logging()


if __name__ == "__main__":
    # Ejemplo de uso
    loader = ConfigLoader()
    
    # Mostrar resumen
    loader.print_config_summary()
    
    # Crear archivo .env de ejemplo si no existe
    if not Path(".env").exists():
        print("\n📝 Creando archivo .env de ejemplo...")
        loader.create_env_file()
    
    # Mostrar configuración para LLM
    print("\n🤖 Configuración para SemanticSearchEngine:")
    llm_config = loader.get_llm_config_dict()
    for key, value in llm_config.items():
        print(f"  {key}: {value}")