"""
Configuración centralizada de modelos para el sistema de búsqueda semántica de audio.
Permite seleccionar entre diferentes modelos para cada tarea específica.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos disponibles"""
    SPEECH_TO_TEXT = "speech_to_text"
    AUDIO_EMBEDDING = "audio_embedding"
    AUDIO_EVENT_DETECTION = "audio_event_detection"
    TEXT_EMBEDDING = "text_embedding"


class SpeechToTextModel(Enum):
    """Modelos disponibles para conversión speech-to-text"""
    WHISPER_TINY = "whisper_tiny"
    WHISPER_BASE = "whisper_base"
    WHISPER_SMALL = "whisper_small"
    WHISPER_MEDIUM = "whisper_medium"
    WHISPER_LARGE = "whisper_large"
    WHISPER_LARGE_V2 = "whisper_large_v2"
    WHISPER_LARGE_V3 = "whisper_large_v3"


class AudioEmbeddingModel(Enum):
    """Modelos disponibles para embeddings de audio"""
    YAMNET = "yamnet"
    CLAP_LAION = "clap_laion"
    CLAP_MUSIC = "clap_music"


class AudioEventDetectionModel(Enum):
    """Modelos disponibles para detección de eventos de audio"""
    YAMNET = "yamnet"
    CLAP_LAION = "clap_laion"


@dataclass
class WhisperConfig:
    """Configuración específica para modelos Whisper"""
    model_name: str = "base"
    device: str = "auto"  # "auto", "cpu", "cuda"
    language: str | None = None  # None para detección automática
    temperature: float = 0.0
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4


@dataclass
class YAMNetConfig:
    """Configuración específica para YAMNet"""
    model_url: str = "https://tfhub.dev/google/yamnet/1"
    sample_rate: int = 16000
    hop_length: int = 512
    frame_length: int = 1024


@dataclass
class CLAPConfig:
    """Configuración específica para modelos CLAP"""
    model_name: str = "laion/clap-htsat-unfused"  # Modelo por defecto
    device: str = "auto"  # "auto", "cpu", "cuda"
    enable_fusion: bool = False
    amodel: str = "HTSAT-tiny"  # Arquitectura de audio
    tmodel: str = "roberta"     # Arquitectura de texto
    cache_dir: str | None = None
    # Parámetros de chunking con overlapping
    chunk_duration: float = 6.0  # Duración del chunk en segundos
    overlap_duration: float = 2.0  # Solapamiento entre chunks en segundos
    hop_duration: float | None = None  # Paso entre chunks (chunk_duration - overlap_duration si None)


@dataclass
class ModelsConfiguration:
    """Configuración principal de todos los modelos"""

    # Modelos activos por defecto
    default_speech_to_text: SpeechToTextModel = SpeechToTextModel.WHISPER_BASE
    default_audio_embedding: AudioEmbeddingModel = AudioEmbeddingModel.CLAP_LAION
    default_audio_event_detection: AudioEventDetectionModel = AudioEventDetectionModel.CLAP_LAION

    # Configuraciones específicas
    whisper_config: WhisperConfig = field(default_factory=WhisperConfig)
    yamnet_config: YAMNetConfig = field(default_factory=YAMNetConfig)
    clap_config: CLAPConfig = field(default_factory=CLAPConfig)

    # Configuraciones de fallback
    fallback_models: dict[ModelType, list[str]] = field(default_factory=lambda: {
        ModelType.SPEECH_TO_TEXT: ["whisper_base", "whisper_tiny"],
        ModelType.AUDIO_EMBEDDING: ["clap_laion", "clap_music", "yamnet"],
        ModelType.AUDIO_EVENT_DETECTION: ["clap_laion", "yamnet"],
    })

    # Configuraciones generales
    cache_models: bool = True
    model_cache_dir: str = "./models_cache"
    max_memory_usage: str | None = None  # "4GB", "8GB", etc.

    def get_whisper_model_name(self) -> str:
        """Obtiene el nombre del modelo Whisper para cargar"""
        model_mapping = {
            SpeechToTextModel.WHISPER_TINY: "tiny",
            SpeechToTextModel.WHISPER_BASE: "base",
            SpeechToTextModel.WHISPER_SMALL: "small",
            SpeechToTextModel.WHISPER_MEDIUM: "medium",
            SpeechToTextModel.WHISPER_LARGE: "large",
            SpeechToTextModel.WHISPER_LARGE_V2: "large-v2",
            SpeechToTextModel.WHISPER_LARGE_V3: "large-v3",
        }
        return model_mapping.get(self.default_speech_to_text, "base")

    def get_clap_model_name(self) -> str:
        """Obtiene el nombre del modelo CLAP para cargar"""
        clap_models = {
            AudioEmbeddingModel.CLAP_LAION: "laion/clap-htsat-unfused",
            AudioEmbeddingModel.CLAP_MUSIC: "laion/clap-htsat-fused",
        }
        return clap_models.get(self.default_audio_embedding, "laion/clap-htsat-unfused")

    def is_model_available(self, model_type: ModelType, model_name: str) -> bool:
        """Verifica si un modelo específico está disponible"""
        try:
            if model_type == ModelType.SPEECH_TO_TEXT and "whisper" in model_name:
                import whisper
                return True
            if model_type == ModelType.AUDIO_EMBEDDING or model_type == ModelType.AUDIO_EVENT_DETECTION:
                if model_name == "yamnet":
                    import tensorflow_hub
                    return True
                if "clap" in model_name:
                    import laion_clap
                    return True
        except ImportError:
            return False
        return False

    def get_available_models(self, model_type: ModelType) -> list[str]:
        """Obtiene lista de modelos disponibles para un tipo específico"""
        available = []

        if model_type == ModelType.SPEECH_TO_TEXT:
            try:
                import whisper
                available.extend([
                    "whisper_tiny", "whisper_base", "whisper_small",
                    "whisper_medium", "whisper_large", "whisper_large_v2", "whisper_large_v3"
                ])
            except ImportError:
                logger.warning("Whisper no disponible")

        elif model_type == ModelType.AUDIO_EMBEDDING:
            try:
                import tensorflow_hub
                available.append("yamnet")
            except ImportError:
                logger.warning("TensorFlow Hub no disponible")

            try:
                import laion_clap
                available.extend(["clap_laion", "clap_music"])
            except ImportError:
                logger.warning("LAION CLAP no disponible")

        elif model_type == ModelType.AUDIO_EVENT_DETECTION:
            # Similar a audio embedding
            try:
                import tensorflow_hub
                available.append("yamnet")
            except ImportError:
                logger.warning("TensorFlow Hub no disponible")

            try:
                import laion_clap
                available.extend(["clap_laion"])
            except ImportError:
                logger.warning("LAION CLAP no disponible")

        return available

    def validate_configuration(self) -> dict[str, Any]:
        """Valida la configuración actual y retorna un reporte"""
        validation_report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "available_models": {},
            "recommended_changes": []
        }

        # Validar cada tipo de modelo
        for model_type in ModelType:
            available = self.get_available_models(model_type)
            validation_report["available_models"][model_type.value] = available

            if not available:
                validation_report["errors"].append(
                    f"No hay modelos disponibles para {model_type.value}"
                )
                validation_report["valid"] = False

        # Validar modelo de speech-to-text seleccionado
        if not self.is_model_available(ModelType.SPEECH_TO_TEXT, self.default_speech_to_text.value):
            validation_report["errors"].append(
                f"Modelo speech-to-text seleccionado no disponible: {self.default_speech_to_text.value}"
            )
            validation_report["valid"] = False

        # Validar modelo de audio embedding seleccionado
        if not self.is_model_available(ModelType.AUDIO_EMBEDDING, self.default_audio_embedding.value):
            validation_report["errors"].append(
                f"Modelo audio embedding seleccionado no disponible: {self.default_audio_embedding.value}"
            )
            validation_report["valid"] = False

        # Recomendaciones
        audio_models = validation_report["available_models"].get(ModelType.AUDIO_EMBEDDING.value, [])
        if not any("clap" in model for model in audio_models):
            validation_report["recommended_changes"].append(
                "Considera instalar LAION CLAP para mejores embeddings de audio: poetry install"
            )
        if "yamnet" not in audio_models:
            validation_report["recommended_changes"].append(
                "Considera instalar YAMNet para embeddings de audio: poetry install --extras yamnet"
            )

        return validation_report


class ModelsConfigLoader:
    """Cargador de configuración de modelos desde variables de entorno y archivos"""

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file
        self._config = None

    def load_config(self) -> ModelsConfiguration:
        """Carga la configuración de modelos"""
        if self._config is None:
            self._config = self._build_config_from_env()
        return self._config

    def _build_config_from_env(self) -> ModelsConfiguration:
        """Construye configuración desde variables de entorno"""

        # Modelos por defecto desde variables de entorno
        speech_to_text = os.getenv("DEFAULT_SPEECH_TO_TEXT_MODEL", "whisper_base")
        audio_embedding = os.getenv("DEFAULT_AUDIO_EMBEDDING_MODEL", "clap_laion")
        audio_event_detection = os.getenv("DEFAULT_AUDIO_EVENT_DETECTION_MODEL", "clap_laion")

        # Mapear strings a enums
        try:
            speech_model = SpeechToTextModel(speech_to_text)
        except ValueError:
            logger.warning(f"Modelo speech-to-text inválido: {speech_to_text}, usando whisper_base")
            speech_model = SpeechToTextModel.WHISPER_BASE

        try:
            embedding_model = AudioEmbeddingModel(audio_embedding)
        except ValueError:
            logger.warning(f"Modelo audio embedding inválido: {audio_embedding}, usando clap_laion")
            embedding_model = AudioEmbeddingModel.CLAP_LAION

        try:
            event_model = AudioEventDetectionModel(audio_event_detection)
        except ValueError:
            logger.warning(f"Modelo event detection inválido: {audio_event_detection}, usando clap_laion")
            event_model = AudioEventDetectionModel.CLAP_LAION

        # Configuraciones específicas
        whisper_config = WhisperConfig(
            model_name=os.getenv("WHISPER_MODEL_SIZE", "base"),
            device=os.getenv("WHISPER_DEVICE", "auto"),
            language=os.getenv("WHISPER_LANGUAGE"),
            temperature=float(os.getenv("WHISPER_TEMPERATURE", "0.0")),
        )

        yamnet_config = YAMNetConfig(
            model_url=os.getenv("YAMNET_MODEL_URL", "https://tfhub.dev/google/yamnet/1"),
            sample_rate=int(os.getenv("YAMNET_SAMPLE_RATE", "16000")),
        )

        clap_config = CLAPConfig(
            model_name=os.getenv("CLAP_MODEL_NAME", "laion/clap-htsat-unfused"),
            device=os.getenv("CLAP_DEVICE", "auto"),
            enable_fusion=os.getenv("CLAP_ENABLE_FUSION", "false").lower() == "true",
            chunk_duration=float(os.getenv("CLAP_CHUNK_DURATION", "6.0")),
            overlap_duration=float(os.getenv("CLAP_OVERLAP_DURATION", "2.0")),
            hop_duration=float(os.getenv("CLAP_HOP_DURATION", "0")) or None,
        )

        return ModelsConfiguration(
            default_speech_to_text=speech_model,
            default_audio_embedding=embedding_model,
            default_audio_event_detection=event_model,
            whisper_config=whisper_config,
            yamnet_config=yamnet_config,
            clap_config=clap_config,
            cache_models=os.getenv("CACHE_MODELS", "true").lower() == "true",
            model_cache_dir=os.getenv("MODEL_CACHE_DIR", "./models_cache"),
        )

    def print_config_summary(self):
        """Imprime un resumen de la configuración de modelos"""
        config = self.load_config()

        logger.info("🤖 Configuración de Modelos")
        logger.info("=" * 50)

        logger.info(f"\n🎤 Speech-to-Text: {config.default_speech_to_text.value}")
        logger.info(f"   Modelo Whisper: {config.get_whisper_model_name()}")
        logger.info(f"   Device: {config.whisper_config.device}")

        logger.info(f"\n🔊 Audio Embedding: {config.default_audio_embedding.value}")
        if config.default_audio_embedding == AudioEmbeddingModel.YAMNET:
            logger.info(f"   URL: {config.yamnet_config.model_url}")
        elif "clap" in config.default_audio_embedding.value:
            logger.info(f"   Modelo: {config.get_clap_model_name()}")

        logger.info(f"\n🎯 Event Detection: {config.default_audio_event_detection.value}")

        logger.info(f"\n💾 Cache: {config.cache_models}")
        logger.info(f"📁 Cache Dir: {config.model_cache_dir}")

        # Validación
        validation = config.validate_configuration()
        if validation["valid"]:
            logger.info("\n✅ Configuración válida")
        else:
            logger.warning("\n⚠️  Problemas en la configuración:")
            for error in validation["errors"]:
                logger.error(f"   ❌ {error}")

        if validation["recommended_changes"]:
            logger.info("\n💡 Recomendaciones:")
            for rec in validation["recommended_changes"]:
                logger.info(f"   📌 {rec}")

    def create_env_template(self, output_path: str = ".env.models"):
        """Crea un archivo de template para configuración de modelos"""
        template = """# ================================
# Configuración de Modelos de IA
# ================================

# ================================
# Modelos por Defecto
# ================================
# Opciones para speech-to-text: whisper_tiny, whisper_base, whisper_small, whisper_medium, whisper_large, whisper_large_v2, whisper_large_v3
DEFAULT_SPEECH_TO_TEXT_MODEL=whisper_base

# Opciones para audio embedding: clap_laion (recomendado), clap_music, yamnet
DEFAULT_AUDIO_EMBEDDING_MODEL=clap_laion

# Opciones para detección de eventos: clap_laion (recomendado), yamnet
DEFAULT_AUDIO_EVENT_DETECTION_MODEL=clap_laion

# ================================
# Configuración Whisper
# ================================
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=auto
WHISPER_LANGUAGE=
WHISPER_TEMPERATURE=0.0

# ================================
# Configuración YAMNet
# ================================
YAMNET_MODEL_URL=https://tfhub.dev/google/yamnet/1
YAMNET_SAMPLE_RATE=16000

# ================================
# Configuración CLAP
# ================================
CLAP_MODEL_NAME=laion/clap-htsat-unfused
CLAP_DEVICE=auto
CLAP_ENABLE_FUSION=false
# Parámetros de chunking con overlapping (en segundos)
CLAP_CHUNK_DURATION=6.0
CLAP_OVERLAP_DURATION=2.0
CLAP_HOP_DURATION=4.0

# ================================
# Configuraciones Generales
# ================================
CACHE_MODELS=true
MODEL_CACHE_DIR=./models_cache
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)

        logger.info(f"✅ Template de configuración de modelos creado: {output_path}")


# Instancia global del cargador
models_config_loader = ModelsConfigLoader()


def get_models_config() -> ModelsConfiguration:
    """Función conveniente para obtener la configuración de modelos"""
    return models_config_loader.load_config()


def get_available_models_info() -> dict[str, list[str]]:
    """Obtiene información de modelos disponibles"""
    config = get_models_config()
    return {
        model_type.value: config.get_available_models(model_type)
        for model_type in ModelType
    }


if __name__ == "__main__":
    # Ejemplo de uso y diagnóstico
    loader = ModelsConfigLoader()

    print("🤖 Sistema de Configuración de Modelos")
    print("=" * 50)

    # Mostrar configuración actual
    loader.print_config_summary()

    # Mostrar modelos disponibles
    print("\n📋 Modelos Disponibles:")
    available = get_available_models_info()
    for model_type, models in available.items():
        print(f"  {model_type}: {', '.join(models) if models else 'Ninguno disponible'}")

    # Crear template si no existe
    if not os.path.exists(".env.models"):
        print("\n📝 Creando template de configuración...")
        loader.create_env_template()
