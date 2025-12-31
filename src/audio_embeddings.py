from abc import ABC, abstractmethod
import os
import warnings

import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

# TensorFlow imports (opcional - solo para YAMNet)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    hub = None
    logger.warning("⚠️  TensorFlow no disponible. YAMNet no estará disponible.")
    logger.warning("   Para usar YAMNet: poetry install --extras yamnet")

try:
    from .models_config import AudioEmbeddingModel, YAMNetConfig, get_models_config
except ImportError:
    # Fallback para ejecución desde diferentes directorios
    try:
        from models_config import AudioEmbeddingModel, YAMNetConfig, get_models_config
    except ImportError:
        # Definiciones mínimas para evitar errores
        def get_models_config():
            return None

        class YAMNetConfig:
            def __init__(self):
                self.model_url = "https://tfhub.dev/google/yamnet/1"
                self.sample_rate = 16000

        class AudioEmbeddingModel:
            YAMNET = "yamnet"
            CLAP_LAION = "clap_laion"
            CLAP_MUSIC = "clap_music"


class BaseAudioEmbedding(ABC):
    """
    Clase base abstracta para generadores de embeddings de audio.
    Todos los modelos de embeddings deben heredar de esta clase e implementar sus métodos.
    """

    def __init__(self):
        """Inicializa el generador de embeddings de audio"""
        self.model = None
        self.embedding_dim = None
        self.model_name = None
        self.sample_rate = None

    @abstractmethod
    def _load_model(self):
        """Carga el modelo de embeddings"""

    @abstractmethod
    def preprocess_audio(self, audio_path: str, target_sr: int | None = None) -> np.ndarray:
        """
        Preprocesa un archivo de audio según los requisitos del modelo

        Args:
            audio_path: Ruta al archivo de audio
            target_sr: Frecuencia de muestreo objetivo (usa configuración si None)

        Returns:
            Audio preprocesado como array numpy
        """

    @abstractmethod
    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """
        Genera embedding para un archivo de audio

        Args:
            audio_path: Ruta al archivo de audio

        Returns:
            Array numpy con el embedding del audio
        """

    def generate_embeddings_batch(self, audio_paths: list[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de archivos de audio

        Args:
            audio_paths: Lista de rutas a archivos de audio

        Returns:
            Array numpy con embeddings de todos los archivos
        """
        embeddings = []

        for i, audio_path in enumerate(audio_paths):
            if os.environ.get('MCP_MODE') != '1':
                logger.info(f"Procesando audio {i+1}/{len(audio_paths)}: {audio_path}")
            embedding = self.generate_embedding(audio_path)
            embeddings.append(embedding)

        return np.array(embeddings)

    def process_transcription_dataframe(self, df: pd.DataFrame,
                                      temp_audio_dir: str = "temp_audio") -> pd.DataFrame:
        """
        Procesa un DataFrame con transcripciones y genera embeddings de audio

        Args:
            df: DataFrame con información de segmentos de audio
            temp_audio_dir: Directorio temporal para archivos de audio

        Returns:
            DataFrame con embeddings de audio añadidos
        """
        if not os.path.exists(temp_audio_dir):
            os.makedirs(temp_audio_dir)

        # Generar archivos de audio temporales para cada segmento
        audio_paths = []
        valid_indices = []

        for idx, row in df.iterrows():
            try:
                # Crear archivo temporal para el segmento
                temp_audio_path = os.path.join(temp_audio_dir, f"segment_{idx}.wav")

                # Cargar audio original y extraer segmento
                audio, sr = librosa.load(row['source_file'], sr=self.sample_rate, mono=True)
                start_sample = int(row['start_time'] * sr)
                end_sample = int(row['end_time'] * sr)

                segment_audio = audio[start_sample:end_sample]

                # Guardar segmento
                import soundfile as sf
                sf.write(temp_audio_path, segment_audio, sr)

                audio_paths.append(temp_audio_path)
                valid_indices.append(idx)

            except Exception as e:
                logger.error(f"Error procesando segmento {idx}: {e}")
                continue

        if not audio_paths:
            logger.error("No se pudieron procesar segmentos de audio")
            return df

        # Generar embeddings
        if os.environ.get('MCP_MODE') != '1':
            logger.info(f"Generando embeddings de audio para {len(audio_paths)} segmentos...")
        embeddings = self.generate_embeddings_batch(audio_paths)

        # Añadir embeddings al DataFrame
        result_df = df.copy()
        result_df['audio_embedding'] = None

        for i, idx in enumerate(valid_indices):
            result_df.at[idx, 'audio_embedding'] = embeddings[i].tolist()

        result_df['audio_embedding_model'] = self.model_name
        result_df['audio_embedding_dim'] = self.embedding_dim

        # Limpiar archivos temporales
        for audio_path in audio_paths:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # Filtrar filas que tengan embeddings
        result_df = result_df[result_df['audio_embedding'].notna()]

        return result_df

    def calculate_similarity(self, query_embedding: np.ndarray,
                           audio_embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula la similitud coseno entre embeddings de audio

        Args:
            query_embedding: Embedding de consulta
            audio_embeddings: Array con embeddings de audio

        Returns:
            Array con las similitudes
        """
        # Normalizar embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        audio_norms = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)

        # Calcular similitud coseno
        similarities = np.dot(audio_norms, query_norm)

        return similarities

    def search_similar_audio(self, query_audio_path: str, df: pd.DataFrame,
                           top_k: int = 5) -> pd.DataFrame:
        """
        Busca audios similares a una consulta de audio

        Args:
            query_audio_path: Ruta al archivo de audio de consulta
            df: DataFrame con embeddings de audio
            top_k: Número de resultados a retornar

        Returns:
            DataFrame con los resultados más similares
        """
        # Generar embedding de la consulta
        query_embedding = self.generate_embedding(query_audio_path)

        # Obtener embeddings de los audios
        audio_embeddings = np.array(df['audio_embedding'].tolist())

        # Calcular similitudes
        similarities = self.calculate_similarity(query_embedding, audio_embeddings)

        # Obtener top_k resultados
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Crear DataFrame resultado
        result_df = df.iloc[top_indices].copy()
        result_df['audio_similarity_score'] = similarities[top_indices]
        result_df['query_audio'] = query_audio_path

        # Ordenar por similitud
        result_df = result_df.sort_values('audio_similarity_score', ascending=False)

        return result_df


class YAMNetEmbedding(BaseAudioEmbedding):
    """
    Implementación de embeddings de audio usando YAMNet de TensorFlow Hub
    """

    def __init__(self, config: YAMNetConfig | None = None):
        """
        Inicializa el generador de embeddings YAMNet

        Args:
            config: Configuración específica de YAMNet - usa configuración global si None
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow no está instalado. Para usar YAMNet:\n"
                "  poetry install --extras yamnet\n"
                "O usa CLAP como alternativa: poetry install"
            )

        super().__init__()

        # Cargar configuración
        if config is None:
            models_config = get_models_config()
            self.config = models_config.yamnet_config if models_config else YAMNetConfig()
        else:
            self.config = config

        self.model_url = self.config.model_url
        self.model_name = "YAMNet"
        self.embedding_dim = 1024  # YAMNet produce embeddings de 1024 dimensiones
        self.sample_rate = self.config.sample_rate
        self._load_model()

    def _load_model(self):
        """
        Carga el modelo YAMNet desde TensorFlow Hub
        """
        try:
            logger.info("Cargando modelo YAMNet...")
            self.model = hub.load(self.model_url)
            logger.info("Modelo YAMNet cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando YAMNet: {e}")
            raise RuntimeError(f"No se pudo cargar YAMNet: {e}")

    def preprocess_audio(self, audio_path: str, target_sr: int | None = None) -> np.ndarray:
        """
        Preprocesa un archivo de audio para YAMNet

        Args:
            audio_path: Ruta al archivo de audio
            target_sr: Frecuencia de muestreo objetivo (usa configuración si None)

        Returns:
            Audio preprocesado como array numpy
        """
        # Usar frecuencia de muestreo de la configuración si no se especifica
        if target_sr is None:
            target_sr = self.sample_rate

        # Cargar audio
        audio, _sr = librosa.load(audio_path, sr=target_sr, mono=True)

        # YAMNet espera audio en el rango [-1, 1]
        audio = librosa.util.normalize(audio)

        # Convertir a float32 para TensorFlow
        audio = audio.astype(np.float32)

        return audio

    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """
        Genera embedding para un archivo de audio usando YAMNet

        Args:
            audio_path: Ruta al archivo de audio

        Returns:
            Array numpy con el embedding del audio
        """
        if self.model is None:
            raise RuntimeError("Modelo YAMNet no está cargado")

        try:
            # Preprocesar audio
            audio = self.preprocess_audio(audio_path)

            # Generar embedding con YAMNet
            # YAMNet retorna: (scores, embeddings, spectrogram)
            _scores, embeddings, _spectrogram = self.model(audio)

            # Promediar embeddings a lo largo del tiempo
            embedding = tf.reduce_mean(embeddings, axis=0)
            return embedding.numpy()

        except Exception as e:
            logger.error(f"Error generando embedding para {audio_path}: {e}")
            raise RuntimeError(f"Error procesando audio: {e}")


# Alias para compatibilidad con código existente
# Nota: Usa get_audio_embedding_generator() para obtener la instancia adecuada
# Solo YAMNet si TensorFlow está disponible, de lo contrario lanzará error
AudioEmbeddingGenerator = YAMNetEmbedding


# Función para obtener el generador de embeddings
def get_audio_embedding_generator() -> BaseAudioEmbedding:
    """
    Factory function que retorna el generador de embeddings configurado

    Soporta los siguientes modelos:
    - YAMNet: Embeddings generales de audio (1024 dim, 16kHz) - requiere TensorFlow
    - CLAP: Embeddings alineados audio-texto (512 dim, 48kHz)

    Returns:
        Instancia de BaseAudioEmbedding con el modelo configurado

    Raises:
        ImportError: Si ningún modelo está disponible
    """
    # Usar configuración si está disponible
    models_config = get_models_config()

    if models_config is not None and hasattr(models_config, 'default_audio_embedding'):
        model_type = models_config.default_audio_embedding

        # CLAP Models
        if model_type in [AudioEmbeddingModel.CLAP_LAION, AudioEmbeddingModel.CLAP_MUSIC]:
            try:
                from .clap_audio_embeddings import CLAPEmbedding
                return CLAPEmbedding()
            except ImportError:
                try:
                    from clap_audio_embeddings import CLAPEmbedding
                    return CLAPEmbedding()
                except ImportError:
                    logger.error("⚠️  CLAP no disponible")
                    raise ImportError(
                        "CLAP no está disponible. Instala laion-clap:\n"
                        "  poetry install\n"
                        "O usa YAMNet: poetry install --extras yamnet"
                    )

    # Intentar usar YAMNet
    if TENSORFLOW_AVAILABLE:
        try:
            return YAMNetEmbedding()
        except Exception as e:
            logger.error(f"⚠️  Error cargando YAMNet: {e}")
            raise ImportError(
                f"Error cargando YAMNet: {e}\n"
                "Verifica la instalación de TensorFlow: poetry install --extras yamnet"
            )

    raise ImportError(
        "No hay generador de embeddings de audio disponible.\n"
        "Opciones:\n"
        "  - Instala TensorFlow para YAMNet: poetry install --extras yamnet\n"
        "  - O usa CLAP (ya incluido): poetry install"
    )


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso del generador de embeddings de audio
    embedder = get_audio_embedding_generator()

    # Datos de ejemplo
    sample_data = {
        'text': [
            "El presidente anunció nuevas medidas económicas",
            "Los mercados financieros mostraron volatilidad"
        ],
        'start_time': [0, 10],
        'end_time': [10, 20],
        'source_file': ['audio1.wav', 'audio1.wav']
    }

    df = pd.DataFrame(sample_data)

    # Generar embeddings reales
    # df_with_embeddings = embedder.process_transcription_dataframe(df)
    # logger.info(f"DataFrame con embeddings: {df_with_embeddings.columns.tolist()}")

    logger.info("Módulo de embeddings de audio listo.")
