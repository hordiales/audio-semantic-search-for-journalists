"""
Integración del modelo CLAP (Contrastive Language-Audio Pre-training) para embeddings de audio.
CLAP permite generar embeddings de audio que están alineados con representaciones de texto,
lo que es ideal para búsqueda semántica de audio con consultas en lenguaje natural.
"""

import logging
import os
from typing import Any
import warnings

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings('ignore')

from .audio_embeddings import BaseAudioEmbedding
from .models_config import CLAPConfig, get_models_config

logger = logging.getLogger(__name__)

# Verificar disponibilidad de CLAP
try:
    import laion_clap
    import torch
    CLAP_AVAILABLE = True
    logger.info("✅ LAION CLAP disponible")
except ImportError:
    CLAP_AVAILABLE = False
    logger.warning("⚠️  LAION CLAP no disponible. Instala con: pip install laion-clap")


def _build_checkpoint_args(config: CLAPConfig) -> dict[str, Any]:
    """
    Resuelve los argumentos de checkpoint respetando model_name y cache_dir.
    Prioridad:
    1) Si model_name es ruta de archivo existente, usar ckpt_path.
    2) Si model_name coincide con conocidos, usar model_id correspondiente.
    3) Caso contrario, solo aplicar download_root si existe.
    """
    args: dict[str, Any] = {}
    if config.cache_dir:
        os.makedirs(config.cache_dir, exist_ok=True)
        args["download_root"] = config.cache_dir

    model_map = {
        "laion/clap-htsat-unfused": 1,
        "clap-htsat-unfused": 1,
        "laion/clap-htsat-fused": 2,
        "clap-htsat-fused": 2,
    }

    if config.model_name:
        if os.path.isfile(config.model_name):
            args["ckpt_path"] = config.model_name
            return args

        mapped_id = model_map.get(config.model_name)
        if mapped_id is not None:
            args["model_id"] = mapped_id

    return args


class CLAPEmbedding(BaseAudioEmbedding):
    """
    Implementación de embeddings de audio usando CLAP (Contrastive Language-Audio Pre-training)

    CLAP es superior a YAMNet para búsqueda semántica porque:
    - Los embeddings están alineados con representaciones de texto
    - Permite búsqueda directa con consultas en lenguaje natural
    - Mejor comprensión del contenido semántico del audio
    """

    def __init__(self, config: CLAPConfig | None = None):
        """
        Inicializa el generador de embeddings CLAP

        Args:
            config: Configuración específica de CLAP, usa por defecto si None
        """
        if not CLAP_AVAILABLE:
            raise RuntimeError("LAION CLAP no está disponible. Instala con: pip install laion-clap")

        super().__init__()

        # Cargar configuración
        if config is None:
            models_config = get_models_config()
            self.config = models_config.clap_config if models_config else CLAPConfig()
        else:
            self.config = config

        self.device = self._get_device()
        self.model_name = "CLAP"
        self.embedding_dim = 512  # CLAP produce embeddings de 512 dimensiones
        self.sample_rate = 48000  # CLAP usa 48kHz
        self._load_model()

    def _get_device(self) -> str:
        """Determina el device a usar"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            return "cpu"
        return self.config.device

    def _load_model(self):
        """Carga el modelo CLAP"""
        try:
            logger.info(f"🔄 Cargando modelo CLAP: {self.config.model_name}")

            # Inicializar modelo CLAP
            self.model = laion_clap.CLAP_Module(
                enable_fusion=self.config.enable_fusion,
                device=self.device,
                amodel=self.config.amodel,
                tmodel=self.config.tmodel
            )

            ckpt_args = _build_checkpoint_args(self.config)
            if ckpt_args:
                self.model.load_ckpt(**ckpt_args)
            else:
                self.model.load_ckpt()

            # Asegurar modo inferencia
            if hasattr(self.model, "eval"):
                self.model.eval()

            logger.info(f"✅ Modelo CLAP cargado en {self.device}")

        except Exception as e:
            logger.error(f"❌ Error cargando CLAP: {e}")
            raise RuntimeError(f"No se pudo cargar CLAP: {e}")

    def preprocess_audio(self, audio_path: str, target_sr: int = 48000) -> np.ndarray:
        """
        Preprocesa un archivo de audio para CLAP

        Args:
            audio_path: Ruta al archivo de audio
            target_sr: Frecuencia de muestreo objetivo (CLAP usa 48kHz)

        Returns:
            Audio preprocesado como array numpy
        """
        # Cargar audio con la frecuencia de muestreo correcta
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        # CLAP espera audio normalizado
        audio = librosa.util.normalize(audio)

        # Convertir a float32
        audio = audio.astype(np.float32)

        return audio

    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """
        Genera embedding para un archivo de audio usando CLAP

        Args:
            audio_path: Ruta al archivo de audio

        Returns:
            Array numpy con el embedding del audio (512 dimensiones)
        """
        if self.model is None:
            raise RuntimeError("Modelo CLAP no está cargado")

        try:
            # Preprocesar audio
            audio = self.preprocess_audio(audio_path)

            # Generar embedding con CLAP
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                audio_embed = self.model.get_audio_embedding_from_data(
                    x=audio_tensor,
                    use_tensor=True
                )

            embedding = audio_embed.cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(embedding) + 1e-10
            embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"❌ Error generando embedding para {audio_path}: {e}")
            raise RuntimeError(f"Error procesando audio con CLAP: {e}")

    def generate_embeddings_batch(self, audio_paths: list[str], batch_size: int = 8) -> np.ndarray:
        """
        Genera embeddings para una lista de archivos de audio en lotes

        Args:
            audio_paths: Lista de rutas a archivos de audio
            batch_size: Tamaño del lote para procesamiento eficiente

        Returns:
            Array numpy con embeddings de todos los archivos
        """
        embeddings = []

        # Procesar en lotes para eficiencia
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            batch_embeddings = []

            for audio_path in batch_paths:
                try:
                    if os.environ.get('MCP_MODE') != '1':
                        logger.info(f"📊 Procesando {i+len(batch_embeddings)+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")

                    embedding = self.generate_embedding(audio_path)
                    batch_embeddings.append(embedding)

                except Exception as e:
                    logger.warning(f"⚠️  Error procesando {audio_path}: {e}")
                    # Agregar embedding cero para mantener consistencia
                    batch_embeddings.append(np.zeros(self.embedding_dim))

            embeddings.extend(batch_embeddings)

            # Limpieza de memoria GPU si es necesario
            if self.device.startswith('cuda') and len(batch_embeddings) > 0:
                torch.cuda.empty_cache()

        return np.array(embeddings)

    def process_transcription_dataframe(self, df: pd.DataFrame,
                                      temp_audio_dir: str = "temp_audio_clap") -> pd.DataFrame:
        """
        Procesa un DataFrame con transcripciones y genera embeddings de audio usando CLAP

        Args:
            df: DataFrame con información de segmentos de audio
            temp_audio_dir: Directorio temporal para archivos de audio

        Returns:
            DataFrame con embeddings de audio CLAP añadidos
        """
        if not os.path.exists(temp_audio_dir):
            os.makedirs(temp_audio_dir)

        # Generar archivos de audio temporales para cada segmento
        audio_paths = []
        valid_indices = []

        for idx, row in df.iterrows():
            try:
                # Crear archivo temporal para el segmento
                temp_audio_path = os.path.join(temp_audio_dir, f"clap_segment_{idx}.wav")

                # Cargar audio original y extraer segmento
                audio, sr = librosa.load(row['source_file'], sr=48000, mono=True)  # CLAP usa 48kHz
                start_sample = int(row['start_time'] * sr)
                end_sample = int(row['end_time'] * sr)

                segment_audio = audio[start_sample:end_sample]

                # Guardar segmento
                sf.write(temp_audio_path, segment_audio, sr)

                audio_paths.append(temp_audio_path)
                valid_indices.append(idx)

            except Exception as e:
                logger.error(f"❌ Error procesando segmento {idx}: {e}")
                continue

        if not audio_paths:
            logger.error("❌ No se pudieron procesar segmentos de audio para CLAP")
            return df

        # Generar embeddings
        if os.environ.get('MCP_MODE') != '1':
            logger.info(f"🔄 Generando embeddings CLAP para {len(audio_paths)} segmentos...")

        embeddings = self.generate_embeddings_batch(audio_paths)

        # Añadir embeddings al DataFrame
        result_df = df.copy()
        result_df['audio_embedding_clap'] = None

        for i, idx in enumerate(valid_indices):
            result_df.at[idx, 'audio_embedding_clap'] = embeddings[i].tolist()

        result_df['audio_embedding_model'] = "CLAP"
        result_df['audio_embedding_dim'] = self.embedding_dim

        # Limpiar archivos temporales
        for audio_path in audio_paths:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # Filtrar filas que tengan embeddings
        result_df = result_df[result_df['audio_embedding_clap'].notna()]

        if os.environ.get('MCP_MODE') != '1':
            logger.info(f"✅ {len(result_df)} segmentos procesados con embeddings CLAP")

        return result_df

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding de texto usando CLAP (para búsqueda semántica)

        Args:
            text: Texto para generar embedding

        Returns:
            Array numpy con el embedding de texto
        """
        if self.model is None:
            raise RuntimeError("Modelo CLAP no está cargado")

        try:
            with torch.inference_mode():
                text_embed = self.model.get_text_embedding([text])

            embedding = text_embed.cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(embedding) + 1e-10
            embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"❌ Error generando embedding de texto: {e}")
            raise RuntimeError(f"Error procesando texto con CLAP: {e}")

    def search_by_text_query(self, query: str, df: pd.DataFrame,
                           top_k: int = 5, embedding_column: str = 'audio_embedding_clap') -> pd.DataFrame:
        """
        Busca audios similares usando una consulta de texto (funcionalidad única de CLAP)

        Args:
            query: Consulta en texto natural
            df: DataFrame con embeddings de audio CLAP
            top_k: Número de resultados a retornar
            embedding_column: Nombre de la columna con embeddings

        Returns:
            DataFrame con los resultados más similares
        """
        if embedding_column not in df.columns:
            raise ValueError(f"Columna {embedding_column} no encontrada en el DataFrame")

        # Generar embedding de la consulta de texto
        query_embedding = self.generate_text_embedding(query)

        # Obtener embeddings de audio
        audio_embeddings = np.array(df[embedding_column].tolist())

        # Calcular similitudes coseno
        similarities = np.dot(audio_embeddings, query_embedding)

        # Obtener top_k resultados
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Crear DataFrame resultado
        result_df = df.iloc[top_indices].copy()
        result_df['text_similarity_score'] = similarities[top_indices]
        result_df['query_text'] = query

        # Ordenar por similitud
        result_df = result_df.sort_values('text_similarity_score', ascending=False)

        return result_df

    def calculate_similarity(self, query_embedding: np.ndarray,
                           audio_embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula la similitud coseno entre embeddings

        Args:
            query_embedding: Embedding de consulta
            audio_embeddings: Array con embeddings de audio

        Returns:
            Array con las similitudes
        """
        # Los embeddings ya están normalizados, usar producto punto directo
        similarities = np.dot(audio_embeddings, query_embedding)
        return similarities

    def generate_overlapping_chunks(
        self,
        audio_path: str,
        chunk_duration: float | None = None,
        overlap_duration: float | None = None,
        hop_duration: float | None = None,
        temp_audio_dir: str = "temp_audio_clap"
    ) -> pd.DataFrame:
        """
        Genera chunks de audio con overlapping y sus embeddings CLAP

        Args:
            audio_path: Ruta al archivo de audio
            chunk_duration: Duración del chunk en segundos (usa config si None)
            overlap_duration: Solapamiento entre chunks en segundos (usa config si None)
            hop_duration: Paso entre chunks en segundos (usa config si None, o calcula como chunk - overlap)
            temp_audio_dir: Directorio temporal para archivos de audio

        Returns:
            DataFrame con información de chunks y sus embeddings
        """
        # Usar parámetros de configuración si no se proporcionan
        chunk_duration = chunk_duration or self.config.chunk_duration
        overlap_duration = overlap_duration or self.config.overlap_duration

        # Calcular hop_duration si no se proporciona
        if hop_duration is None:
            if self.config.hop_duration is not None:
                hop_duration = self.config.hop_duration
            else:
                hop_duration = chunk_duration - overlap_duration

        # Validar parámetros
        if chunk_duration <= 0:
            raise ValueError("chunk_duration debe ser mayor que 0")
        if overlap_duration < 0:
            raise ValueError("overlap_duration no puede ser negativo")
        if overlap_duration >= chunk_duration:
            raise ValueError("overlap_duration debe ser menor que chunk_duration")
        if hop_duration <= 0:
            raise ValueError("hop_duration debe ser mayor que 0")

        if not os.path.exists(temp_audio_dir):
            os.makedirs(temp_audio_dir)

        # Cargar audio completo
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        total_duration = len(audio) / sr

        # Generar chunks con overlapping
        chunks_info = []
        current_time = 0.0
        chunk_id = 0

        while current_time < total_duration:
            start_time = current_time
            end_time = min(current_time + chunk_duration, total_duration)

            # Calcular muestras
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Extraer chunk
            chunk_audio = audio[start_sample:end_sample]
            actual_duration = len(chunk_audio) / sr

            # Guardar chunk temporal
            temp_chunk_path = os.path.join(
                temp_audio_dir,
                f"clap_chunk_{chunk_id:05d}.wav"
            )
            sf.write(temp_chunk_path, chunk_audio, sr)

            chunks_info.append({
                'chunk_id': chunk_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': actual_duration,
                'temp_path': temp_chunk_path,
                'source_file': audio_path
            })

            # Avanzar con hop_duration
            current_time += hop_duration
            chunk_id += 1

        if not chunks_info:
            logger.warning(f"⚠️  No se generaron chunks para {audio_path}")
            return pd.DataFrame()

        logger.info(
            f"🔄 Generados {len(chunks_info)} chunks con overlapping "
            f"(chunk={chunk_duration}s, overlap={overlap_duration}s, hop={hop_duration}s)"
        )

        # Generar embeddings para todos los chunks
        chunk_paths = [chunk['temp_path'] for chunk in chunks_info]
        embeddings = self.generate_embeddings_batch(chunk_paths)

        # Crear DataFrame con resultados
        result_data = []
        for i, chunk_info in enumerate(chunks_info):
            result_data.append({
                'chunk_id': chunk_info['chunk_id'],
                'start_time': chunk_info['start_time'],
                'end_time': chunk_info['end_time'],
                'duration': chunk_info['duration'],
                'source_file': chunk_info['source_file'],
                'audio_embedding_clap': embeddings[i].tolist(),
                'audio_embedding_model': 'CLAP',
                'audio_embedding_dim': self.embedding_dim
            })

        result_df = pd.DataFrame(result_data)

        # Limpiar archivos temporales
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

        logger.info(f"✅ {len(result_df)} chunks procesados con embeddings CLAP")

        return result_df


# Alias para compatibilidad con código existente
CLAPAudioEmbeddingGenerator = CLAPEmbedding


def get_clap_embedding_generator(config: CLAPConfig | None = None) -> CLAPEmbedding:
    """
    Factory function para obtener una instancia del generador CLAP

    Args:
        config: Configuración específica de CLAP

    Returns:
        Instancia del generador de embeddings CLAP
    """
    return CLAPEmbedding(config)


# Función de compatibilidad para reemplazar YAMNet
def get_audio_embedding_generator():
    """
    Función de compatibilidad que retorna CLAP o YAMNet según configuración

    Returns:
        Generador de embeddings de audio
    """
    models_config = get_models_config()

    if models_config.default_audio_embedding.value.startswith('clap') and CLAP_AVAILABLE:
        return get_clap_embedding_generator()
    # Fallback a YAMNet
    from .audio_embeddings import get_audio_embedding_generator as get_yamnet
    return get_yamnet()


if __name__ == "__main__":
    # Ejemplo de uso y prueba
    if CLAP_AVAILABLE:
        print("🎵 Generador de Embeddings CLAP")
        print("=" * 50)

        try:
            embedder = get_clap_embedding_generator()
            print("✅ Modelo CLAP inicializado correctamente")
            print(f"📐 Dimensiones de embedding: {embedder.embedding_dim}")
            print(f"🖥️  Device: {embedder.device}")

            # Ejemplo de embedding de texto
            text_query = "música clásica con piano"
            text_embedding = embedder.generate_text_embedding(text_query)
            print(f"🔤 Embedding de texto generado: {text_embedding.shape}")

            print("\n💡 Características de CLAP:")
            print("  🎯 Búsqueda semántica con texto natural")
            print("  🔄 Embeddings alineados audio-texto")
            print("  📊 Mejor para contenido musical y hablado")
            print("  🚀 Ideal para consultas descriptivas")

        except Exception as e:
            print(f"❌ Error inicializando CLAP: {e}")
            print("💡 Asegúrate de tener instalado: pip install laion-clap")
    else:
        print("❌ LAION CLAP no disponible")
        print("💡 Para instalar: pip install laion-clap")
