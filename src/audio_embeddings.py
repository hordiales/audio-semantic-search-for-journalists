import numpy as np
import pandas as pd
import librosa
from typing import List, Dict, Optional
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
import logging
import sys

logger = logging.getLogger(__name__)

# TensorFlow imports requeridos
import tensorflow as tf
import tensorflow_hub as hub

from .models_config import get_models_config, YAMNetConfig, AudioEmbeddingModel


class AudioEmbeddingGenerator:
    """
    Clase para generar embeddings de audio usando YAMNet de TensorFlow Hub
    """
    
    def __init__(self, config: Optional[YAMNetConfig] = None):
        """
        Inicializa el generador de embeddings de audio
        
        Args:
            config: Configuración específica de YAMNet - usa configuración global si None
        """
        # Cargar configuración
        if config is None:
            models_config = get_models_config()
            self.config = models_config.yamnet_config
        else:
            self.config = config
            
        self.model_url = self.config.model_url
        self.model = None
        self.embedding_dim = 1024  # YAMNet produce embeddings de 1024 dimensiones
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
    
    def preprocess_audio(self, audio_path: str, target_sr: Optional[int] = None) -> np.ndarray:
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
            target_sr = self.config.sample_rate
        
        # Cargar audio
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # YAMNet espera audio en el rango [-1, 1]
        audio = librosa.util.normalize(audio)
        
        # Convertir a float32 para TensorFlow
        audio = audio.astype(np.float32)
        
        return audio
    
    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """
        Genera embedding para un archivo de audio
        
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
            scores, embeddings, spectrogram = self.model(audio)
            
            # Promediar embeddings a lo largo del tiempo
            embedding = tf.reduce_mean(embeddings, axis=0)
            return embedding.numpy()
            
        except Exception as e:
            logger.error(f"Error generando embedding para {audio_path}: {e}")
            # Error en procesamiento de audio
            raise RuntimeError(f"Error procesando audio: {e}")
    
    def generate_embeddings_batch(self, audio_paths: List[str]) -> np.ndarray:
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
                audio, sr = librosa.load(row['source_file'], sr=16000, mono=True)
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
        
        result_df['audio_embedding_model'] = "YAMNet"
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
    




# Función para obtener el generador de embeddings
def get_audio_embedding_generator():
    """
    Retorna el generador de embeddings configurado (YAMNet o CLAP según configuración)
    
    Returns:
        Instancia del generador de embeddings
    """
    models_config = get_models_config()
    
    # Verificar si se debe usar CLAP
    if models_config.default_audio_embedding == AudioEmbeddingModel.CLAP_LAION:
        try:
            from .clap_audio_embeddings import get_clap_embedding_generator
            return get_clap_embedding_generator()
        except ImportError:
            logger.warning("⚠️  CLAP no disponible, usando YAMNet como fallback")
            return AudioEmbeddingGenerator()
    elif models_config.default_audio_embedding == AudioEmbeddingModel.CLAP_MUSIC:
        try:
            from .clap_audio_embeddings import get_clap_embedding_generator
            return get_clap_embedding_generator()
        except ImportError:
            logger.warning("⚠️  CLAP no disponible, usando YAMNet como fallback")
            return AudioEmbeddingGenerator()
    else:
        # Usar YAMNet por defecto
        return AudioEmbeddingGenerator()


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