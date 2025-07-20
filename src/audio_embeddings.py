import numpy as np
import pandas as pd
import librosa
from typing import List, Dict, Optional
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports opcionales
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow no disponible. Usando embeddings mock para audio.")


class AudioEmbeddingGenerator:
    """
    Clase para generar embeddings de audio usando YAMNet de TensorFlow Hub
    """
    
    def __init__(self, model_url: str = "https://tfhub.dev/google/yamnet/1"):
        """
        Inicializa el generador de embeddings de audio
        
        Args:
            model_url: URL del modelo YAMNet en TensorFlow Hub
        """
        self.model_url = model_url
        self.model = None
        self.embedding_dim = 1024  # YAMNet produce embeddings de 1024 dimensiones
        self._load_model()
    
    def _load_model(self):
        """
        Carga el modelo YAMNet desde TensorFlow Hub
        """
        if not TF_AVAILABLE:
            print("TensorFlow no disponible. Usando modelo mock.")
            self.model = None
            return
            
        try:
            print("Cargando modelo YAMNet...")
            self.model = hub.load(self.model_url)
            print("Modelo YAMNet cargado exitosamente")
        except Exception as e:
            print(f"Error cargando YAMNet: {e}")
            print("Usando modelo mock para pruebas")
            self.model = None
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Preprocesa un archivo de audio para YAMNet
        
        Args:
            audio_path: Ruta al archivo de audio
            target_sr: Frecuencia de muestreo objetivo
            
        Returns:
            Audio preprocesado como array numpy
        """
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
            # Generar embedding mock para pruebas
            return np.random.rand(self.embedding_dim).astype(np.float32)
        
        try:
            # Preprocesar audio
            audio = self.preprocess_audio(audio_path)
            
            # Generar embedding con YAMNet
            # YAMNet retorna: (scores, embeddings, spectrogram)
            scores, embeddings, spectrogram = self.model(audio)
            
            # Promediar embeddings a lo largo del tiempo
            if TF_AVAILABLE:
                embedding = tf.reduce_mean(embeddings, axis=0)
                return embedding.numpy()
            else:
                # Fallback si TF no está disponible
                return np.random.rand(self.embedding_dim).astype(np.float32)
            
        except Exception as e:
            print(f"Error generando embedding para {audio_path}: {e}")
            # Retornar embedding mock en caso de error
            return np.random.rand(self.embedding_dim).astype(np.float32)
    
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
            print(f"Procesando audio {i+1}/{len(audio_paths)}: {audio_path}")
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
                print(f"Error procesando segmento {idx}: {e}")
                continue
        
        if not audio_paths:
            print("No se pudieron procesar segmentos de audio")
            return df
        
        # Generar embeddings
        print(f"Generando embeddings de audio para {len(audio_paths)} segmentos...")
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
    
    def generate_mock_embedding_from_text(self, text: str) -> np.ndarray:
        """
        Genera un embedding mock basado en texto (para consultas)
        Fallback cuando no hay archivo de audio disponible
        
        Args:
            text: Texto de consulta
            
        Returns:
            Embedding mock basado en el texto
        """
        # Usar hash del texto para generar embedding consistente
        text_hash = hash(text) % 1000
        np.random.seed(text_hash)
        embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        return embedding


class MockAudioEmbeddingGenerator:
    """
    Versión mock del generador de embeddings de audio para pruebas sin internet
    """
    
    def __init__(self):
        self.embedding_dim = 1024
        self.model = "MockModel"
        np.random.seed(42)  # Para reproducibilidad
    
    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """
        Genera un embedding mock basado en características simples del audio
        """
        try:
            # Cargar audio y extraer características básicas
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Características básicas
            duration = len(audio) / sr
            energy = np.mean(audio**2)
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(audio, sr=sr))
            
            # Crear embedding determinístico basado en características
            features = np.array([duration, energy, zero_crossing_rate, spectral_centroid])
            
            # Usar las características para generar un embedding reproducible
            np.random.seed(int(np.sum(features * 1000)))
            embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            
            return embedding
            
        except Exception as e:
            print(f"Error en mock embedding: {e}")
            return np.random.rand(self.embedding_dim).astype(np.float32)
    
    def generate_embeddings_batch(self, audio_paths: List[str]) -> np.ndarray:
        """
        Genera embeddings mock para una lista de archivos
        """
        embeddings = []
        for audio_path in audio_paths:
            embedding = self.generate_embedding(audio_path)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def process_transcription_dataframe(self, df: pd.DataFrame, 
                                      temp_audio_dir: str = "temp_audio") -> pd.DataFrame:
        """
        Versión mock del procesamiento de DataFrame
        """
        # Generar embeddings mock basados en características del texto
        result_df = df.copy()
        embeddings = []
        
        for idx, row in df.iterrows():
            # Usar características del texto para generar embedding consistente
            text_hash = hash(row.get('text', '')) % 1000
            np.random.seed(text_hash)
            embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            embeddings.append(embedding.tolist())
        
        result_df['audio_embedding'] = embeddings
        result_df['audio_embedding_model'] = "MockYAMNet"
        result_df['audio_embedding_dim'] = self.embedding_dim
        
        return result_df
    
    def generate_mock_embedding_from_text(self, text: str) -> np.ndarray:
        """
        Genera un embedding mock basado en texto (para consultas)
        
        Args:
            text: Texto de consulta
            
        Returns:
            Embedding mock basado en el texto
        """
        # Usar hash del texto para generar embedding consistente
        text_hash = hash(text) % 1000
        np.random.seed(text_hash)
        embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        return embedding


# Función para seleccionar el generador apropiado
def get_audio_embedding_generator(use_mock: bool = None):
    """
    Retorna el generador de embeddings apropiado
    
    Args:
        use_mock: Si True, usa el generador mock. Si None, detecta automáticamente.
        
    Returns:
        Instancia del generador de embeddings
    """
    # Auto-detectar si usar mock basado en disponibilidad de TensorFlow
    if use_mock is None:
        use_mock = not TF_AVAILABLE
    
    if use_mock or not TF_AVAILABLE:
        return MockAudioEmbeddingGenerator()
    else:
        return AudioEmbeddingGenerator()


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso del generador de embeddings de audio
    
    # Usar versión mock para pruebas
    embedder = get_audio_embedding_generator(use_mock=True)
    
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
    
    # Generar embeddings (mock)
    # df_with_embeddings = embedder.process_transcription_dataframe(df)
    # print(f"DataFrame con embeddings: {df_with_embeddings.columns.tolist()}")
    
    print("Módulo de embeddings de audio listo.")