import faiss
import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict, Optional, Tuple
import json


class VectorIndexManager:
    """
    Clase para gestionar índices vectoriales usando FAISS
    """
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "L2"):
        """
        Inicializa el gestor de índices vectoriales
        
        Args:
            embedding_dim: Dimensión de los embeddings
            index_type: Tipo de índice ('L2', 'IP' para inner product, 'cosine')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.text_index = None
        self.audio_index = None
        self.text_metadata = None
        self.audio_metadata = None
        self.is_trained = False
        
    def _create_index(self, embedding_dim: int, index_type: str = "L2") -> faiss.Index:
        """
        Crea un índice FAISS
        
        Args:
            embedding_dim: Dimensión de los embeddings
            index_type: Tipo de índice
            
        Returns:
            Índice FAISS
        """
        if index_type == "L2":
            return faiss.IndexFlatL2(embedding_dim)
        elif index_type == "IP":
            return faiss.IndexFlatIP(embedding_dim)
        elif index_type == "cosine":
            # Para similitud coseno, usar inner product con vectores normalizados
            return faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(f"Tipo de índice no soportado: {index_type}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normaliza embeddings para similitud coseno
        
        Args:
            embeddings: Array de embeddings
            
        Returns:
            Embeddings normalizados
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    
    def create_text_index(self, df: pd.DataFrame, embedding_column: str = 'text_embedding') -> bool:
        """
        Crea índice para embeddings de texto
        
        Args:
            df: DataFrame con embeddings de texto
            embedding_column: Nombre de la columna con embeddings
            
        Returns:
            True si se creó exitosamente
        """
        if embedding_column not in df.columns:
            print(f"Columna {embedding_column} no encontrada")
            return False
        
        # Extraer embeddings
        embeddings = np.array(df[embedding_column].tolist()).astype(np.float32)
        
        # Verificar dimensiones
        if embeddings.shape[1] != self.embedding_dim:
            print(f"Dimensión de embeddings ({embeddings.shape[1]}) no coincide con la esperada ({self.embedding_dim})")
            # Actualizar dimensión
            self.embedding_dim = embeddings.shape[1]
        
        # Normalizar si es necesario
        if self.index_type == "cosine":
            embeddings = self._normalize_embeddings(embeddings)
        
        # Crear índice
        self.text_index = self._create_index(embeddings.shape[1], self.index_type)
        
        # Añadir vectores al índice
        self.text_index.add(embeddings)
        
        # Guardar metadatos
        self.text_metadata = df.copy()
        self.text_metadata['faiss_id'] = range(len(df))
        
        print(f"Índice de texto creado con {len(embeddings)} vectores")
        return True
    
    def create_audio_index(self, df: pd.DataFrame, embedding_column: str = 'audio_embedding') -> bool:
        """
        Crea índice para embeddings de audio
        
        Args:
            df: DataFrame con embeddings de audio
            embedding_column: Nombre de la columna con embeddings
            
        Returns:
            True si se creó exitosamente
        """
        if embedding_column not in df.columns:
            print(f"Columna {embedding_column} no encontrada")
            return False
        
        # Filtrar filas con embeddings válidos
        valid_df = df[df[embedding_column].notna()].copy()
        
        if len(valid_df) == 0:
            print("No se encontraron embeddings de audio válidos")
            return False
        
        # Extraer embeddings
        embeddings = np.array(valid_df[embedding_column].tolist()).astype(np.float32)
        
        # Normalizar si es necesario
        if self.index_type == "cosine":
            embeddings = self._normalize_embeddings(embeddings)
        
        # Crear índice
        self.audio_index = self._create_index(embeddings.shape[1], self.index_type)
        
        # Añadir vectores al índice
        self.audio_index.add(embeddings)
        
        # Guardar metadatos
        self.audio_metadata = valid_df.copy()
        self.audio_metadata['faiss_id'] = range(len(valid_df))
        
        print(f"Índice de audio creado con {len(embeddings)} vectores")
        return True
    
    def search_text_index(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Busca en el índice de texto
        
        Args:
            query_embedding: Embedding de la consulta
            k: Número de resultados a retornar
            
        Returns:
            Tupla con (distancias, índices)
        """
        if self.text_index is None:
            raise ValueError("Índice de texto no inicializado")
        
        # Normalizar consulta si es necesario
        if self.index_type == "cosine":
            query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))
        else:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Buscar
        distances, indices = self.text_index.search(query_embedding, k)
        
        return distances[0], indices[0]
    
    def search_audio_index(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Busca en el índice de audio
        
        Args:
            query_embedding: Embedding de la consulta
            k: Número de resultados a retornar
            
        Returns:
            Tupla con (distancias, índices)
        """
        if self.audio_index is None:
            raise ValueError("Índice de audio no inicializado")
        
        # Normalizar consulta si es necesario
        if self.index_type == "cosine":
            query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))
        else:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Buscar
        distances, indices = self.audio_index.search(query_embedding, k)
        
        return distances[0], indices[0]
    
    def get_text_results(self, distances: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
        """
        Obtiene resultados de búsqueda en texto con metadatos
        
        Args:
            distances: Distancias retornadas por FAISS
            indices: Índices retornados por FAISS
            
        Returns:
            DataFrame con resultados
        """
        if self.text_metadata is None:
            raise ValueError("Metadatos de texto no disponibles")
        
        # Filtrar índices válidos
        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        valid_distances = distances[valid_mask]
        
        if len(valid_indices) == 0:
            return pd.DataFrame()
        
        # Obtener resultados
        results = self.text_metadata.iloc[valid_indices].copy()
        
        # Convertir distancias a scores de similitud
        if self.index_type == "L2":
            # Para L2, menor distancia = mayor similitud
            scores = 1 / (1 + valid_distances)
        elif self.index_type in ["IP", "cosine"]:
            # Para inner product, mayor valor = mayor similitud
            scores = valid_distances
        else:
            scores = valid_distances
        
        results['similarity_score'] = scores
        results['faiss_distance'] = valid_distances
        
        return results.sort_values('similarity_score', ascending=False)
    
    def get_audio_results(self, distances: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
        """
        Obtiene resultados de búsqueda en audio con metadatos
        
        Args:
            distances: Distancias retornadas por FAISS
            indices: Índices retornados por FAISS
            
        Returns:
            DataFrame con resultados
        """
        if self.audio_metadata is None:
            raise ValueError("Metadatos de audio no disponibles")
        
        # Filtrar índices válidos
        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        valid_distances = distances[valid_mask]
        
        if len(valid_indices) == 0:
            return pd.DataFrame()
        
        # Obtener resultados
        results = self.audio_metadata.iloc[valid_indices].copy()
        
        # Convertir distancias a scores de similitud
        if self.index_type == "L2":
            scores = 1 / (1 + valid_distances)
        elif self.index_type in ["IP", "cosine"]:
            scores = valid_distances
        else:
            scores = valid_distances
        
        results['audio_similarity_score'] = scores
        results['faiss_distance'] = valid_distances
        
        return results.sort_values('audio_similarity_score', ascending=False)
    
    def save_indices(self, base_path: str):
        """
        Guarda los índices y metadatos en archivos
        
        Args:
            base_path: Ruta base para guardar los archivos
        """
        # Crear directorio si no existe
        os.makedirs(base_path, exist_ok=True)
        
        # Guardar índices FAISS
        if self.text_index is not None:
            faiss.write_index(self.text_index, os.path.join(base_path, "text_index.faiss"))
            
        if self.audio_index is not None:
            faiss.write_index(self.audio_index, os.path.join(base_path, "audio_index.faiss"))
        
        # Guardar metadatos
        if self.text_metadata is not None:
            self.text_metadata.to_pickle(os.path.join(base_path, "text_metadata.pkl"))
            
        if self.audio_metadata is not None:
            self.audio_metadata.to_pickle(os.path.join(base_path, "audio_metadata.pkl"))
        
        # Guardar configuración
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'has_text_index': self.text_index is not None,
            'has_audio_index': self.audio_index is not None
        }
        
        with open(os.path.join(base_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Índices guardados en: {base_path}")
    
    def load_indices(self, base_path: str):
        """
        Carga los índices y metadatos desde archivos
        
        Args:
            base_path: Ruta base donde están los archivos
        """
        # Cargar configuración
        config_path = os.path.join(base_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.embedding_dim = config['embedding_dim']
            self.index_type = config['index_type']
        
        # Cargar índices FAISS
        text_index_path = os.path.join(base_path, "text_index.faiss")
        if os.path.exists(text_index_path):
            self.text_index = faiss.read_index(text_index_path)
        
        audio_index_path = os.path.join(base_path, "audio_index.faiss")
        if os.path.exists(audio_index_path):
            self.audio_index = faiss.read_index(audio_index_path)
        
        # Cargar metadatos
        text_metadata_path = os.path.join(base_path, "text_metadata.pkl")
        if os.path.exists(text_metadata_path):
            self.text_metadata = pd.read_pickle(text_metadata_path)
        
        audio_metadata_path = os.path.join(base_path, "audio_metadata.pkl")
        if os.path.exists(audio_metadata_path):
            self.audio_metadata = pd.read_pickle(audio_metadata_path)
        
        print(f"Índices cargados desde: {base_path}")
    
    def get_index_stats(self) -> Dict:
        """
        Retorna estadísticas de los índices
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'text_index_size': self.text_index.ntotal if self.text_index else 0,
            'audio_index_size': self.audio_index.ntotal if self.audio_index else 0,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type
        }
        
        return stats


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso del gestor de índices
    
    # Crear datos de ejemplo
    sample_text_data = {
        'text': ['Ejemplo 1', 'Ejemplo 2', 'Ejemplo 3'],
        'text_embedding': [
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist()
        ],
        'start_time': [0, 10, 20],
        'source_file': ['audio1.wav'] * 3
    }
    
    df = pd.DataFrame(sample_text_data)
    
    # Crear gestor de índices
    index_manager = VectorIndexManager(embedding_dim=384, index_type="cosine")
    
    # Crear índice de texto
    # index_manager.create_text_index(df)
    
    # Buscar
    # query_embedding = np.random.rand(384)
    # distances, indices = index_manager.search_text_index(query_embedding, k=3)
    # results = index_manager.get_text_results(distances, indices)
    # print(results[['text', 'similarity_score']])
    
    print("Módulo de indexación vectorial listo.")