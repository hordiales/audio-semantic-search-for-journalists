"""
Sistema de bases de datos vectoriales configurable para Audio Semantic Search.
Soporta FAISS, ChromaDB y Supabase (pgvector) con una interfaz unificada.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

class VectorDBType(Enum):
    """Tipos de bases de datos vectoriales soportadas"""
    FAISS = "faiss"
    CHROMADB = "chromadb"
    SUPABASE = "supabase"
    MEMORY = "memory"  # Fallback para desarrollo

@dataclass
class VectorDocument:
    """Documento con embedding vectorial y metadatos"""
    id: str
    embedding: np.ndarray
    text: str
    metadata: dict[str, Any]
    category: str | None = None
    timestamp: float | None = None
    audio_file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convierte a diccionario serializable"""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "category": self.category,
            "timestamp": self.timestamp,
            "audio_file_path": self.audio_file_path,
            # embedding se maneja por separado según la DB
        }

@dataclass
class SearchResult:
    """Resultado de búsqueda vectorial"""
    document: VectorDocument
    similarity_score: float
    distance: float
    rank: int

@dataclass
class VectorDBConfig:
    """Configuración para bases de datos vectoriales"""
    db_type: VectorDBType

    # Configuración común
    embedding_dimension: int = 512
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product

    # FAISS específico
    faiss_index_type: str = "flat"  # flat, ivf, hnsw
    faiss_index_path: str | None = None
    faiss_gpu: bool = False
    faiss_nprobe: int = 10

    # ChromaDB específico
    chromadb_path: str | None = None
    chromadb_collection_name: str = "audio_embeddings"
    chromadb_distance_function: str = "cosine"

    # Supabase específico
    supabase_url: str | None = None
    supabase_key: str | None = None
    supabase_table_name: str = "audio_embeddings"
    supabase_connection_pool_size: int = 10

class VectorDatabaseInterface(ABC):
    """Interfaz abstracta para bases de datos vectoriales"""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.is_initialized = False
        self._document_count = 0

    @abstractmethod
    def initialize(self) -> bool:
        """Inicializa la base de datos"""

    @abstractmethod
    def add_documents(self, documents: list[VectorDocument]) -> bool:
        """Añade documentos a la base de datos"""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10,
               filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """Busca documentos similares"""

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Elimina un documento"""

    @abstractmethod
    def get_document(self, document_id: str) -> VectorDocument | None:
        """Obtiene un documento por ID"""

    @abstractmethod
    def update_document(self, document: VectorDocument) -> bool:
        """Actualiza un documento"""

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Obtiene estadísticas de la base de datos"""

    @abstractmethod
    def save_index(self, path: str) -> bool:
        """Guarda el índice (si aplica)"""

    @abstractmethod
    def load_index(self, path: str) -> bool:
        """Carga el índice (si aplica)"""

    @abstractmethod
    def clear(self) -> bool:
        """Limpia toda la base de datos"""

    def get_document_count(self) -> int:
        """Retorna el número de documentos"""
        return self._document_count

    def is_ready(self) -> bool:
        """Verifica si la DB está lista para usar"""
        return self.is_initialized

class MemoryVectorDatabase(VectorDatabaseInterface):
    """Implementación en memoria para desarrollo y fallback"""

    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self.documents: dict[str, VectorDocument] = {}
        self.embeddings: np.ndarray | None = None
        self.document_ids: list[str] = []

    def initialize(self) -> bool:
        """Inicializa la base de datos en memoria"""
        try:
            logger.info("🧠 Inicializando base de datos vectorial en memoria...")
            self.is_initialized = True
            logger.info("✅ Base de datos en memoria inicializada")
            return True
        except Exception as e:
            logger.error(f"❌ Error inicializando base de datos en memoria: {e}")
            return False

    def add_documents(self, documents: list[VectorDocument]) -> bool:
        """Añade documentos a la memoria"""
        try:
            for doc in documents:
                self.documents[doc.id] = doc
                self.document_ids.append(doc.id)

            # Reconstruir matriz de embeddings
            if documents:
                embeddings_list = [self.documents[doc_id].embedding for doc_id in self.document_ids]
                self.embeddings = np.vstack(embeddings_list)
                self._document_count = len(self.document_ids)

            logger.info(f"✅ {len(documents)} documentos añadidos a memoria")
            return True
        except Exception as e:
            logger.error(f"❌ Error añadiendo documentos a memoria: {e}")
            return False

    def search(self, query_embedding: np.ndarray, k: int = 10,
               filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """Busca documentos similares en memoria"""
        if self.embeddings is None or len(self.document_ids) == 0:
            return []

        try:
            # Calcular similitudes
            if self.config.similarity_metric == "cosine":
                # Normalizar vectores
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                similarities = np.dot(embeddings_norm, query_norm)
                distances = 1 - similarities
            else:
                # Distancia euclidiana
                distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
                similarities = 1 / (1 + distances)

            # Aplicar filtros
            valid_indices = list(range(len(self.document_ids)))
            if filters:
                valid_indices = []
                for i, doc_id in enumerate(self.document_ids):
                    doc = self.documents[doc_id]
                    if self._matches_filters(doc, filters):
                        valid_indices.append(i)

            if not valid_indices:
                return []

            # Obtener top-k
            valid_similarities = similarities[valid_indices]
            valid_distances = distances[valid_indices]

            # Ordenar por similitud descendente
            sorted_indices = np.argsort(valid_similarities)[::-1][:k]

            results = []
            for rank, idx in enumerate(sorted_indices):
                original_idx = valid_indices[idx]
                doc_id = self.document_ids[original_idx]
                doc = self.documents[doc_id]

                result = SearchResult(
                    document=doc,
                    similarity_score=float(valid_similarities[idx]),
                    distance=float(valid_distances[idx]),
                    rank=rank + 1
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda en memoria: {e}")
            return []

    def _matches_filters(self, doc: VectorDocument, filters: dict[str, Any]) -> bool:
        """Verifica si un documento coincide con los filtros"""
        for key, value in filters.items():
            if (key == "category" and doc.category != value) or (key in doc.metadata and doc.metadata[key] != value):
                return False
        return True

    def delete_document(self, document_id: str) -> bool:
        """Elimina un documento de memoria"""
        if document_id in self.documents:
            del self.documents[document_id]
            if document_id in self.document_ids:
                self.document_ids.remove(document_id)

            # Reconstruir embeddings
            if self.document_ids:
                embeddings_list = [self.documents[doc_id].embedding for doc_id in self.document_ids]
                self.embeddings = np.vstack(embeddings_list)
            else:
                self.embeddings = None

            self._document_count = len(self.document_ids)
            return True
        return False

    def get_document(self, document_id: str) -> VectorDocument | None:
        """Obtiene un documento por ID"""
        return self.documents.get(document_id)

    def update_document(self, document: VectorDocument) -> bool:
        """Actualiza un documento en memoria"""
        if document.id in self.documents:
            self.documents[document.id] = document
            # Reconstruir embeddings si es necesario
            embeddings_list = [self.documents[doc_id].embedding for doc_id in self.document_ids]
            self.embeddings = np.vstack(embeddings_list)
            return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Obtiene estadísticas de la base de datos"""
        return {
            "db_type": "memory",
            "document_count": self._document_count,
            "embedding_dimension": self.config.embedding_dimension,
            "similarity_metric": self.config.similarity_metric,
            "is_initialized": self.is_initialized,
            "memory_usage_mb": self.embeddings.nbytes / 1024 / 1024 if self.embeddings is not None else 0
        }

    def save_index(self, path: str) -> bool:
        """Guarda la base de datos en disco"""
        try:
            import pickle
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "documents": self.documents,
                "document_ids": self.document_ids,
                "embeddings": self.embeddings,
                "config": self.config
            }

            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"💾 Base de datos en memoria guardada en: {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando base de datos: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Carga la base de datos desde disco"""
        try:
            import pickle
            load_path = Path(path)

            if not load_path.exists():
                logger.warning(f"⚠️  Archivo no encontrado: {path}")
                return False

            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            self.documents = data["documents"]
            self.document_ids = data["document_ids"]
            self.embeddings = data["embeddings"]
            self._document_count = len(self.document_ids)

            logger.info(f"📂 Base de datos cargada desde: {path}")
            logger.info(f"📊 {self._document_count} documentos cargados")
            return True
        except Exception as e:
            logger.error(f"❌ Error cargando base de datos: {e}")
            return False

    def clear(self) -> bool:
        """Limpia toda la base de datos"""
        self.documents.clear()
        self.document_ids.clear()
        self.embeddings = None
        self._document_count = 0
        logger.info("🧹 Base de datos en memoria limpiada")
        return True

# Funciones utilitarias
def create_vector_database(config: VectorDBConfig) -> VectorDatabaseInterface:
    """Factory para crear instancias de bases de datos vectoriales"""

    if config.db_type == VectorDBType.FAISS:
        try:
            from .vector_db_faiss import FAISSVectorDatabase
        except ImportError:
            from vector_db_faiss import FAISSVectorDatabase
        return FAISSVectorDatabase(config)
    if config.db_type == VectorDBType.CHROMADB:
        try:
            from .vector_db_chromadb import ChromaVectorDatabase
        except ImportError:
            from vector_db_chromadb import ChromaVectorDatabase
        return ChromaVectorDatabase(config)
    if config.db_type == VectorDBType.SUPABASE:
        try:
            from .vector_db_supabase import SupabaseVectorDatabase
        except ImportError:
            from vector_db_supabase import SupabaseVectorDatabase
        return SupabaseVectorDatabase(config)
    if config.db_type == VectorDBType.MEMORY:
        return MemoryVectorDatabase(config)
    raise ValueError(f"Tipo de base de datos no soportado: {config.db_type}")

def get_default_config(db_type: VectorDBType, embedding_dimension: int = 512) -> VectorDBConfig:
    """Obtiene configuración por defecto para un tipo de DB"""

    base_config = VectorDBConfig(
        db_type=db_type,
        embedding_dimension=embedding_dimension,
        similarity_metric="cosine"
    )

    if db_type == VectorDBType.FAISS:
        base_config.faiss_index_type = "flat"
        base_config.faiss_index_path = "data/faiss_index.bin"
    elif db_type == VectorDBType.CHROMADB:
        base_config.chromadb_path = "data/chromadb"
        base_config.chromadb_collection_name = "audio_embeddings"
    elif db_type == VectorDBType.SUPABASE:
        # Estas se deben configurar por el usuario
        base_config.supabase_table_name = "audio_embeddings"

    return base_config

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    # Crear configuración de ejemplo
    config = get_default_config(VectorDBType.MEMORY, embedding_dimension=256)

    # Crear base de datos
    db = create_vector_database(config)

    # Inicializar
    if db.initialize():
        print("✅ Base de datos inicializada")

        # Crear documentos de ejemplo
        docs = []
        for i in range(5):
            embedding = np.random.randn(256)
            embedding = embedding / np.linalg.norm(embedding)  # Normalizar

            doc = VectorDocument(
                id=f"doc_{i}",
                embedding=embedding,
                text=f"Documento de ejemplo {i}",
                metadata={"category": f"cat_{i % 3}", "source": "test"},
                category=f"categoria_{i % 3}"
            )
            docs.append(doc)

        # Añadir documentos
        if db.add_documents(docs):
            print(f"✅ {len(docs)} documentos añadidos")

            # Realizar búsqueda
            query = np.random.randn(256)
            query = query / np.linalg.norm(query)

            results = db.search(query, k=3)
            print(f"🔍 Encontrados {len(results)} resultados")

            for result in results:
                print(f"  📄 {result.document.text} (similitud: {result.similarity_score:.3f})")

            # Estadísticas
            stats = db.get_statistics()
            print(f"📊 Estadísticas: {stats}")

    else:
        print("❌ Error inicializando base de datos")
