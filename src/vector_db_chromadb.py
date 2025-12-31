"""
Implementación de base de datos vectorial usando ChromaDB
Ideal para prototipado y desarrollo de aplicaciones de búsqueda semántica.
"""

import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    from .vector_database_interface import (
        SearchResult,
        VectorDatabaseInterface,
        VectorDBConfig,
        VectorDocument,
    )
except ImportError:
    from vector_database_interface import (
        SearchResult,
        VectorDatabaseInterface,
        VectorDBConfig,
        VectorDocument,
    )

logger = logging.getLogger(__name__)

# Imports condicionales para ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("⚠️  ChromaDB no disponible. Instala con: pip install chromadb")

class ChromaVectorDatabase(VectorDatabaseInterface):
    """Implementación de base de datos vectorial usando ChromaDB"""

    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self.client = None
        self.collection = None

        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB no está disponible. Instala con: pip install chromadb")

    def initialize(self) -> bool:
        """Inicializa ChromaDB"""
        try:
            logger.info("🍊 Inicializando ChromaDB...")

            # Configurar cliente
            if self.config.chromadb_path:
                # Cliente persistente
                chroma_path = Path(self.config.chromadb_path)
                chroma_path.mkdir(parents=True, exist_ok=True)

                settings = Settings(
                    persist_directory=str(chroma_path),
                    anonymized_telemetry=False
                )
                self.client = chromadb.PersistentClient(
                    path=str(chroma_path),
                    settings=settings
                )
                logger.info(f"📁 Cliente persistente: {chroma_path}")
            else:
                # Cliente en memoria
                self.client = chromadb.EphemeralClient()
                logger.info("🧠 Cliente en memoria")

            # Crear o obtener colección
            distance_function = self._get_distance_function()

            try:
                # Intentar obtener colección existente
                self.collection = self.client.get_collection(
                    name=self.config.chromadb_collection_name
                )
                logger.info(f"📚 Colección existente cargada: {self.config.chromadb_collection_name}")

            except Exception:
                # Crear nueva colección
                self.collection = self.client.create_collection(
                    name=self.config.chromadb_collection_name,
                    metadata={"hnsw:space": distance_function}
                )
                logger.info(f"📚 Nueva colección creada: {self.config.chromadb_collection_name}")

            # Obtener recuento actual
            self._document_count = self.collection.count()

            self.is_initialized = True
            logger.info(f"✅ ChromaDB inicializado con {self._document_count} documentos")
            return True

        except Exception as e:
            logger.error(f"❌ Error inicializando ChromaDB: {e}")
            return False

    def _get_distance_function(self) -> str:
        """Mapea la métrica de similitud a función de distancia de ChromaDB"""
        if self.config.similarity_metric == "cosine":
            return "cosine"
        if self.config.similarity_metric == "euclidean":
            return "l2"
        if self.config.similarity_metric == "dot_product":
            return "ip"  # inner product
        logger.warning(f"⚠️  Métrica no reconocida: {self.config.similarity_metric}, usando cosine")
        return "cosine"

    def add_documents(self, documents: list[VectorDocument]) -> bool:
        """Añade documentos a ChromaDB"""
        if not self.is_initialized:
            logger.error("❌ ChromaDB no está inicializado")
            return False

        try:
            # Preparar datos para ChromaDB
            embeddings = []
            ids = []
            metadatas = []
            documents_texts = []

            for doc in documents:
                # Verificar dimensión
                if doc.embedding.shape[0] != self.config.embedding_dimension:
                    logger.error(f"❌ Dimensión incorrecta: esperada {self.config.embedding_dimension}, obtenida {doc.embedding.shape[0]}")
                    continue

                embeddings.append(doc.embedding.tolist())
                ids.append(doc.id)
                documents_texts.append(doc.text)

                # Preparar metadatos (ChromaDB requiere tipos específicos)
                metadata = {}
                if doc.category:
                    metadata["category"] = doc.category
                if doc.timestamp:
                    metadata["timestamp"] = float(doc.timestamp)
                if doc.audio_file_path:
                    metadata["audio_file_path"] = doc.audio_file_path

                # Añadir metadatos adicionales
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        # Convertir tipos complejos a string
                        metadata[key] = str(value)

                metadatas.append(metadata)

            if not embeddings:
                logger.warning("⚠️  No se encontraron documentos válidos para añadir")
                return False

            # Añadir a ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=documents_texts,
                metadatas=metadatas,
                ids=ids
            )

            self._document_count = self.collection.count()
            logger.info(f"✅ {len(embeddings)} documentos añadidos a ChromaDB")
            return True

        except Exception as e:
            logger.error(f"❌ Error añadiendo documentos a ChromaDB: {e}")
            return False

    def search(self, query_embedding: np.ndarray, k: int = 10,
               filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """Busca documentos similares usando ChromaDB"""
        if not self.is_initialized:
            return []

        try:
            start_time = time.time()

            # Preparar query
            query_embeddings = [query_embedding.tolist()]

            # Preparar filtros Where clause
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)

            # Realizar búsqueda
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances", "embeddings"]
            )

            search_time = time.time() - start_time

            # Convertir resultados
            search_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for rank, (doc_id, distance, text, metadata, embedding) in enumerate(zip(
                    results["ids"][0],
                    results["distances"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results.get("embeddings", [[]] * len(results["ids"][0]))[0] if results.get("embeddings") else [None] * len(results["ids"][0]), strict=False
                )):
                    # Convertir distancia a similitud
                    if self.config.similarity_metric == "cosine":
                        # ChromaDB devuelve distancia coseno (0=idéntico, 2=opuesto)
                        similarity_score = 1.0 - (distance / 2.0)
                    elif self.config.similarity_metric == "euclidean":
                        # Distancia euclidiana
                        similarity_score = 1.0 / (1.0 + distance)
                    else:
                        similarity_score = 1.0 - distance

                    # Reconstruir VectorDocument
                    doc_embedding = np.array(embedding) if embedding else query_embedding
                    doc = VectorDocument(
                        id=doc_id,
                        embedding=doc_embedding,
                        text=text,
                        metadata=metadata or {},
                        category=metadata.get("category") if metadata else None,
                        timestamp=metadata.get("timestamp") if metadata else None,
                        audio_file_path=metadata.get("audio_file_path") if metadata else None
                    )

                    result = SearchResult(
                        document=doc,
                        similarity_score=similarity_score,
                        distance=distance,
                        rank=rank + 1
                    )
                    search_results.append(result)

            logger.debug(f"🔍 Búsqueda ChromaDB completada en {search_time:.3f}s, {len(search_results)} resultados")
            return search_results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda ChromaDB: {e}")
            return []

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Construye cláusula WHERE para ChromaDB"""
        where_clause = {}

        for key, value in filters.items():
            if isinstance(value, list):
                # Filtro de lista (IN)
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                # Filtros complejos (rangos, etc.)
                where_clause[key] = value
            else:
                # Filtro de igualdad
                where_clause[key] = {"$eq": value}

        return where_clause

    def delete_document(self, document_id: str) -> bool:
        """Elimina un documento de ChromaDB"""
        if not self.is_initialized:
            return False

        try:
            self.collection.delete(ids=[document_id])
            self._document_count = self.collection.count()
            logger.info(f"🗑️  Documento {document_id} eliminado")
            return True

        except Exception as e:
            logger.error(f"❌ Error eliminando documento {document_id}: {e}")
            return False

    def get_document(self, document_id: str) -> VectorDocument | None:
        """Obtiene un documento por ID"""
        if not self.is_initialized:
            return None

        try:
            results = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas", "embeddings"]
            )

            if results["ids"] and len(results["ids"]) > 0:
                doc_id = results["ids"][0]
                text = results["documents"][0] if results["documents"] else ""
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                embedding = np.array(results["embeddings"][0]) if results["embeddings"] else np.zeros(self.config.embedding_dimension)

                return VectorDocument(
                    id=doc_id,
                    embedding=embedding,
                    text=text,
                    metadata=metadata,
                    category=metadata.get("category"),
                    timestamp=metadata.get("timestamp"),
                    audio_file_path=metadata.get("audio_file_path")
                )

            return None

        except Exception as e:
            logger.error(f"❌ Error obteniendo documento {document_id}: {e}")
            return None

    def update_document(self, document: VectorDocument) -> bool:
        """Actualiza un documento en ChromaDB"""
        if not self.is_initialized:
            return False

        try:
            # ChromaDB maneja updates como upserts
            metadata = {}
            if document.category:
                metadata["category"] = document.category
            if document.timestamp:
                metadata["timestamp"] = float(document.timestamp)
            if document.audio_file_path:
                metadata["audio_file_path"] = document.audio_file_path

            for key, value in document.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)

            self.collection.upsert(
                embeddings=[document.embedding.tolist()],
                documents=[document.text],
                metadatas=[metadata],
                ids=[document.id]
            )

            logger.info(f"📝 Documento {document.id} actualizado")
            return True

        except Exception as e:
            logger.error(f"❌ Error actualizando documento {document.id}: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Obtiene estadísticas de ChromaDB"""
        stats = {
            "db_type": "chromadb",
            "collection_name": self.config.chromadb_collection_name,
            "document_count": self._document_count,
            "embedding_dimension": self.config.embedding_dimension,
            "similarity_metric": self.config.similarity_metric,
            "distance_function": self._get_distance_function(),
            "is_initialized": self.is_initialized,
            "persistent": self.config.chromadb_path is not None,
            "storage_path": self.config.chromadb_path
        }

        if self.is_initialized and self.collection:
            try:
                # Obtener metadatos de la colección
                collection_metadata = self.collection.metadata
                if collection_metadata:
                    stats["collection_metadata"] = collection_metadata

                # Obtener estadísticas adicionales
                stats["actual_count"] = self.collection.count()

            except Exception as e:
                logger.debug(f"No se pudieron obtener estadísticas adicionales: {e}")

        return stats

    def save_index(self, path: str) -> bool:
        """ChromaDB maneja persistencia automáticamente"""
        if self.config.chromadb_path:
            logger.info(f"💾 ChromaDB usa persistencia automática en: {self.config.chromadb_path}")
            return True
        logger.warning("⚠️  ChromaDB en memoria, no se puede guardar")
        return False

    def load_index(self, path: str) -> bool:
        """ChromaDB carga automáticamente desde el directorio persistente"""
        if self.config.chromadb_path and Path(self.config.chromadb_path).exists():
            logger.info(f"📂 ChromaDB carga automáticamente desde: {self.config.chromadb_path}")
            return True
        logger.warning(f"⚠️  Directorio no encontrado: {path}")
        return False

    def clear(self) -> bool:
        """Limpia la colección ChromaDB"""
        if not self.is_initialized:
            return False

        try:
            # Eliminar colección existente
            self.client.delete_collection(name=self.config.chromadb_collection_name)

            # Recrear colección vacía
            distance_function = self._get_distance_function()
            self.collection = self.client.create_collection(
                name=self.config.chromadb_collection_name,
                metadata={"hnsw:space": distance_function}
            )

            self._document_count = 0
            logger.info("🧹 Colección ChromaDB limpiada")
            return True

        except Exception as e:
            logger.error(f"❌ Error limpiando ChromaDB: {e}")
            return False

    def get_all_documents(self, limit: int | None = None) -> list[VectorDocument]:
        """Obtiene todos los documentos de la colección"""
        if not self.is_initialized:
            return []

        try:
            # ChromaDB tiene límite en get(), usar paginación si es necesario
            if limit is None:
                limit = self._document_count

            results = self.collection.get(
                limit=min(limit, 1000),  # ChromaDB típicamente limita a 1000
                include=["documents", "metadatas", "embeddings"]
            )

            documents = []
            if results["ids"]:
                for doc_id, text, metadata, embedding in zip(
                    results["ids"],
                    results["documents"] or [],
                    results["metadatas"] or [],
                    results["embeddings"] or [], strict=False
                ):
                    doc = VectorDocument(
                        id=doc_id,
                        embedding=np.array(embedding) if embedding else np.zeros(self.config.embedding_dimension),
                        text=text or "",
                        metadata=metadata or {},
                        category=metadata.get("category") if metadata else None,
                        timestamp=metadata.get("timestamp") if metadata else None,
                        audio_file_path=metadata.get("audio_file_path") if metadata else None
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"❌ Error obteniendo todos los documentos: {e}")
            return []

    def get_collections(self) -> list[str]:
        """Obtiene lista de colecciones disponibles"""
        if not self.is_initialized:
            return []

        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]

        except Exception as e:
            logger.error(f"❌ Error obteniendo colecciones: {e}")
            return []
