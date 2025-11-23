"""
Implementación de base de datos vectorial usando FAISS
Optimizada para búsquedas rápidas de similitud en embeddings de audio.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import pickle
from pathlib import Path
import time

try:
    from .vector_database_interface import VectorDatabaseInterface, VectorDBConfig, VectorDocument, SearchResult
except ImportError:
    from vector_database_interface import VectorDatabaseInterface, VectorDBConfig, VectorDocument, SearchResult

logger = logging.getLogger(__name__)

# Imports condicionales para FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️  FAISS no disponible. Instala con: pip install faiss-cpu o faiss-gpu")

class FAISSVectorDatabase(VectorDatabaseInterface):
    """Implementación de base de datos vectorial usando FAISS"""

    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self.index = None
        self.documents: Dict[str, VectorDocument] = {}
        self.id_to_faiss_id: Dict[str, int] = {}
        self.faiss_id_to_id: Dict[int, str] = {}
        self.next_faiss_id = 0

        if not FAISS_AVAILABLE:
            raise ImportError("FAISS no está disponible. Instala faiss-cpu o faiss-gpu.")

    def initialize(self) -> bool:
        """Inicializa el índice FAISS"""
        try:
            logger.info(f"🚀 Inicializando FAISS con dimensión {self.config.embedding_dimension}")

            # Crear índice según configuración
            if self.config.faiss_index_type == "flat":
                if self.config.similarity_metric == "cosine":
                    # Para similitud coseno, usar dot product con vectores normalizados
                    self.index = faiss.IndexFlatIP(self.config.embedding_dimension)
                else:
                    # Distancia euclidiana
                    self.index = faiss.IndexFlatL2(self.config.embedding_dimension)

            elif self.config.faiss_index_type == "ivf":
                # Índice IVF para datasets grandes
                nlist = min(100, max(10, int(np.sqrt(1000))))  # Heurística para nlist
                quantizer = faiss.IndexFlatL2(self.config.embedding_dimension)

                if self.config.similarity_metric == "cosine":
                    self.index = faiss.IndexIVFFlat(quantizer, self.config.embedding_dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                else:
                    self.index = faiss.IndexIVFFlat(quantizer, self.config.embedding_dimension, nlist, faiss.METRIC_L2)

                # Configurar nprobe
                self.index.nprobe = self.config.faiss_nprobe

            elif self.config.faiss_index_type == "hnsw":
                # Índice HNSW para búsquedas rápidas
                if self.config.similarity_metric == "cosine":
                    self.index = faiss.IndexHNSWFlat(self.config.embedding_dimension, 32, faiss.METRIC_INNER_PRODUCT)
                else:
                    self.index = faiss.IndexHNSWFlat(self.config.embedding_dimension, 32, faiss.METRIC_L2)

            else:
                raise ValueError(f"Tipo de índice FAISS no soportado: {self.config.faiss_index_type}")

            # Configurar GPU si está disponible y solicitado
            if self.config.faiss_gpu and faiss.get_num_gpus() > 0:
                logger.info("🎮 Configurando FAISS para GPU")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

            # Cargar índice existente si se especifica
            if self.config.faiss_index_path and Path(self.config.faiss_index_path).exists():
                self.load_index(self.config.faiss_index_path)

            self.is_initialized = True
            logger.info(f"✅ FAISS inicializado: {self.config.faiss_index_type}")
            return True

        except Exception as e:
            logger.error(f"❌ Error inicializando FAISS: {e}")
            return False

    def add_documents(self, documents: List[VectorDocument]) -> bool:
        """Añade documentos al índice FAISS"""
        if not self.is_initialized:
            logger.error("❌ FAISS no está inicializado")
            return False

        try:
            # Preparar embeddings
            embeddings = []
            document_ids = []

            for doc in documents:
                # Verificar que el embedding tenga la dimensión correcta
                if doc.embedding.shape[0] != self.config.embedding_dimension:
                    logger.error(f"❌ Dimensión incorrecta: esperada {self.config.embedding_dimension}, obtenida {doc.embedding.shape[0]}")
                    continue

                embedding = doc.embedding.copy()

                # Normalizar para similitud coseno
                if self.config.similarity_metric == "cosine":
                    embedding = embedding / np.linalg.norm(embedding)

                embeddings.append(embedding)
                document_ids.append(doc.id)

                # Almacenar documento y mapeo de IDs
                self.documents[doc.id] = doc
                self.id_to_faiss_id[doc.id] = self.next_faiss_id
                self.faiss_id_to_id[self.next_faiss_id] = doc.id
                self.next_faiss_id += 1

            if not embeddings:
                logger.warning("⚠️  No se añadieron documentos válidos")
                return False

            # Convertir a array numpy
            embeddings_array = np.vstack(embeddings).astype(np.float32)

            # Entrenar índice si es necesario (para IVF)
            if self.config.faiss_index_type == "ivf" and not self.index.is_trained:
                logger.info("🏋️  Entrenando índice IVF...")
                self.index.train(embeddings_array)

            # Añadir al índice
            self.index.add(embeddings_array)
            self._document_count += len(embeddings)

            logger.info(f"✅ {len(embeddings)} documentos añadidos a FAISS")
            return True

        except Exception as e:
            logger.error(f"❌ Error añadiendo documentos a FAISS: {e}")
            return False

    def search(self, query_embedding: np.ndarray, k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Busca documentos similares usando FAISS"""
        if not self.is_initialized or self.index.ntotal == 0:
            return []

        try:
            # Preparar query
            query = query_embedding.copy().astype(np.float32)

            if self.config.similarity_metric == "cosine":
                query = query / np.linalg.norm(query)

            # Expandir dimensiones para FAISS
            query = query.reshape(1, -1)

            # Realizar búsqueda
            start_time = time.time()

            if filters:
                # Búsqueda con filtros (menos eficiente pero necesaria)
                return self._search_with_filters(query, k, filters)
            else:
                # Búsqueda directa
                distances, indices = self.index.search(query, k)

            search_time = time.time() - start_time

            # Convertir resultados
            results = []
            for rank, (distance, faiss_idx) in enumerate(zip(distances[0], indices[0])):
                if faiss_idx == -1:  # No hay más resultados
                    break

                doc_id = self.faiss_id_to_id.get(faiss_idx)
                if doc_id is None:
                    continue

                doc = self.documents.get(doc_id)
                if doc is None:
                    continue

                # Convertir distancia a similitud
                if self.config.similarity_metric == "cosine":
                    # FAISS devuelve inner product para IP, convertir a similitud coseno
                    similarity_score = float(distance)
                    distance_score = 1.0 - similarity_score
                else:
                    # Distancia euclidiana
                    distance_score = float(distance)
                    similarity_score = 1.0 / (1.0 + distance_score)

                result = SearchResult(
                    document=doc,
                    similarity_score=similarity_score,
                    distance=distance_score,
                    rank=rank + 1
                )
                results.append(result)

            logger.debug(f"🔍 Búsqueda FAISS completada en {search_time:.3f}s, {len(results)} resultados")
            return results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda FAISS: {e}")
            return []

    def _search_with_filters(self, query: np.ndarray, k: int,
                           filters: Dict[str, Any]) -> List[SearchResult]:
        """Búsqueda con filtros aplicados post-procesamiento"""
        # Buscar más resultados para compensar filtrado
        search_k = min(k * 10, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)

        results = []
        for rank, (distance, faiss_idx) in enumerate(zip(distances[0], indices[0])):
            if len(results) >= k or faiss_idx == -1:
                break

            doc_id = self.faiss_id_to_id.get(faiss_idx)
            if doc_id is None:
                continue

            doc = self.documents.get(doc_id)
            if doc is None:
                continue

            # Aplicar filtros
            if not self._matches_filters(doc, filters):
                continue

            # Convertir distancia a similitud
            if self.config.similarity_metric == "cosine":
                similarity_score = float(distance)
                distance_score = 1.0 - similarity_score
            else:
                distance_score = float(distance)
                similarity_score = 1.0 / (1.0 + distance_score)

            result = SearchResult(
                document=doc,
                similarity_score=similarity_score,
                distance=distance_score,
                rank=len(results) + 1
            )
            results.append(result)

        return results

    def _matches_filters(self, doc: VectorDocument, filters: Dict[str, Any]) -> bool:
        """Verifica si un documento coincide con los filtros"""
        for key, value in filters.items():
            if key == "category" and doc.category != value:
                return False
            elif key in doc.metadata and doc.metadata[key] != value:
                return False
        return True

    def delete_document(self, document_id: str) -> bool:
        """Elimina un documento (FAISS no soporta eliminación directa)"""
        logger.warning("⚠️  FAISS no soporta eliminación directa de documentos")
        logger.info("💡 Para eliminar documentos, reconstruye el índice sin los documentos no deseados")

        # Eliminar de estructuras internas pero mantener en FAISS
        if document_id in self.documents:
            faiss_id = self.id_to_faiss_id.get(document_id)
            if faiss_id is not None:
                del self.faiss_id_to_id[faiss_id]
                del self.id_to_faiss_id[document_id]
            del self.documents[document_id]
            self._document_count = len(self.documents)
            return True

        return False

    def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Obtiene un documento por ID"""
        return self.documents.get(document_id)

    def update_document(self, document: VectorDocument) -> bool:
        """Actualiza un documento (requiere eliminación y adición)"""
        if document.id in self.documents:
            # Actualizar metadatos en memoria
            self.documents[document.id] = document
            logger.info(f"📝 Metadatos actualizados para {document.id}")
            logger.warning("⚠️  Para actualizar embedding, elimina y añade el documento")
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice FAISS"""
        stats = {
            "db_type": "faiss",
            "index_type": self.config.faiss_index_type,
            "document_count": self._document_count,
            "faiss_ntotal": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.config.embedding_dimension,
            "similarity_metric": self.config.similarity_metric,
            "is_initialized": self.is_initialized,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            "gpu_enabled": self.config.faiss_gpu,
        }

        if self.config.faiss_index_type == "ivf":
            stats["nprobe"] = self.config.faiss_nprobe
            stats["nlist"] = getattr(self.index, 'nlist', 'unknown')

        return stats

    def save_index(self, path: str) -> bool:
        """Guarda el índice FAISS en disco"""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Guardar índice FAISS
            if self.config.faiss_gpu:
                # Mover a CPU antes de guardar
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(save_path))
            else:
                faiss.write_index(self.index, str(save_path))

            # Guardar metadatos y mapeos
            metadata_path = save_path.with_suffix('.metadata.pkl')
            metadata = {
                "documents": self.documents,
                "id_to_faiss_id": self.id_to_faiss_id,
                "faiss_id_to_id": self.faiss_id_to_id,
                "next_faiss_id": self.next_faiss_id,
                "config": self.config,
                "_document_count": self._document_count
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"💾 Índice FAISS guardado en: {path}")
            logger.info(f"💾 Metadatos guardados en: {metadata_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error guardando índice FAISS: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Carga el índice FAISS desde disco"""
        try:
            load_path = Path(path)
            if not load_path.exists():
                logger.warning(f"⚠️  Archivo de índice no encontrado: {path}")
                return False

            # Cargar índice FAISS
            self.index = faiss.read_index(str(load_path))

            # Configurar GPU si es necesario
            if self.config.faiss_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

            # Cargar metadatos
            metadata_path = load_path.with_suffix('.metadata.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                self.documents = metadata["documents"]
                self.id_to_faiss_id = metadata["id_to_faiss_id"]
                self.faiss_id_to_id = metadata["faiss_id_to_id"]
                self.next_faiss_id = metadata["next_faiss_id"]
                self._document_count = metadata["_document_count"]

                logger.info(f"📂 Índice FAISS cargado desde: {path}")
                logger.info(f"📊 {self._document_count} documentos cargados")
            else:
                logger.warning(f"⚠️  Metadatos no encontrados: {metadata_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Error cargando índice FAISS: {e}")
            return False

    def clear(self) -> bool:
        """Limpia el índice FAISS"""
        try:
            # Reinicializar índice
            self.documents.clear()
            self.id_to_faiss_id.clear()
            self.faiss_id_to_id.clear()
            self.next_faiss_id = 0
            self._document_count = 0

            # Recrear índice vacío
            self.initialize()

            logger.info("🧹 Índice FAISS limpiado")
            return True

        except Exception as e:
            logger.error(f"❌ Error limpiando índice FAISS: {e}")
            return False

    def rebuild_index(self, remove_document_ids: Optional[List[str]] = None) -> bool:
        """Reconstruye el índice excluyendo documentos específicos"""
        try:
            if remove_document_ids:
                # Remover documentos especificados
                for doc_id in remove_document_ids:
                    if doc_id in self.documents:
                        del self.documents[doc_id]

            # Crear nuevo índice
            old_index = self.index
            self.initialize()

            # Re-añadir todos los documentos
            if self.documents:
                documents = list(self.documents.values())
                self.documents.clear()
                self.id_to_faiss_id.clear()
                self.faiss_id_to_id.clear()
                self.next_faiss_id = 0
                self._document_count = 0

                success = self.add_documents(documents)
                if success:
                    logger.info(f"✅ Índice FAISS reconstruido con {len(documents)} documentos")
                    return True

            return False

        except Exception as e:
            logger.error(f"❌ Error reconstruyendo índice FAISS: {e}")
            return False