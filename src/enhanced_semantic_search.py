"""
Sistema de búsqueda semántica mejorado con soporte para múltiples bases de datos vectoriales.
Integra FAISS, ChromaDB y Supabase con el framework existente de embeddings de audio.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

try:
    from .vector_database_config import (
        ConfigurationPreset,
        VectorDatabaseConfigurator,
        get_configurator,
    )
    from .vector_database_interface import (
        SearchResult,
        VectorDatabaseInterface,
        VectorDBType,
        VectorDocument,
        create_vector_database,
    )
except ImportError:
    from vector_database_config import (
        ConfigurationPreset,
        VectorDatabaseConfigurator,
        get_configurator,
    )
    from vector_database_interface import (
        SearchResult,
        VectorDatabaseInterface,
        VectorDBType,
        VectorDocument,
        create_vector_database,
    )

logger = logging.getLogger(__name__)

# Imports existentes del sistema
try:
    from .audio_embeddings import get_audio_embedding_generator
    from .models_config import AudioEmbeddingModel, get_models_config
    AUDIO_EMBEDDINGS_AVAILABLE = True
except ImportError:
    AUDIO_EMBEDDINGS_AVAILABLE = False
    logger.warning("⚠️  Módulos de embeddings de audio no disponibles")

class EnhancedSemanticSearch:
    """
    Sistema de búsqueda semántica mejorado con soporte para múltiples bases de datos vectoriales
    """

    def __init__(self, configurator: VectorDatabaseConfigurator | None = None):
        """
        Inicializa el sistema de búsqueda semántica

        Args:
            configurator: Configurador de bases de datos vectoriales (usa global si None)
        """
        self.configurator = configurator or get_configurator()
        self.vector_db: VectorDatabaseInterface | None = None
        self.fallback_db: VectorDatabaseInterface | None = None

        # Generadores de embeddings
        self.audio_embedding_generator = None
        self.text_embedding_generator = None

        # Estadísticas
        self.search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "avg_search_time": 0.0,
            "fallback_uses": 0
        }

        self._initialize_components()

    def _initialize_components(self):
        """Inicializa todos los componentes del sistema"""
        try:
            # Inicializar base de datos principal
            self._initialize_vector_database()

            # Inicializar generadores de embeddings si están disponibles
            if AUDIO_EMBEDDINGS_AVAILABLE:
                self._initialize_embedding_generators()

            logger.info("✅ Sistema de búsqueda semántica mejorado inicializado")

        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")

    def _initialize_vector_database(self):
        """Inicializa la base de datos vectorial principal y fallback"""
        try:
            # Base de datos principal
            main_config = self.configurator.get_vector_db_config()
            self.vector_db = create_vector_database(main_config)

            if not self.vector_db.initialize():
                raise Exception(f"No se pudo inicializar {main_config.db_type.value}")

            logger.info(f"✅ Base de datos principal inicializada: {main_config.db_type.value}")

            # Base de datos de fallback si está habilitada
            if self.configurator.settings.auto_fallback:
                fallback_config = self.configurator.get_vector_db_config(
                    self.configurator.settings.fallback_database
                )

                if fallback_config.db_type != main_config.db_type:
                    self.fallback_db = create_vector_database(fallback_config)
                    if self.fallback_db.initialize():
                        logger.info(f"✅ Base de datos de fallback inicializada: {fallback_config.db_type.value}")
                    else:
                        logger.warning(f"⚠️  No se pudo inicializar fallback: {fallback_config.db_type.value}")
                        self.fallback_db = None

        except Exception as e:
            logger.error(f"❌ Error inicializando bases de datos vectoriales: {e}")
            # Usar memoria como fallback final
            from .vector_database_interface import MemoryVectorDatabase, VectorDBConfig
            fallback_config = VectorDBConfig(
                db_type=VectorDBType.MEMORY,
                embedding_dimension=self.configurator.settings.embedding_dimension
            )
            self.vector_db = MemoryVectorDatabase(fallback_config)
            self.vector_db.initialize()
            logger.info("🧠 Usando base de datos en memoria como fallback final")

    def _initialize_embedding_generators(self):
        """Inicializa generadores de embeddings"""
        try:
            # Generador de embeddings de audio
            self.audio_embedding_generator = get_audio_embedding_generator()
            logger.info("🎵 Generador de embeddings de audio inicializado")

            # Generador de embeddings de texto (si está disponible)
            try:
                from sentence_transformers import SentenceTransformer
                self.text_embedding_generator = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("📝 Generador de embeddings de texto inicializado")
            except ImportError:
                logger.warning("⚠️  SentenceTransformers no disponible para embeddings de texto")

        except Exception as e:
            logger.error(f"❌ Error inicializando generadores de embeddings: {e}")

    def add_audio_documents(self, audio_data: pd.DataFrame | list[dict[str, Any]]) -> bool:
        """
        Añade documentos de audio a la base de datos vectorial

        Args:
            audio_data: DataFrame o lista de diccionarios con información de audio

        Returns:
            True si se añadieron exitosamente
        """
        try:
            if isinstance(audio_data, pd.DataFrame):
                audio_data = audio_data.to_dict('records')

            documents = []
            batch_size = self.configurator.settings.batch_size

            logger.info(f"📊 Procesando {len(audio_data)} documentos de audio...")

            for i, record in enumerate(audio_data):
                # Generar embedding si no existe
                embedding = record.get('audio_embedding')
                if embedding is None:
                    if self.audio_embedding_generator is None:
                        logger.error("❌ Generador de embeddings de audio no disponible")
                        continue

                    audio_path = record.get('audio_file_path') or record.get('source_file')
                    if not audio_path:
                        logger.warning(f"⚠️  No se encontró ruta de audio para registro {i}")
                        continue

                    # Generar embedding
                    try:
                        embedding = self.audio_embedding_generator.generate_embedding(audio_path)
                    except Exception as e:
                        logger.error(f"❌ Error generando embedding para {audio_path}: {e}")
                        continue

                elif isinstance(embedding, list):
                    embedding = np.array(embedding)

                # Crear documento vectorial
                doc_id = record.get('id') or f"audio_{i}"
                text = record.get('text', '') or record.get('transcript', '')

                metadata = {
                    'start_time': record.get('start_time'),
                    'end_time': record.get('end_time'),
                    'confidence': record.get('confidence'),
                    'speaker': record.get('speaker'),
                    'emotion': record.get('emotion'),
                    'sentiment': record.get('sentiment_label'),
                    'source': record.get('source', 'audio_processing')
                }

                # Filtrar valores None
                metadata = {k: v for k, v in metadata.items() if v is not None}

                doc = VectorDocument(
                    id=doc_id,
                    embedding=embedding,
                    text=text,
                    metadata=metadata,
                    category=record.get('category'),
                    timestamp=record.get('timestamp') or time.time(),
                    audio_file_path=record.get('audio_file_path') or record.get('source_file')
                )

                documents.append(doc)

                # Procesar en lotes
                if len(documents) >= batch_size:
                    if not self._add_documents_batch(documents):
                        return False
                    documents = []

            # Procesar lote final
            if documents:
                if not self._add_documents_batch(documents):
                    return False

            logger.info(f"✅ {len(audio_data)} documentos de audio procesados exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error añadiendo documentos de audio: {e}")
            return False

    def _add_documents_batch(self, documents: list[VectorDocument]) -> bool:
        """Añade un lote de documentos con fallback automático"""
        try:
            # Intentar con base de datos principal
            if self.vector_db.add_documents(documents):
                return True

            # Intentar con fallback si está disponible
            if self.fallback_db:
                logger.warning("⚠️  Base de datos principal falló, usando fallback")
                self.search_stats["fallback_uses"] += 1

                if self.fallback_db.add_documents(documents):
                    return True

            return False

        except Exception as e:
            logger.error(f"❌ Error en lote de documentos: {e}")
            return False

    def search_by_text(self, query: str, k: int = 10,
                      filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """
        Busca documentos usando consulta de texto

        Args:
            query: Texto de búsqueda
            k: Número de resultados a retornar
            filters: Filtros adicionales

        Returns:
            Lista de resultados ordenados por similitud
        """
        try:
            start_time = time.time()
            self.search_stats["total_searches"] += 1

            # Generar embedding de texto
            if self.text_embedding_generator is None:
                logger.error("❌ Generador de embeddings de texto no disponible")
                return []

            query_embedding = self.text_embedding_generator.encode([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalizar

            # Realizar búsqueda
            results = self._search_with_fallback(query_embedding, k, filters)

            # Actualizar estadísticas
            search_time = time.time() - start_time
            self._update_search_stats(search_time, len(results) > 0)

            logger.debug(f"🔍 Búsqueda por texto completada: {len(results)} resultados en {search_time:.3f}s")
            return results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda por texto: {e}")
            return []

    def search_by_audio(self, audio_path: str, k: int = 10,
                       filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """
        Busca documentos usando archivo de audio

        Args:
            audio_path: Ruta al archivo de audio de consulta
            k: Número de resultados a retornar
            filters: Filtros adicionales

        Returns:
            Lista de resultados ordenados por similitud
        """
        try:
            start_time = time.time()
            self.search_stats["total_searches"] += 1

            # Generar embedding de audio
            if self.audio_embedding_generator is None:
                logger.error("❌ Generador de embeddings de audio no disponible")
                return []

            query_embedding = self.audio_embedding_generator.generate_embedding(audio_path)

            # Realizar búsqueda
            results = self._search_with_fallback(query_embedding, k, filters)

            # Actualizar estadísticas
            search_time = time.time() - start_time
            self._update_search_stats(search_time, len(results) > 0)

            logger.debug(f"🔍 Búsqueda por audio completada: {len(results)} resultados en {search_time:.3f}s")
            return results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda por audio: {e}")
            return []

    def search_by_embedding(self, embedding: np.ndarray, k: int = 10,
                          filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """
        Busca documentos usando embedding directo

        Args:
            embedding: Vector de embedding
            k: Número de resultados a retornar
            filters: Filtros adicionales

        Returns:
            Lista de resultados ordenados por similitud
        """
        try:
            start_time = time.time()
            self.search_stats["total_searches"] += 1

            # Realizar búsqueda
            results = self._search_with_fallback(embedding, k, filters)

            # Actualizar estadísticas
            search_time = time.time() - start_time
            self._update_search_stats(search_time, len(results) > 0)

            logger.debug(f"🔍 Búsqueda por embedding completada: {len(results)} resultados en {search_time:.3f}s")
            return results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda por embedding: {e}")
            return []

    def _search_with_fallback(self, query_embedding: np.ndarray, k: int,
                            filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """Realiza búsqueda con fallback automático"""
        try:
            # Intentar con base de datos principal
            results = self.vector_db.search(query_embedding, k, filters)
            if results:
                return results

            # Intentar con fallback si está disponible
            if self.fallback_db:
                logger.warning("⚠️  Base de datos principal no retornó resultados, usando fallback")
                self.search_stats["fallback_uses"] += 1
                results = self.fallback_db.search(query_embedding, k, filters)

            return results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda con fallback: {e}")
            return []

    def _update_search_stats(self, search_time: float, success: bool):
        """Actualiza estadísticas de búsqueda"""
        if success:
            self.search_stats["successful_searches"] += 1

        # Actualizar tiempo promedio
        total_searches = self.search_stats["total_searches"]
        current_avg = self.search_stats["avg_search_time"]
        self.search_stats["avg_search_time"] = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )

    def get_similar_documents(self, document_id: str, k: int = 5) -> list[SearchResult]:
        """
        Encuentra documentos similares a uno específico

        Args:
            document_id: ID del documento de referencia
            k: Número de documentos similares a retornar

        Returns:
            Lista de documentos similares
        """
        try:
            # Obtener documento de referencia
            doc = self.vector_db.get_document(document_id)
            if not doc:
                if self.fallback_db:
                    doc = self.fallback_db.get_document(document_id)

            if not doc:
                logger.error(f"❌ Documento no encontrado: {document_id}")
                return []

            # Buscar documentos similares
            results = self._search_with_fallback(doc.embedding, k + 1)  # +1 para excluir el mismo documento

            # Filtrar el documento original
            filtered_results = [r for r in results if r.document.id != document_id]
            return filtered_results[:k]

        except Exception as e:
            logger.error(f"❌ Error buscando documentos similares: {e}")
            return []

    def get_statistics(self) -> dict[str, Any]:
        """Obtiene estadísticas completas del sistema"""
        stats = {
            "search_statistics": self.search_stats.copy(),
            "vector_db_stats": self.vector_db.get_statistics() if self.vector_db else {},
            "fallback_db_stats": self.fallback_db.get_statistics() if self.fallback_db else {},
            "configuration": self.configurator.get_status_report(),
            "embeddings_available": {
                "audio": self.audio_embedding_generator is not None,
                "text": self.text_embedding_generator is not None
            }
        }

        return stats

    def switch_database(self, new_db_type: VectorDBType, migrate_data: bool = True) -> bool:
        """
        Cambia la base de datos vectorial activa

        Args:
            new_db_type: Nuevo tipo de base de datos
            migrate_data: Si migrar datos existentes

        Returns:
            True si el cambio fue exitoso
        """
        try:
            logger.info(f"🔄 Cambiando base de datos a: {new_db_type.value}")

            # Migrar datos si se solicita
            migrated_data = []
            if migrate_data and self.vector_db.get_document_count() > 0:
                logger.info("📦 Migrando datos existentes...")
                # Para migración completa, necesitaríamos implementar get_all_documents()
                # Por ahora, solo registramos la intención
                logger.warning("⚠️  Migración de datos no implementada completamente")

            # Actualizar configuración
            self.configurator.switch_database(new_db_type)

            # Reinicializar sistema
            old_db = self.vector_db
            self._initialize_vector_database()

            # Añadir datos migrados si existen
            if migrated_data:
                self._add_documents_batch(migrated_data)

            logger.info(f"✅ Base de datos cambiada exitosamente a: {new_db_type.value}")
            return True

        except Exception as e:
            logger.error(f"❌ Error cambiando base de datos: {e}")
            return False

    def save_index(self, path: str | None = None) -> bool:
        """Guarda el índice de la base de datos"""
        try:
            if path is None:
                timestamp = int(time.time())
                path = f"{self.configurator.settings.data_directory}/backup_{timestamp}"

            Path(path).parent.mkdir(parents=True, exist_ok=True)

            success = self.vector_db.save_index(path)
            if success:
                logger.info(f"💾 Índice guardado en: {path}")

            return success

        except Exception as e:
            logger.error(f"❌ Error guardando índice: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Carga un índice guardado"""
        try:
            success = self.vector_db.load_index(path)
            if success:
                logger.info(f"📂 Índice cargado desde: {path}")

            return success

        except Exception as e:
            logger.error(f"❌ Error cargando índice: {e}")
            return False

# Función de conveniencia para crear instancia configurada
def create_enhanced_search(preset: ConfigurationPreset | None = None,
                         config_file: str | None = None) -> EnhancedSemanticSearch:
    """
    Crea una instancia configurada del sistema de búsqueda semántica

    Args:
        preset: Preset de configuración a aplicar
        config_file: Archivo de configuración personalizado

    Returns:
        Instancia configurada de EnhancedSemanticSearch
    """
    configurator = get_configurator(config_file)

    if preset:
        configurator.apply_preset(preset)

    return EnhancedSemanticSearch(configurator)

# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Crear sistema de búsqueda para desarrollo
    search_system = create_enhanced_search(ConfigurationPreset.DEVELOPMENT)

    # Mostrar estadísticas
    stats = search_system.get_statistics()
    print(f"📊 Estadísticas del sistema: {json.dumps(stats, indent=2)}")

    # Ejemplo de búsqueda por texto
    if search_system.text_embedding_generator:
        results = search_system.search_by_text("política económica", k=5)
        print(f"🔍 Resultados de búsqueda: {len(results)}")

    logger.info("✅ Sistema de búsqueda semántica mejorado listo")
