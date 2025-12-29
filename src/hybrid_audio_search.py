#!/usr/bin/env python3
"""
Sistema híbrido de búsqueda de audio que combina:
1. Búsqueda por palabras clave (siempre funciona)
2. Búsqueda con embeddings YAMNet reales (si están disponibles)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from improved_audio_search import ImprovedAudioSearch
from vector_indexing import VectorIndexManager


class HybridAudioSearch:
    """Búsqueda híbrida que combina palabras clave + embeddings YAMNet"""

    def __init__(self, dataset_dir: str, logger=None):
        """
        Inicializa el sistema híbrido
        
        Args:
            dataset_dir: Directorio del dataset
            logger: A logger instance (optional)
        """
        self.dataset_dir = Path(dataset_dir)
        self.logger = logger or logging.getLogger(__name__)

        # Inicializar búsqueda por palabras clave (siempre disponible)
        self.keyword_search = ImprovedAudioSearch()

        # Verificar si hay embeddings YAMNet reales
        self.has_real_yamnet = self._check_real_yamnet_availability()

        # Cargar índice vectorial si está disponible
        self.index_manager = None
        if self.has_real_yamnet:
            self._load_vector_index()

        self.logger.info("🎵 Sistema híbrido inicializado:")
        self.logger.info("  🔑 Búsqueda por palabras clave: ✅ Disponible")
        self.logger.info(f"  🧠 Búsqueda con YAMNet real: {'✅ Disponible' if self.has_real_yamnet else '❌ No disponible'}")

    def _check_real_yamnet_availability(self) -> bool:
        """Verifica si el dataset tiene embeddings YAMNet reales"""
        try:
            # Verificar manifiesto
            manifest_file = self.dataset_dir / "final" / "dataset_manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)

                config = manifest.get('config', {})
                audio_model = config.get('audio_embedding_model', 'YAMNet')

                if audio_model == "YAMNet":
                    self.logger.info("✅ Detectado YAMNet real en manifiesto")
                    return True

            # Verificar metadatos de índices
            indices_metadata = self.dataset_dir / "indices" / "indices_metadata.json"
            if indices_metadata.exists():
                with open(indices_metadata) as f:
                    metadata = json.load(f)

                audio_model = metadata.get('audio_model', 'YAMNet')
                if audio_model == "YAMNet":
                    self.logger.info("✅ Detectado YAMNet real en índices")
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"⚠️  Error verificando YAMNet: {e}")
            return False

    def _load_vector_index(self):
        """Carga el índice vectorial si está disponible"""
        try:
            indices_dir = self.dataset_dir / "indices"
            if indices_dir.exists():
                # Cargar dimensión desde metadatos
                metadata_file = indices_dir / "indices_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    embedding_dim = metadata.get('embedding_dimension', 384)
                else:
                    embedding_dim = 1024  # YAMNet default

                self.index_manager = VectorIndexManager(embedding_dim=embedding_dim)
                self.index_manager.load_indices(str(indices_dir))
                self.logger.info("✅ Índices vectoriales cargados")
        except Exception as e:
            self.logger.warning(f"⚠️  Error cargando índices: {e}")
            self.index_manager = None

    def search_by_keywords(self, df: pd.DataFrame, query: str, k: int = 10) -> list[dict]:
        """
        Búsqueda por palabras clave (método 1)
        
        Args:
            df: DataFrame con transcripciones
            query: Consulta de texto
            k: Número de resultados
            
        Returns:
            Lista de resultados con scores de palabras clave
        """
        results = self.keyword_search.search_audio_by_text(df, query, k)

        # Marcar como búsqueda por palabras clave
        for result in results:
            result['search_method'] = 'keywords'
            result['keyword_score'] = result['score']

        return results

    def search_by_yamnet_embeddings(self, df: pd.DataFrame, query_audio_file: str, k: int = 10) -> list[dict]:
        """
        Búsqueda PURA por embeddings YAMNet usando archivo de audio como consulta
        
        Args:
            df: DataFrame con embeddings YAMNet
            query_audio_file: Ruta al archivo de audio de consulta
            k: Número de resultados
            
        Returns:
            Lista de resultados con scores de embeddings de audio
        """
        if not self.has_real_yamnet or not self.index_manager:
            return []

        try:
            # Cargar modelo YAMNet para generar embedding de consulta
            from audio_embeddings import get_audio_embedding_generator

            audio_embedder = get_audio_embedding_generator()

            # Generar embedding del archivo de consulta
            query_embedding = audio_embedder.generate_embedding(query_audio_file)

            if query_embedding is None:
                self.logger.error("❌ No se pudo generar embedding del archivo de consulta")
                return []

            # Buscar con índice FAISS
            distances, indices = self.index_manager.search_audio_index(query_embedding, k)

            results = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx < len(df):
                    row = df.iloc[idx]
                    result = {
                        'rank': i + 1,
                        'score': float(1 / (1 + distance)),
                        'text': row['text'],
                        'source_file': row['source_file'],
                        'start_time': row['start_time'],
                        'end_time': row['end_time'],
                        'duration': row['duration'],
                        'search_method': 'yamnet_pure',
                        'embedding_score': float(1 / (1 + distance)),
                        'query_audio_file': query_audio_file
                    }
                    results.append(result)

            return results

        except Exception as e:
            self.logger.warning(f"⚠️  Error en búsqueda por embeddings YAMNet: {e}")
            return []

    def search_by_yamnet_similarity(self, df: pd.DataFrame, reference_segment: dict, k: int = 10) -> list[dict]:
        """
        Búsqueda por similitud YAMNet usando un segmento de referencia del dataset
        
        Args:
            df: DataFrame con embeddings YAMNet
            reference_segment: Segmento de referencia con embedding
            k: Número de resultados
            
        Returns:
            Lista de resultados similares
        """
        if not self.has_real_yamnet or not self.index_manager:
            return []

        try:
            # Usar embedding del segmento de referencia
            if 'audio_embedding' not in reference_segment:
                self.logger.error("❌ Segmento de referencia no tiene embedding de audio")
                return []

            query_embedding = np.array(reference_segment['audio_embedding'])

            # Buscar con índice FAISS
            distances, indices = self.index_manager.search_audio_index(query_embedding, k + 1)  # +1 para excluir el mismo

            results = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx < len(df):
                    row = df.iloc[idx]

                    # Excluir el segmento de referencia
                    if (row['source_file'] == reference_segment.get('source_file') and
                        abs(row['start_time'] - reference_segment.get('start_time', -1)) < 0.1):
                        continue

                    result = {
                        'rank': len(results) + 1,
                        'score': float(1 / (1 + distance)),
                        'text': row['text'],
                        'source_file': row['source_file'],
                        'start_time': row['start_time'],
                        'end_time': row['end_time'],
                        'duration': row['duration'],
                        'search_method': 'yamnet_similarity',
                        'embedding_score': float(1 / (1 + distance)),
                        'reference_segment': f"{reference_segment.get('source_file', 'unknown')} @ {reference_segment.get('start_time', 0):.1f}s"
                    }
                    results.append(result)

                    if len(results) >= k:
                        break

            return results

        except Exception as e:
            self.logger.warning(f"⚠️  Error en búsqueda por similitud YAMNet: {e}")
            return []

    def _generate_query_embedding_from_classes(self, audio_classes: list[str]) -> np.ndarray | None:
        """
        Genera embedding de consulta basado en clases de audio
        Requiere audio de ejemplo real para funcionar correctamente
        """
        # Esta función requiere implementación con audio real
        # No se puede generar embeddings sin archivo de audio
        return None

    def search_hybrid(self, df: pd.DataFrame, query: str, k: int = 10,
                     keyword_weight: float = 0.7) -> list[dict]:
        """
        Búsqueda híbrida que combina ambos métodos
        
        Args:
            df: DataFrame con datos
            query: Consulta de texto
            k: Número de resultados
            keyword_weight: Peso para búsqueda por palabras clave (0-1)
            
        Returns:
            Lista de resultados combinados y rankeados
        """
        self.logger.info(f"🔍 Búsqueda híbrida para: '{query}'")

        # Método 1: Búsqueda por palabras clave
        keyword_results = self.search_by_keywords(df, query, k * 2)
        self.logger.info(f"🔑 Palabras clave: {len(keyword_results)} resultados")

        # Método 2: Búsqueda por embeddings (si disponible)
        embedding_results = []
        if self.has_real_yamnet:
            embedding_results = self.search_by_yamnet_embeddings(df, query, k * 2)
            self.logger.info(f"🧠 Embeddings YAMNet: {len(embedding_results)} resultados")
        else:
            self.logger.info("🧠 Embeddings YAMNet: No disponible")

        # Combinar resultados
        combined_results = self._combine_results(
            keyword_results,
            embedding_results,
            keyword_weight
        )

        # Tomar top k
        final_results = combined_results[:k]

        # Reasignar ranks
        for i, result in enumerate(final_results):
            result['rank'] = i + 1

        self.logger.info(f"🎯 Resultados finales: {len(final_results)}")
        return final_results

    def _combine_results(self, keyword_results: list[dict],
                        embedding_results: list[dict],
                        keyword_weight: float) -> list[dict]:
        """Combina y rankea resultados de ambos métodos"""

        embedding_weight = 1 - keyword_weight
        combined_scores = {}

        # Procesar resultados de palabras clave
        for result in keyword_results:
            key = f"{result['source_file']}_{result['start_time']}"
            combined_scores[key] = {
                'keyword_score': result['score'],
                'embedding_score': 0.0,
                'data': result,
                'methods': ['keywords']
            }

        # Procesar resultados de embeddings
        for result in embedding_results:
            key = f"{result['source_file']}_{result['start_time']}"
            if key in combined_scores:
                combined_scores[key]['embedding_score'] = result['score']
                combined_scores[key]['methods'].append('yamnet_embeddings')
            else:
                combined_scores[key] = {
                    'keyword_score': 0.0,
                    'embedding_score': result['score'],
                    'data': result,
                    'methods': ['yamnet_embeddings']
                }

        # Calcular scores combinados
        final_results = []
        for key, scores in combined_scores.items():
            combined_score = (keyword_weight * scores['keyword_score'] +
                            embedding_weight * scores['embedding_score'])

            result = scores['data'].copy()
            result.update({
                'score': combined_score,
                'keyword_score': scores['keyword_score'],
                'embedding_score': scores['embedding_score'],
                'search_methods': scores['methods'],
                'combination_weights': {
                    'keywords': keyword_weight,
                    'embeddings': embedding_weight
                }
            })

            final_results.append(result)

        # Ordenar por score combinado
        final_results.sort(key=lambda x: x['score'], reverse=True)

        return final_results

    def get_search_capabilities(self) -> dict:
        """Retorna las capacidades de búsqueda disponibles"""
        return {
            'keyword_search': True,
            'yamnet_embeddings': self.has_real_yamnet,
            'hybrid_search': True,
            'vector_index_available': self.index_manager is not None,
            'audio_classes_available': len(self.keyword_search.get_available_audio_classes())
        }

# Función de conveniencia
def create_hybrid_search(dataset_dir: str) -> HybridAudioSearch:
    """Crea una instancia del buscador híbrido"""
    return HybridAudioSearch(dataset_dir)

if __name__ == "__main__":
    # Ejemplo de uso
    import sys

    if len(sys.argv) != 2:
        logging.error("Uso: python hybrid_audio_search.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]

    # Crear buscador híbrido
    search_engine = create_hybrid_search(dataset_dir)

    # Mostrar capacidades
    capabilities = search_engine.get_search_capabilities()
    logging.info("\n🎯 Capacidades de búsqueda:")
    for capability, available in capabilities.items():
        status = "✅" if available else "❌"
        logging.info(f"  {status} {capability.replace('_', ' ').title()}: {available}")

    # Cargar dataset para prueba
    df = pd.read_pickle(Path(dataset_dir) / "final" / "complete_dataset.pkl")

    # Prueba de búsqueda
    query = "aplausos"
    logging.info(f"\n🔍 Prueba de búsqueda híbrida para: '{query}'")
    results = search_engine.search_hybrid(df, query, k=3)

    for result in results:
        logging.info(f"\n🏆 Rank {result['rank']} - Score: {result['score']:.3f}")
        logging.info(f"📝 Texto: {result['text'][:80]}...")
        logging.info(f"🔍 Métodos: {', '.join(result.get('search_methods', []))}")
        if 'keyword_score' in result:
            logging.info(f"🔑 Score palabras: {result['keyword_score']:.3f}")
        if 'embedding_score' in result:
            logging.info(f"🧠 Score embeddings: {result['embedding_score']:.3f}")
