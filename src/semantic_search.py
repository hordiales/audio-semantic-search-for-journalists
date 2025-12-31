import logging
import os

import pandas as pd

from audio_embeddings import get_audio_embedding_generator
from audio_transcription import AudioTranscriber
from sentiment_analysis import SentimentAnalyzer
from text_embeddings import TextEmbeddingGenerator, TextPreprocessor
from vector_indexing import VectorIndexManager

# Cargar configuración automáticamente
try:
    from config_loader import get_config
    SYSTEM_CONFIG = get_config()
except ImportError:
    SYSTEM_CONFIG = None


class SemanticSearchEngine:
    """
    Motor de búsqueda semántica para contenido de audio transcrito
    """

    def __init__(self, config: dict | None = None, logger=None):
        """
        Inicializa el motor de búsqueda semántica

        Args:
            config: Configuración del motor de búsqueda
            logger: A logger instance (optional)
        """
        self.config = config or self._default_config()
        self.log = logger.info if logger else print
        self.log_error = logger.error if logger else lambda msg: print(msg, file=sys.stderr)

        # Inicializar componentes
        self.transcriber = AudioTranscriber(model_name=self.config['whisper_model'])
        self.text_embedder = TextEmbeddingGenerator(model_name=self.config['text_embedding_model'])
        self.audio_embedder = get_audio_embedding_generator()
        self.sentiment_analyzer = SentimentAnalyzer(logger=logger)
        self.index_manager = VectorIndexManager(
            embedding_dim=self.text_embedder.embedding_dim,
            index_type=self.config['index_type']
        )

        # Estado del sistema
        self.is_indexed = False
        self.processed_data = None

    def _default_config(self) -> dict:
        """
        Configuración por defecto del sistema usando configuración centralizada

        Returns:
            Diccionario con configuración por defecto
        """
        # Usar configuración del sistema si está disponible
        if SYSTEM_CONFIG:
            return {
                'whisper_model': SYSTEM_CONFIG.default_whisper_model,
                'text_embedding_model': SYSTEM_CONFIG.default_text_model,
                'index_type': SYSTEM_CONFIG.index_type,
                'segmentation_method': SYSTEM_CONFIG.segmentation_method,
                'min_silence_len': SYSTEM_CONFIG.min_silence_len,
                'silence_thresh': SYSTEM_CONFIG.silence_thresh,
                'segment_duration': SYSTEM_CONFIG.segment_duration,
                'top_k_results': SYSTEM_CONFIG.max_results_default,
                'combine_scores': True,
                'text_weight': SYSTEM_CONFIG.text_weight,
                'audio_weight': SYSTEM_CONFIG.audio_weight
            }

        # Fallback a configuración manual si no hay sistema de configuración
        return {
            'whisper_model': 'base',
            'text_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'index_type': 'cosine',
            'segmentation_method': 'silence',
            'min_silence_len': 500,
            'silence_thresh': -40,
            'segment_duration': 10.0,
            'top_k_results': 5,
            'combine_scores': True,
            'text_weight': 0.7,
            'audio_weight': 0.3
        }

    def process_audio_files(self, audio_files: list[str],
                           output_dir: str = "processed_data") -> pd.DataFrame:
        """
        Procesa archivos de audio: transcripción y generación de embeddings

        Args:
            audio_files: Lista de rutas a archivos de audio
            output_dir: Directorio para guardar datos procesados

        Returns:
            DataFrame con todos los datos procesados
        """
        self.log(f"Procesando {len(audio_files)} archivos de audio...")

        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)

        # Paso 1: Transcripción
        self.log("Paso 1: Transcripción de audio...")
        if self.config['segmentation_method'] == 'silence':
            transcription_df = self.transcriber.process_multiple_files(
                audio_files,
                segmentation_method='silence',
                min_silence_len=self.config['min_silence_len'],
                silence_thresh=self.config['silence_thresh']
            )
        else:
            transcription_df = self.transcriber.process_multiple_files(
                audio_files,
                segmentation_method='time',
                segment_duration=self.config['segment_duration']
            )

        if len(transcription_df) == 0:
            self.log("No se pudo procesar ningún archivo de audio")
            return pd.DataFrame()

        self.log(f"Transcripción completada: {len(transcription_df)} segmentos")

        # Paso 2: Preprocesamiento de texto
        self.log("Paso 2: Preprocesamiento de texto...")
        transcription_df = TextPreprocessor.preprocess_dataframe(transcription_df)

        # Paso 3: Generación de embeddings de texto
        self.log("Paso 3: Generación de embeddings de texto...")
        text_embeddings_df = self.text_embedder.process_transcription_dataframe(transcription_df)

        # Paso 4: Generación de embeddings de audio
        self.log("Paso 4: Generación de embeddings de audio...")
        audio_df = self.audio_embedder.process_transcription_dataframe(text_embeddings_df)

        # Paso 5: Análisis de sentimientos
        self.log("Paso 5: Análisis de sentimientos...")
        full_df = self.sentiment_analyzer.process_dataframe(audio_df)

        # Guardar datos procesados
        output_path = os.path.join(output_dir, "processed_segments.pkl")
        full_df.to_pickle(output_path)
        self.log(f"Datos procesados guardados en: {output_path}")

        # Guardar también en formato JSON para inspección
        json_path = os.path.join(output_dir, "processed_segments.json")
        # Convertir embeddings a listas para JSON
        json_df = full_df.copy()
        for col in ['text_embedding', 'audio_embedding']:
            if col in json_df.columns:
                json_df[col] = json_df[col].apply(lambda x: x if isinstance(x, list) else [])

        json_df.to_json(json_path, orient='records', indent=2)

        self.processed_data = full_df
        return full_df

    def create_indices(self, df: pd.DataFrame = None,
                      index_dir: str = "indices") -> bool:
        """
        Crea índices vectoriales para búsqueda

        Args:
            df: DataFrame con embeddings (usa self.processed_data si es None)
            index_dir: Directorio para guardar índices

        Returns:
            True si se crearon exitosamente
        """
        if df is None:
            df = self.processed_data

        if df is None or len(df) == 0:
            self.log_error("No hay datos para indexar")
            return False

        self.log("Creando índices vectoriales...")

        # Crear índice de texto
        text_success = self.index_manager.create_text_index(df)

        # Crear índice de audio
        audio_success = self.index_manager.create_audio_index(df)

        if text_success or audio_success:
            # Guardar índices
            self.index_manager.save_indices(index_dir)
            self.is_indexed = True
            self.log("Índices creados exitosamente")
            return True
        self.log_error("Error creando índices")
        return False

    def load_indices(self, index_dir: str = "indices") -> bool:
        """
        Carga índices previamente creados

        Args:
            index_dir: Directorio con los índices

        Returns:
            True si se cargaron exitosamente
        """
        try:
            self.index_manager.load_indices(index_dir)
            self.is_indexed = True
            self.log("Índices cargados exitosamente")
            return True
        except Exception as e:
            self.log_error(f"Error cargando índices: {e}")
            return False

    def search(self, query: str, search_type: str = "combined",
               top_k: int | None = None, sentiment_filter: str | None = None) -> pd.DataFrame:
        """
        Realiza búsqueda semántica

        Args:
            query: Consulta en lenguaje natural
            search_type: Tipo de búsqueda ('text', 'audio', 'combined', 'sentiment')
            top_k: Número de resultados (usa config si es None)
            sentiment_filter: Filtro de sentimiento (ej: "feliz", "triste", "enojado")

        Returns:
            DataFrame con resultados ordenados por relevancia
        """
        if not self.is_indexed:
            raise ValueError("Sistema no indexado. Ejecutar create_indices() primero.")

        if top_k is None:
            top_k = self.config['top_k_results']

        results = []

        # Búsqueda por sentimiento
        if search_type == 'sentiment':
            sentiment_results = self._search_sentiment(query, top_k)
            if len(sentiment_results) > 0:
                sentiment_results['search_type'] = 'sentiment'
                return sentiment_results
            return pd.DataFrame()

        # Búsqueda en texto
        if search_type in ['text', 'combined']:
            text_results = self._search_text(query, top_k)
            if len(text_results) > 0:
                text_results['search_type'] = 'text'
                results.append(text_results)

        # Búsqueda en audio (por ahora solo usando embeddings de audio)
        if search_type in ['audio', 'combined']:
            # Para búsqueda de audio, podríamos usar el embedding de texto de la consulta
            # como proxy para buscar en embeddings de audio similares
            audio_results = self._search_audio_by_text(query, top_k)
            if len(audio_results) > 0:
                audio_results['search_type'] = 'audio'
                results.append(audio_results)

        # Aplicar filtro de sentimiento si se especifica
        if sentiment_filter and results:
            filtered_results = []
            for result_df in results:
                filtered_df = self._apply_sentiment_filter(result_df, sentiment_filter)
                if len(filtered_df) > 0:
                    filtered_results.append(filtered_df)
            results = filtered_results

        # Combinar resultados
        if not results:
            return pd.DataFrame()

        if search_type == 'combined' and len(results) > 1:
            combined_results = self._combine_results(results[0], results[1])
            return combined_results
        return results[0]

    def _search_text(self, query: str, top_k: int) -> pd.DataFrame:
        """
        Busca en el índice de texto

        Args:
            query: Consulta
            top_k: Número de resultados

        Returns:
            DataFrame con resultados
        """
        # Generar embedding de la consulta
        query_embedding = self.text_embedder.generate_query_embedding(query)

        if query_embedding.size == 0:
            return pd.DataFrame()

        # Buscar en índice
        try:
            distances, indices = self.index_manager.search_text_index(query_embedding, top_k)
            results = self.index_manager.get_text_results(distances, indices)
            results['query'] = query
            return results
        except Exception as e:
            self.log_error(f"Error en búsqueda de texto: {e}")
            return pd.DataFrame()

    def _search_audio_by_text(self, query: str, top_k: int) -> pd.DataFrame:
        """
        Busca en el índice de audio usando consulta de texto
        (implementación simplificada)

        Args:
            query: Consulta
            top_k: Número de resultados

        Returns:
            DataFrame con resultados
        """
        # Por ahora, buscar en texto y retornar como resultados de audio
        # En una implementación completa, se podría usar un modelo cross-modal
        text_results = self._search_text(query, top_k)

        if len(text_results) > 0:
            audio_results = text_results.copy()
            audio_results.rename(columns={'similarity_score': 'audio_similarity_score'}, inplace=True)
            return audio_results

        return pd.DataFrame()

    def _search_sentiment(self, query: str, top_k: int) -> pd.DataFrame:
        """
        Busca textos por sentimiento/estado de ánimo

        Args:
            query: Consulta de sentimiento (ej: "feliz", "triste", "enojado")
            top_k: Número de resultados

        Returns:
            DataFrame con resultados filtrados por sentimiento
        """
        if self.processed_data is None:
            self.log_error("No hay datos procesados disponibles")
            return pd.DataFrame()

        # Verificar si el dataset tiene análisis de sentimientos
        sentiment_columns = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'dominant_sentiment']
        if not any(col in self.processed_data.columns for col in sentiment_columns):
            self.log("Datos no tienen análisis de sentimientos. Procesando...")
            self.processed_data = self.sentiment_analyzer.process_dataframe(self.processed_data)

        # Buscar por sentimiento
        try:
            results = self.sentiment_analyzer.search_by_sentiment(
                self.processed_data,
                query,
                threshold=0.4,  # Umbral más permisivo
                top_k=top_k
            )
            return results
        except Exception as e:
            self.log_error(f"Error en búsqueda de sentimientos: {e}")
            return pd.DataFrame()

    def _apply_sentiment_filter(self, df: pd.DataFrame, sentiment_filter: str) -> pd.DataFrame:
        """
        Aplica filtro de sentimiento a resultados existentes

        Args:
            df: DataFrame con resultados de búsqueda
            sentiment_filter: Filtro de sentimiento

        Returns:
            DataFrame filtrado por sentimiento
        """
        try:
            # Verificar si hay análisis de sentimientos en los datos
            sentiment_columns = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
            if not any(col in df.columns for col in sentiment_columns):
                # Aplicar análisis de sentimientos a los resultados
                df = self.sentiment_analyzer.process_dataframe(df)

            # Aplicar filtro
            filtered_df = self.sentiment_analyzer.search_by_sentiment(
                df,
                sentiment_filter,
                threshold=0.3,  # Umbral más bajo para filtros
                top_k=len(df)  # Mantener todos los resultados que pasen el filtro
            )

            return filtered_df

        except Exception as e:
            self.log_error(f"Error aplicando filtro de sentimientos: {e}")
            return df  # Retornar resultados originales si hay error

    def _combine_results(self, text_results: pd.DataFrame,
                        audio_results: pd.DataFrame) -> pd.DataFrame:
        """
        Combina resultados de búsqueda de texto y audio

        Args:
            text_results: Resultados de búsqueda de texto
            audio_results: Resultados de búsqueda de audio

        Returns:
            DataFrame con resultados combinados
        """
        # Crear diccionario para combinar scores
        combined_scores = {}

        # Procesar resultados de texto
        for idx, row in text_results.iterrows():
            key = (row.get('segment_id', idx), row.get('source_file', ''))
            combined_scores[key] = {
                'text_score': row.get('similarity_score', 0),
                'audio_score': 0,
                'data': row
            }

        # Procesar resultados de audio
        for idx, row in audio_results.iterrows():
            key = (row.get('segment_id', idx), row.get('source_file', ''))
            if key in combined_scores:
                combined_scores[key]['audio_score'] = row.get('audio_similarity_score', 0)
            else:
                combined_scores[key] = {
                    'text_score': 0,
                    'audio_score': row.get('audio_similarity_score', 0),
                    'data': row
                }

        # Calcular scores combinados
        combined_results = []
        for key, scores in combined_scores.items():
            combined_score = (
                scores['text_score'] * self.config['text_weight'] +
                scores['audio_score'] * self.config['audio_weight']
            )

            result_row = scores['data'].copy()
            result_row['combined_score'] = combined_score
            result_row['text_score'] = scores['text_score']
            result_row['audio_score'] = scores['audio_score']
            result_row['search_type'] = 'combined'

            combined_results.append(result_row)

        # Crear DataFrame y ordenar
        if combined_results:
            combined_df = pd.DataFrame(combined_results)
            combined_df = combined_df.sort_values('combined_score', ascending=False)
            return combined_df.head(self.config['top_k_results'])

        return pd.DataFrame()

    def get_available_sentiments(self) -> list[str]:
        """
        Retorna lista de sentimientos disponibles para búsqueda

        Returns:
            Lista de estados de ánimo soportados
        """
        return self.sentiment_analyzer.get_available_moods()

    def get_sentiment_distribution(self) -> dict[str, int]:
        """
        Obtiene la distribución de sentimientos en el dataset actual

        Returns:
            Diccionario con conteo de sentimientos
        """
        if self.processed_data is None:
            return {}

        return self.sentiment_analyzer.get_sentiment_distribution(self.processed_data)

    def get_system_stats(self) -> dict:
        """
        Retorna estadísticas del sistema

        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'is_indexed': self.is_indexed,
            'processed_segments': len(self.processed_data) if self.processed_data is not None else 0,
            'config': self.config,
            'available_sentiments': self.get_available_sentiments(),
            'sentiment_distribution': self.get_sentiment_distribution()
        }

        if self.is_indexed:
            stats.update(self.index_manager.get_index_stats())

        return stats

    def export_results(self, results: pd.DataFrame, output_path: str):
        """
        Exporta resultados de búsqueda

        Args:
            results: DataFrame con resultados
            output_path: Ruta del archivo de salida
        """
        # Seleccionar columnas relevantes para exportar
        export_columns = [
            'text', 'start_time', 'end_time', 'source_file',
            'similarity_score', 'audio_similarity_score', 'combined_score',
            'query', 'search_type'
        ]

        available_columns = [col for col in export_columns if col in results.columns]
        export_df = results[available_columns]

        # Exportar según extensión
        if output_path.endswith('.csv'):
            export_df.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            export_df.to_json(output_path, orient='records', indent=2)
        elif output_path.endswith('.xlsx'):
            export_df.to_excel(output_path, index=False)
        else:
            export_df.to_pickle(output_path)

        self.log(f"Resultados exportados a: {output_path}")


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración personalizada
    config = {
        'whisper_model': 'base',
        'segmentation_method': 'time',
        'segment_duration': 15.0,
        'top_k_results': 10
    }

    # Crear motor de búsqueda
    search_engine = SemanticSearchEngine(config)

    # Ejemplo de procesamiento
    # audio_files = ['audio1.wav', 'audio2.mp3']
    # processed_df = search_engine.process_audio_files(audio_files)
    # search_engine.create_indices(processed_df)

    # Ejemplo de búsqueda
    # results = search_engine.search("economía y inflación", search_type="combined")
    # print(results[['text', 'combined_score', 'start_time', 'source_file']])

    logging.info("Motor de búsqueda semántica listo.")
