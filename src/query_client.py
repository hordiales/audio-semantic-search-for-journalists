#!/usr/bin/env python3
"""
Cliente de consola para hacer consultas al dataset de audio
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import List, Dict, Tuple
import json
from datetime import datetime
import cmd

# Importar componentes del sistema
from text_embeddings import TextEmbeddingGenerator
from audio_embeddings import get_audio_embedding_generator
from vector_indexing import VectorIndexManager
from semantic_audio_mapping import semantic_mapper, validate_audio_query, suggest_audio_queries
from audioset_ontology import AUDIOSET_CLASSES
from improved_audio_search import ImprovedAudioSearch
from hybrid_audio_search import HybridAudioSearch
from search_config import SearchConfig
from semantic_search import SemanticSearchEngine
from sentiment_analysis import SentimentAnalyzer

class AudioDatasetClient:
    """Cliente para consultas al dataset de audio"""
    
    def __init__(self, dataset_dir: str, config_file: str = None):
        """
        Inicializa el cliente
        
        Args:
            dataset_dir: Directorio del dataset
            config_file: Archivo de configuración (opcional)
        """
        self.dataset_dir = Path(dataset_dir)
        self.df = None
        self.text_embedder = None
        self.audio_embedder = None
        self.index_manager = None
        self.manifest = None
        self.sentiment_search_engine = None
        self.sentiment_enabled = False
        
        # Cargar configuración
        if config_file:
            self.config = SearchConfig.load_from_file(config_file)
        else:
            config_path = self.dataset_dir / "search_config.json"
            self.config = SearchConfig.load_from_file(str(config_path))
        
        # Cargar dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Carga el dataset y componentes"""
        print("🔄 Cargando dataset...")
        
        # Cargar dataset principal
        dataset_file = self.dataset_dir / "final" / "complete_dataset.pkl"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_file}")
        
        self.df = pd.read_pickle(dataset_file)
        print(f"✅ Dataset cargado: {len(self.df):,} segmentos")
        
        # Cargar manifiesto
        manifest_file = self.dataset_dir / "final" / "dataset_manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                self.manifest = json.load(f)
        
        # Inicializar embedders
        if self.manifest and 'config' in self.manifest:
            config = self.manifest['config']
            text_model = config.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2')
        else:
            text_model = 'sentence-transformers/all-MiniLM-L6-v2'
        
        print(f"🧠 Inicializando embedders...")
        self.text_embedder = TextEmbeddingGenerator(model_name=text_model)
        self.audio_embedder = get_audio_embedding_generator()
        
        # Cargar índices vectoriales
        indices_dir = self.dataset_dir / "indices"
        if indices_dir.exists():
            print(f"🔍 Cargando índices vectoriales...")
            self.index_manager = VectorIndexManager(embedding_dim=self.text_embedder.embedding_dim)
            self.index_manager.load_indices(str(indices_dir))
            print(f"✅ Índices cargados")
        else:
            print(f"⚠️  No se encontraron índices vectoriales")
        
        # Inicializar sistemas de búsqueda de audio
        self.improved_audio_search = ImprovedAudioSearch()
        self.hybrid_audio_search = HybridAudioSearch(str(self.dataset_dir))
        print(f"🎵 Sistemas de búsqueda de audio inicializados")
        
        # Intentar cargar sentiment search si el dataset lo soporta
        self._initialize_sentiment_search()
    
    def search_text(self, query: str, k: int = 5) -> List[Dict]:
        """
        Busca por texto usando embeddings
        
        Args:
            query: Consulta de texto
            k: Número de resultados
            
        Returns:
            Lista de resultados
        """
        print(f"🔍 Buscando: '{query}'")
        
        # Generar embedding de la consulta
        query_embedding = self.text_embedder.generate_query_embedding(query)
        
        if self.index_manager and self.index_manager.text_index is not None:
            # Usar índice FAISS
            distances, indices = self.index_manager.search_text_index(query_embedding, k)
            results = []
            
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                row = self.df.iloc[idx]
                result = {
                    'rank': i + 1,
                    'score': float(1 / (1 + distance)),  # Convertir distancia a score
                    'text': row['text'],
                    'source_file': row['source_file'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'duration': row['duration']
                }
                results.append(result)
            
            # Filtrar por score
            filtered_results = self.config.filter_results_by_score(results, 'text')
            
            if len(filtered_results) < len(results):
                print(f"🔍 Filtrados {len(results) - len(filtered_results)} resultados por umbral de score ({self.config.min_text_score})")
            
            return filtered_results
        else:
            # Búsqueda por similaridad coseno manual
            print("⚠️  Usando búsqueda manual (sin índices)")
            
            # Calcular similaridades
            embeddings = np.stack(self.df['text_embedding'].values)
            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Obtener top k
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for i, idx in enumerate(top_indices):
                row = self.df.iloc[idx]
                result = {
                    'rank': i + 1,
                    'score': float(similarities[idx]),
                    'text': row['text'],
                    'source_file': row['source_file'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'duration': row['duration']
                }
                results.append(result)
            
            # Filtrar por score
            filtered_results = self.config.filter_results_by_score(results, 'text')
            
            if len(filtered_results) < len(results):
                print(f"🔍 Filtrados {len(results) - len(filtered_results)} resultados por umbral de score ({self.config.min_text_score})")
            
            return filtered_results
    
    def search_audio(self, query_text: str, k: int = 5) -> List[Dict]:
        """
        Busca por audio usando palabras clave en transcripciones
        
        Args:
            query_text: Texto para buscar clases de audio relacionadas
            k: Número de resultados
            
        Returns:
            Lista de resultados basados en palabras clave reales
        """
        print(f"🔊 Buscando audio para: '{query_text}' (búsqueda por palabras clave)")
        
        # Usar el sistema de búsqueda mejorado
        results = self.improved_audio_search.search_audio_by_text(self.df, query_text, k)
        
        # Filtrar por score
        filtered_results = self.config.filter_results_by_score(results, 'keyword')
        
        if filtered_results:
            print(f"✅ Encontrados {len(filtered_results)} segmentos con contenido de audio relevante")
            
            if len(filtered_results) < len(results):
                print(f"🔍 Filtrados {len(results) - len(filtered_results)} resultados por umbral de score ({self.config.min_keyword_score})")
            
            # Mostrar clases de audio detectadas
            detected_classes = set()
            for result in filtered_results:
                if 'audio_class' in result:
                    detected_classes.add(result['audio_class'])
            
            if detected_classes:
                print(f"🎵 Clases de audio detectadas: {', '.join(detected_classes)}")
        else:
            print("❌ No se encontraron segmentos con palabras clave de audio relevantes")
            
            # Mostrar clases disponibles
            available_classes = self.improved_audio_search.get_available_audio_classes()
            print("💡 Clases de audio disponibles:")
            for audio_class in available_classes[:10]:
                keywords = self.improved_audio_search.get_keywords_for_class(audio_class)
                print(f"  • {audio_class}: {', '.join(keywords[:3])}...")
            
            if len(available_classes) > 10:
                print(f"  ... y {len(available_classes) - 10} más")
        
        return filtered_results
    
    def search_combined(self, query: str, k: int = 5, text_weight: float = 0.7) -> List[Dict]:
        """
        Búsqueda combinada de texto y audio
        
        Args:
            query: Consulta
            k: Número de resultados
            text_weight: Peso para texto (audio_weight = 1 - text_weight)
            
        Returns:
            Lista de resultados combinados
        """
        print(f"🔄 Búsqueda combinada: '{query}'")
        
        # Obtener resultados de texto y audio
        text_results = self.search_text(query, k * 2)  # Obtener más para combinar
        audio_results = self.search_audio(query, k * 2)
        
        # Combinar puntuaciones
        combined_scores = {}
        audio_weight = 1 - text_weight
        
        # Puntuaciones de texto
        for result in text_results:
            file_time_key = f"{result['source_file']}_{result['start_time']}"
            combined_scores[file_time_key] = {
                'text_score': result['score'],
                'audio_score': 0.0,
                'data': result
            }
        
        # Puntuaciones de audio
        for result in audio_results:
            file_time_key = f"{result['source_file']}_{result['start_time']}"
            if file_time_key in combined_scores:
                combined_scores[file_time_key]['audio_score'] = result['score']
            else:
                combined_scores[file_time_key] = {
                    'text_score': 0.0,
                    'audio_score': result['score'],
                    'data': result
                }
        
        # Calcular puntuaciones combinadas
        final_results = []
        for key, scores in combined_scores.items():
            combined_score = (text_weight * scores['text_score'] + 
                            audio_weight * scores['audio_score'])
            
            result = scores['data'].copy()
            result['score'] = combined_score
            result['text_score'] = scores['text_score']
            result['audio_score'] = scores['audio_score']
            final_results.append(result)
        
        # Ordenar por puntuación combinada
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Reasignar ranks
        for i, result in enumerate(final_results[:k]):
            result['rank'] = i + 1
        
        return final_results[:k]
    
    def _initialize_sentiment_search(self):
        """Inicializa el sistema de búsqueda por sentimientos si es posible"""
        try:
            # Verificar si el dataset tiene análisis de sentimientos
            sentiment_columns = [col for col in self.df.columns if 'sentiment' in col.lower()] if self.df is not None else []
            if self.df is not None and len(sentiment_columns) > 0:
                print("🎭 Inicializando sistema de análisis de sentimientos...")
                
                # Configurar semantic search engine para sentiment
                config = {
                    'whisper_model': 'base',
                    'text_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'index_type': 'cosine',
                    'top_k_results': 10
                }
                
                self.sentiment_search_engine = SemanticSearchEngine(config)
                self.sentiment_search_engine.processed_data = self.df
                
                # Crear índices si es necesario
                if 'text_embedding' in self.df.columns:
                    self.sentiment_search_engine.create_indices(self.df)
                
                self.sentiment_enabled = True
                print("✅ Sistema de sentimientos habilitado")
            else:
                print("⚠️  Dataset sin análisis de sentimientos - funcionalidad limitada")
        except Exception as e:
            print(f"⚠️  No se pudo inicializar sentiment search: {e}")
            self.sentiment_enabled = False
    
    def search_by_sentiment(self, sentiment: str, k: int = 5) -> List[Dict]:
        """Busca contenido por sentimiento/estado de ánimo"""
        if not self.sentiment_enabled:
            print("❌ Sistema de sentimientos no disponible")
            return []
        
        try:
            print(f"🎭 Buscando contenido con sentimiento: '{sentiment}'")
            
            results_df = self.sentiment_search_engine.search(
                sentiment,
                search_type="sentiment", 
                top_k=k
            )
            
            if len(results_df) == 0:
                return []
            
            # Convertir a formato estándar
            results = []
            for i, (_, row) in enumerate(results_df.iterrows()):
                result = {
                    'rank': i + 1,
                    'score': row.get('similarity_score', 0.0),
                    'text': row.get('text', ''),
                    'source_file': row.get('source_file', ''),
                    'start_time': row.get('start_time', 0),
                    'end_time': row.get('end_time', 0),
                    'duration': row.get('duration', 0),
                    'sentiment_score': row.get('sentiment_positive', 0.0),
                    'dominant_sentiment': row.get('dominant_sentiment', 'UNKNOWN'),
                    'search_method': 'sentiment'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error en búsqueda por sentimiento: {e}")
            return []
    
    def search_combined_with_sentiment(self, query: str, sentiment_filter: str = None, k: int = 5) -> List[Dict]:
        """Búsqueda combinada de texto con filtro de sentimiento"""
        if not self.sentiment_enabled:
            print("❌ Sistema de sentimientos no disponible")
            return self.search_text(query, k)  # Fallback a búsqueda de texto
        
        try:
            print(f"🔍 Búsqueda combinada: '{query}'" + (f" + sentimiento '{sentiment_filter}'" if sentiment_filter else ""))
            
            results_df = self.sentiment_search_engine.search(
                query,
                search_type="text",
                top_k=k,
                sentiment_filter=sentiment_filter
            )
            
            if len(results_df) == 0:
                return []
            
            # Convertir a formato estándar
            results = []
            for i, (_, row) in enumerate(results_df.iterrows()):
                result = {
                    'rank': i + 1,
                    'score': row.get('similarity_score', 0.0),
                    'text': row.get('text', ''),
                    'source_file': row.get('source_file', ''),
                    'start_time': row.get('start_time', 0),
                    'end_time': row.get('end_time', 0),
                    'duration': row.get('duration', 0),
                    'sentiment_score': row.get('sentiment_positive', 0.0),
                    'dominant_sentiment': row.get('dominant_sentiment', 'UNKNOWN'),
                    'search_method': 'text_with_sentiment'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error en búsqueda combinada: {e}")
            return []
    
    def analyze_content_mood(self, topic: str) -> Dict:
        """Analiza el estado de ánimo general del contenido sobre un tema"""
        if not self.sentiment_enabled:
            print("❌ Sistema de sentimientos no disponible")
            return {}
        
        try:
            print(f"📊 Analizando estado de ánimo sobre: '{topic}'")
            
            # Buscar contenido sobre el tema
            results_df = self.sentiment_search_engine.search(topic, search_type="text", top_k=50)
            
            if len(results_df) == 0:
                return {'error': f'No se encontró contenido sobre {topic}'}
            
            # Analizar distribución de sentimientos
            sentiments = []
            for _, row in results_df.iterrows():
                if hasattr(row, 'dominant_sentiment') and row.dominant_sentiment:
                    sentiments.append(row.dominant_sentiment)
            
            if not sentiments:
                return {'error': 'No hay datos de sentimiento disponibles'}
            
            # Calcular estadísticas
            sentiment_counts = pd.Series(sentiments).value_counts()
            total = len(sentiments)
            
            positive_count = sentiment_counts.get('POSITIVE', 0)
            negative_count = sentiment_counts.get('NEGATIVE', 0)
            neutral_count = sentiment_counts.get('NEUTRAL', 0)
            
            # Determinar estado de ánimo general
            if positive_count > negative_count:
                overall_mood = "POSITIVE"
                mood_emoji = "😊"
            elif negative_count > positive_count:
                overall_mood = "NEGATIVE"
                mood_emoji = "😢"
            else:
                overall_mood = "NEUTRAL"
                mood_emoji = "😐"
            
            return {
                'topic': topic,
                'total_segments': total,
                'overall_mood': overall_mood,
                'mood_emoji': mood_emoji,
                'distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'percentages': {
                    'positive': round(positive_count/total*100, 1),
                    'negative': round(negative_count/total*100, 1),
                    'neutral': round(neutral_count/total*100, 1)
                }
            }
            
        except Exception as e:
            print(f"❌ Error en análisis de estado de ánimo: {e}")
            return {'error': str(e)}
    
    def get_available_sentiments(self) -> List[str]:
        """Obtiene lista de sentimientos disponibles"""
        if not self.sentiment_enabled:
            return []
        
        try:
            return self.sentiment_search_engine.get_available_sentiments()
        except:
            return ['feliz', 'triste', 'enojado', 'neutral', 'optimista', 'preocupado']
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del dataset"""
        return {
            'total_segments': len(self.df),
            'unique_files': self.df['source_file'].nunique(),
            'total_duration': self.df['duration'].sum(),
            'avg_segment_duration': self.df['duration'].mean(),
            'text_avg_length': self.df['text'].str.len().mean(),
            'date_range': {
                'min': self.df['start_time'].min(),
                'max': self.df['end_time'].max()
            }
        }
    
    def print_results(self, results: List[Dict], show_details: bool = True):
        """Imprime resultados de búsqueda"""
        if not results:
            print("❌ No se encontraron resultados")
            return
        
        print(f"\n📊 Encontrados {len(results)} resultados:")
        print("-" * 80)
        
        for result in results:
            score = result['score']
            print(f"\n🏆 Rank {result['rank']} - Score: {score:.3f}")
            
            # Mostrar interpretación del score si está configurado
            if self.config.show_score_details:
                method = result.get('search_method', 'text')
                interpretation = self.config.get_score_interpretation(score, method)
                print(f"📊 Calidad: {interpretation}")
            
            print(f"📁 Archivo: {result['source_file']}")
            print(f"⏱️  Tiempo: {result['start_time']:.1f}s - {result['end_time']:.1f}s ({result['duration']:.1f}s)")
            
            if show_details:
                # Mostrar scores individuales si están disponibles
                if 'text_score' in result and 'audio_score' in result:
                    print(f"📝 Score texto: {result['text_score']:.3f} | 🔊 Score audio: {result['audio_score']:.3f}")
                
                # Mostrar información de sentimiento si está disponible
                if 'sentiment_score' in result and result['sentiment_score'] is not None:
                    print(f"🎭 Sentimiento: {result['sentiment_score']:.3f}")
                if 'dominant_sentiment' in result and result['dominant_sentiment']:
                    emoji = {"POSITIVE": "😊", "NEGATIVE": "😢", "NEUTRAL": "😐"}.get(result['dominant_sentiment'], "❓")
                    print(f"💭 Estado: {emoji} {result['dominant_sentiment']}")
                
                # Mostrar métodos de búsqueda utilizados
                if 'search_methods' in result and self.config.show_method_breakdown:
                    methods_str = ', '.join(result['search_methods'])
                    print(f"🔍 Métodos: {methods_str}")
                
                # Mostrar clases de audio coincidentes
                if 'matched_audio_classes' in result and result['matched_audio_classes']:
                    classes_str = ', '.join(result['matched_audio_classes'])
                    print(f"🎵 Clases de audio: {classes_str}")
                
                # Mostrar palabras clave encontradas
                if 'matched_keywords' in result and result['matched_keywords']:
                    keywords_str = ', '.join(result['matched_keywords'])
                    print(f"🔑 Palabras clave: {keywords_str}")
                
                # Mostrar clase de audio detectada
                if 'audio_class' in result:
                    print(f"🔊 Tipo de audio: {result['audio_class']}")
                
                # Mostrar consulta de audio si está disponible
                if 'audio_query' in result:
                    print(f"🔍 Consulta audio: {result['audio_query']}")
            
            # Mostrar texto (limitado)
            text = result['text']
            text_length = self.config.truncate_text_length
            if len(text) > text_length:
                text = text[:text_length] + "..."
            print(f"📝 Texto: {text}")
            
            if show_details:
                print(f"🔗 Contexto: {result['source_file']} @ {result['start_time']:.1f}s")

class InteractiveClient(cmd.Cmd):
    """Cliente interactivo de consola"""
    
    intro = """
🎵 Cliente Interactivo de Búsqueda de Audio
==========================================
Comandos disponibles:
  search <consulta>        - Búsqueda por texto semántico
  audio <consulta>         - Búsqueda por palabras clave de audio
  yamnet <archivo>         - Búsqueda YAMNet pura con archivo de audio
  similar <índice>         - Encontrar similares a un segmento
  hybrid <consulta>        - Búsqueda híbrida (texto + audio)
  sentiment <emoción>      - Búsqueda por sentimiento/emoción
  mood <query> [sentimiento] - Búsqueda con filtro de sentimiento
  analyze <tema>           - Analizar estado de ánimo sobre un tema
  sentiments               - Listar sentimientos disponibles
  suggest <consulta>       - Sugerencias de audio para texto
  validate <consulta>      - Validar consulta de audio
  audioset                 - Mostrar clases de AudioSet
  browse [número]          - Explorar segmentos del dataset
  find <texto>             - Buscar texto para obtener índices
  config                   - Mostrar configuración de búsqueda
  threshold <método> <valor> - Cambiar umbral de score
  capabilities             - Mostrar capacidades del sistema
  stats                    - Estadísticas del dataset
  help                     - Ayuda
  quit                     - Salir

Ejemplos:
  search economía política
  audio aplausos
  yamnet manifestación
  sentiment feliz
  mood política optimista
  analyze elecciones
  sentiments
  similar 150
  browse 10
  find aplausos
  config
  threshold text 0.6
  suggest política
  validate applause
"""
    
    prompt = "🔍 > "
    
    def __init__(self, client: AudioDatasetClient):
        super().__init__()
        self.client = client
    
    def do_search(self, arg):
        """Búsqueda por texto"""
        if not arg:
            print("❌ Proporciona una consulta: search <consulta>")
            return
        
        try:
            results = self.client.search_text(arg)
            self.client.print_results(results)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_audio(self, arg):
        """Búsqueda por audio"""
        if not arg:
            print("❌ Proporciona una consulta: audio <consulta>")
            return
        
        try:
            results = self.client.search_audio(arg)
            self.client.print_results(results)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_yamnet(self, arg):
        """Búsqueda YAMNet pura con archivo de audio"""
        if not arg:
            print("❌ Proporciona ruta del archivo de audio: yamnet <archivo_audio>")
            return
        
        try:
            # Verificar si el archivo existe
            audio_file = Path(arg)
            if not audio_file.exists():
                print(f"❌ Archivo no encontrado: {arg}")
                return
            
            # Usar búsqueda pura por embeddings YAMNet
            results = self.client.hybrid_audio_search.search_by_yamnet_embeddings(self.client.df, str(audio_file))
            if results:
                self.client.print_results(results)
            else:
                print("❌ No se encontraron resultados YAMNet (verificar embeddings disponibles)")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_similar(self, arg):
        """Encontrar similares a un segmento por índice"""
        if not arg:
            print("❌ Proporciona un índice: similar <índice>")
            return
        
        try:
            idx = int(arg)
            if idx < 0 or idx >= len(self.client.df):
                print(f"❌ Índice fuera de rango. Use 0-{len(self.client.df)-1}")
                return
            
            # Obtener segmento de referencia
            reference_segment = self.client.df.iloc[idx].to_dict()
            
            # Buscar similares usando YAMNet
            results = self.client.hybrid_audio_search.search_by_yamnet_similarity(self.client.df, reference_segment)
            if results:
                print(f"🔍 Buscando similares a: {reference_segment['source_file']} @ {reference_segment['start_time']:.1f}s")
                print(f"📝 Texto referencia: {reference_segment['text'][:80]}...")
                self.client.print_results(results)
            else:
                print("❌ No se encontraron similares (verificar embeddings YAMNet)")
        except ValueError:
            print("❌ El índice debe ser un número")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_hybrid(self, arg):
        """Búsqueda híbrida (texto + audio)"""
        if not arg:
            print("❌ Proporciona una consulta: hybrid <consulta>")
            return
        
        try:
            results = self.client.hybrid_audio_search.search_hybrid(self.client.df, arg)
            self.client.print_results(results)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_combined(self, arg):
        """Búsqueda combinada (método legacy)"""
        if not arg:
            print("❌ Proporciona una consulta: combined <consulta>")
            return
        
        try:
            results = self.client.search_combined(arg)
            self.client.print_results(results)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_capabilities(self, arg):
        """Muestra capacidades del sistema"""
        try:
            capabilities = self.client.hybrid_audio_search.get_search_capabilities()
            print("\n🎯 Capacidades del Sistema de Búsqueda:")
            print("-" * 40)
            
            status_icon = lambda x: "✅" if x else "❌"
            print(f"{status_icon(capabilities['keyword_search'])} Búsqueda por palabras clave")
            print(f"{status_icon(capabilities['yamnet_embeddings'])} Embeddings YAMNet reales")
            print(f"{status_icon(capabilities['hybrid_search'])} Búsqueda híbrida")
            print(f"{status_icon(capabilities['vector_index_available'])} Índices vectoriales")
            print(f"{status_icon(self.client.sentiment_enabled)} Análisis de sentimientos")
            print(f"📊 Clases de audio disponibles: {capabilities['audio_classes_available']}")
            
            if self.client.sentiment_enabled:
                sentiments_count = len(self.client.get_available_sentiments())
                print(f"🎭 Sentimientos disponibles: {sentiments_count}")
            
            print("\n🚀 Comandos disponibles:")
            print("  • 'search <consulta>' para búsqueda semántica")
            print("  • 'audio <consulta>' para búsqueda por palabras clave")
            
            if capabilities['yamnet_embeddings']:
                print("  • 'yamnet <archivo>' para búsqueda pura YAMNet con audio")
                print("  • 'similar <índice>' para encontrar segmentos similares")
                print("  • 'hybrid <consulta>' para búsqueda híbrida avanzada")
            
            if self.client.sentiment_enabled:
                print("  • 'sentiment <emoción>' para búsqueda por sentimiento")
                print("  • 'mood <query> [sentimiento]' para búsqueda con filtro emocional")
                print("  • 'analyze <tema>' para análisis de estado de ánimo")
                print("  • 'sentiments' para ver emociones disponibles")
            
            if not capabilities['yamnet_embeddings']:
                print("\n⚠️  Para habilitar YAMNet, ejecuta el procesamiento completo")
            
            if not self.client.sentiment_enabled:
                print("\n⚠️  Para habilitar sentimientos, procesa el dataset con análisis emocional")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_stats(self, arg):
        """Muestra estadísticas del dataset"""
        try:
            stats = self.client.get_stats()
            print("\n📊 Estadísticas del Dataset:")
            print(f"  📄 Total segmentos: {stats['total_segments']:,}")
            print(f"  📁 Archivos únicos: {stats['unique_files']:,}")
            print(f"  ⏱️  Duración total: {stats['total_duration']:.1f}s ({stats['total_duration']/3600:.1f}h)")
            print(f"  📊 Segmento promedio: {stats['avg_segment_duration']:.1f}s")
            print(f"  📝 Texto promedio: {stats['text_avg_length']:.1f} caracteres")
            
            # Mostrar estadísticas de sentimientos si están disponibles
            if self.client.sentiment_enabled and 'dominant_sentiment' in self.client.df.columns:
                sentiment_counts = self.client.df['dominant_sentiment'].value_counts()
                print(f"\n🎭 Distribución de Sentimientos:")
                for sentiment, count in sentiment_counts.head(5).items():
                    percentage = (count / len(self.client.df) * 100)
                    emoji = {"POSITIVE": "😊", "NEGATIVE": "😢", "NEUTRAL": "😐"}.get(sentiment, "❓")
                    print(f"  {emoji} {sentiment}: {count:,} ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_suggest(self, arg):
        """Sugiere consultas de audio para un texto"""
        if not arg:
            print("❌ Proporciona un texto: suggest <texto>")
            return
        
        try:
            suggestions = suggest_audio_queries(arg)
            if suggestions:
                print(f"\n💡 Sugerencias de audio para '{arg}':")
                for i, suggestion in enumerate(suggestions[:10], 1):
                    print(f"  {i:2d}. {suggestion['query']:<20} - {suggestion['name']}")
                    if suggestion['description']:
                        print(f"      {suggestion['description']}")
            else:
                print(f"❌ No se encontraron sugerencias para '{arg}'")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_validate(self, arg):
        """Valida una consulta de audio"""
        if not arg:
            print("❌ Proporciona una consulta: validate <consulta>")
            return
        
        try:
            validation = validate_audio_query(arg)
            if validation['valid']:
                info = validation['info']
                print(f"✅ '{arg}' es una clase AudioSet válida")
                print(f"📝 Nombre: {info.get('name', arg)}")
                print(f"📖 Descripción: {info.get('description', 'N/A')}")
            else:
                print(f"❌ '{arg}' no es una clase AudioSet válida")
                
                if validation.get('similar_classes'):
                    print("\n🔍 Clases similares:")
                    for cls in validation['similar_classes'][:5]:
                        print(f"  • {cls['class']} - {cls['name']}")
                
                if validation.get('semantic_suggestions'):
                    print("\n💡 Sugerencias semánticas:")
                    for suggestion in validation['semantic_suggestions'][:5]:
                        print(f"  • {suggestion['query']} - {suggestion['name']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_audioset(self, arg):
        """Muestra información sobre AudioSet"""
        try:
            from audioset_ontology import AUDIOSET_CATEGORIES, get_all_categories
            
            if not arg:
                print("\n🎵 Categorías de AudioSet:")
                categories = get_all_categories()
                for i, category in enumerate(categories, 1):
                    class_count = len(AUDIOSET_CATEGORIES[category])
                    print(f"  {i:2d}. {category.replace('_', ' ').title():<20} ({class_count} clases)")
                print(f"\nUsa 'audioset <categoría>' para ver las clases específicas")
            else:
                category = arg.lower().replace(' ', '_')
                if category in AUDIOSET_CATEGORIES:
                    classes = AUDIOSET_CATEGORIES[category]
                    print(f"\n🎵 Clases de audio en '{category.replace('_', ' ').title()}':")
                    for i, class_name in enumerate(classes, 1):
                        info = AUDIOSET_CLASSES.get(class_name, {})
                        name = info.get('name', class_name)
                        print(f"  {i:2d}. {class_name:<20} - {name}")
                else:
                    print(f"❌ Categoría '{arg}' no encontrada")
                    print("💡 Categorías disponibles:", ', '.join(get_all_categories()))
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_quit(self, arg):
        """Salir del cliente"""
        print("👋 ¡Hasta luego!")
        return True
    
    def do_browse(self, arg):
        """Explorar segmentos del dataset"""
        try:
            limit = int(arg) if arg else 20
            
            print(f"\n📋 Explorando {limit} segmentos del dataset:")
            print("=" * 100)
            print(f"{'Índice':<6} {'Archivo':<20} {'Tiempo':<12} {'Duración':<8} {'Texto':<40}")
            print("-" * 100)
            
            for i, (idx, row) in enumerate(self.client.df.head(limit).iterrows()):
                if i >= limit:
                    break
                    
                archivo = row['source_file']
                if len(archivo) > 19:
                    archivo = archivo[:16] + "..."
                
                tiempo = f"{row['start_time']:.1f}-{row['end_time']:.1f}s"
                duracion = f"{row['duration']:.1f}s"
                
                texto = row['text']
                if len(texto) > 39:
                    texto = texto[:36] + "..."
                
                print(f"{idx:<6} {archivo:<20} {tiempo:<12} {duracion:<8} {texto:<40}")
            
            print(f"\n💡 Para buscar similares a un segmento: similar <índice>")
            print(f"   Ejemplo: similar {self.client.df.index[0]}")
            
        except ValueError:
            print("❌ El número debe ser un entero")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_find(self, arg):
        """Buscar texto para obtener índices"""
        if not arg:
            print("❌ Proporciona texto a buscar: find <texto>")
            return
        
        try:
            # Buscar en el texto
            mask = self.client.df['text'].str.contains(arg, case=False, na=False)
            results = self.client.df[mask].head(10)
            
            if len(results) == 0:
                print(f"❌ No se encontraron segmentos con '{arg}'")
                return
            
            print(f"\n🔍 Encontrados {len(results)} segmentos con '{arg}':")
            print("=" * 100)
            print(f"{'Índice':<6} {'Archivo':<20} {'Tiempo':<12} {'Duración':<8} {'Texto':<40}")
            print("-" * 100)
            
            for idx, row in results.iterrows():
                archivo = row['source_file']
                if len(archivo) > 19:
                    archivo = archivo[:16] + "..."
                
                tiempo = f"{row['start_time']:.1f}-{row['end_time']:.1f}s"
                duracion = f"{row['duration']:.1f}s"
                
                texto = row['text']
                if len(texto) > 39:
                    texto = texto[:36] + "..."
                
                print(f"{idx:<6} {archivo:<20} {tiempo:<12} {duracion:<8} {texto:<40}")
            
            print(f"\n💡 Usa cualquiera de estos índices con similar:")
            print(f"   similar {results.index[0]}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_config(self, arg):
        """Mostrar configuración de búsqueda"""
        try:
            config = self.client.config
            print("\\n⚙️  Configuración de Búsqueda:")
            print("=" * 50)
            
            print(f"🎯 Modo de calidad: {config.quality_mode}")
            print(f"📊 Resultados por defecto: {config.default_results_count}")
            print(f"📏 Longitud máxima texto: {config.truncate_text_length}")
            
            print("\\n🔍 Umbrales de Score:")
            print(f"  📝 Texto: {config.min_text_score:.2f}")
            print(f"  🔊 Audio: {config.min_audio_score:.2f}")
            print(f"  🎵 YAMNet: {config.min_yamnet_score:.2f}")
            print(f"  🔑 Palabras clave: {config.min_keyword_score:.2f}")
            print(f"  🔄 Híbrida: {config.min_hybrid_score:.2f}")
            
            print("\\n⚖️  Pesos Híbridos:")
            print(f"  📝 Texto: {config.hybrid_text_weight:.2f}")
            print(f"  🔊 Audio: {config.hybrid_audio_weight:.2f}")
            
            print("\\n🎛️  Opciones:")
            print(f"  📊 Detalles de score: {'✅' if config.show_score_details else '❌'}")
            print(f"  🔍 Desglose métodos: {'✅' if config.show_method_breakdown else '❌'}")
            
            print("\\n💡 Usar 'threshold <método> <valor>' para cambiar umbrales")
            print("   Métodos: text, audio, yamnet, keyword, hybrid")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_threshold(self, arg):
        """Cambiar umbral de score"""
        if not arg:
            print("❌ Uso: threshold <método> <valor>")
            print("   Métodos: text, audio, yamnet, keyword, hybrid")
            print("   Valor: 0.0 - 1.0")
            return
        
        try:
            parts = arg.split()
            if len(parts) != 2:
                print("❌ Uso: threshold <método> <valor>")
                return
            
            method, value_str = parts
            value = float(value_str)
            
            if not 0.0 <= value <= 1.0:
                print("❌ El valor debe estar entre 0.0 y 1.0")
                return
            
            # Cambiar umbral
            config = self.client.config
            if method == 'text':
                old_value = config.min_text_score
                config.min_text_score = value
            elif method == 'audio':
                old_value = config.min_audio_score
                config.min_audio_score = value
            elif method == 'yamnet':
                old_value = config.min_yamnet_score
                config.min_yamnet_score = value
            elif method == 'keyword':
                old_value = config.min_keyword_score
                config.min_keyword_score = value
            elif method == 'hybrid':
                old_value = config.min_hybrid_score
                config.min_hybrid_score = value
            else:
                print(f"❌ Método desconocido: {method}")
                print("   Métodos disponibles: text, audio, yamnet, keyword, hybrid")
                return
            
            print(f"✅ Umbral {method} cambiado: {old_value:.2f} → {value:.2f}")
            
            # Mostrar interpretación
            interpretation = config.get_score_interpretation(value, method)
            print(f"📊 Nuevo umbral considera '{interpretation}' como mínimo")
            
        except ValueError:
            print("❌ El valor debe ser un número válido")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_sentiment(self, arg):
        """Búsqueda por sentimiento"""
        if not arg:
            print("❌ Proporciona un sentimiento: sentiment <emoción>")
            print("💡 Ejemplos: feliz, triste, optimista, preocupado, neutral")
            return
        
        try:
            results = self.client.search_by_sentiment(arg)
            if results:
                self.client.print_results(results)
            else:
                print(f"❌ No se encontraron resultados para sentimiento '{arg}'")
                if self.client.sentiment_enabled:
                    sentiments = self.client.get_available_sentiments()
                    if sentiments:
                        print(f"💡 Sentimientos disponibles: {', '.join(sentiments[:10])}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_mood(self, arg):
        """Búsqueda con filtro de sentimiento"""
        if not arg:
            print("❌ Uso: mood <consulta> [sentimiento]")
            print("   Ejemplos: mood política optimista")
            print("             mood economía (sin filtro)")
            return
        
        parts = arg.split()
        if len(parts) == 1:
            query = parts[0]
            sentiment_filter = None
        elif len(parts) >= 2:
            query = parts[0]
            sentiment_filter = parts[1]
        else:
            print("❌ Formato incorrecto")
            return
        
        try:
            results = self.client.search_combined_with_sentiment(query, sentiment_filter)
            if results:
                self.client.print_results(results)
            else:
                search_desc = f"'{query}'"
                if sentiment_filter:
                    search_desc += f" con sentimiento '{sentiment_filter}'"
                print(f"❌ No se encontraron resultados para {search_desc}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_analyze(self, arg):
        """Analizar estado de ánimo sobre un tema"""
        if not arg:
            print("❌ Proporciona un tema: analyze <tema>")
            print("   Ejemplos: analyze política, analyze economía")
            return
        
        try:
            analysis = self.client.analyze_content_mood(arg)
            
            if 'error' in analysis:
                print(f"❌ {analysis['error']}")
                return
            
            # Mostrar análisis
            print(f"\n📊 ANÁLISIS DE ESTADO DE ÁNIMO: {analysis['topic']}")
            print("=" * 50)
            print(f"{analysis['mood_emoji']} Estado general: {analysis['overall_mood']}")
            print(f"📁 Total segmentos: {analysis['total_segments']}")
            print(f"😊 Positivo: {analysis['distribution']['positive']} ({analysis['percentages']['positive']}%)")
            print(f"😢 Negativo: {analysis['distribution']['negative']} ({analysis['percentages']['negative']}%)")
            print(f"😐 Neutral: {analysis['distribution']['neutral']} ({analysis['percentages']['neutral']}%)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_sentiments(self, arg):
        """Listar sentimientos disponibles"""
        try:
            if not self.client.sentiment_enabled:
                print("❌ Sistema de sentimientos no disponible")
                print("💡 El dataset debe tener análisis de sentimientos procesado")
                return
            
            sentiments = self.client.get_available_sentiments()
            
            if not sentiments:
                print("❌ No hay sentimientos disponibles")
                return
            
            # Categorizar sentimientos básicos
            positive = [s for s in sentiments if any(p in s.lower() for p in ['feliz', 'alegre', 'optimista', 'contento', 'happy', 'entusiasta', 'positivo'])]
            negative = [s for s in sentiments if any(n in s.lower() for n in ['triste', 'enojado', 'preocupado', 'sad', 'angry', 'frustrado', 'negativo'])]
            neutral = [s for s in sentiments if any(n in s.lower() for n in ['neutral', 'calmado', 'tranquilo', 'calm', 'sereno'])]
            other = [s for s in sentiments if s not in positive + negative + neutral]
            
            print(f"\n🎭 SENTIMIENTOS DISPONIBLES ({len(sentiments)} total)")
            print("=" * 50)
            
            if positive:
                print(f"😊 POSITIVOS ({len(positive)}):")
                print("   " + ", ".join(positive))
                print()
            
            if negative:
                print(f"😢 NEGATIVOS ({len(negative)}):")
                print("   " + ", ".join(negative))
                print()
            
            if neutral:
                print(f"😐 NEUTRALES ({len(neutral)}):")
                print("   " + ", ".join(neutral))
                print()
            
            if other:
                print(f"❓ OTROS ({len(other)}):")
                print("   " + ", ".join(other[:20]))  # Limitar para no saturar
                if len(other) > 20:
                    print(f"   ... y {len(other) - 20} más")
            
            print("💡 Usa 'sentiment <emoción>' para buscar por sentimiento")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def do_exit(self, arg):
        """Salir del cliente"""
        return self.do_quit(arg)

def main():
    parser = argparse.ArgumentParser(description="Cliente de consultas para dataset de audio")
    parser.add_argument("dataset_dir", help="Directorio del dataset")
    parser.add_argument("--query", "-q", help="Consulta directa (no interactivo)")
    parser.add_argument("--type", "-t", choices=['text', 'audio', 'combined'], 
                       default='text', help="Tipo de búsqueda")
    parser.add_argument("--results", "-k", type=int, default=5, 
                       help="Número de resultados (default: 5)")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Modo interactivo")
    parser.add_argument("--load-real", action="store_true",
                       help="Cargar dataset real con análisis de sentimientos")
    
    args = parser.parse_args()
    
    try:
        # Inicializar cliente
        client = AudioDatasetClient(args.dataset_dir)
        
        # Mostrar stats
        stats = client.get_stats()
        print(f"📊 Dataset cargado: {stats['total_segments']:,} segmentos de {stats['unique_files']:,} archivos")
        
        # Cargar dataset real con sentimientos si se solicita
        if args.load_real:
            print("🎭 Modo de análisis de sentimientos activado")
            if hasattr(client, 'sentiment_enabled') and client.sentiment_enabled:
                print("✅ Sistema de sentimientos listo")
            else:
                print("⚠️  Dataset sin análisis de sentimientos")
        
        if args.interactive or not args.query:
            # Modo interactivo
            interactive_client = InteractiveClient(client)
            interactive_client.cmdloop()
        else:
            # Consulta directa
            if args.type == 'text':
                results = client.search_text(args.query, args.results)
            elif args.type == 'audio':
                results = client.search_audio(args.query, args.results)
            elif args.type == 'combined':
                results = client.search_combined(args.query, args.results)
            
            client.print_results(results)
    
    except KeyboardInterrupt:
        print("\n\n👋 Saliendo...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()