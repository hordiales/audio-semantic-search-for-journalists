"""
Sentiment Analysis Module for Text Sentiment Search
Enables searching by different mood states like angry, happy, sad, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
from datetime import datetime

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Cargar configuración automáticamente
try:
    from config_loader import get_config
    SYSTEM_CONFIG = get_config()
except ImportError:
    SYSTEM_CONFIG = None


class SentimentAnalyzer:
    """
    Analizador de sentimientos para búsqueda por estados de ánimo
    """
    
    # Mapeo de sentimientos en español a categorías estándar
    SENTIMENT_MAPPING = {
        # Emociones positivas
        'feliz': 'POSITIVE',
        'alegre': 'POSITIVE', 
        'contento': 'POSITIVE',
        'optimista': 'POSITIVE',
        'esperanzado': 'POSITIVE',
        'entusiasta': 'POSITIVE',
        'satisfecho': 'POSITIVE',
        'eufórico': 'POSITIVE',
        
        # Emociones negativas
        'triste': 'NEGATIVE',
        'enojado': 'NEGATIVE',
        'enfadado': 'NEGATIVE',
        'furioso': 'NEGATIVE',
        'molesto': 'NEGATIVE',
        'frustrado': 'NEGATIVE',
        'deprimido': 'NEGATIVE',
        'melancólico': 'NEGATIVE',
        'pesimista': 'NEGATIVE',
        'desanimado': 'NEGATIVE',
        'ansioso': 'NEGATIVE',
        'preocupado': 'NEGATIVE',
        'estresado': 'NEGATIVE',
        'irritado': 'NEGATIVE',
        
        # Emociones neutrales
        'neutral': 'NEUTRAL',
        'calmado': 'NEUTRAL',
        'tranquilo': 'NEUTRAL',
        'sereno': 'NEUTRAL',
        'equilibrado': 'NEUTRAL',
        
        # Sinónimos en inglés
        'happy': 'POSITIVE',
        'sad': 'NEGATIVE',
        'angry': 'NEGATIVE',
        'neutral': 'NEUTRAL',
        'positive': 'POSITIVE',
        'negative': 'NEGATIVE',
        'excited': 'POSITIVE',
        'disappointed': 'NEGATIVE',
        'worried': 'NEGATIVE',
        'calm': 'NEUTRAL'
    }
    
    def __init__(self, model_name: str = None):
        """
        Inicializa el analizador de sentimientos
        
        Args:
            model_name: Nombre del modelo de análisis de sentimientos
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library es requerida para análisis de sentimientos. Instala con: pip install transformers")
        
        self.model_name = model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
        try:
            self._initialize_model()
        except Exception as e:
            raise RuntimeError(f"Error inicializando modelo de sentimientos: {e}")
    
    def _initialize_model(self):
        """Inicializa el modelo de análisis de sentimientos"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library no disponible")
        
        try:
            # Intentar usar un modelo en español si está disponible
            spanish_models = [
                "pysentimiento/robertuito-sentiment-analysis",
                "finiteautomata/beto-sentiment-analysis", 
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            ]
            
            model_to_use = self.model_name
            if self.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest":
                # Probar modelos en español primero
                for spanish_model in spanish_models[:-1]:
                    try:
                        test_pipeline = pipeline("sentiment-analysis", model=spanish_model)
                        model_to_use = spanish_model
                        break
                    except:
                        continue
            
            self.pipeline = pipeline("sentiment-analysis", model=model_to_use)
            logging.info(f"Modelo de sentimientos inicializado: {model_to_use}")
            
        except Exception as e:
            logging.error(f"Error inicializando modelo: {e}")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analiza el sentimiento de un texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con scores de sentimiento
        """
        
        try:
            # Limpiar texto
            text = text.strip()
            if not text:
                return {'POSITIVE': 0.33, 'NEGATIVE': 0.33, 'NEUTRAL': 0.34}
            
            # Truncar texto si es muy largo
            if len(text) > 512:
                text = text[:512]
            
            # Realizar análisis
            result = self.pipeline(text)
            
            # Normalizar resultados según el modelo
            return self._normalize_sentiment_result(result)
            
        except Exception as e:
            logging.error(f"Error en análisis de sentimientos: {e}")
            raise RuntimeError(f"Error analizando sentimiento: {e}")
    
    def _normalize_sentiment_result(self, result: List[Dict]) -> Dict[str, float]:
        """
        Normaliza resultados de diferentes modelos a formato estándar
        
        Args:
            result: Resultado del modelo de sentimientos
            
        Returns:
            Diccionario normalizado con POSITIVE, NEGATIVE, NEUTRAL
        """
        if not result:
            return {'POSITIVE': 0.33, 'NEGATIVE': 0.33, 'NEUTRAL': 0.34}
        
        # Tomar el primer resultado
        sentiment_result = result[0] if isinstance(result, list) else result
        
        label = sentiment_result.get('label', '').upper()
        score = sentiment_result.get('score', 0.0)
        
        # Mapear etiquetas comunes a nuestro formato
        label_mapping = {
            'POSITIVE': 'POSITIVE',
            'POS': 'POSITIVE', 
            'LABEL_2': 'POSITIVE',  # RoBERTa
            '2': 'POSITIVE',
            
            'NEGATIVE': 'NEGATIVE',
            'NEG': 'NEGATIVE',
            'LABEL_0': 'NEGATIVE',  # RoBERTa
            '0': 'NEGATIVE',
            
            'NEUTRAL': 'NEUTRAL',
            'NEU': 'NEUTRAL',
            'LABEL_1': 'NEUTRAL',  # RoBERTa
            '1': 'NEUTRAL'
        }
        
        normalized_label = label_mapping.get(label, 'NEUTRAL')
        
        # Crear distribución de probabilidades
        sentiment_scores = {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 0.0}
        sentiment_scores[normalized_label] = score
        
        # Distribuir el resto de probabilidad
        remaining_prob = 1.0 - score
        other_labels = [l for l in sentiment_scores.keys() if l != normalized_label]
        for other_label in other_labels:
            sentiment_scores[other_label] = remaining_prob / len(other_labels)
        
        return sentiment_scores
    
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Procesa un DataFrame añadiendo análisis de sentimientos
        
        Args:
            df: DataFrame con textos
            text_column: Nombre de la columna con texto
            
        Returns:
            DataFrame con columnas de sentimiento añadidas
        """
        if text_column not in df.columns:
            raise ValueError(f"Columna '{text_column}' no encontrada en DataFrame")
        
        print(f"Analizando sentimientos de {len(df)} textos...")
        
        results = []
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0:
                print(f"Procesando texto {idx + 1}/{len(df)}")
            
            sentiment_scores = self.analyze_text(str(text))
            results.append(sentiment_scores)
        
        # Añadir columnas de sentimiento
        df_result = df.copy()
        df_result['sentiment_positive'] = [r['POSITIVE'] for r in results]
        df_result['sentiment_negative'] = [r['NEGATIVE'] for r in results]
        df_result['sentiment_neutral'] = [r['NEUTRAL'] for r in results]
        
        # Determinar sentimiento dominante
        def get_dominant_sentiment(row):
            scores = {
                'POSITIVE': row['sentiment_positive'],
                'NEGATIVE': row['sentiment_negative'], 
                'NEUTRAL': row['sentiment_neutral']
            }
            return max(scores, key=scores.get)
        
        df_result['dominant_sentiment'] = df_result.apply(get_dominant_sentiment, axis=1)
        
        print("Análisis de sentimientos completado")
        return df_result
    
    def search_by_sentiment(self, df: pd.DataFrame, mood_query: str, 
                           threshold: float = 0.5, top_k: int = 10) -> pd.DataFrame:
        """
        Busca textos por sentimiento/estado de ánimo
        
        Args:
            df: DataFrame con análisis de sentimientos
            mood_query: Consulta de estado de ánimo (ej: "feliz", "triste", "enojado")
            threshold: Umbral mínimo de score de sentimiento
            top_k: Número máximo de resultados
            
        Returns:
            DataFrame con resultados filtrados por sentimiento
        """
        # Mapear consulta a sentimiento
        mood_lower = mood_query.lower().strip()
        target_sentiment = self.SENTIMENT_MAPPING.get(mood_lower)
        
        if not target_sentiment:
            # Buscar coincidencias parciales
            for mood_key, sentiment in self.SENTIMENT_MAPPING.items():
                if mood_key in mood_lower or mood_lower in mood_key:
                    target_sentiment = sentiment
                    break
        
        if not target_sentiment:
            print(f"No se pudo mapear '{mood_query}' a un sentimiento conocido")
            return pd.DataFrame()
        
        # Filtrar por sentimiento
        sentiment_column = f'sentiment_{target_sentiment.lower()}'
        
        if sentiment_column not in df.columns:
            print(f"Columna {sentiment_column} no encontrada. Ejecutar process_dataframe primero.")
            return pd.DataFrame()
        
        # Filtrar por umbral y ordenar por score
        filtered_df = df[df[sentiment_column] >= threshold].copy()
        filtered_df = filtered_df.sort_values(sentiment_column, ascending=False)
        
        # Añadir información de la consulta
        filtered_df['sentiment_query'] = mood_query
        filtered_df['sentiment_type'] = target_sentiment
        filtered_df['sentiment_score'] = filtered_df[sentiment_column]
        
        return filtered_df.head(top_k)
    
    def get_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Obtiene la distribución de sentimientos en el dataset
        
        Args:
            df: DataFrame con análisis de sentimientos
            
        Returns:
            Diccionario con conteo de sentimientos
        """
        if 'dominant_sentiment' not in df.columns:
            return {}
        
        return df['dominant_sentiment'].value_counts().to_dict()
    
    def get_available_moods(self) -> List[str]:
        """
        Retorna lista de estados de ánimo disponibles para búsqueda
        
        Returns:
            Lista de estados de ánimo soportados
        """
        return list(self.SENTIMENT_MAPPING.keys())


# Funciones de utilidad
def analyze_sentiment_dataset(df: pd.DataFrame, text_column: str = 'text', 
                            model_name: str = None) -> pd.DataFrame:
    """
    Función de conveniencia para analizar sentimientos en un dataset
    
    Args:
        df: DataFrame con textos
        text_column: Columna con texto a analizar
        model_name: Modelo de sentimientos a usar
        
    Returns:
        DataFrame con análisis de sentimientos
    """
    analyzer = SentimentAnalyzer(model_name=model_name)
    return analyzer.process_dataframe(df, text_column)


def search_by_mood(df: pd.DataFrame, mood: str, threshold: float = 0.5, 
                  top_k: int = 10) -> pd.DataFrame:
    """
    Función de conveniencia para buscar por estado de ánimo
    
    Args:
        df: DataFrame con análisis de sentimientos
        mood: Estado de ánimo a buscar
        threshold: Umbral mínimo de score
        top_k: Número de resultados
        
    Returns:
        DataFrame con resultados
    """
    analyzer = SentimentAnalyzer()
    return analyzer.search_by_sentiment(df, mood, threshold, top_k)


# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    sample_data = {
        'text': [
            "Estoy muy feliz con los resultados obtenidos",
            "Me siento triste por la situación actual", 
            "Esto me enoja mucho, es una situación terrible",
            "Es una noticia neutral, sin mayor impacto",
            "¡Excelente trabajo! Me emociona ver el progreso",
            "La situación económica es preocupante y genera ansiedad"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Inicializar analizador
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers no está disponible. No se pueden ejecutar las pruebas.")
        exit(1)
    
    analyzer = SentimentAnalyzer()
    
    # Procesar sentimientos
    df_with_sentiment = analyzer.process_dataframe(df)
    print("Resultados del análisis:")
    print(df_with_sentiment[['text', 'dominant_sentiment', 'sentiment_positive', 'sentiment_negative']])
    
    # Buscar por estado de ánimo
    happy_results = analyzer.search_by_sentiment(df_with_sentiment, "feliz", threshold=0.5)
    print(f"\nTextos 'felices' encontrados: {len(happy_results)}")
    
    # Mostrar distribución
    distribution = analyzer.get_sentiment_distribution(df_with_sentiment)
    print(f"\nDistribución de sentimientos: {distribution}")
    
    # Mostrar estados de ánimo disponibles
    moods = analyzer.get_available_moods()
    print(f"\nEstados de ánimo disponibles: {moods[:10]}...")  # Primeros 10