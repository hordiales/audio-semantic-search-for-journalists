#!/usr/bin/env python3
"""
Sistema de búsqueda de audio mejorado
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import re

import logging

class ImprovedAudioSearch:
    """Búsqueda de audio mejorada que funciona sin embeddings reales"""
    
    def __init__(self, logger=None):
        """Inicializa el sistema de búsqueda mejorado"""
        self.logger = logger or logging.getLogger(__name__)
        self.audio_keywords_mapping = self._build_audio_keywords()
    
    def _build_audio_keywords(self) -> Dict[str, List[str]]:
        """Construye mapeo de clases de audio a palabras clave en español"""
        return {
            # Aplausos y ovaciones
            'applause': [
                'aplauso', 'aplausos', 'palmada', 'palmadas', 'ovación', 
                'ovaciones', 'aplaude', 'aplaudir', 'standing ovation',
                'ovaciona', 'ovacionan', 'palmas'
            ],
            
            # Risas
            'laughter': [
                'risa', 'risas', 'reír', 'reírse', 'carcajada', 'carcajadas',
                'risotada', 'riendo', 'ríe', 'reímos', 'sonrisa', 'hilaridad'
            ],
            
            # Multitudes
            'crowd': [
                'multitud', 'multitudes', 'gente', 'público', 'audiencia',
                'masa', 'masas', 'muchedumbre', 'gentío', 'concurrencia',
                'asistentes', 'seguidores', 'manifestantes', 'concentración'
            ],
            
            # Gritos y vociferaciones
            'shouting': [
                'grito', 'gritos', 'gritar', 'grita', 'gritan', 'chillar',
                'chilla', 'vocear', 'vocifera', 'exclamar', 'exclamación',
                'alarido', 'berrido', 'bramido'
            ],
            
            # Abucheos
            'booing': [
                'abucheo', 'abucheos', 'abuchea', 'abuchean', 'silbar',
                'silbido', 'silbidos', 'rechifla', 'pitar', 'pitada'
            ],
            
            # Vítores y celebración
            'cheering': [
                'vítor', 'vítores', 'vitorea', 'vitorean', 'celebrar',
                'celebración', 'festejo', 'aclamación', 'aclamar',
                'hurra', 'bravo', 'ole'
            ],
            
            # Cánticos
            'chanting': [
                'cántico', 'cánticos', 'cantar', 'canta', 'cantan',
                'coro', 'coros', 'corear', 'himno', 'consigna',
                'eslogan', 'lema'
            ],
            
            # Música
            'music': [
                'música', 'musical', 'canción', 'canciones', 'melodía',
                'himno', 'himnos', 'banda', 'orquesta', 'instrumento',
                'tocar', 'interpretar', 'sonido musical'
            ],
            
            # Discurso y habla
            'speech': [
                'discurso', 'discursos', 'habla', 'hablar', 'dice', 'dijo',
                'pronuncia', 'declara', 'manifiesta', 'expresa', 'comenta',
                'intervención', 'alocución', 'arenga'
            ],
            
            # Conversación
            'conversation': [
                'conversación', 'conversa', 'diálogo', 'dialoga', 'charla',
                'charlar', 'plática', 'intercambio', 'debate', 'discusión'
            ],
            
            # Entrevista
            'interview': [
                'entrevista', 'entrevista', 'pregunta', 'preguntas',
                'responde', 'respuesta', 'declaraciones', 'testimonio'
            ],
            
            # Silencio
            'silence': [
                'silencio', 'silencioso', 'callado', 'mudo', 'sin ruido',
                'quieto', 'calma', 'tranquilo', 'pausa'
            ],
            
            # Ruido
            'noise': [
                'ruido', 'ruidoso', 'estruendo', 'estrépito', 'alboroto',
                'bullicio', 'barullo', 'jaleo', 'escándalo'
            ],
            
            # Sirenas
            'siren': [
                'sirena', 'sirenas', 'ambulancia', 'policía', 'bomberos',
                'emergencia', 'alarma', 'señal de alarma'
            ],
            
            # Vehículos
            'vehicle': [
                'vehículo', 'coche', 'auto', 'automóvil', 'camión',
                'autobús', 'moto', 'motocicleta', 'tráfico', 'motor'
            ],
            
            # Construcción
            'construction': [
                'construcción', 'obra', 'obras', 'taladro', 'martillo',
                'sierra', 'maquinaria', 'excavadora', 'grúa'
            ]
        }
    
    def search_by_keywords(self, df: pd.DataFrame, audio_class: str, k: int = 10) -> List[Dict]:
        """
        Busca segmentos por palabras clave relacionadas con clases de audio
        
        Args:
            df: DataFrame con transcripciones
            audio_class: Clase de audio a buscar
            k: Número de resultados
            
        Returns:
            Lista de resultados ordenados por relevancia
        """
        keywords = self.audio_keywords_mapping.get(audio_class, [])
        
        if not keywords:
            return []
        
        scored_segments = []
        
        for idx, row in df.iterrows():
            text = row['text'].lower()
            score = 0.0
            matched_keywords = []
            
            # Buscar palabras clave exactas
            for keyword in keywords:
                if keyword in text:
                    matched_keywords.append(keyword)
                    # Score más alto para coincidencias exactas
                    score += 1.0
                    
                    # Bonus si la palabra está al principio o es prominente
                    if text.startswith(keyword) or f" {keyword} " in text:
                        score += 0.5
            
            # Buscar palabras clave parciales
            for keyword in keywords:
                # Buscar variaciones de la palabra
                pattern = rf'\b{re.escape(keyword[:-1])}\w*\b'
                if re.search(pattern, text):
                    score += 0.3
            
            # Bonus por múltiples coincidencias
            if len(matched_keywords) > 1:
                score += 0.5 * (len(matched_keywords) - 1)
            
            if score > 0:
                scored_segments.append({
                    'idx': idx,
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'row': row
                })
        
        # Ordenar por score
        scored_segments.sort(key=lambda x: x['score'], reverse=True)
        
        # Formatear resultados
        results = []
        for i, segment in enumerate(scored_segments[:k]):
            row = segment['row']
            result = {
                'rank': i + 1,
                'score': segment['score'],
                'text': row['text'],
                'source_file': row['source_file'],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'duration': row['duration'],
                'matched_keywords': segment['matched_keywords'],
                'audio_class': audio_class
            }
            results.append(result)
        
        return results
    
    def search_audio_by_text(self, df: pd.DataFrame, query: str, k: int = 10) -> List[Dict]:
        """
        Busca audio basado en texto usando mapeo semántico + palabras clave
        
        Args:
            df: DataFrame con transcripciones
            query: Consulta de texto
            k: Número de resultados
            
        Returns:
            Lista de resultados combinados
        """
        query_lower = query.lower()
        all_results = []
        
        # Buscar en cada clase de audio
        for audio_class, keywords in self.audio_keywords_mapping.items():
            # Verificar si la consulta está relacionada con esta clase
            class_score = 0
            
            # Consulta directa
            if query_lower == audio_class or query_lower in keywords:
                class_score = 1.0
            # Consulta parcial
            elif any(keyword in query_lower for keyword in keywords):
                class_score = 0.8
            elif any(query_lower in keyword for keyword in keywords):
                class_score = 0.6
            
            if class_score > 0:
                results = self.search_by_keywords(df, audio_class, k * 2)
                
                # Ajustar scores por relevancia de clase
                for result in results:
                    result['score'] *= class_score
                    result['class_relevance'] = class_score
                
                all_results.extend(results)
        
        # Ordenar todos los resultados
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Eliminar duplicados (mismo segmento)
        seen = set()
        unique_results = []
        
        for result in all_results:
            key = (result['source_file'], result['start_time'])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        # Reasignar ranks
        for i, result in enumerate(unique_results[:k]):
            result['rank'] = i + 1
        
        return unique_results[:k]
    
    def get_available_audio_classes(self) -> List[str]:
        """Obtiene las clases de audio disponibles"""
        return list(self.audio_keywords_mapping.keys())
    
    def get_keywords_for_class(self, audio_class: str) -> List[str]:
        """Obtiene las palabras clave para una clase de audio"""
        return self.audio_keywords_mapping.get(audio_class, [])

# Función de conveniencia
def create_improved_audio_search():
    """Crea una instancia del buscador de audio mejorado"""
    return ImprovedAudioSearch()

if __name__ == "__main__":
    # Ejemplo de uso
    search_engine = ImprovedAudioSearch()
    
    search_engine.logger.info("🔍 Clases de audio disponibles:")
    for audio_class in search_engine.get_available_audio_classes():
        keywords = search_engine.get_keywords_for_class(audio_class)[:3]
        search_engine.logger.info(f"  {audio_class}: {', '.join(keywords)}...")
    
    search_engine.logger.info(f"\n💡 Ejemplo de búsqueda por 'aplausos':")
    keywords = search_engine.get_keywords_for_class('applause')
    search_engine.logger.info(f"Palabras clave: {', '.join(keywords)}")