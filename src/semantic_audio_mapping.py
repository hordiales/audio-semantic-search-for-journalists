#!/usr/bin/env python3
"""
Mapeo semántico entre conceptos de texto y clases de audio
"""

import logging

from typing import List, Dict, Set
from audioset_ontology import AUDIOSET_CLASSES, AUDIOSET_CATEGORIES

class SemanticAudioMapper:
    """Mapeador semántico entre texto y audio"""
    
    def __init__(self):
        """Inicializa el mapeador con las reglas semánticas"""
        self.text_to_audio_mapping = self._build_semantic_mapping()
    
    def _build_semantic_mapping(self) -> Dict[str, List[str]]:
        """Construye el mapeo semántico texto → audio"""
        return {
            # Conceptos políticos y sociales
            "discurso": ["speech", "male_speech", "female_speech", "presentation", "lecture"],
            "conferencia": ["speech", "presentation", "lecture", "conference", "microphone"],
            "rueda de prensa": ["speech", "interview", "news", "microphone", "camera"],
            "entrevista": ["interview", "conversation", "speech", "microphone"],
            "debate": ["debate", "argument", "discussion", "speech", "conversation"],
            "presentación": ["presentation", "speech", "lecture", "microphone"],
            "seminario": ["seminar", "lecture", "presentation", "speech"],
            "reunión": ["meeting", "discussion", "conversation", "speech"],
            "asamblea": ["meeting", "speech", "discussion", "crowd"],
            
            # Eventos políticos
            "manifestación": ["protest", "demonstration", "crowd", "chanting", "shouting"],
            "protesta": ["protest", "demonstration", "crowd", "chanting", "shouting"],
            "marcha": ["protest", "demonstration", "crowd", "chanting"],
            "mitin": ["rally", "speech", "crowd", "applause", "cheering"],
            "campaña": ["rally", "speech", "crowd", "applause", "music"],
            "elecciones": ["speech", "news", "interview", "crowd", "applause"],
            "votación": ["news", "announcement", "crowd", "discussion"],
            
            # Reacciones del público
            "aplausos": ["applause", "clapping", "cheering", "crowd"],
            "ovación": ["applause", "cheering", "crowd", "standing_ovation"],
            "abucheos": ["booing", "crowd", "shouting", "protest"],
            "silbidos": ["whistling", "booing", "crowd"],
            "gritos": ["shouting", "yelling", "crowd", "protest"],
            "vítores": ["cheering", "applause", "crowd", "celebration"],
            "coros": ["chanting", "singing", "crowd", "protest"],
            "cánticos": ["chanting", "singing", "crowd", "protest"],
            
            # Medios de comunicación
            "noticias": ["news", "broadcast", "speech", "interview"],
            "noticiario": ["news", "broadcast", "speech", "television"],
            "telediario": ["news", "broadcast", "television", "speech"],
            "radio": ["radio", "broadcast", "speech", "music"],
            "televisión": ["television", "broadcast", "news", "speech"],
            "transmisión": ["broadcast", "speech", "news", "radio"],
            "programa": ["broadcast", "speech", "interview", "music"],
            "reportaje": ["news", "interview", "speech", "broadcast"],
            "documental": ["narration", "speech", "interview", "background_music"],
            
            # Tipos de comunicación
            "anuncio": ["announcement", "speech", "broadcast", "news"],
            "comunicado": ["announcement", "speech", "news", "broadcast"],
            "declaración": ["speech", "announcement", "news", "interview"],
            "testimonio": ["speech", "interview", "narration", "conversation"],
            "comentario": ["commentary", "speech", "discussion", "interview"],
            "análisis": ["commentary", "speech", "discussion", "interview"],
            "opinión": ["commentary", "speech", "discussion", "interview"],
            
            # Eventos y ceremonias
            "ceremonia": ["speech", "music", "applause", "crowd"],
            "inauguración": ["speech", "music", "applause", "crowd"],
            "clausura": ["speech", "music", "applause", "crowd"],
            "homenaje": ["speech", "music", "applause", "crowd"],
            "conmemoración": ["speech", "music", "applause", "crowd"],
            "celebración": ["music", "applause", "cheering", "crowd"],
            "festival": ["music", "crowd", "applause", "celebration"],
            "concierto": ["music", "singing", "applause", "crowd"],
            
            # Instituciones y lugares
            "congreso": ["speech", "debate", "discussion", "applause"],
            "parlamento": ["speech", "debate", "discussion", "applause"],
            "senado": ["speech", "debate", "discussion", "applause"],
            "tribunal": ["speech", "discussion", "debate", "announcement"],
            "juzgado": ["speech", "discussion", "announcement", "conversation"],
            "audiencia": ["speech", "discussion", "conversation", "crowd"],
            "universidad": ["lecture", "speech", "discussion", "crowd"],
            "escuela": ["lecture", "speech", "discussion", "child_speech"],
            "aula": ["lecture", "speech", "discussion", "conversation"],
            
            # Situaciones de conflicto
            "conflicto": ["argument", "shouting", "protest", "crowd"],
            "crisis": ["news", "speech", "interview", "urgent_broadcast"],
            "emergencia": ["siren", "alarm", "news", "speech"],
            "accidente": ["siren", "alarm", "news", "emergency"],
            "catástrofe": ["siren", "alarm", "news", "emergency"],
            "tragedia": ["news", "speech", "crying", "siren"],
            
            # Transporte y lugares públicos
            "aeropuerto": ["aircraft", "announcement", "crowd", "luggage"],
            "estación": ["train", "announcement", "crowd", "loudspeaker"],
            "metro": ["train", "announcement", "crowd", "underground"],
            "autobús": ["bus", "engine", "crowd", "announcement"],
            "tráfico": ["car", "truck", "engine", "horn"],
            "calle": ["traffic", "crowd", "footsteps", "urban"],
            "plaza": ["crowd", "footsteps", "speech", "urban"],
            "mercado": ["crowd", "speech", "commerce", "activity"],
            
            # Tecnología y oficina
            "teléfono": ["telephone", "phone_ringing", "conversation", "voice"],
            "ordenador": ["computer", "keyboard", "mouse", "electronic"],
            "impresora": ["printer", "office", "machinery", "electronic"],
            "oficina": ["keyboard", "phone", "conversation", "office"],
            "reunión": ["meeting", "discussion", "conversation", "office"],
            
            # Naturaleza y clima
            "lluvia": ["rain", "water", "storm", "weather"],
            "tormenta": ["thunder", "rain", "wind", "storm"],
            "viento": ["wind", "weather", "nature", "outdoor"],
            "mar": ["ocean", "water", "waves", "nature"],
            "río": ["water", "flow", "nature", "stream"],
            "bosque": ["bird", "wind", "nature", "forest"],
            "campo": ["bird", "wind", "nature", "rural"],
            
            # Animales
            "perro": ["dog", "animal", "barking", "pet"],
            "gato": ["cat", "animal", "meowing", "pet"],
            "pájaro": ["bird", "animal", "chirping", "nature"],
            "caballo": ["horse", "animal", "galloping", "rural"],
            
            # Emociones y reacciones
            "risa": ["laughter", "joy", "happiness", "crowd"],
            "llanto": ["crying", "sadness", "emotion", "distress"],
            "alegría": ["laughter", "cheering", "applause", "music"],
            "tristeza": ["crying", "sorrow", "quiet", "melancholy"],
            "miedo": ["screaming", "panic", "alarm", "emergency"],
            "sorpresa": ["gasp", "exclamation", "sudden", "reaction"],
            
            # Actividades laborales
            "construcción": ["construction", "drilling", "hammer", "machinery"],
            "fábrica": ["machinery", "industrial", "work", "production"],
            "taller": ["machinery", "hammer", "saw", "work"],
            "obras": ["construction", "drilling", "machinery", "work"],
            
            # Conceptos abstractos (menos obvios)
            "economía": ["news", "speech", "interview", "analysis"],
            "política": ["speech", "debate", "news", "interview"],
            "educación": ["lecture", "speech", "discussion", "school"],
            "cultura": ["music", "speech", "applause", "arts"],
            "sociedad": ["crowd", "discussion", "speech", "community"],
            "historia": ["narration", "speech", "documentary", "lecture"],
            "ciencia": ["lecture", "speech", "discussion", "laboratory"],
            "tecnología": ["computer", "electronic", "machinery", "innovation"],
            "arte": ["music", "applause", "creative", "performance"],
            "deportes": ["crowd", "cheering", "applause", "stadium"],
            "salud": ["speech", "interview", "medical", "hospital"],
            "medio ambiente": ["nature", "wind", "water", "outdoor"],
            
            # Situaciones específicas del periodismo
            "breaking news": ["news", "urgent", "broadcast", "announcement"],
            "última hora": ["news", "urgent", "broadcast", "announcement"],
            "flash informativo": ["news", "urgent", "broadcast", "announcement"],
            "directo": ["broadcast", "live", "news", "speech"],
            "en vivo": ["broadcast", "live", "news", "speech"],
            "corresponsal": ["news", "interview", "broadcast", "reporter"],
            "enviado especial": ["news", "interview", "broadcast", "reporter"],
            "redacción": ["news", "office", "keyboard", "phone"],
            "estudio": ["broadcast", "news", "microphone", "television"],
            "plató": ["broadcast", "television", "news", "studio"],
        }
    
    def get_audio_classes_for_text(self, text: str) -> List[str]:
        """
        Obtiene las clases de audio relevantes para un texto
        
        Args:
            text: Texto de consulta
            
        Returns:
            Lista de clases de audio relacionadas
        """
        text_lower = text.lower()
        audio_classes = set()
        
        # Búsqueda exacta
        if text_lower in self.text_to_audio_mapping:
            audio_classes.update(self.text_to_audio_mapping[text_lower])
        
        # Búsqueda parcial
        for concept, classes in self.text_to_audio_mapping.items():
            if concept in text_lower or text_lower in concept:
                audio_classes.update(classes)
        
        return list(audio_classes)
    
    def get_semantic_score(self, text: str, audio_class: str) -> float:
        """
        Calcula la relevancia semántica entre texto y clase de audio
        
        Args:
            text: Texto de consulta
            audio_class: Clase de audio
            
        Returns:
            Score de relevancia (0-1)
        """
        relevant_classes = self.get_audio_classes_for_text(text)
        
        if audio_class in relevant_classes:
            # Calcular posición en la lista (más relevante = mayor score)
            position = relevant_classes.index(audio_class)
            return max(0.1, 1.0 - (position * 0.2))  # Decremento de 0.2 por posición
        
        return 0.0
    
    def suggest_audio_queries(self, text: str) -> List[Dict]:
        """
        Sugiere consultas de audio basadas en el texto
        
        Args:
            text: Texto de consulta
            
        Returns:
            Lista de sugerencias con scores
        """
        audio_classes = self.get_audio_classes_for_text(text)
        suggestions = []
        
        for audio_class in audio_classes[:10]:  # Top 10
            info = AUDIOSET_CLASSES.get(audio_class, {})
            score = self.get_semantic_score(text, audio_class)
            
            suggestion = {
                "class": audio_class,
                "name": info.get("name", audio_class),
                "description": info.get("description", ""),
                "score": score,
                "query": audio_class
            }
            suggestions.append(suggestion)
        
        # Ordenar por score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions
    
    def validate_audio_query(self, query: str) -> Dict:
        """
        Valida si una consulta de audio es válida
        
        Args:
            query: Consulta de audio
            
        Returns:
            Diccionario con validación y sugerencias
        """
        query_lower = query.lower()
        
        # Verificar si es una clase de audio válida
        if query_lower in AUDIOSET_CLASSES:
            return {
                "valid": True,
                "class": query_lower,
                "info": AUDIOSET_CLASSES[query_lower],
                "suggestions": []
            }
        
        # Buscar clases similares
        similar_classes = []
        for class_name, info in AUDIOSET_CLASSES.items():
            if (query_lower in class_name.lower() or 
                query_lower in info.get("name", "").lower()):
                similar_classes.append({
                    "class": class_name,
                    "name": info.get("name", class_name),
                    "description": info.get("description", "")
                })
        
        # Obtener sugerencias semánticas
        semantic_suggestions = self.suggest_audio_queries(query)
        
        return {
            "valid": False,
            "message": f"'{query}' no es una clase de audio válida",
            "similar_classes": similar_classes,
            "semantic_suggestions": semantic_suggestions
        }
    
    def get_mapping_stats(self) -> Dict:
        """Obtiene estadísticas del mapeo"""
        total_concepts = len(self.text_to_audio_mapping)
        total_mappings = sum(len(classes) for classes in self.text_to_audio_mapping.values())
        
        return {
            "total_text_concepts": total_concepts,
            "total_audio_mappings": total_mappings,
            "avg_mappings_per_concept": total_mappings / total_concepts if total_concepts > 0 else 0
        }

# Instancia global
semantic_mapper = SemanticAudioMapper()

# Funciones de conveniencia
def get_audio_classes_for_text(text: str) -> List[str]:
    """Función de conveniencia para obtener clases de audio"""
    return semantic_mapper.get_audio_classes_for_text(text)

def suggest_audio_queries(text: str) -> List[Dict]:
    """Función de conveniencia para sugerir consultas"""
    return semantic_mapper.suggest_audio_queries(text)

def validate_audio_query(query: str) -> Dict:
    """Función de conveniencia para validar consultas"""
    return semantic_mapper.validate_audio_query(query)

if __name__ == "__main__":
    # Ejemplos de uso
    mapper = SemanticAudioMapper()
    
    # Ejemplo 1: Consulta política
    logging.info("🔍 Consulta: 'discurso político'")
    classes = mapper.get_audio_classes_for_text("discurso político")
    logging.info(f"Clases de audio: {classes}")
    
    # Ejemplo 2: Sugerencias
    logging.info("\n💡 Sugerencias para 'manifestación':")
    suggestions = mapper.suggest_audio_queries("manifestación")
    for suggestion in suggestions[:5]:
        logging.info(f"  • {suggestion['query']} (score: {suggestion['score']:.2f}) - {suggestion['name']}")
    
    # Ejemplo 3: Validación
    logging.info("\n✅ Validación de 'aplausos':")
    validation = mapper.validate_audio_query("aplausos")
    logging.info(f"Válido: {validation['valid']}")
    
    # Estadísticas
    stats = mapper.get_mapping_stats()
    logging.info(f"\n📊 Estadísticas:")
    logging.info(f"  Conceptos de texto: {stats['total_text_concepts']}")
    logging.info(f"  Mapeos de audio: {stats['total_audio_mappings']}")
    logging.info(f"  Promedio por concepto: {stats['avg_mappings_per_concept']:.1f}")