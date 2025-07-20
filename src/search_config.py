#!/usr/bin/env python3
"""
Configuración de parámetros de búsqueda y filtros de score
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import json
from pathlib import Path

@dataclass
class SearchConfig:
    """Configuración de parámetros de búsqueda"""
    
    # Umbrales de score
    min_text_score: float = 0.3
    min_audio_score: float = 0.3
    min_hybrid_score: float = 0.3
    min_keyword_score: float = 0.3
    min_yamnet_score: float = 0.5
    
    # Configuración de búsqueda
    default_results_count: int = 5
    max_results_count: int = 50
    
    # Pesos para búsqueda híbrida
    hybrid_text_weight: float = 0.7
    hybrid_audio_weight: float = 0.3
    
    # Configuración de visualización
    show_score_details: bool = True
    show_method_breakdown: bool = True
    truncate_text_length: int = 150
    
    # Configuración de calidad
    quality_mode: str = "balanced"  # "permissive", "balanced", "strict"
    
    def __post_init__(self):
        """Validar configuración después de inicialización"""
        self._validate_config()
        self._apply_quality_mode()
    
    def _validate_config(self):
        """Validar que los valores de configuración sean válidos"""
        # Validar umbrales
        thresholds = [
            self.min_text_score, self.min_audio_score, self.min_hybrid_score,
            self.min_keyword_score, self.min_yamnet_score
        ]
        
        for threshold in thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Los umbrales deben estar entre 0.0 y 1.0, got {threshold}")
        
        # Validar pesos
        if abs(self.hybrid_text_weight + self.hybrid_audio_weight - 1.0) > 0.001:
            raise ValueError("Los pesos híbridos deben sumar 1.0")
        
        # Validar conteos
        if self.default_results_count > self.max_results_count:
            raise ValueError("default_results_count no puede ser mayor que max_results_count")
    
    def _apply_quality_mode(self):
        """Aplicar configuración predefinida según el modo de calidad"""
        if self.quality_mode == "permissive":
            self.min_text_score = 0.2
            self.min_audio_score = 0.2
            self.min_hybrid_score = 0.2
            self.min_keyword_score = 0.2
            self.min_yamnet_score = 0.3
        elif self.quality_mode == "balanced":
            self.min_text_score = 0.4
            self.min_audio_score = 0.4
            self.min_hybrid_score = 0.4
            self.min_keyword_score = 0.4
            self.min_yamnet_score = 0.5
        elif self.quality_mode == "strict":
            self.min_text_score = 0.6
            self.min_audio_score = 0.7
            self.min_hybrid_score = 0.6
            self.min_keyword_score = 0.6
            self.min_yamnet_score = 0.7
    
    def get_threshold_for_method(self, method: str) -> float:
        """Obtener umbral para un método específico"""
        thresholds = {
            'text': self.min_text_score,
            'audio': self.min_audio_score,
            'hybrid': self.min_hybrid_score,
            'keyword': self.min_keyword_score,
            'yamnet': self.min_yamnet_score,
            'yamnet_pure': self.min_yamnet_score,
            'yamnet_similarity': self.min_yamnet_score
        }
        return thresholds.get(method, 0.3)
    
    def filter_results_by_score(self, results: list, method: str) -> list:
        """Filtrar resultados por umbral de score"""
        min_score = self.get_threshold_for_method(method)
        
        filtered_results = []
        for result in results:
            if result.get('score', 0) >= min_score:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_score_interpretation(self, score: float, method: str) -> str:
        """Obtener interpretación textual del score"""
        if method in ['text', 'hybrid']:
            if score >= 0.8:
                return "Excelente"
            elif score >= 0.6:
                return "Bueno"
            elif score >= 0.4:
                return "Regular"
            elif score >= 0.2:
                return "Pobre"
            else:
                return "Sin relación"
        
        elif method in ['yamnet', 'yamnet_pure', 'yamnet_similarity', 'audio']:
            if score >= 0.85:
                return "Excelente"
            elif score >= 0.7:
                return "Bueno"
            elif score >= 0.5:
                return "Regular"
            elif score >= 0.3:
                return "Pobre"
            else:
                return "Sin relación"
        
        elif method in ['keyword']:
            if score >= 0.9:
                return "Excelente"
            elif score >= 0.7:
                return "Bueno"
            elif score >= 0.5:
                return "Regular"
            elif score >= 0.3:
                return "Pobre"
            else:
                return "Sin relación"
        
        return "Desconocido"
    
    def to_dict(self) -> Dict:
        """Convertir configuración a diccionario"""
        return {
            'thresholds': {
                'text': self.min_text_score,
                'audio': self.min_audio_score,
                'hybrid': self.min_hybrid_score,
                'keyword': self.min_keyword_score,
                'yamnet': self.min_yamnet_score
            },
            'search': {
                'default_results': self.default_results_count,
                'max_results': self.max_results_count
            },
            'hybrid_weights': {
                'text': self.hybrid_text_weight,
                'audio': self.hybrid_audio_weight
            },
            'display': {
                'show_score_details': self.show_score_details,
                'show_method_breakdown': self.show_method_breakdown,
                'truncate_text_length': self.truncate_text_length
            },
            'quality_mode': self.quality_mode
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SearchConfig':
        """Crear configuración desde diccionario"""
        config = cls()
        
        # Cargar umbrales
        if 'thresholds' in config_dict:
            thresholds = config_dict['thresholds']
            config.min_text_score = thresholds.get('text', config.min_text_score)
            config.min_audio_score = thresholds.get('audio', config.min_audio_score)
            config.min_hybrid_score = thresholds.get('hybrid', config.min_hybrid_score)
            config.min_keyword_score = thresholds.get('keyword', config.min_keyword_score)
            config.min_yamnet_score = thresholds.get('yamnet', config.min_yamnet_score)
        
        # Cargar configuración de búsqueda
        if 'search' in config_dict:
            search_config = config_dict['search']
            config.default_results_count = search_config.get('default_results', config.default_results_count)
            config.max_results_count = search_config.get('max_results', config.max_results_count)
        
        # Cargar pesos híbridos
        if 'hybrid_weights' in config_dict:
            weights = config_dict['hybrid_weights']
            config.hybrid_text_weight = weights.get('text', config.hybrid_text_weight)
            config.hybrid_audio_weight = weights.get('audio', config.hybrid_audio_weight)
        
        # Cargar configuración de visualización
        if 'display' in config_dict:
            display_config = config_dict['display']
            config.show_score_details = display_config.get('show_score_details', config.show_score_details)
            config.show_method_breakdown = display_config.get('show_method_breakdown', config.show_method_breakdown)
            config.truncate_text_length = display_config.get('truncate_text_length', config.truncate_text_length)
        
        # Cargar modo de calidad
        if 'quality_mode' in config_dict:
            config.quality_mode = config_dict['quality_mode']
        
        # Validar y aplicar configuración
        config._validate_config()
        config._apply_quality_mode()
        
        return config
    
    def save_to_file(self, file_path: str):
        """Guardar configuración en archivo JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'SearchConfig':
        """Cargar configuración desde archivo JSON"""
        if not Path(file_path).exists():
            # Si no existe el archivo, crear configuración por defecto
            config = cls()
            config.save_to_file(file_path)
            return config
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

# Configuraciones predefinidas
DEFAULT_CONFIG = SearchConfig()

PERMISSIVE_CONFIG = SearchConfig(
    min_text_score=0.2,
    min_audio_score=0.2,
    min_hybrid_score=0.2,
    min_keyword_score=0.2,
    min_yamnet_score=0.3,
    quality_mode="permissive"
)

BALANCED_CONFIG = SearchConfig(
    min_text_score=0.4,
    min_audio_score=0.4,
    min_hybrid_score=0.4,
    min_keyword_score=0.4,
    min_yamnet_score=0.5,
    quality_mode="balanced"
)

STRICT_CONFIG = SearchConfig(
    min_text_score=0.6,
    min_audio_score=0.7,
    min_hybrid_score=0.6,
    min_keyword_score=0.6,
    min_yamnet_score=0.7,
    quality_mode="strict"
)

def get_config_for_use_case(use_case: str) -> SearchConfig:
    """Obtener configuración recomendada para un caso de uso"""
    configs = {
        'research': PERMISSIVE_CONFIG,
        'exploration': PERMISSIVE_CONFIG,
        'production': BALANCED_CONFIG,
        'balanced': BALANCED_CONFIG,
        'precision': STRICT_CONFIG,
        'strict': STRICT_CONFIG
    }
    
    return configs.get(use_case, BALANCED_CONFIG)

if __name__ == "__main__":
    # Ejemplo de uso
    print("📊 Configuraciones de Búsqueda Disponibles:")
    print("=" * 50)
    
    configs = {
        'Permisiva': PERMISSIVE_CONFIG,
        'Balanceada': BALANCED_CONFIG,
        'Estricta': STRICT_CONFIG
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Texto: {config.min_text_score}")
        print(f"  Audio: {config.min_audio_score}")
        print(f"  YAMNet: {config.min_yamnet_score}")
        print(f"  Híbrida: {config.min_hybrid_score}")
    
    # Guardar configuración por defecto
    DEFAULT_CONFIG.save_to_file("search_config.json")
    print(f"\n💾 Configuración por defecto guardada en: search_config.json")