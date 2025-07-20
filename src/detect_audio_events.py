#!/usr/bin/env python3
"""
Script específico para detectar eventos de audio como risas, música, aplausos
usando YAMNet real y agregar esta información al dataset
"""

import pandas as pd
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioEventDetector:
    """Detector especializado de eventos de audio usando YAMNet"""
    
    def __init__(self):
        """Inicializa el detector de eventos"""
        self.model = None
        self.class_names = None
        self._load_yamnet()
        self._setup_event_mapping()
    
    def _load_yamnet(self):
        """Carga YAMNet desde TensorFlow Hub"""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            
            logger.info("🔄 Cargando YAMNet para detección de eventos...")
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            # Cargar nombres de clases si están disponibles
            try:
                import csv
                class_map_path = self.model.class_map_path().numpy()
                self.class_names = []
                with tf.io.gfile.GFile(class_map_path) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.class_names.append(row['display_name'])
                logger.info(f"✅ YAMNet cargado con {len(self.class_names)} clases")
            except:
                logger.info("✅ YAMNet cargado (sin nombres de clases)")
                
        except Exception as e:
            logger.error(f"❌ Error cargando YAMNet: {e}")
            raise
    
    def _setup_event_mapping(self):
        """Configura el mapeo de eventos de interés periodístico"""
        
        # Eventos de interés para periodismo
        self.events_of_interest = {
            'laughter': {
                'keywords': ['laughter', 'laugh', 'chuckle', 'giggle'],
                'description': 'Risas y carcajadas',
                'importance': 'high'
            },
            'applause': {
                'keywords': ['applause', 'clapping', 'clap'],
                'description': 'Aplausos y ovaciones',
                'importance': 'high'
            },
            'music': {
                'keywords': ['music', 'musical', 'song', 'melody', 'instrumental'],
                'description': 'Música de fondo o ambiental',
                'importance': 'medium'
            },
            'singing': {
                'keywords': ['singing', 'vocal', 'song', 'choir'],
                'description': 'Canto o voces cantando',
                'importance': 'medium'
            },
            'crowd': {
                'keywords': ['crowd', 'audience', 'people', 'chatter', 'hubbub'],
                'description': 'Sonidos de multitud',
                'importance': 'high'
            },
            'speech': {
                'keywords': ['speech', 'conversation', 'talking', 'voice'],
                'description': 'Habla y conversación',
                'importance': 'low'  # Menos importante porque ya se transcribe
            },
            'cheering': {
                'keywords': ['cheer', 'shouting', 'yelling', 'celebration'],
                'description': 'Vítores y celebraciones',
                'importance': 'high'
            },
            'booing': {
                'keywords': ['boo', 'booing', 'disapproval'],
                'description': 'Abucheos',
                'importance': 'high'
            }
        }
    
    def detect_events_in_audio(self, audio_path: str) -> Dict:
        """
        Detecta eventos específicos en un archivo de audio
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            Dict con eventos detectados y metadatos
        """
        try:
            import librosa
            
            # Cargar audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio = librosa.util.normalize(audio).astype(np.float32)
            
            # Procesar con YAMNet
            scores, embeddings, spectrogram = self.model(audio)
            scores_np = scores.numpy()
            
            # Analizar scores a lo largo del tiempo
            frame_events = []
            for frame_idx, frame_scores in enumerate(scores_np):
                frame_events.append(self._analyze_frame_scores(frame_scores, frame_idx))
            
            # Consolidar eventos detectados
            consolidated_events = self._consolidate_events(frame_events)
            
            # Calcular estadísticas generales
            avg_scores = np.mean(scores_np, axis=0)
            max_scores = np.max(scores_np, axis=0)
            
            # Análisis temporal de eventos
            temporal_analysis = self._analyze_temporal_patterns(frame_events)
            
            return {
                'events_detected': consolidated_events,
                'temporal_analysis': temporal_analysis,
                'audio_duration': len(audio) / sr,
                'total_frames': len(scores_np),
                'max_confidence': float(np.max(max_scores)),
                'processing_success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error detectando eventos en {audio_path}: {e}")
            return {
                'events_detected': {},
                'temporal_analysis': {},
                'audio_duration': 0,
                'total_frames': 0,
                'max_confidence': 0,
                'processing_success': False,
                'error': str(e)
            }
    
    def _analyze_frame_scores(self, scores: np.ndarray, frame_idx: int) -> Dict:
        """Analiza scores de un frame específico"""
        events_in_frame = {}
        
        # Umbral dinámico basado en la distribución de scores
        threshold = np.mean(scores) + 2 * np.std(scores)
        threshold = max(threshold, 0.1)  # Mínimo 0.1
        
        for event_name, event_info in self.events_of_interest.items():
            max_score = 0
            detected_classes = []
            
            # Si tenemos nombres de clases, buscar por keywords
            if self.class_names:
                for i, class_name in enumerate(self.class_names):
                    if any(keyword.lower() in class_name.lower() 
                          for keyword in event_info['keywords']):
                        if scores[i] > max_score:
                            max_score = scores[i]
                            detected_classes.append((class_name, scores[i]))
            else:
                # Mapeo aproximado por índices (menos preciso)
                approximate_indices = self._get_approximate_indices(event_name)
                for idx in approximate_indices:
                    if idx < len(scores) and scores[idx] > max_score:
                        max_score = scores[idx]
            
            if max_score > threshold:
                events_in_frame[event_name] = {
                    'confidence': float(max_score),
                    'frame': frame_idx,
                    'detected_classes': detected_classes,
                    'importance': event_info['importance']
                }
        
        return events_in_frame
    
    def _get_approximate_indices(self, event_name: str) -> List[int]:
        """Mapeo aproximado de eventos a índices de AudioSet"""
        # Nota: Estos son índices aproximados, en un sistema real
        # necesitarías el mapeo exacto de YAMNet a AudioSet
        mapping = {
            'laughter': [0, 1, 2, 3],
            'applause': [20, 21, 22],
            'music': [40, 41, 42, 43, 44, 45],
            'singing': [50, 51, 52],
            'crowd': [70, 71, 72],
            'speech': [100, 101, 102, 103],
            'cheering': [120, 121],
            'booing': [125, 126]
        }
        return mapping.get(event_name, [])
    
    def _consolidate_events(self, frame_events: List[Dict]) -> Dict:
        """Consolida eventos detectados a lo largo de todos los frames"""
        consolidated = {}
        
        for event_name in self.events_of_interest.keys():
            detections = []
            confidences = []
            
            for frame_event in frame_events:
                if event_name in frame_event:
                    detections.append(frame_event[event_name])
                    confidences.append(frame_event[event_name]['confidence'])
            
            if detections:
                consolidated[event_name] = {
                    'detected': True,
                    'max_confidence': max(confidences),
                    'avg_confidence': np.mean(confidences),
                    'detection_count': len(detections),
                    'detection_percentage': len(detections) / len(frame_events) * 100,
                    'importance': self.events_of_interest[event_name]['importance'],
                    'description': self.events_of_interest[event_name]['description']
                }
            else:
                consolidated[event_name] = {
                    'detected': False,
                    'max_confidence': 0.0,
                    'avg_confidence': 0.0,
                    'detection_count': 0,
                    'detection_percentage': 0.0,
                    'importance': self.events_of_interest[event_name]['importance'],
                    'description': self.events_of_interest[event_name]['description']
                }
        
        return consolidated
    
    def _analyze_temporal_patterns(self, frame_events: List[Dict]) -> Dict:
        """Analiza patrones temporales de eventos"""
        analysis = {
            'event_timeline': [],
            'peak_moments': [],
            'dominant_events': [],
            'event_transitions': []
        }
        
        # Timeline de eventos por frame
        for i, frame_event in enumerate(frame_events):
            if frame_event:
                analysis['event_timeline'].append({
                    'frame': i,
                    'events': list(frame_event.keys()),
                    'max_confidence': max(e['confidence'] for e in frame_event.values())
                })
        
        # Identificar momentos pico (alta actividad de eventos)
        confidences_by_frame = []
        for frame_event in frame_events:
            if frame_event:
                avg_conf = np.mean([e['confidence'] for e in frame_event.values()])
                confidences_by_frame.append(avg_conf)
            else:
                confidences_by_frame.append(0)
        
        if confidences_by_frame:
            threshold = np.mean(confidences_by_frame) + np.std(confidences_by_frame)
            peaks = [i for i, conf in enumerate(confidences_by_frame) if conf > threshold]
            analysis['peak_moments'] = peaks
        
        return analysis


class DatasetEventProcessor:
    """Procesador para agregar detección de eventos al dataset"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.detector = AudioEventDetector()
    
    def process_dataset_events(self, overwrite: bool = False) -> bool:
        """
        Procesa el dataset agregando detección de eventos de audio
        
        Args:
            overwrite: Si sobreescribir análisis existente
            
        Returns:
            True si el procesamiento fue exitoso
        """
        try:
            # Cargar dataset
            dataset_file = self.dataset_dir / "final" / "complete_dataset.pkl"
            logger.info(f"📊 Cargando dataset desde: {dataset_file}")
            df = pd.read_pickle(dataset_file)
            
            # Determinar qué segmentos procesar
            if overwrite or 'audio_events_detailed' not in df.columns:
                segments_to_process = df.index.tolist()
            else:
                segments_to_process = df[df['audio_events_detailed'].isna()].index.tolist()
            
            logger.info(f"🔄 Procesando eventos de audio en {len(segments_to_process)} segmentos...")
            
            processed_count = 0
            
            # Procesar cada segmento
            for idx in segments_to_process:
                row = df.iloc[idx]
                
                try:
                    audio_file = row['source_file']
                    if not pd.isna(audio_file) and Path(audio_file).exists():
                        
                        # Detectar eventos
                        events_result = self.detector.detect_events_in_audio(audio_file)
                        
                        if events_result['processing_success']:
                            # Agregar información detallada de eventos
                            df.at[idx, 'audio_events_detailed'] = json.dumps(events_result)
                            
                            # Agregar columnas específicas para eventos importantes
                            events = events_result['events_detected']
                            df.at[idx, 'has_laughter'] = events.get('laughter', {}).get('detected', False)
                            df.at[idx, 'has_applause'] = events.get('applause', {}).get('detected', False)
                            df.at[idx, 'has_music'] = events.get('music', {}).get('detected', False)
                            df.at[idx, 'has_crowd'] = events.get('crowd', {}).get('detected', False)
                            df.at[idx, 'has_cheering'] = events.get('cheering', {}).get('detected', False)
                            
                            # Confidence scores para eventos principales
                            df.at[idx, 'laughter_confidence'] = events.get('laughter', {}).get('max_confidence', 0.0)
                            df.at[idx, 'applause_confidence'] = events.get('applause', {}).get('max_confidence', 0.0)
                            df.at[idx, 'music_confidence'] = events.get('music', {}).get('max_confidence', 0.0)
                            
                            processed_count += 1
                        
                        if processed_count % 50 == 0:
                            logger.info(f"📊 Progreso: {processed_count}/{len(segments_to_process)} segmentos")
                            
                except Exception as e:
                    logger.warning(f"⚠️  Error procesando segmento {idx}: {e}")
                    continue
            
            # Guardar dataset actualizado
            df.to_pickle(dataset_file)
            
            # Guardar resumen de eventos
            self._save_events_summary(df)
            
            logger.info(f"✅ Procesamiento de eventos completado:")
            logger.info(f"  ✅ Segmentos procesados: {processed_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento de eventos: {e}")
            return False
    
    def _save_events_summary(self, df: pd.DataFrame):
        """Guarda un resumen de eventos detectados"""
        try:
            summary = {
                'total_segments': len(df),
                'segments_with_laughter': df['has_laughter'].sum() if 'has_laughter' in df.columns else 0,
                'segments_with_applause': df['has_applause'].sum() if 'has_applause' in df.columns else 0,
                'segments_with_music': df['has_music'].sum() if 'has_music' in df.columns else 0,
                'segments_with_crowd': df['has_crowd'].sum() if 'has_crowd' in df.columns else 0,
                'segments_with_cheering': df['has_cheering'].sum() if 'has_cheering' in df.columns else 0,
                'analysis_date': datetime.now().isoformat()
            }
            
            summary_file = self.dataset_dir / "final" / "audio_events_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📄 Resumen de eventos guardado: {summary_file}")
            
            # Imprimir estadísticas
            logger.info("📊 Estadísticas de eventos detectados:")
            for event, count in summary.items():
                if event.startswith('segments_with_'):
                    event_name = event.replace('segments_with_', '').title()
                    percentage = (count / summary['total_segments']) * 100 if summary['total_segments'] > 0 else 0
                    logger.info(f"  🎵 {event_name}: {count} segmentos ({percentage:.1f}%)")
            
        except Exception as e:
            logger.warning(f"⚠️  No se pudo guardar resumen de eventos: {e}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Detectar eventos de audio en el dataset")
    parser.add_argument("dataset_dir", help="Directorio del dataset")
    parser.add_argument("--overwrite", action="store_true", help="Sobreescribir análisis existente")
    
    args = parser.parse_args()
    
    print("🎵 Detector de Eventos de Audio para Dataset")
    print("=" * 50)
    print(f"📁 Dataset: {args.dataset_dir}")
    print(f"🔄 Overwrite: {args.overwrite}")
    print()
    
    try:
        processor = DatasetEventProcessor(args.dataset_dir)
        success = processor.process_dataset_events(overwrite=args.overwrite)
        
        if success:
            print("✅ ¡Detección de eventos de audio completada!")
            print(f"📁 Dataset actualizado en: {args.dataset_dir}/final/complete_dataset.pkl")
            print()
            print("🔍 Nuevas columnas agregadas:")
            print("  • has_laughter, has_applause, has_music, has_crowd, has_cheering")
            print("  • laughter_confidence, applause_confidence, music_confidence")
            print("  • audio_events_detailed (JSON con análisis completo)")
            print()
            print("🚀 Ahora puedes buscar por eventos específicos:")
            print("   python src/query_client.py ./dataset --interactive")
            print("   🔍 > risas")
            print("   🔍 > aplausos")
            print("   🔍 > música")
            return 0
        else:
            print("❌ Error en la detección de eventos")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())