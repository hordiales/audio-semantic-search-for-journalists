import librosa
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

import logging
import sys

try:
    from .models_config import get_models_config, WhisperConfig
except ImportError:
    from models_config import get_models_config, WhisperConfig

# Configuración de logging
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[handler])


class AudioTranscriber:
    """Transcripción de audio usando Whisper con configuración centralizada"""
    
    def __init__(self, model_name: Optional[str] = None, config: Optional[WhisperConfig] = None):
        """
        Inicializa el transcriptor
        
        Args:
            model_name: Nombre del modelo Whisper (tiny, base, small, medium, large) - opcional
            config: Configuración específica de Whisper - usa configuración global si None
        """
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper no está disponible. Instala con: pip install openai-whisper")
        
        # Cargar configuración
        if config is None:
            models_config = get_models_config()
            self.config = models_config.whisper_config
            # Usar el modelo configurado globalmente si no se especifica uno
            if model_name is None:
                model_name = models_config.get_whisper_model_name()
        else:
            self.config = config
            if model_name is None:
                model_name = config.model_name
        
        self.model_name = model_name
        self.device = self._get_device()
        self.model = whisper.load_model(model_name, device=self.device)
        logging.info(f"Modelo Whisper '{model_name}' cargado en {self.device}")
    
    def _get_device(self) -> str:
        """Determina el device a usar basado en la configuración"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return self.config.device
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Carga un archivo de audio y lo convierte al formato requerido por Whisper
        
        Args:
            file_path: Ruta al archivo de audio
            
        Returns:
            Audio normalizado como array numpy
        """
        # Whisper requiere 16kHz mono
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        return audio
    
    def segment_by_silence(self, audio_path: str, min_silence_len: int = 500, 
                          silence_thresh: int = -40) -> List[Dict]:
        """
        Segmenta el audio basándose en silencios preservando timestamps originales
        
        Args:
            audio_path: Ruta al archivo de audio
            min_silence_len: Duración mínima del silencio en ms
            silence_thresh: Umbral de silencio en dB
            
        Returns:
            Lista de diccionarios con información de segmentos con timestamps correctos
        """
        audio = AudioSegment.from_file(audio_path)
        
        # PASO 1: Detectar silencios para obtener las posiciones originales
        silence_ranges = detect_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
        # PASO 2: Calcular los rangos de audio (no-silencio) con timestamps originales
        audio_ranges = []
        start_pos = 0
        
        for silence_start, silence_end in silence_ranges:
            if start_pos < silence_start:
                # Hay audio antes de este silencio
                audio_ranges.append((start_pos, silence_start))
            start_pos = silence_end
        
        # Añadir el último segmento si existe audio después del último silencio
        if start_pos < len(audio):
            audio_ranges.append((start_pos, len(audio)))
        
        # Si no hay silencios detectados, todo el audio es un segmento
        if not silence_ranges and len(audio) > 0:
            audio_ranges = [(0, len(audio))]
        
        # PASO 3: Crear segmentos con timestamps originales correctos
        segments = []
        
        for i, (start_ms, end_ms) in enumerate(audio_ranges):
            # Extraer el chunk con las posiciones originales
            chunk = audio[start_ms:end_ms]
            
            # Calcular timestamps en segundos (basados en posiciones originales)
            start_time = start_ms / 1000.0
            end_time = end_ms / 1000.0
            duration = (end_ms - start_ms) / 1000.0
            
            # Guardar segmento temporal
            temp_path = f"temp_segment_{i}.wav"
            chunk.export(temp_path, format="wav")
            
            segment_info = {
                'segment_id': i,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'temp_path': temp_path,
                'source_file': audio_path,
                'original_start_ms': start_ms,  # Para debugging
                'original_end_ms': end_ms       # Para debugging
            }
            
            segments.append(segment_info)
        
        logging.info(f"Segmentación por silencio: {len(segments)} segmentos detectados")
        if len(segments) > 0:
            total_audio_time = sum(seg['duration'] for seg in segments)
            original_duration = len(audio) / 1000.0
            logging.info(f"Tiempo total de audio: {total_audio_time:.2f}s de {original_duration:.2f}s originales")
        
        return segments
    
    def segment_by_time(self, audio_path: str, segment_duration: float = 10.0) -> List[Dict]:
        """
        Segmenta el audio en intervalos fijos de tiempo
        
        Args:
            audio_path: Ruta al archivo de audio
            segment_duration: Duración de cada segmento en segundos
            
        Returns:
            Lista de diccionarios con información de segmentos
        """
        audio = AudioSegment.from_file(audio_path)
        total_duration = len(audio) / 1000.0  # Duración total en segundos
        
        segments = []
        current_time = 0
        segment_id = 0
        
        while current_time < total_duration:
            start_ms = int(current_time * 1000)
            end_ms = int(min((current_time + segment_duration) * 1000, len(audio)))
            
            chunk = audio[start_ms:end_ms]
            duration = (end_ms - start_ms) / 1000.0
            
            # Guardar segmento temporal
            temp_path = f"temp_segment_{segment_id}.wav"
            chunk.export(temp_path, format="wav")
            
            segment_info = {
                'segment_id': segment_id,
                'start_time': current_time,
                'end_time': current_time + duration,
                'duration': duration,
                'temp_path': temp_path,
                'source_file': audio_path
            }
            
            segments.append(segment_info)
            current_time += segment_duration
            segment_id += 1
            
        return segments
    
    def transcribe_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Transcribe una lista de segmentos de audio
        
        Args:
            segments: Lista de diccionarios con información de segmentos
            
        Returns:
            Lista de segmentos con transcripciones añadidas
        """
        transcribed_segments = []
        
        for segment in segments:
            try:
                # Transcribir el segmento usando configuración
                transcribe_options = {
                    'language': self.config.language,
                    'temperature': self.config.temperature,
                    'no_speech_threshold': self.config.no_speech_threshold,
                    'logprob_threshold': self.config.logprob_threshold,
                    'compression_ratio_threshold': self.config.compression_ratio_threshold
                }
                
                # Filtrar opciones None
                transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
                
                result = self.model.transcribe(segment['temp_path'], **transcribe_options)
                
                # Añadir información de transcripción
                segment_with_text = segment.copy()
                segment_with_text.update({
                    'text': result['text'].strip(),
                    'language': result['language'],
                    'confidence': getattr(result, 'confidence', None),
                    'whisper_model': self.model_name,
                    'whisper_config': transcribe_options
                })
                
                transcribed_segments.append(segment_with_text)
                
                # Limpiar archivo temporal
                if os.path.exists(segment['temp_path']):
                    os.remove(segment['temp_path'])
                    
            except Exception as e:
                logging.error(f"Error transcribiendo segmento {segment['segment_id']}: {e}")
                continue
        
        return transcribed_segments
    
    def process_audio_file(self, file_path: str, segmentation_method: str = "silence",
                          **kwargs) -> pd.DataFrame:
        """
        Procesa un archivo de audio completo: segmenta y transcribe
        
        Args:
            file_path: Ruta al archivo de audio
            segmentation_method: 'silence' o 'time'
            min_silence_len: Duración mínima de silencio para segmentar (ms)
            silence_thresh: Umbral de silencio (dBFS)
            segment_duration: Duración de cada segmento en segundos (para método 'time')
            
        Returns:
            DataFrame con segmentos y transcripciones
        """
        logging.info(f"Procesando archivo: {file_path}")
        
        # Cargar y segmentar audio
        if segmentation_method == 'silence':
            segments = self.segment_by_silence(file_path, **kwargs)
        elif segmentation_method == "time":
            segments = self.segment_by_time(file_path, **kwargs)
        else:
            raise ValueError("segmentation_method debe ser 'silence' o 'time'")
        
        logging.info(f"Encontrados {len(segments)} segmentos")
        
        # Transcribir segmentos
        transcribed_segments = self.transcribe_segments(segments)
        
        # Convertir a DataFrame
        df = pd.DataFrame(transcribed_segments)
        
        # Filtrar segmentos vacíos o muy cortos
        df = df[df['text'].str.len() > 3]
        
        return df
    
    def process_multiple_files(self, file_paths: List[str], 
                             segmentation_method: str = "silence",
                             **kwargs) -> pd.DataFrame:
        """
        Procesa múltiples archivos de audio
        
        Args:
            file_paths: Lista de rutas a archivos de audio
            segmentation_method: Método de segmentación
            **kwargs: Argumentos adicionales
            
        Returns:
            DataFrame combinado con todos los segmentos
        """
        all_segments = []
        
        for file_path in file_paths:
            try:
                segments_df = self.process_audio_file(file_path, segmentation_method, **kwargs)
                all_segments.append(segments_df)
            except Exception as e:
                logging.error(f"Error procesando {file_path}: {e}")
                continue
        
        if all_segments:
            combined_df = pd.concat(all_segments, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso del transcriptor
    transcriber = AudioTranscriber(model_name="base")
    
    # Procesar un archivo individual
    # df = transcriber.process_audio_file("ejemplo.wav", segmentation_method="silence")
    # print(df.head())
    
    # Procesar múltiples archivos
    # files = ["audio1.wav", "audio2.mp3"]
    # df = transcriber.process_multiple_files(files)
    # print(f"Total de segmentos: {len(df)}")
    
    logging.info("Módulo de transcripción listo. Usar AudioTranscriber para procesar archivos.")