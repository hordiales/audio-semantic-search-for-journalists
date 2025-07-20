import whisper
import librosa
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence


class AudioTranscriber:
    """
    Clase para transcribir audio usando Whisper y segmentarlo por pausas o tiempo
    """
    
    def __init__(self, model_name: str = "base"):
        """
        Inicializa el transcriptor con el modelo Whisper especificado
        
        Args:
            model_name: Nombre del modelo Whisper ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model = whisper.load_model(model_name)
        self.model_name = model_name
        
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
        Segmenta el audio basándose en silencios
        
        Args:
            audio_path: Ruta al archivo de audio
            min_silence_len: Duración mínima del silencio en ms
            silence_thresh: Umbral de silencio en dB
            
        Returns:
            Lista de diccionarios con información de segmentos
        """
        audio = AudioSegment.from_file(audio_path)
        
        # Dividir por silencios
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=100  # Mantener 100ms de silencio
        )
        
        segments = []
        current_time = 0
        
        for i, chunk in enumerate(chunks):
            duration = len(chunk) / 1000.0  # Duración en segundos
            
            # Guardar segmento temporal
            temp_path = f"temp_segment_{i}.wav"
            chunk.export(temp_path, format="wav")
            
            segment_info = {
                'segment_id': i,
                'start_time': current_time,
                'end_time': current_time + duration,
                'duration': duration,
                'temp_path': temp_path,
                'source_file': audio_path
            }
            
            segments.append(segment_info)
            current_time += duration
            
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
                # Transcribir el segmento
                result = self.model.transcribe(segment['temp_path'])
                
                # Añadir información de transcripción
                segment_with_text = segment.copy()
                segment_with_text.update({
                    'text': result['text'].strip(),
                    'language': result['language'],
                    'confidence': getattr(result, 'confidence', None)
                })
                
                transcribed_segments.append(segment_with_text)
                
                # Limpiar archivo temporal
                if os.path.exists(segment['temp_path']):
                    os.remove(segment['temp_path'])
                    
            except Exception as e:
                print(f"Error transcribiendo segmento {segment['segment_id']}: {e}")
                continue
        
        return transcribed_segments
    
    def process_audio_file(self, file_path: str, segmentation_method: str = "silence",
                          **kwargs) -> pd.DataFrame:
        """
        Procesa un archivo de audio completo: segmenta y transcribe
        
        Args:
            file_path: Ruta al archivo de audio
            segmentation_method: Método de segmentación ('silence' o 'time')
            **kwargs: Argumentos adicionales para el método de segmentación
            
        Returns:
            DataFrame con información de todos los segmentos transcritos
        """
        print(f"Procesando archivo: {file_path}")
        
        # Segmentar audio
        if segmentation_method == "silence":
            segments = self.segment_by_silence(file_path, **kwargs)
        elif segmentation_method == "time":
            segments = self.segment_by_time(file_path, **kwargs)
        else:
            raise ValueError("segmentation_method debe ser 'silence' o 'time'")
        
        print(f"Encontrados {len(segments)} segmentos")
        
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
                print(f"Error procesando {file_path}: {e}")
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
    
    print("Módulo de transcripción listo. Usar AudioTranscriber para procesar archivos.")