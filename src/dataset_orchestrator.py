#!/usr/bin/env python3
"""
Orquestador para generar dataset desde archivos de audio
Proceso completo: Conversión → Transcripción → Embeddings → Dataset
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from audio_conversion import AudioConverter
from audio_transcription import AudioTranscriber
from text_embeddings import TextEmbeddingGenerator
from audio_embeddings import get_audio_embedding_generator
from vector_indexing import VectorIndexManager
from config_loader import get_config


# Funciones globales para procesamiento paralelo (deben estar fuera de la clase)
def convert_single_file_worker(args):
    """Worker function para convertir un archivo individual"""
    audio_file, config, converted_dir = args
    
    try:
        from audio_conversion import AudioConverter
        audio_converter = AudioConverter()
        
        # Generar nombre de archivo de salida
        input_path = Path(config['input_dir'])
        relative_path = audio_file.relative_to(input_path)
        output_path = converted_dir / relative_path.with_suffix('.wav')
        
        # Crear directorio si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Si ya está en WAV y tiene la configuración correcta, copiar
        if audio_file.suffix.lower() == '.wav':
            # TODO: Verificar sample rate y canales
            shutil.copy2(audio_file, output_path)
        else:
            # Convertir usando AudioConverter
            success = audio_converter.convert(
                str(audio_file),
                str(output_path),
                sample_rate=config['sample_rate'],
                channels=config['channels']
            )
            
            if not success:
                return None
        
        return output_path
        
    except Exception as e:
        print(f"Error convirtiendo {audio_file}: {str(e)}")
        return None


def transcribe_single_file_worker(args):
    """Worker function para transcribir un archivo individual"""
    wav_file, config, transcriptions_dir = args
    
    try:
        from audio_transcription import AudioTranscriber
        transcriber = AudioTranscriber(model_name=config['whisper_model'])
        
        # Configurar parámetros de transcripción
        if config['segmentation_method'] == "silence":
            segments_df = transcriber.process_audio_file(
                str(wav_file),
                segmentation_method="silence",
                min_silence_len=config['min_silence_len'],
                silence_thresh=config['silence_thresh']
            )
        else:
            segments_df = transcriber.process_audio_file(
                str(wav_file),
                segmentation_method="time", 
                segment_duration=config['segment_duration']
            )
        
        if len(segments_df) == 0:
            print(f"No se generaron segmentos para: {wav_file.name}")
            return None
        
        # Convertir a diccionario para JSON
        transcription_data = {
            "file_path": str(wav_file),
            "file_name": wav_file.name,
            "transcription_date": datetime.now().isoformat(),
            "total_segments": len(segments_df),
            "total_duration": segments_df['duration'].sum(),
            "whisper_model": config['whisper_model'],
            "language": config['language'],
            "segments": segments_df.to_dict('records')
        }
        
        # Guardar transcripción individual
        if config['save_intermediate']:
            json_file = transcriptions_dir / f"{wav_file.stem}_transcription.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=2)
        
        return transcription_data
        
    except Exception as e:
        print(f"Error transcribiendo {wav_file}: {str(e)}")
        return None


@dataclass
class ProcessingStats:
    """Estadísticas del procesamiento"""
    total_files: int = 0
    converted_files: int = 0
    transcribed_files: int = 0
    embedded_files: int = 0
    failed_files: int = 0
    total_segments: int = 0
    processing_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class DatasetConfig:
    """Configuración para la generación del dataset"""
    # Directorios
    input_dir: str = "/data"
    output_dir: str = "./dataset"
    temp_dir: str = "./temp_processing"
    
    # Conversión de audio
    target_format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1
    
    # Transcripción
    whisper_model: str = "base"
    language: str = "es"
    batch_size: int = 4
    
    # Segmentación
    segmentation_method: str = "silence"
    min_silence_len: int = 500
    silence_thresh: int = -40
    segment_duration: float = 10.0
    
    # Embeddings
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Paralelización
    max_workers: int = 4
    use_multiprocessing: bool = False  # Desactivado por defecto para mayor estabilidad
    
    # Output
    save_intermediate: bool = True
    compress_output: bool = True


class DatasetOrchestrator:
    """Orquestador principal para generar dataset desde archivos de audio"""
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Inicializa el orquestador
        
        Args:
            config: Configuración del dataset, usa defaults si es None
        """
        self.config = config or self._load_default_config()
        self.stats = ProcessingStats()
        
        # Configurar logging
        self._setup_logging()
        
        # Inicializar componentes
        self.audio_converter = AudioConverter()
        self.transcriber = AudioTranscriber(model_name=self.config.whisper_model)
        self.text_embedder = TextEmbeddingGenerator(model_name=self.config.text_model)
        self.audio_embedder = get_audio_embedding_generator()
        
        # Crear directorios
        self._create_directories()
        
        self.logger.info(f"Orquestador inicializado - Input: {self.config.input_dir}")
    
    def _load_default_config(self) -> DatasetConfig:
        """Carga configuración por defecto desde el sistema"""
        system_config = get_config()
        
        return DatasetConfig(
            whisper_model=system_config.default_whisper_model,
            text_model=system_config.default_text_model,
            segmentation_method=system_config.segmentation_method,
            min_silence_len=system_config.min_silence_len,
            silence_thresh=system_config.silence_thresh,
            segment_duration=system_config.segment_duration,
            max_workers=min(mp.cpu_count(), 8)
        )
    
    def _setup_logging(self):
        """Configura el sistema de logging"""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _create_directories(self):
        """Crea los directorios necesarios"""
        dirs = [
            self.config.output_dir,
            self.config.temp_dir,
            f"{self.config.output_dir}/converted",
            f"{self.config.output_dir}/transcriptions", 
            f"{self.config.output_dir}/embeddings",
            f"{self.config.output_dir}/indices",
            f"{self.config.output_dir}/final"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def discover_audio_files(self) -> List[Path]:
        """
        Descubre archivos de audio en el directorio de entrada
        
        Returns:
            Lista de rutas a archivos de audio encontrados
        """
        input_path = Path(self.config.input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Directorio de entrada no existe: {input_path}")
        
        # Extensiones de audio soportadas
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.rglob(f"*{ext}"))
            audio_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        # Ordenar por nombre para procesamiento consistente
        audio_files.sort()
        
        self.logger.info(f"Encontrados {len(audio_files)} archivos de audio")
        self.stats.total_files = len(audio_files)
        
        return audio_files
    
    def step1_convert_audio(self, audio_files: List[Path]) -> List[Path]:
        """
        Paso 1: Convierte todos los archivos a formato WAV
        
        Args:
            audio_files: Lista de archivos de audio a convertir
            
        Returns:
            Lista de archivos WAV convertidos
        """
        self.logger.info("=== PASO 1: Conversión de Audio ===")
        
        converted_dir = Path(self.config.output_dir) / "converted"
        converted_files = []
        
        # Preparar argumentos para workers
        worker_args = [
            (audio_file, asdict(self.config), converted_dir) 
            for audio_file in audio_files
        ]
        
        # Procesar archivos en paralelo
        if self.config.use_multiprocessing and len(audio_files) > 1:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                results = list(executor.map(convert_single_file_worker, worker_args))
        else:
            results = [convert_single_file_worker(args) for args in worker_args]
        
        # Filtrar resultados exitosos y actualizar estadísticas
        converted_files = []
        for result in results:
            if result is not None:
                converted_files.append(result)
                self.stats.converted_files += 1
                self.logger.info(f"Convertido: {result.name}")
            else:
                self.stats.failed_files += 1
        
        self.logger.info(f"Conversión completada: {len(converted_files)}/{len(audio_files)} archivos")
        return converted_files
    
    def step2_transcribe_audio(self, wav_files: List[Path]) -> Dict[str, any]:
        """
        Paso 2: Transcribe archivos WAV usando speech-to-text
        
        Args:
            wav_files: Lista de archivos WAV a transcribir
            
        Returns:
            Diccionario con transcripciones y metadatos
        """
        self.logger.info("=== PASO 2: Transcripción de Audio ===")
        
        transcriptions_dir = Path(self.config.output_dir) / "transcriptions"
        all_transcriptions = []
        
        # Preparar argumentos para workers
        worker_args = [
            (wav_file, asdict(self.config), transcriptions_dir) 
            for wav_file in wav_files
        ]
        
        # Procesar archivos (transcripción es intensiva en GPU, usar menos workers)
        transcription_workers = min(2, self.config.max_workers)
        
        if self.config.use_multiprocessing and len(wav_files) > 1:
            with ThreadPoolExecutor(max_workers=transcription_workers) as executor:
                results = list(executor.map(transcribe_single_file_worker, worker_args))
        else:
            results = [transcribe_single_file_worker(args) for args in worker_args]
        
        # Filtrar resultados exitosos y actualizar estadísticas
        for result in results:
            if result is not None:
                all_transcriptions.append(result)
                self.stats.transcribed_files += 1
                self.stats.total_segments += result["total_segments"]
                self.logger.info(f"Transcrito: {result['file_name']} - {result['total_segments']} segmentos")
            else:
                self.stats.failed_files += 1
        
        # Guardar transcripciones consolidadas
        consolidated_data = {
            "dataset_info": {
                "creation_date": datetime.now().isoformat(),
                "total_files": len(all_transcriptions),
                "total_segments": sum(t["total_segments"] for t in all_transcriptions),
                "total_duration": sum(t["total_duration"] for t in all_transcriptions),
                "config": asdict(self.config)
            },
            "transcriptions": all_transcriptions
        }
        
        # Guardar archivo consolidado
        consolidated_file = transcriptions_dir / "all_transcriptions.json"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Transcripción completada: {len(all_transcriptions)} archivos, {self.stats.total_segments} segmentos")
        return consolidated_data
    
    def step3_generate_embeddings(self, transcription_data: Dict) -> pd.DataFrame:
        """
        Paso 3: Genera embeddings de texto y audio
        
        Args:
            transcription_data: Datos de transcripción del paso anterior
            
        Returns:
            DataFrame con todos los datos incluyendo embeddings
        """
        self.logger.info("=== PASO 3: Generación de Embeddings ===")
        
        embeddings_dir = Path(self.config.output_dir) / "embeddings"
        
        # Convertir transcripciones a DataFrame
        all_segments = []
        for transcription in transcription_data["transcriptions"]:
            for segment in transcription["segments"]:
                segment["source_file"] = transcription["file_name"]
                segment["file_path"] = transcription["file_path"]
                all_segments.append(segment)
        
        df = pd.DataFrame(all_segments)
        
        if len(df) == 0:
            raise ValueError("No hay segmentos para procesar")
        
        self.logger.info(f"Generando embeddings para {len(df)} segmentos")
        
        # Paso 3.1: Embeddings de texto
        self.logger.info("Generando embeddings de texto...")
        df_with_text_embeddings = self.text_embedder.process_transcription_dataframe(df)
        
        # Paso 3.2: Embeddings de audio
        self.logger.info("Generando embeddings de audio...")
        df_with_all_embeddings = self.audio_embedder.process_transcription_dataframe(df_with_text_embeddings)
        
        # Guardar DataFrame con embeddings
        embeddings_file = embeddings_dir / "embeddings_data.pkl"
        df_with_all_embeddings.to_pickle(embeddings_file)
        
        # Guardar también versión CSV (sin embeddings para legibilidad)
        csv_df = df_with_all_embeddings.drop(columns=['text_embedding', 'audio_embedding'], errors='ignore')
        csv_file = embeddings_dir / "segments_metadata.csv"
        csv_df.to_csv(csv_file, index=False)
        
        self.stats.embedded_files = len(df_with_all_embeddings)
        self.logger.info(f"Embeddings generados para {len(df_with_all_embeddings)} segmentos")
        
        return df_with_all_embeddings
    
    def step4_create_vector_indices(self, df_with_embeddings: pd.DataFrame) -> Dict[str, str]:
        """
        Paso 4: Crea índices vectoriales para búsqueda
        
        Args:
            df_with_embeddings: DataFrame con embeddings generados
            
        Returns:
            Diccionario con rutas de los índices creados
        """
        self.logger.info("=== PASO 4: Creación de Índices Vectoriales ===")
        
        indices_dir = Path(self.config.output_dir) / "indices"
        
        # Inicializar gestor de índices
        embedding_dim = self.text_embedder.embedding_dim
        index_manager = VectorIndexManager(embedding_dim=embedding_dim)
        
        # Crear índice de texto
        self.logger.info("Creando índice de texto...")
        text_success = index_manager.create_text_index(df_with_embeddings)
        
        # Crear índice de audio
        self.logger.info("Creando índice de audio...")
        audio_success = index_manager.create_audio_index(df_with_embeddings)
        
        if not (text_success or audio_success):
            raise RuntimeError("No se pudo crear ningún índice vectorial")
        
        # Guardar índices
        index_manager.save_indices(str(indices_dir))
        
        # Crear archivo de metadatos de índices
        indices_metadata = {
            "creation_date": datetime.now().isoformat(),
            "total_vectors": len(df_with_embeddings),
            "embedding_dimension": embedding_dim,
            "text_index_created": text_success,
            "audio_index_created": audio_success,
            "text_model": self.config.text_model,
            "audio_model": "yamnet"
        }
        
        metadata_file = indices_dir / "indices_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(indices_metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Índices creados - Texto: {text_success}, Audio: {audio_success}")
        
        return {
            "indices_dir": str(indices_dir),
            "text_index": str(indices_dir / "text_index.faiss") if text_success else None,
            "audio_index": str(indices_dir / "audio_index.faiss") if audio_success else None,
            "metadata": str(metadata_file)
        }
    
    def step5_create_final_dataset(self, df_with_embeddings: pd.DataFrame, indices_info: Dict) -> Dict[str, str]:
        """
        Paso 5: Crea el dataset final consolidado
        
        Args:
            df_with_embeddings: DataFrame con todos los datos
            indices_info: Información de los índices creados
            
        Returns:
            Diccionario con rutas del dataset final
        """
        self.logger.info("=== PASO 5: Creación de Dataset Final ===")
        
        final_dir = Path(self.config.output_dir) / "final"
        
        # Guardar dataset completo
        dataset_file = final_dir / "complete_dataset.pkl"
        df_with_embeddings.to_pickle(dataset_file)
        
        # Crear versión CSV sin embeddings
        csv_df = df_with_embeddings.drop(columns=['text_embedding', 'audio_embedding'], errors='ignore')
        csv_file = final_dir / "dataset_metadata.csv"
        csv_df.to_csv(csv_file, index=False)
        
        # Crear manifiesto del dataset
        # Convertir stats a dict serializable
        stats_dict = asdict(self.stats)
        if stats_dict.get('start_time'):
            stats_dict['start_time'] = stats_dict['start_time'].isoformat()
        if stats_dict.get('end_time'):
            stats_dict['end_time'] = stats_dict['end_time'].isoformat()
        
        manifest = {
            "dataset_info": {
                "name": f"audio_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "creation_date": datetime.now().isoformat(),
                "version": "1.0",
                "description": "Dataset de audio procesado con transcripciones y embeddings"
            },
            "statistics": stats_dict,
            "config": asdict(self.config),
            "data_files": {
                "complete_dataset": str(dataset_file),
                "metadata_csv": str(csv_file),
                "indices": indices_info
            },
            "schema": {
                "required_columns": ["text", "start_time", "end_time", "duration", "source_file"],
                "embedding_columns": ["text_embedding", "audio_embedding"],
                "metadata_columns": ["language", "confidence", "segment_id"]
            }
        }
        
        manifest_file = final_dir / "dataset_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        # Crear archivo README para el dataset
        readme_content = f"""# Dataset de Audio Procesado

## Información General
- **Fecha de creación**: {manifest['dataset_info']['creation_date']}
- **Total de archivos procesados**: {self.stats.total_files}
- **Total de segmentos**: {self.stats.total_segments}
- **Tiempo de procesamiento**: {self.stats.processing_time:.2f} segundos

## Archivos Principales
- `complete_dataset.pkl`: Dataset completo con embeddings
- `dataset_metadata.csv`: Metadatos en formato CSV (sin embeddings)
- `dataset_manifest.json`: Manifiesto completo del dataset

## Estructura del Dataset
Cada fila representa un segmento de audio con:
- **text**: Transcripción del segmento
- **start_time/end_time**: Tiempos en segundos
- **duration**: Duración del segmento
- **source_file**: Archivo de audio original
- **text_embedding**: Vector embedding del texto
- **audio_embedding**: Vector embedding del audio

## Uso del Dataset
```python
import pandas as pd

# Cargar dataset completo
df = pd.read_pickle('complete_dataset.pkl')

# O solo metadatos
df_meta = pd.read_csv('dataset_metadata.csv')
```

## Configuración Utilizada
- **Modelo Whisper**: {self.config.whisper_model}
- **Modelo de texto**: {self.config.text_model}
- **Segmentación**: {self.config.segmentation_method}
- **Sample rate**: {self.config.sample_rate} Hz
"""
        
        readme_file = final_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        self.logger.info(f"Dataset final creado en: {final_dir}")
        
        return {
            "dataset_dir": str(final_dir),
            "complete_dataset": str(dataset_file),
            "metadata_csv": str(csv_file),
            "manifest": str(manifest_file),
            "readme": str(readme_file)
        }
    
    def run_full_pipeline(self) -> Dict[str, any]:
        """
        Ejecuta el pipeline completo de procesamiento
        
        Returns:
            Diccionario con resultados y estadísticas
        """
        self.logger.info("🚀 INICIANDO PIPELINE COMPLETO DE PROCESAMIENTO")
        self.stats.start_time = datetime.now()
        
        try:
            # Paso 0: Descubrir archivos
            audio_files = self.discover_audio_files()
            
            if not audio_files:
                raise ValueError("No se encontraron archivos de audio para procesar")
            
            # Paso 1: Conversión
            wav_files = self.step1_convert_audio(audio_files)
            
            # Paso 2: Transcripción
            transcription_data = self.step2_transcribe_audio(wav_files)
            
            # Paso 3: Embeddings
            df_with_embeddings = self.step3_generate_embeddings(transcription_data)
            
            # Paso 4: Índices vectoriales
            indices_info = self.step4_create_vector_indices(df_with_embeddings)
            
            # Paso 5: Dataset final
            final_dataset = self.step5_create_final_dataset(df_with_embeddings, indices_info)
            
            # Completar estadísticas
            self.stats.end_time = datetime.now()
            self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            # Limpiar directorio temporal si se desea
            if not self.config.save_intermediate:
                shutil.rmtree(self.config.temp_dir, ignore_errors=True)
            
            result = {
                "success": True,
                "stats": asdict(self.stats),
                "dataset": final_dataset,
                "indices": indices_info,
                "message": f"Pipeline completado exitosamente - {self.stats.total_segments} segmentos procesados"
            }
            
            self.logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            return result
            
        except Exception as e:
            self.stats.end_time = datetime.now()
            self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            self.logger.error(f"❌ ERROR EN PIPELINE: {str(e)}", exc_info=True)
            
            return {
                "success": False,
                "error": str(e),
                "stats": asdict(self.stats),
                "message": f"Pipeline falló después de {self.stats.processing_time:.2f} segundos"
            }
    
    def cleanup(self):
        """Limpia recursos y archivos temporales"""
        if hasattr(self, 'config') and Path(self.config.temp_dir).exists():
            shutil.rmtree(self.config.temp_dir, ignore_errors=True)
        
        self.logger.info("Limpieza completada")


if __name__ == "__main__":
    # Ejemplo de uso
    config = DatasetConfig(
        input_dir="/data",
        output_dir="./audio_dataset",
        whisper_model="base",
        max_workers=4,
    )
    
    orchestrator = DatasetOrchestrator(config)
    
    try:
        result = orchestrator.run_full_pipeline()
        
        if result["success"]:
            print(f"\n✅ Dataset creado exitosamente!")
            print(f"📊 Estadísticas: {result['stats']['total_segments']} segmentos en {result['stats']['processing_time']:.2f}s")
            print(f"📁 Dataset: {result['dataset']['dataset_dir']}")
        else:
            print(f"\n❌ Error: {result['error']}")
            
    finally:
        orchestrator.cleanup()