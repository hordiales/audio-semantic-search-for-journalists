#!/usr/bin/env python3
"""
Pipeline simplificado para generar dataset desde archivos de audio
Versión más estable sin multiprocesamiento complejo
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import shutil

import pandas as pd
from tqdm import tqdm

from audio_conversion import AudioConverter
from audio_embeddings import get_audio_embedding_generator
from audio_transcription import AudioTranscriber
from config_loader import get_config
from text_embeddings import TextEmbeddingGenerator
from vector_indexing import VectorIndexManager


class SimpleDatasetPipeline:
    """Pipeline simplificado para generar dataset"""

    def __init__(self, input_dir: str, output_dir: str, **kwargs):
        """
        Inicializa el pipeline

        Args:
            input_dir: Directorio con archivos de audio
            output_dir: Directorio de salida
            **kwargs: Configuración adicional
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Configuración
        system_config = get_config()
        self.config = {
            'whisper_model': kwargs.get('whisper_model', system_config.default_whisper_model),
            'text_model': kwargs.get('text_model', system_config.default_text_model),
            'sample_rate': kwargs.get('sample_rate', 16000),
            'channels': kwargs.get('channels', 1),
            'segmentation_method': kwargs.get('segmentation_method', system_config.segmentation_method),
            'min_silence_len': kwargs.get('min_silence_len', system_config.min_silence_len),
            'silence_thresh': kwargs.get('silence_thresh', system_config.silence_thresh),
            'segment_duration': kwargs.get('segment_duration', system_config.segment_duration),
            'language': kwargs.get('language', 'es'),
            'save_intermediate': kwargs.get('save_intermediate', True)
        }

        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "converted").mkdir(exist_ok=True)
        (self.output_dir / "transcriptions").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "indices").mkdir(exist_ok=True)
        (self.output_dir / "final").mkdir(exist_ok=True)

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "pipeline.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Inicializar componentes
        self.audio_converter = AudioConverter()
        self.transcriber = AudioTranscriber(model_name=self.config['whisper_model'])
        self.text_embedder = TextEmbeddingGenerator(model_name=self.config['text_model'])
        self.audio_embedder = get_audio_embedding_generator()

        # Loggear modelos utilizados
        self.logger.info("=" * 60)
        self.logger.info("CONFIGURACIÓN DE MODELOS")
        self.logger.info("=" * 60)
        self.logger.info(f"🤖 Modelo Whisper: {self.config['whisper_model']}")
        self.logger.info(f"📝 Modelo de texto: {self.config['text_model']}")
        self.logger.info(f"🎵 Modelo de audio: {getattr(self.audio_embedder, 'model_name', 'Desconocido')}")
        self.logger.info(f"📊 Dimensión embeddings texto: {self.text_embedder.embedding_dim}")
        self.logger.info(f"📊 Dimensión embeddings audio: {getattr(self.audio_embedder, 'embedding_dim', 'N/A')}")
        self.logger.info("=" * 60)

        # Estadísticas
        self.stats = {
            'start_time': datetime.now(),
            'total_files': 0,
            'converted_files': 0,
            'transcribed_files': 0,
            'total_segments': 0,
            'failed_files': 0,
            'errors': []
        }

    def discover_audio_files(self) -> list[Path]:
        """Descubre archivos de audio en el directorio"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directorio no existe: {self.input_dir}")

        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(self.input_dir.rglob(f"*{ext}"))
            audio_files.extend(self.input_dir.rglob(f"*{ext.upper()}"))

        audio_files.sort()
        self.stats['total_files'] = len(audio_files)
        self.logger.info(f"Encontrados {len(audio_files)} archivos de audio")

        return audio_files

    def convert_audio_files(self, audio_files: list[Path]) -> list[Path]:
        """Convierte archivos de audio a WAV"""
        self.logger.info("=== PASO 1: Conversión de Audio ===")

        converted_dir = self.output_dir / "converted"
        converted_files = []

        for audio_file in tqdm(audio_files, desc="Convirtiendo archivos"):
            try:
                # Generar ruta de salida
                relative_path = audio_file.relative_to(self.input_dir)
                output_path = converted_dir / relative_path.with_suffix('.wav')

                # Crear directorio si no existe
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Convertir o copiar
                if audio_file.suffix.lower() == '.wav':
                    shutil.copy2(audio_file, output_path)
                else:
                    success = self.audio_converter.convert(
                        str(audio_file),
                        str(output_path),
                        sample_rate=self.config['sample_rate'],
                        channels=self.config['channels']
                    )

                    if not success:
                        self.stats['failed_files'] += 1
                        self.stats['errors'].append(f"Conversión falló: {audio_file}")
                        continue

                converted_files.append(output_path)
                self.stats['converted_files'] += 1

            except Exception as e:
                self.logger.error(f"Error convirtiendo {audio_file}: {e!s}")
                self.stats['failed_files'] += 1
                self.stats['errors'].append(f"Error convirtiendo {audio_file}: {e!s}")

        self.logger.info(f"Conversión completada: {len(converted_files)}/{len(audio_files)} archivos")
        return converted_files

    def transcribe_audio_files(self, wav_files: list[Path]) -> list[dict]:
        """Transcribe archivos WAV"""
        self.logger.info("=== PASO 2: Transcripción de Audio ===")
        self.logger.info(f"Usando modelo Whisper: {self.config['whisper_model']}")
        self.logger.info(f"Método de segmentación: {self.config['segmentation_method']}")

        transcriptions_dir = self.output_dir / "transcriptions"
        all_transcriptions = []

        for wav_file in tqdm(wav_files, desc="Transcribiendo archivos"):
            try:
                # Transcribir archivo
                if self.config['segmentation_method'] == "silence":
                    segments_df = self.transcriber.process_audio_file(
                        str(wav_file),
                        segmentation_method="silence",
                        min_silence_len=self.config['min_silence_len'],
                        silence_thresh=self.config['silence_thresh']
                    )
                else:
                    segments_df = self.transcriber.process_audio_file(
                        str(wav_file),
                        segmentation_method="time",
                        segment_duration=self.config['segment_duration']
                    )

                if len(segments_df) == 0:
                    self.logger.warning(f"No se generaron segmentos para: {wav_file.name}")
                    continue

                # Crear datos de transcripción
                transcription_data = {
                    "file_path": str(wav_file),
                    "file_name": wav_file.name,
                    "transcription_date": datetime.now().isoformat(),
                    "total_segments": len(segments_df),
                    "total_duration": float(segments_df['duration'].sum()),
                    "whisper_model": self.config['whisper_model'],
                    "language": self.config['language'],
                    "segments": segments_df.to_dict('records')
                }

                # Guardar transcripción individual
                if self.config['save_intermediate']:
                    json_file = transcriptions_dir / f"{wav_file.stem}_transcription.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(transcription_data, f, ensure_ascii=False, indent=2)

                all_transcriptions.append(transcription_data)
                self.stats['transcribed_files'] += 1
                self.stats['total_segments'] += len(segments_df)

            except Exception as e:
                self.logger.error(f"Error transcribiendo {wav_file}: {e!s}")
                self.stats['failed_files'] += 1
                self.stats['errors'].append(f"Error transcribiendo {wav_file}: {e!s}")

        # Guardar transcripciones consolidadas
        consolidated_data = {
            "dataset_info": {
                "creation_date": datetime.now().isoformat(),
                "total_files": len(all_transcriptions),
                "total_segments": self.stats['total_segments'],
                "config": self.config
            },
            "transcriptions": all_transcriptions
        }

        consolidated_file = transcriptions_dir / "all_transcriptions.json"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Transcripción completada: {len(all_transcriptions)} archivos, {self.stats['total_segments']} segmentos")
        return all_transcriptions

    def generate_embeddings(self, transcriptions: list[dict]) -> pd.DataFrame:
        """Genera embeddings de texto y audio"""
        self.logger.info("=== PASO 3: Generación de Embeddings ===")

        # Convertir a DataFrame
        all_segments = []
        for transcription in transcriptions:
            for segment in transcription["segments"]:
                # Mantener la ruta completa del archivo WAV convertido
                segment["source_file"] = transcription["file_path"]
                segment["original_file_name"] = transcription["file_name"]
                all_segments.append(segment)

        df = pd.DataFrame(all_segments)

        if len(df) == 0:
            raise ValueError("No hay segmentos para procesar")

        self.logger.info(f"Generando embeddings para {len(df)} segmentos")

        # Embeddings de texto
        self.logger.info(f"Generando embeddings de texto usando modelo: {self.config['text_model']}")
        df_with_text = self.text_embedder.process_transcription_dataframe(df)

        # Embeddings de audio
        audio_model_name = getattr(self.audio_embedder, 'model_name', 'Desconocido')
        self.logger.info(f"Generando embeddings de audio usando modelo: {audio_model_name}")
        df_with_all = self.audio_embedder.process_transcription_dataframe(df_with_text)

        # Guardar embeddings
        embeddings_dir = self.output_dir / "embeddings"
        embeddings_file = embeddings_dir / "embeddings_data.pkl"
        df_with_all.to_pickle(embeddings_file)

        # Guardar CSV sin embeddings
        csv_df = df_with_all.drop(columns=['text_embedding', 'audio_embedding'], errors='ignore')
        csv_file = embeddings_dir / "segments_metadata.csv"
        csv_df.to_csv(csv_file, index=False)

        self.logger.info(f"Embeddings generados para {len(df_with_all)} segmentos")
        return df_with_all

    def create_vector_indices(self, df: pd.DataFrame) -> dict:
        """Crea índices vectoriales"""
        self.logger.info("=== PASO 4: Creación de Índices Vectoriales ===")

        indices_dir = self.output_dir / "indices"

        # Crear índices
        index_manager = VectorIndexManager(embedding_dim=self.text_embedder.embedding_dim)

        text_success = index_manager.create_text_index(df)
        audio_success = index_manager.create_audio_index(df)

        if text_success or audio_success:
            index_manager.save_indices(str(indices_dir))

            # Obtener nombre real del modelo de audio
            audio_model_name = getattr(self.audio_embedder, 'model_name', 'yamnet')

            # Guardar metadatos
            metadata = {
                "creation_date": datetime.now().isoformat(),
                "total_vectors": len(df),
                "embedding_dimension": self.text_embedder.embedding_dim,
                "text_index_created": text_success,
                "audio_index_created": audio_success,
                "text_model": self.config['text_model'],
                "audio_model": audio_model_name
            }

            metadata_file = indices_dir / "indices_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Índices creados - Texto: {text_success}, Audio: {audio_success}")
            self.logger.info(f"Modelos utilizados - Texto: {self.config['text_model']}, Audio: {audio_model_name}")
            return metadata
        raise RuntimeError("No se pudo crear ningún índice")

    def create_final_dataset(self, df: pd.DataFrame) -> dict:
        """Crea el dataset final"""
        self.logger.info("=== PASO 5: Creación de Dataset Final ===")

        final_dir = self.output_dir / "final"

        # Guardar dataset completo
        dataset_file = final_dir / "complete_dataset.pkl"
        df.to_pickle(dataset_file)

        # CSV sin embeddings
        csv_df = df.drop(columns=['text_embedding', 'audio_embedding'], errors='ignore')
        csv_file = final_dir / "dataset_metadata.csv"
        csv_df.to_csv(csv_file, index=False)

        # Estadísticas finales
        self.stats['end_time'] = datetime.now()
        self.stats['processing_time'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        # Convertir stats a dict serializable
        stats_dict = self.stats.copy()
        if stats_dict.get('start_time'):
            stats_dict['start_time'] = stats_dict['start_time'].isoformat()
        if stats_dict.get('end_time'):
            stats_dict['end_time'] = stats_dict['end_time'].isoformat()

        # Obtener información de modelos del DataFrame si está disponible
        audio_model_name = getattr(self.audio_embedder, 'model_name', 'Desconocido')
        audio_embedding_dim = getattr(self.audio_embedder, 'embedding_dim', None)

        # Extraer información de modelos del DataFrame si existe
        text_model_from_df = None
        text_dim_from_df = None
        audio_model_from_df = None
        audio_dim_from_df = None

        if 'embedding_model' in df.columns:
            text_model_from_df = df['embedding_model'].iloc[0] if len(df) > 0 else None
        if 'embedding_dim' in df.columns:
            text_dim_from_df = int(df['embedding_dim'].iloc[0]) if len(df) > 0 else None
        if 'audio_embedding_model' in df.columns:
            audio_model_from_df = df['audio_embedding_model'].iloc[0] if len(df) > 0 else None
        if 'audio_embedding_dim' in df.columns:
            audio_dim_from_df = int(df['audio_embedding_dim'].iloc[0]) if len(df) > 0 else None

        # Usar información del DataFrame si está disponible, sino usar la de los objetos
        final_text_model = text_model_from_df or self.config['text_model']
        final_text_dim = text_dim_from_df or self.text_embedder.embedding_dim
        final_audio_model = audio_model_from_df or audio_model_name
        final_audio_dim = audio_dim_from_df or audio_embedding_dim

        # Manifiesto
        manifest = {
            "dataset_info": {
                "name": f"audio_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "creation_date": datetime.now().isoformat(),
                "total_files": self.stats['transcribed_files'],
                "total_segments": self.stats['total_segments'],
                "processing_time": self.stats['processing_time']
            },
            "config": self.config,
            "models_used": {
                "transcription": {
                    "model": self.config['whisper_model'],
                    "language": self.config.get('language', 'es'),
                    "segmentation_method": self.config['segmentation_method'],
                    "segmentation_params": {
                        "min_silence_len": self.config.get('min_silence_len'),
                        "silence_thresh": self.config.get('silence_thresh'),
                        "segment_duration": self.config.get('segment_duration')
                    } if self.config['segmentation_method'] == 'time' else {
                        "min_silence_len": self.config.get('min_silence_len'),
                        "silence_thresh": self.config.get('silence_thresh')
                    }
                },
                "text_embeddings": {
                    "model": final_text_model,
                    "embedding_dimension": final_text_dim,
                    "model_type": "sentence-transformers"
                },
                "audio_embeddings": {
                    "model": final_audio_model,
                    "embedding_dimension": final_audio_dim,
                    "model_type": "audio_embedding"
                }
            },
            "statistics": stats_dict,
            "files": {
                "complete_dataset": str(dataset_file),
                "metadata_csv": str(csv_file)
            }
        }

        manifest_file = final_dir / "dataset_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Dataset final creado en: {final_dir}")
        return manifest

    def run_pipeline(self) -> dict:
        """Ejecuta el pipeline completo"""
        self.logger.info("🚀 INICIANDO PIPELINE DE DATASET")

        try:
            # Paso 1: Descubrir archivos
            audio_files = self.discover_audio_files()

            if not audio_files:
                raise ValueError("No se encontraron archivos de audio")

            # Paso 2: Convertir
            wav_files = self.convert_audio_files(audio_files)

            # Paso 3: Transcribir
            transcriptions = self.transcribe_audio_files(wav_files)

            # Paso 4: Embeddings
            df_with_embeddings = self.generate_embeddings(transcriptions)

            # Paso 5: Índices
            indices_metadata = self.create_vector_indices(df_with_embeddings)

            # Paso 6: Dataset final
            manifest = self.create_final_dataset(df_with_embeddings)

            self.logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")

            return {
                "success": True,
                "stats": self.stats,
                "manifest": manifest,
                "indices": indices_metadata
            }

        except Exception as e:
            self.logger.error(f"❌ ERROR EN PIPELINE: {e!s}")

            return {
                "success": False,
                "error": str(e),
                "stats": self.stats
            }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline simplificado de dataset de audio")
    parser.add_argument("--input", "-i", required=True, help="Directorio de entrada")
    parser.add_argument("--output", "-o", default="./dataset", help="Directorio de salida")
    parser.add_argument("--whisper-model", default="base", help="Modelo Whisper")
    parser.add_argument("--segmentation", default="silence", choices=["silence", "time"])

    args = parser.parse_args()

    # Crear pipeline
    pipeline = SimpleDatasetPipeline(
        input_dir=args.input,
        output_dir=args.output,
        whisper_model=args.whisper_model,
        segmentation_method=args.segmentation
    )

    # Ejecutar
    result = pipeline.run_pipeline()

    if result["success"]:
        logging.info("\n✅ Pipeline completado exitosamente!")
        logging.info(f"📊 Segmentos procesados: {result['stats']['total_segments']}")
        logging.info(f"⏱️ Tiempo: {result['stats']['processing_time']:.2f} segundos")
        logging.info(f"📁 Dataset: {pipeline.output_dir}")
    else:
        logging.error(f"\n❌ Error: {result['error']}")
        exit(1)
