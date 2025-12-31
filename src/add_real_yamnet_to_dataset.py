#!/usr/bin/env python3
"""
Script para agregar embeddings de YAMNet REAL al dataset existente
También detecta eventos de audio como risas, música, aplausos
"""

import argparse
import builtins
import contextlib
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealYAMNetProcessor:
    """Procesador de audio con YAMNet real para detección de eventos"""

    def __init__(self):
        """Inicializa el procesador YAMNet"""
        self.model = None
        self.class_names = None
        self._load_yamnet()
        self._load_audioset_classes()

    def _load_yamnet(self):
        """Carga el modelo YAMNet real"""
        try:
            import pandas as pd
            import tensorflow as tf
            import tensorflow_hub as hub

            logger.info("🔄 Cargando modelo YAMNet desde TensorFlow Hub...")
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')

            # Cargar los nombres de las clases desde el archivo CSV
            class_map_path = self.model.class_map_path().numpy().decode('utf-8')
            self.class_names = pd.read_csv(class_map_path)['display_name'].tolist()

            logger.info("✅ Modelo YAMNet cargado exitosamente")

        except ImportError:
            logger.error("❌ TensorFlow no está disponible")
            raise RuntimeError("TensorFlow requerido para YAMNet real")
        except Exception as e:
            logger.error(f"❌ Error cargando YAMNet: {e}")
            raise RuntimeError(f"No se pudo cargar YAMNet: {e}")

    def _load_audioset_classes(self):
        """Carga las clases de AudioSet"""
        try:
            # Importar las clases de AudioSet desde nuestro módulo
            from audioset_ontology import AUDIOSET_CLASSES
            self.audioset_classes = AUDIOSET_CLASSES
            logger.info(f"✅ Cargadas {len(AUDIOSET_CLASSES)} clases de AudioSet")

        except ImportError:
            logger.warning("⚠️  No se pudo cargar audioset_ontology, usando detección básica")
            self.audioset_classes = {}

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocesa audio para YAMNet (16kHz, mono)"""
        try:
            # Cargar audio a 16kHz mono
            audio, _sr = librosa.load(audio_path, sr=16000, mono=True)

            # Normalizar
            audio = librosa.util.normalize(audio)

            # Convertir a float32
            audio = audio.astype(np.float32)

            return audio

        except Exception as e:
            logger.error(f"❌ Error preprocesando {audio_path}: {e}")
            raise

    def extract_audio_features(self, audio_path: str) -> dict:
        """
        Extrae embeddings y detecta eventos de audio usando YAMNet

        Returns:
            Dict con embedding, scores, y eventos detectados
        """
        try:
            # Preprocesar audio
            audio = self.preprocess_audio(audio_path)

            # Procesar con YAMNet
            scores, embeddings, _spectrogram = self.model(audio)

            # Convertir a numpy
            scores_np = scores.numpy()
            embeddings_np = embeddings.numpy()

            # Promedio de embeddings a lo largo del tiempo
            avg_embedding = np.mean(embeddings_np, axis=0)

            # Promedio de scores para clasificación
            avg_scores = np.mean(scores_np, axis=0)

            # Detectar eventos de audio específicos
            audio_events = self._detect_audio_events(avg_scores)

            # Calcular confianza general
            max_confidence = float(np.max(avg_scores))

            return {
                'embedding': avg_embedding.tolist(),
                'embedding_dim': len(avg_embedding),
                'scores': avg_scores.tolist(),
                'max_confidence': max_confidence,
                'audio_events': audio_events,
                'model': 'YAMNet_real',
                'processing_success': True
            }

        except Exception as e:
            logger.error(f"❌ Error procesando {audio_path}: {e}")
            return {
                'embedding': None,
                'embedding_dim': 1024,
                'scores': None,
                'max_confidence': 0.0,
                'audio_events': {},
                'model': 'YAMNet_real',
                'processing_success': False,
                'error': str(e)
            }

    def _detect_audio_events(self, scores: np.ndarray) -> dict:
        """
        Detecta eventos específicos de audio basado en scores de YAMNet

        Args:
            scores: Array de 521 scores de AudioSet

        Returns:
            Dict con eventos detectados y sus confianzas
        """
        events = {}

        # Direct mapping from desired event names to YAMNet class names
        # These should be exact matches or very close to YAMNet's actual class_names
        yamnet_event_mapping = {
            'laughter': ['Laughter', 'Giggle', 'Chuckle'],
            'applause': ['Applause', 'Clapping'],
            'music': ['Music'],
            'crowd': ['Crowd', 'Human crowd'],
            'cheering': ['Cheering', 'Yell', 'Shout'],
            'speech': ['Speech', 'Human speech', 'Spoken words'],
            'silence': ['Silence', 'Quiet'],
            'noise': ['Noise', 'Ambient noise', 'Background noise']
        }

        threshold = 0.1  # Umbral mínimo de confianza para detectar un evento

        for event_name, yamnet_classes in yamnet_event_mapping.items():
            max_event_score = 0.0

            for yamnet_class_name in yamnet_classes:
                try:
                    # Find the exact index of the YAMNet class name
                    class_index = np.where(self.class_names == yamnet_class_name)[0][0]
                    if class_index < len(scores):
                        max_event_score = max(max_event_score, scores[class_index])
                except IndexError:
                    # YAMNet class name not found in the loaded model's class_names
                    logger.debug(f"YAMNet class '{yamnet_class_name}' not found in model's class names.")
                    continue

            if max_event_score > threshold:
                events[event_name] = {
                    'confidence': float(max_event_score),
                    'detected': True
                }
            else:
                events[event_name] = {
                    'confidence': float(max_event_score),
                    'detected': False
                }

        return events


class DatasetAudioProcessor:
    """Procesador principal para agregar análisis de audio real al dataset"""

    def __init__(self, dataset_dir: str, batch_size: int = 16):
        """
        Args:
            dataset_dir: Directorio del dataset
            batch_size: Tamaño de lote para procesamiento
        """
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.yamnet_processor = RealYAMNetProcessor()

        # Verificar estructura del dataset
        self._verify_dataset_structure()

    def _verify_dataset_structure(self):
        """Verifica que el dataset tenga la estructura correcta"""
        dataset_file = self.dataset_dir / "final" / "complete_dataset.pkl"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_file}")

        manifest_file = self.dataset_dir / "final" / "dataset_manifest.json"
        if not manifest_file.exists():
            logger.warning("⚠️  Archivo de manifiesto no encontrado")

    def process_dataset(self, overwrite: bool = False, backup: bool = True) -> bool:
        """
        Procesa el dataset completo agregando análisis de audio real

        Args:
            overwrite: Si sobreescribir embeddings existentes
            backup: Si crear backup antes de modificar

        Returns:
            True si el procesamiento fue exitoso
        """
        try:
            # Cargar dataset
            dataset_file = self.dataset_dir / "final" / "complete_dataset.pkl"
            logger.info(f"📊 Cargando dataset desde: {dataset_file}")
            df = pd.read_pickle(dataset_file)

            logger.info(f"📈 Dataset cargado: {len(df)} segmentos")
            logger.info(f"📋 Columnas: {list(df.columns)}")

            # Crear backup si se solicita
            if backup:
                self._create_backup(df)

            # Verificar qué segmentos necesitan procesamiento
            segments_to_process = self._identify_segments_to_process(df, overwrite)

            if not segments_to_process:
                logger.info("✅ Todos los segmentos ya tienen análisis de audio real")
                return True

            logger.info(f"🔄 Procesando {len(segments_to_process)} segmentos...")

            # Procesar en lotes
            processed_count = 0
            failed_count = 0

            for i in range(0, len(segments_to_process), self.batch_size):
                batch_indices = segments_to_process[i:i + self.batch_size]
                batch_results = self._process_batch(df, batch_indices)

                # Actualizar DataFrame
                for idx, result in zip(batch_indices, batch_results, strict=False):
                    if result['processing_success']:
                        df.at[idx, 'audio_embedding'] = result['embedding']
                        df.at[idx, 'audio_embedding_model'] = result['model']
                        df.at[idx, 'audio_embedding_dim'] = result['embedding_dim']

                        # Agregar información de eventos de audio
                        df.at[idx, 'audio_events'] = json.dumps(result['audio_events'])
                        df.at[idx, 'audio_max_confidence'] = result['max_confidence']

                        processed_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"⚠️  Falló procesamiento del segmento {idx}: {result.get('error', 'Error desconocido')}")

                # Progreso
                current = min(i + self.batch_size, len(segments_to_process))
                logger.info(f"📊 Progreso: {current}/{len(segments_to_process)} segmentos procesados")

            # Guardar dataset actualizado
            self._save_updated_dataset(df)

            # Actualizar manifiesto
            self._update_manifest(processed_count, failed_count)

            logger.info("✅ Procesamiento completado:")
            logger.info(f"  ✅ Procesados exitosamente: {processed_count}")
            logger.info(f"  ❌ Fallidos: {failed_count}")

            return True

        except Exception as e:
            logger.error(f"❌ Error en procesamiento del dataset: {e}")
            return False

    def _identify_segments_to_process(self, df: pd.DataFrame, overwrite: bool) -> list[int]:
        """Identifica qué segmentos necesitan procesamiento"""
        if overwrite:
            # Procesar todos los segmentos que tengan archivo de audio
            return df[df['source_file'].notna()].index.tolist()

        # Solo procesar segmentos sin embeddings reales
        needs_processing = []

        for idx, row in df.iterrows():
            # Verificar si tiene archivo de audio
            if pd.isna(row.get('source_file')):
                continue

            # Verificar si ya tiene embedding real
            model = row.get('audio_embedding_model', '')
            if model != 'YAMNet_real':
                needs_processing.append(idx)

        return needs_processing

    def _process_batch(self, df: pd.DataFrame, batch_indices: list[int]) -> list[dict]:
        """Procesa un lote de segmentos"""
        results = []

        for idx in batch_indices:
            row = df.iloc[idx]

            try:
                # Obtener ruta del archivo de audio
                audio_file = row['source_file']

                # Verificar que el archivo existe
                if not os.path.exists(audio_file):
                    logger.warning(f"⚠️  Archivo no encontrado: {audio_file}")
                    results.append({
                        'processing_success': False,
                        'error': 'Archivo no encontrado'
                    })
                    continue

                # Procesar segmento específico si es necesario
                segment_audio_path = self._extract_segment_audio(row)

                # Extraer características con YAMNet
                result = self.yamnet_processor.extract_audio_features(segment_audio_path)
                results.append(result)

                # Limpiar archivo temporal si se creó
                if segment_audio_path != audio_file:
                    with contextlib.suppress(builtins.BaseException):
                        os.remove(segment_audio_path)

            except Exception as e:
                logger.error(f"❌ Error procesando segmento {idx}: {e}")
                results.append({
                    'processing_success': False,
                    'error': str(e)
                })

        return results

    def _extract_segment_audio(self, row: pd.DataFrame) -> str:
        """
        Extrae el segmento de audio específico si es necesario

        Args:
            row: Fila del DataFrame con información del segmento

        Returns:
            Ruta al archivo de audio (original o segmento extraído)
        """
        audio_file = row['source_file']
        start_time = row.get('start_time', 0)
        end_time = row.get('end_time', None)

        # Si no hay tiempos específicos, usar archivo completo
        if end_time is None or start_time == 0:
            return audio_file

        try:
            # Crear archivo temporal para el segmento
            temp_dir = self.dataset_dir / "temp_audio"
            temp_dir.mkdir(exist_ok=True)

            segment_id = row.get('segment_id', f"seg_{start_time}_{end_time}")
            temp_file = temp_dir / f"{segment_id}_temp.wav"

            # Cargar y extraer segmento
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr) if end_time else len(audio)

            segment_audio = audio[start_sample:end_sample]

            # Guardar segmento temporal
            sf.write(temp_file, segment_audio, sr)

            return str(temp_file)

        except Exception as e:
            logger.warning(f"⚠️  No se pudo extraer segmento, usando archivo completo: {e}")
            return audio_file

    def _create_backup(self, df: pd.DataFrame):
        """Crea backup del dataset antes de modificar"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.dataset_dir / "backups"
        backup_dir.mkdir(exist_ok=True)

        backup_file = backup_dir / f"complete_dataset_backup_{timestamp}.pkl"
        df.to_pickle(backup_file)

        logger.info(f"💾 Backup creado: {backup_file}")

    def _save_updated_dataset(self, df: pd.DataFrame):
        """Guarda el dataset actualizado"""
        dataset_file = self.dataset_dir / "final" / "complete_dataset.pkl"
        df.to_pickle(dataset_file)

        # También guardar como CSV para inspección
        csv_file = self.dataset_dir / "final" / "dataset_with_real_audio.csv"
        df.to_csv(csv_file, index=False)

        logger.info(f"💾 Dataset actualizado guardado: {dataset_file}")

    def _update_manifest(self, processed_count: int, failed_count: int):
        """Actualiza el manifiesto del dataset"""
        manifest_file = self.dataset_dir / "final" / "dataset_manifest.json"

        try:
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)
            else:
                manifest = {}

            # Actualizar información de procesamiento de audio
            manifest['real_audio_processing'] = {
                'processed': True,
                'processed_date': datetime.now().isoformat(),
                'processor_version': '1.0',
                'model': 'YAMNet_real',
                'successful_segments': processed_count,
                'failed_segments': failed_count,
                'embedding_dimension': 1024
            }

            manifest['last_updated'] = datetime.now().isoformat()

            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

            logger.info(f"📄 Manifiesto actualizado: {manifest_file}")

        except Exception as e:
            logger.warning(f"⚠️  No se pudo actualizar manifiesto: {e}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Agregar análisis de audio real (YAMNet) al dataset")
    parser.add_argument("dataset_dir", help="Directorio del dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamaño de lote para procesamiento")
    parser.add_argument("--overwrite", action="store_true", help="Sobreescribir embeddings existentes")
    parser.add_argument("--no-backup", action="store_true", help="No crear backup")

    args = parser.parse_args()

    logger.info("🎵 Procesador de Audio Real (YAMNet) para Dataset")
    logger.info("=" * 50)
    logger.info(f"📁 Dataset: {args.dataset_dir}")
    logger.info(f"📊 Batch size: {args.batch_size}")
    logger.info(f"🔄 Overwrite: {args.overwrite}")
    logger.info(f"💾 Backup: {not args.no_backup}")
    logger.info("")

    try:
        # Crear procesador
        processor = DatasetAudioProcessor(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size
        )

        # Procesar dataset
        success = processor.process_dataset(
            overwrite=args.overwrite,
            backup=not args.no_backup
        )

        if success:
            logger.info("✅ ¡Análisis de audio real agregado exitosamente!")
            logger.info(f"📁 Dataset actualizado en: {args.dataset_dir}/final/complete_dataset.pkl")
            logger.info("")
            logger.info("🚀 Ahora puedes usar:")
            logger.info("   python src/query_client.py ./dataset --interactive --load-real")
            return 0
        logger.error("❌ Error en el procesamiento del dataset")
        return 1

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Procesamiento interrumpido por el usuario")
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
