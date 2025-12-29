#!/usr/bin/env python3
"""
Script para reanudar el pipeline desde el paso 5
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys

import pandas as pd

from dataset_orchestrator import DatasetConfig, DatasetOrchestrator


def resume_from_step5(dataset_dir: str):
    """Reanuda el pipeline desde el paso 5"""

    dataset_path = Path(dataset_dir)

    # Verificar que los archivos necesarios existen
    embeddings_file = dataset_path / "embeddings" / "embeddings_data.pkl"
    indices_dir = dataset_path / "indices"

    if not embeddings_file.exists():
        logging.error(f"❌ Error: No se encontró {embeddings_file}")
        return False

    if not indices_dir.exists():
        logging.error(f"❌ Error: No se encontró {indices_dir}")
        return False

    logging.info(f"✅ Encontrados archivos necesarios en {dataset_dir}")

    # Cargar DataFrame con embeddings
    logging.info("📊 Cargando DataFrame con embeddings...")
    df_with_embeddings = pd.read_pickle(embeddings_file)
    logging.info(f"✅ Cargado DataFrame con {len(df_with_embeddings)} segmentos")

    # Cargar metadatos de índices
    indices_metadata_file = indices_dir / "indices_metadata.json"
    if indices_metadata_file.exists():
        with open(indices_metadata_file, encoding='utf-8') as f:
            indices_info = json.load(f)
        logging.info("✅ Metadatos de índices cargados")
    else:
        # Crear metadatos básicos
        indices_info = {
            "indices_dir": str(indices_dir),
            "text_index": str(indices_dir / "text_index.faiss"),
            "audio_index": str(indices_dir / "audio_index.faiss"),
            "metadata": str(indices_metadata_file)
        }
        logging.warning("⚠️  Usando metadatos de índices básicos")

    # Crear configuración básica para el orquestador
    config = DatasetConfig(
        input_dir="./data",  # No importa para este paso
        output_dir=str(dataset_path)
    )

    # Crear orquestador
    orchestrator = DatasetOrchestrator(config)

    # Simular estadísticas
    orchestrator.stats.total_files = len(df_with_embeddings['source_file'].unique())
    orchestrator.stats.total_segments = len(df_with_embeddings)
    orchestrator.stats.converted_files = orchestrator.stats.total_files
    orchestrator.stats.transcribed_files = orchestrator.stats.total_files
    orchestrator.stats.embedded_files = orchestrator.stats.total_files
    orchestrator.stats.start_time = datetime.now()

    try:
        logging.info("🚀 Ejecutando paso 5: Creación de dataset final...")
        final_dataset = orchestrator.step5_create_final_dataset(df_with_embeddings, indices_info)

        logging.info("✅ ¡Dataset final creado exitosamente!")
        logging.info(f"📁 Dataset completo en: {final_dataset['dataset_dir']}")
        logging.info(f"📋 Manifiesto: {final_dataset['manifest']}")

        return True

    except Exception as e:
        logging.error(f"❌ Error en paso 5: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Uso: python resume_pipeline.py <directorio_dataset>")
        logging.error("Ejemplo: python resume_pipeline.py ./dataset")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    success = resume_from_step5(dataset_dir)

    if success:
        logging.info("\n🎉 Pipeline reanudado exitosamente!")
        sys.exit(0)
    else:
        logging.error("\n❌ Error al reanudar pipeline")
        sys.exit(1)
