#!/usr/bin/env python3
"""
Script para agregar análisis de sentimientos a un dataset existente
Procesa datasets ya creados y añade columnas de sentiment analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import logging
import pickle

# Importar módulos del proyecto
from sentiment_analysis import SentimentAnalyzer
from config_loader import get_config

class DatasetSentimentProcessor:
    """Procesador para agregar sentimientos a datasets existentes"""
    
    def __init__(self, dataset_dir: str, batch_size: int = 32):
        """
        Inicializa el procesador
        
        Args:
            dataset_dir: Directorio del dataset
            batch_size: Tamaño de lote para procesamiento
        """
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        
        # Verificar que el dataset existe
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_dir}")
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.dataset_dir / "sentiment_processing.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Inicializar analizador de sentimientos
        try:
            config = get_config()
            self.sentiment_analyzer = SentimentAnalyzer(
                model_name=config.sentiment_model if hasattr(config, 'sentiment_model') else None
            )
        except Exception as e:
            self.logger.warning(f"Usando configuración por defecto para sentimientos: {e}")
            self.sentiment_analyzer = SentimentAnalyzer()
        
        self.logger.info(f"Procesador inicializado - Dataset: {dataset_dir}")
        self.logger.info(f"Batch size: {batch_size}")
    
    def find_dataset_files(self) -> dict:
        """Encuentra los archivos del dataset disponibles"""
        files = {}
        
        # Buscar archivo principal del dataset
        final_dir = self.dataset_dir / "final"
        if final_dir.exists():
            complete_dataset = final_dir / "complete_dataset.pkl"
            if complete_dataset.exists():
                files['complete_dataset'] = complete_dataset
            
            metadata_csv = final_dir / "dataset_metadata.csv"
            if metadata_csv.exists():
                files['metadata_csv'] = metadata_csv
        
        # Buscar archivos de transcripciones
        transcriptions_dir = self.dataset_dir / "transcriptions"
        if transcriptions_dir.exists():
            segments_file = transcriptions_dir / "segments_metadata.csv"
            if segments_file.exists():
                files['segments_metadata'] = segments_file
        
        # Buscar archivos de embeddings
        embeddings_dir = self.dataset_dir / "embeddings"
        if embeddings_dir.exists():
            embeddings_file = embeddings_dir / "embeddings_data.pkl"
            if embeddings_file.exists():
                files['embeddings_data'] = embeddings_file
        
        return files
    
    def load_dataset(self) -> pd.DataFrame:
        """Carga el dataset desde el mejor archivo disponible"""
        files = self.find_dataset_files()
        
        if not files:
            raise FileNotFoundError("No se encontraron archivos de dataset válidos")
        
        # Prioridad: complete_dataset > embeddings_data > segments_metadata > metadata_csv
        if 'complete_dataset' in files:
            self.logger.info("Cargando complete_dataset.pkl...")
            df = pd.read_pickle(files['complete_dataset'])
            source_file = files['complete_dataset']
            
        elif 'embeddings_data' in files:
            self.logger.info("Cargando embeddings_data.pkl...")
            df = pd.read_pickle(files['embeddings_data'])
            source_file = files['embeddings_data']
            
        elif 'segments_metadata' in files:
            self.logger.info("Cargando segments_metadata.csv...")
            df = pd.read_csv(files['segments_metadata'])
            source_file = files['segments_metadata']
            
        elif 'metadata_csv' in files:
            self.logger.info("Cargando dataset_metadata.csv...")
            df = pd.read_csv(files['metadata_csv'])
            source_file = files['metadata_csv']
        
        else:
            raise FileNotFoundError("No se encontró ningún archivo de dataset compatible")
        
        self.logger.info(f"Dataset cargado desde: {source_file}")
        self.logger.info(f"Dimensiones: {df.shape}")
        self.logger.info(f"Columnas: {list(df.columns)}")
        
        return df
    
    def check_existing_sentiment(self, df: pd.DataFrame) -> tuple[bool, list]:
        """Verifica si ya existen columnas de sentimientos"""
        sentiment_columns = [col for col in df.columns if 'sentiment' in col.lower()]
        has_sentiment = len(sentiment_columns) > 0
        
        if has_sentiment:
            self.logger.info(f"Columnas de sentimiento existentes: {sentiment_columns}")
        else:
            self.logger.info("No se encontraron columnas de sentimiento existentes")
        
        return has_sentiment, sentiment_columns
    
    def process_sentiments(self, df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
        """Procesa los sentimientos del dataset"""
        
        # Verificar columna de texto
        if 'text' not in df.columns:
            raise ValueError("El dataset debe tener una columna 'text'")
        
        # Verificar sentimientos existentes
        has_sentiment, existing_cols = self.check_existing_sentiment(df)
        
        if has_sentiment and not overwrite:
            self.logger.warning("El dataset ya tiene análisis de sentimientos")
            self.logger.info("Usa --overwrite para reemplazar el análisis existente")
            return df
        
        if has_sentiment and overwrite:
            self.logger.info("Reemplazando análisis de sentimientos existente...")
            # Remover columnas existentes
            df = df.drop(columns=existing_cols)
        
        # Procesar en lotes
        self.logger.info(f"Procesando {len(df)} segmentos...")
        
        processed_df = self.sentiment_analyzer.process_dataframe(
            df, 
            text_column='text'
        )
        
        self.logger.info("✅ Análisis de sentimientos completado")
        
        # Mostrar estadísticas
        self._show_sentiment_stats(processed_df)
        
        return processed_df
    
    def _show_sentiment_stats(self, df: pd.DataFrame):
        """Muestra estadísticas del análisis de sentimientos"""
        if 'dominant_sentiment' in df.columns:
            sentiment_counts = df['dominant_sentiment'].value_counts()
            total = len(df)
            
            self.logger.info("📊 Distribución de sentimientos:")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total) * 100
                emoji = {"POSITIVE": "😊", "NEGATIVE": "😢", "NEUTRAL": "😐"}.get(sentiment, "❓")
                self.logger.info(f"   {emoji} {sentiment}: {count:,} ({percentage:.1f}%)")
        
        if 'sentiment_score' in df.columns:
            avg_score = df['sentiment_score'].mean()
            self.logger.info(f"📈 Score promedio de sentimiento: {avg_score:.3f}")
    
    def save_dataset(self, df: pd.DataFrame, backup: bool = True):
        """Guarda el dataset procesado"""
        
        # Crear backup si se solicita
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.dataset_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Backup del archivo principal
            final_dir = self.dataset_dir / "final"
            complete_dataset = final_dir / "complete_dataset.pkl"
            
            if complete_dataset.exists():
                backup_file = backup_dir / f"complete_dataset_backup_{timestamp}.pkl"
                import shutil
                shutil.copy2(complete_dataset, backup_file)
                self.logger.info(f"Backup creado: {backup_file}")
        
        # Guardar dataset actualizado
        final_dir = self.dataset_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        output_file = final_dir / "complete_dataset.pkl"
        df.to_pickle(output_file)
        self.logger.info(f"Dataset guardado: {output_file}")
        
        # Actualizar manifest
        self._update_manifest(df)
        
        # Guardar también como CSV para fácil inspección
        csv_file = final_dir / "dataset_with_sentiment.csv"
        # Guardar solo una muestra para CSV (el pickle completo es muy grande)
        sample_df = df.head(1000) if len(df) > 1000 else df
        
        # Remover embeddings del CSV para que sea más ligero
        csv_columns = [col for col in sample_df.columns if 'embedding' not in col.lower()]
        sample_df[csv_columns].to_csv(csv_file, index=False)
        self.logger.info(f"Muestra CSV guardada: {csv_file}")
    
    def _update_manifest(self, df: pd.DataFrame):
        """Actualiza el manifest del dataset"""
        manifest_file = self.dataset_dir / "final" / "dataset_manifest.json"
        
        # Cargar manifest existente o crear uno nuevo
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        else:
            manifest = {}
        
        # Actualizar información
        manifest.update({
            'sentiment_analysis': {
                'processed': True,
                'processed_date': datetime.now().isoformat(),
                'analyzer_version': '1.0',
                'total_segments': len(df)
            },
            'columns': list(df.columns),
            'last_updated': datetime.now().isoformat()
        })
        
        # Guardar manifest actualizado
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Manifest actualizado: {manifest_file}")
    
    def run(self, overwrite: bool = False, backup: bool = True) -> bool:
        """Ejecuta el procesamiento completo"""
        try:
            # Cargar dataset
            df = self.load_dataset()
            
            # Procesar sentimientos
            processed_df = self.process_sentiments(df, overwrite=overwrite)
            
            # Guardar resultado
            self.save_dataset(processed_df, backup=backup)
            
            self.logger.info("🎉 Procesamiento completado exitosamente")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error durante el procesamiento: {e}")
            return False


def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(
        description="Agregar análisis de sentimientos a dataset existente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Procesar dataset existente
python add_sentiment_to_dataset.py ./dataset

# Procesar con análisis de sentimientos
python add_sentiment_to_dataset.py ./dataset

# Reemplazar análisis existente
python add_sentiment_to_dataset.py ./dataset --overwrite

# Procesar sin crear backup
python add_sentiment_to_dataset.py ./dataset --no-backup

# Procesar con batch size específico
python add_sentiment_to_dataset.py ./dataset --batch-size 64
        """
    )
    
    parser.add_argument(
        "dataset_dir",
        help="Directorio del dataset a procesar"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de lote para procesamiento (default: 32)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir análisis de sentimientos existente"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="No crear backup antes de procesar"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar información detallada"
    )
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🎭 Procesador de Sentimientos para Dataset")
    print("=" * 50)
    print(f"📁 Dataset: {args.dataset_dir}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🔄 Overwrite: {args.overwrite}")
    print(f"💾 Backup: {not args.no_backup}")
    print()
    
    try:
        # Crear procesador
        processor = DatasetSentimentProcessor(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
        )
        
        # Ejecutar procesamiento
        success = processor.run(
            overwrite=args.overwrite,
            backup=not args.no_backup
        )
        
        if success:
            print("\n✅ ¡Análisis de sentimientos agregado exitosamente!")
            print(f"📁 Dataset actualizado en: {args.dataset_dir}/final/complete_dataset.pkl")
            print("\n🚀 Ahora puedes usar:")
            print(f"   python query_client.py {args.dataset_dir} --interactive --load-real")
        else:
            print("\n❌ El procesamiento falló. Revisa los logs para más detalles.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Procesamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()