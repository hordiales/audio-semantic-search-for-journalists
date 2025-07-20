#!/usr/bin/env python3
"""
Script para regenerar embeddings de audio usando YAMNet real
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import json
import shutil
from datetime import datetime
from tqdm import tqdm

from audio_embeddings import AudioEmbeddingGenerator, get_audio_embedding_generator
from vector_indexing import VectorIndexManager

class AudioEmbeddingRegeneration:
    """Regenera embeddings de audio usando YAMNet real"""
    
    def __init__(self, dataset_dir: str, use_real_yamnet: bool = True):
        """
        Inicializa el regenerador
        
        Args:
            dataset_dir: Directorio del dataset
            use_real_yamnet: Si usar YAMNet real
        """
        self.dataset_dir = Path(dataset_dir)
        self.use_real_yamnet = use_real_yamnet
        
        # Verificar que el dataset existe
        self.dataset_file = self.dataset_dir / "final" / "complete_dataset.pkl"
        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {self.dataset_file}")
        
        # Crear backup
        self.backup_dir = self.dataset_dir / "backup" / f"before_yamnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Dataset: {self.dataset_dir}")
        print(f"💾 Backup: {self.backup_dir}")
    
    def create_backup(self):
        """Crea backup del dataset actual"""
        print("💾 Creando backup del dataset actual...")
        
        files_to_backup = [
            "final/complete_dataset.pkl",
            "final/dataset_manifest.json",
            "embeddings/embeddings_data.pkl",
            "indices/"
        ]
        
        for file_path in files_to_backup:
            source = self.dataset_dir / file_path
            if source.exists():
                if source.is_dir():
                    shutil.copytree(source, self.backup_dir / file_path, dirs_exist_ok=True)
                else:
                    dest = self.backup_dir / file_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                print(f"  ✅ {file_path}")
        
        print(f"✅ Backup completado en: {self.backup_dir}")
    
    def load_current_dataset(self) -> pd.DataFrame:
        """Carga el dataset actual"""
        print("📊 Cargando dataset actual...")
        df = pd.read_pickle(self.dataset_file)
        print(f"✅ Dataset cargado: {len(df):,} segmentos")
        
        # Verificar que tiene embeddings de texto
        if 'text_embedding' not in df.columns:
            raise ValueError("Dataset no tiene embeddings de texto")
        
        # Verificar archivos de audio
        missing_files = []
        audio_files = df['file_path'].unique()
        
        for audio_file in audio_files[:10]:  # Verificar solo algunos
            if not Path(audio_file).exists():
                missing_files.append(audio_file)
        
        if missing_files:
            print(f"⚠️  Advertencia: {len(missing_files)} archivos de audio no encontrados")
            print("   Esto puede causar errores en la regeneración")
        
        return df
    
    def regenerate_audio_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regenera embeddings de audio"""
        print(f"🎵 Regenerando embeddings de audio...")
        print(f"   Usando: YAMNet real")
        
        # Inicializar generador de embeddings
        audio_embedder = get_audio_embedding_generator()
        
        if self.use_real_yamnet:
            print("🔄 Cargando modelo YAMNet... (puede tomar unos minutos)")
        
        # Procesar dataset
        try:
            result_df = audio_embedder.process_transcription_dataframe(df.copy())
            print(f"✅ Embeddings regenerados para {len(result_df)} segmentos")
            return result_df
            
        except Exception as e:
            print(f"❌ Error regenerando embeddings: {e}")
            print("💡 Verifica que:")
            print("   - TensorFlow está instalado correctamente")
            print("   - Los archivos de audio existen")
            print("   - Hay suficiente espacio en disco")
            raise
    
    def regenerate_indices(self, df: pd.DataFrame):
        """Regenera índices vectoriales"""
        print("🔍 Regenerando índices vectoriales...")
        
        indices_dir = self.dataset_dir / "indices"
        
        # Backup de índices actuales
        if indices_dir.exists():
            backup_indices = self.backup_dir / "indices"
            if not backup_indices.exists():
                shutil.copytree(indices_dir, backup_indices)
        
        # Crear nuevos índices
        embedding_dim = len(df['text_embedding'].iloc[0])
        index_manager = VectorIndexManager(embedding_dim=embedding_dim)
        
        # Crear índices
        text_success = index_manager.create_text_index(df)
        audio_success = index_manager.create_audio_index(df)
        
        if text_success or audio_success:
            index_manager.save_indices(str(indices_dir))
            
            # Guardar metadatos
            metadata = {
                "creation_date": datetime.now().isoformat(),
                "total_vectors": len(df),
                "embedding_dimension": embedding_dim,
                "text_index_created": text_success,
                "audio_index_created": audio_success,
                "audio_model": "YAMNet",
                "regeneration_info": {
                    "regenerated_from_backup": str(self.backup_dir),
                    "regeneration_date": datetime.now().isoformat()
                }
            }
            
            metadata_file = indices_dir / "indices_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Índices regenerados - Texto: {text_success}, Audio: {audio_success}")
        else:
            raise RuntimeError("No se pudo regenerar ningún índice")
    
    def update_manifest(self, df: pd.DataFrame):
        """Actualiza el manifiesto del dataset"""
        print("📋 Actualizando manifiesto...")
        
        manifest_file = self.dataset_dir / "final" / "dataset_manifest.json"
        
        # Cargar manifiesto actual
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        else:
            manifest = {}
        
        # Actualizar información
        if 'config' not in manifest:
            manifest['config'] = {}
        
        manifest['config']['audio_embedding_model'] = "YAMNet"
        
        # Agregar información de regeneración
        manifest['regeneration_info'] = {
            "regenerated_date": datetime.now().isoformat(),
            "backup_location": str(self.backup_dir),
            "audio_model_used": "YAMNet",
            "segments_processed": len(df)
        }
        
        # Guardar manifiesto actualizado
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        print("✅ Manifiesto actualizado")
    
    def save_updated_dataset(self, df: pd.DataFrame):
        """Guarda el dataset actualizado"""
        print("💾 Guardando dataset actualizado...")
        
        # Guardar dataset completo
        df.to_pickle(self.dataset_file)
        
        # Guardar también en embeddings
        embeddings_file = self.dataset_dir / "embeddings" / "embeddings_data.pkl"
        df.to_pickle(embeddings_file)
        
        # Guardar CSV sin embeddings
        csv_df = df.drop(columns=['text_embedding', 'audio_embedding'], errors='ignore')
        csv_file = self.dataset_dir / "final" / "dataset_metadata.csv"
        csv_df.to_csv(csv_file, index=False)
        
        print("✅ Dataset guardado")
    
    def run_regeneration(self):
        """Ejecuta la regeneración completa"""
        print("🚀 INICIANDO REGENERACIÓN DE EMBEDDINGS DE AUDIO")
        print("=" * 60)
        
        try:
            # Paso 1: Backup
            self.create_backup()
            
            # Paso 2: Cargar dataset
            df = self.load_current_dataset()
            
            # Paso 3: Regenerar embeddings de audio
            df_updated = self.regenerate_audio_embeddings(df)
            
            # Paso 4: Regenerar índices
            self.regenerate_indices(df_updated)
            
            # Paso 5: Actualizar manifiesto
            self.update_manifest(df_updated)
            
            # Paso 6: Guardar dataset
            self.save_updated_dataset(df_updated)
            
            print("\n✅ REGENERACIÓN COMPLETADA EXITOSAMENTE")
            print(f"📊 Segmentos procesados: {len(df_updated):,}")
            print(f"🎵 Modelo usado: YAMNet")
            print(f"💾 Backup en: {self.backup_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR EN REGENERACIÓN: {e}")
            print(f"💾 Los datos originales están seguros en: {self.backup_dir}")
            print("🔄 Puedes restaurar desde el backup si es necesario")
            return False

def main():
    parser = argparse.ArgumentParser(description="Regenerar embeddings de audio con YAMNet real")
    parser.add_argument("dataset_dir", help="Directorio del dataset")
    parser.add_argument("--use-real-yamnet", action="store_true", 
                       help="Usar YAMNet real (siempre activo)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Solo verificar requisitos sin regenerar")
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Solo verificar requisitos
        from check_yamnet_requirements import main as check_requirements
        success = check_requirements()
        if success:
            print("\n✅ Sistema listo para YAMNet real")
        else:
            print("\n❌ Sistema no cumple requisitos")
        sys.exit(0 if success else 1)
    
    # Verificar requisitos si se va a usar YAMNet real
    if args.use_real_yamnet:
        print("🔍 Verificando requisitos para YAMNet real...")
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            print(f"✅ TensorFlow {tf.__version__} disponible")
        except ImportError:
            print("❌ TensorFlow no disponible")
            print("💡 Instala con: pip install tensorflow tensorflow-hub")
            sys.exit(1)
    
    # Ejecutar regeneración
    regenerator = AudioEmbeddingRegeneration(
        args.dataset_dir, 
        use_real_yamnet=args.use_real_yamnet
    )
    
    success = regenerator.run_regeneration()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()