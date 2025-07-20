#!/usr/bin/env python3
"""
Script para verificar el dataset generado
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import argparse

def verify_dataset(dataset_dir: str):
    """Verifica la integridad del dataset"""
    
    print("🔍 Verificando Dataset")
    print("=" * 50)
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Error: Directorio {dataset_dir} no existe")
        return False
    
    # Verificar estructura de directorios
    print("\n📁 Estructura de directorios:")
    expected_dirs = ["converted", "transcriptions", "embeddings", "indices", "final"]
    all_dirs_exist = True
    
    for dir_name in expected_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.iterdir()))
            print(f"  ✅ {dir_name:>15}: {file_count} archivos")
        else:
            print(f"  ❌ {dir_name:>15}: NO EXISTE")
            all_dirs_exist = False
    
    if not all_dirs_exist:
        print("\n❌ Estructura de directorios incompleta")
        return False
    
    # Verificar dataset final
    print("\n📊 Dataset Final:")
    final_dir = dataset_path / "final"
    
    # Verificar archivos principales
    files_to_check = [
        ("complete_dataset.pkl", "Dataset completo"),
        ("dataset_metadata.csv", "Metadatos CSV"),
        ("dataset_manifest.json", "Manifiesto")
    ]
    
    for filename, description in files_to_check:
        file_path = final_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size / (1024**2)  # MB
            print(f"  ✅ {description:>20}: {size:.1f} MB")
        else:
            print(f"  ❌ {description:>20}: NO EXISTE")
            return False
    
    # Cargar y verificar dataset
    print("\n🧠 Contenido del Dataset:")
    try:
        # Cargar dataset completo
        df = pd.read_pickle(final_dir / "complete_dataset.pkl")
        print(f"  📄 Total segmentos: {len(df):,}")
        print(f"  📁 Archivos únicos: {df['source_file'].nunique():,}")
        print(f"  ⏱️  Duración total: {df['duration'].sum():.1f} segundos ({df['duration'].sum()/3600:.1f} horas)")
        
        # Verificar columnas requeridas
        required_cols = ['text', 'start_time', 'end_time', 'duration', 'source_file']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ❌ Columnas faltantes: {missing_cols}")
            return False
        else:
            print(f"  ✅ Todas las columnas requeridas presentes")
        
        # Verificar embeddings
        if 'text_embedding' in df.columns and 'audio_embedding' in df.columns:
            text_emb_shape = df['text_embedding'].iloc[0].shape
            audio_emb_shape = df['audio_embedding'].iloc[0].shape
            print(f"  🧠 Embeddings de texto: {text_emb_shape}")
            print(f"  🔊 Embeddings de audio: {audio_emb_shape}")
            
            # Verificar que no hay valores nulos
            text_nulls = df['text_embedding'].isnull().sum()
            audio_nulls = df['audio_embedding'].isnull().sum()
            print(f"  📊 Embeddings de texto nulos: {text_nulls}")
            print(f"  📊 Embeddings de audio nulos: {audio_nulls}")
        else:
            print(f"  ❌ Columnas de embeddings no encontradas")
            return False
        
        # Estadísticas de texto
        print(f"  📝 Texto promedio: {df['text'].str.len().mean():.1f} caracteres")
        print(f"  📝 Segmento promedio: {df['duration'].mean():.1f} segundos")
        
        # Verificar que no hay textos vacíos
        empty_texts = df['text'].str.strip().eq('').sum()
        print(f"  📝 Textos vacíos: {empty_texts}")
        
    except Exception as e:
        print(f"  ❌ Error cargando dataset: {e}")
        return False
    
    # Verificar índices vectoriales
    print("\n🔍 Índices Vectoriales:")
    indices_dir = dataset_path / "indices"
    
    index_files = [
        ("text_index.faiss", "Índice de texto"),
        ("audio_index.faiss", "Índice de audio"),
        ("text_metadata.pkl", "Metadatos de texto"),
        ("audio_metadata.pkl", "Metadatos de audio")
    ]
    
    for filename, description in index_files:
        file_path = indices_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size / (1024**2)  # MB
            print(f"  ✅ {description:>20}: {size:.1f} MB")
        else:
            print(f"  ⚠️  {description:>20}: NO EXISTE")
    
    # Verificar manifiesto
    print("\n📋 Manifiesto:")
    try:
        with open(final_dir / "dataset_manifest.json", 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        print(f"  📅 Fecha creación: {manifest['dataset_info']['creation_date']}")
        print(f"  📊 Total archivos: {manifest['statistics']['converted_files']}")
        print(f"  📊 Total segmentos: {manifest['statistics']['total_segments']}")
        print(f"  ⏱️  Tiempo procesamiento: {manifest['statistics']['processing_time']:.1f} segundos")
        
        if 'config' in manifest:
            config = manifest['config']
            print(f"  🎤 Modelo Whisper: {config.get('whisper_model', 'N/A')}")
            print(f"  🧠 Modelo texto: {config.get('text_model', 'N/A')}")
            print(f"  🔊 Audio mock: {config.get('use_mock_audio', 'N/A')}")
        
    except Exception as e:
        print(f"  ❌ Error leyendo manifiesto: {e}")
        return False
    
    print("\n✅ Dataset verificado exitosamente!")
    return True

def show_sample_data(dataset_dir: str, n_samples: int = 5):
    """Muestra datos de muestra del dataset"""
    
    print(f"\n📋 Muestra de {n_samples} segmentos:")
    print("-" * 80)
    
    dataset_path = Path(dataset_dir)
    df = pd.read_pickle(dataset_path / "final" / "complete_dataset.pkl")
    
    # Tomar muestra aleatoria
    sample_df = df.sample(n=min(n_samples, len(df)))
    
    for idx, row in sample_df.iterrows():
        print(f"\n📄 Segmento {idx}:")
        print(f"  📁 Archivo: {row['source_file']}")
        print(f"  ⏱️  Tiempo: {row['start_time']:.1f}s - {row['end_time']:.1f}s ({row['duration']:.1f}s)")
        print(f"  📝 Texto: {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}")
        if 'confidence' in row:
            print(f"  🎯 Confianza: {row['confidence']:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Verificar dataset de audio procesado")
    parser.add_argument("dataset_dir", help="Directorio del dataset a verificar")
    parser.add_argument("--show-samples", type=int, default=0, help="Mostrar N muestras del dataset")
    parser.add_argument("--quick", action="store_true", help="Verificación rápida (solo estructura)")
    
    args = parser.parse_args()
    
    # Verificar dataset
    success = verify_dataset(args.dataset_dir)
    
    if success and args.show_samples > 0:
        show_sample_data(args.dataset_dir, args.show_samples)
    
    if success:
        print(f"\n🎉 Dataset en {args.dataset_dir} está listo para usar!")
        sys.exit(0)
    else:
        print(f"\n❌ Problemas encontrados en {args.dataset_dir}")
        sys.exit(1)

if __name__ == "__main__":
    main()