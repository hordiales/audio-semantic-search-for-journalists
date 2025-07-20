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
    
    import os
    if os.environ.get('MCP_MODE') != '1':
        print("🔍 Verificando Dataset", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        if os.environ.get('MCP_MODE') != '1':
            print(f"❌ Error: Directorio {dataset_dir} no existe", file=sys.stderr)
        return False
    
    # Verificar estructura de directorios
    if os.environ.get('MCP_MODE') != '1':
        print("\n📁 Estructura de directorios:", file=sys.stderr)
    expected_dirs = ["converted", "transcriptions", "embeddings", "indices", "final"]
    all_dirs_exist = True
    
    for dir_name in expected_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.iterdir()))
            if os.environ.get('MCP_MODE') != '1':
                print(f"  ✅ {dir_name:>15}: {file_count} archivos", file=sys.stderr)
        else:
            if os.environ.get('MCP_MODE') != '1':
                print(f"  ❌ {dir_name:>15}: NO EXISTE", file=sys.stderr)
            all_dirs_exist = False
    
    if not all_dirs_exist:
        if os.environ.get('MCP_MODE') != '1':
            print("\n❌ Estructura de directorios incompleta", file=sys.stderr)
        return False
    
    # Verificar dataset final
    if os.environ.get('MCP_MODE') != '1':
        print("\n📊 Dataset Final:", file=sys.stderr)
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
            if os.environ.get('MCP_MODE') != '1':
                print(f"  ✅ {description:>20}: {size:.1f} MB", file=sys.stderr)
        else:
            if os.environ.get('MCP_MODE') != '1':
                print(f"  ❌ {description:>20}: NO EXISTE", file=sys.stderr)
            return False
    
    # Cargar y verificar dataset
    if os.environ.get('MCP_MODE') != '1':
        print("\n🧠 Contenido del Dataset:", file=sys.stderr)
    try:
        # Cargar dataset completo
        df = pd.read_pickle(final_dir / "complete_dataset.pkl")
        if os.environ.get('MCP_MODE') != '1':
            print(f"  📄 Total segmentos: {len(df):,}", file=sys.stderr)
            print(f"  📁 Archivos únicos: {df['source_file'].nunique():,}", file=sys.stderr)
            print(f"  ⏱️  Duración total: {df['duration'].sum():.1f} segundos ({df['duration'].sum()/3600:.1f} horas)", file=sys.stderr)
        
        # Verificar columnas requeridas
        required_cols = ['text', 'start_time', 'end_time', 'duration', 'source_file']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            if os.environ.get('MCP_MODE') != '1':
                print(f"  ❌ Columnas faltantes: {missing_cols}", file=sys.stderr)
            return False
        else:
            if os.environ.get('MCP_MODE') != '1':
                print(f"  ✅ Todas las columnas requeridas presentes", file=sys.stderr)
        
        # Verificar embeddings
        if 'text_embedding' in df.columns and 'audio_embedding' in df.columns:
            text_emb = df['text_embedding'].iloc[0]
            audio_emb = df['audio_embedding'].iloc[0]
            
            # Obtener dimensiones sin importar si es lista o numpy array
            text_emb_dim = len(text_emb) if isinstance(text_emb, list) else text_emb.shape
            audio_emb_dim = len(audio_emb) if isinstance(audio_emb, list) else audio_emb.shape
            
            if os.environ.get('MCP_MODE') != '1':
                print(f"  🧠 Embeddings de texto: {text_emb_dim} dimensiones", file=sys.stderr)
                print(f"  🔊 Embeddings de audio: {audio_emb_dim} dimensiones", file=sys.stderr)
            
            # Verificar que no hay valores nulos
            text_nulls = df['text_embedding'].isnull().sum()
            audio_nulls = df['audio_embedding'].isnull().sum()
            if os.environ.get('MCP_MODE') != '1':
                print(f"  📊 Embeddings de texto nulos: {text_nulls}", file=sys.stderr)
                print(f"  📊 Embeddings de audio nulos: {audio_nulls}", file=sys.stderr)
        else:
            if os.environ.get('MCP_MODE') != '1':
                print(f"  ❌ Columnas de embeddings no encontradas", file=sys.stderr)
            return False
        
        # Estadísticas de texto
        if os.environ.get('MCP_MODE') != '1':
            print(f"  📝 Texto promedio: {df['text'].str.len().mean():.1f} caracteres", file=sys.stderr)
            print(f"  📝 Segmento promedio: {df['duration'].mean():.1f} segundos", file=sys.stderr)
        
        # Verificar que no hay textos vacíos
        empty_texts = df['text'].str.strip().eq('').sum()
        if os.environ.get('MCP_MODE') != '1':
            print(f"  📝 Textos vacíos: {empty_texts}", file=sys.stderr)
        
    except Exception as e:
        if os.environ.get('MCP_MODE') != '1':
            print(f"  ❌ Error cargando dataset: {e}", file=sys.stderr)
        return False
    
    # Verificar índices vectoriales
    if os.environ.get('MCP_MODE') != '1':
        print("\n🔍 Índices Vectoriales:", file=sys.stderr)
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
            if os.environ.get('MCP_MODE') != '1':
                print(f"  ✅ {description:>20}: {size:.1f} MB", file=sys.stderr)
        else:
            if os.environ.get('MCP_MODE') != '1':
                print(f"  ⚠️  {description:>20}: NO EXISTE", file=sys.stderr)
    
    # Verificar manifiesto
    if os.environ.get('MCP_MODE') != '1':
        print("\n📋 Manifiesto:", file=sys.stderr)
    try:
        with open(final_dir / "dataset_manifest.json", 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        if os.environ.get('MCP_MODE') != '1':
            print(f"  📅 Fecha creación: {manifest['dataset_info']['creation_date']}", file=sys.stderr)
            print(f"  📊 Total archivos: {manifest['statistics']['converted_files']}", file=sys.stderr)
            print(f"  📊 Total segmentos: {manifest['statistics']['total_segments']}", file=sys.stderr)
            print(f"  ⏱️  Tiempo procesamiento: {manifest['statistics']['processing_time']:.1f} segundos", file=sys.stderr)
        
        if 'config' in manifest:
            config = manifest['config']
            if os.environ.get('MCP_MODE') != '1':
                print(f"  🎤 Modelo Whisper: {config.get('whisper_model', 'N/A')}", file=sys.stderr)
                print(f"  🧠 Modelo texto: {config.get('text_model', 'N/A')}", file=sys.stderr)
        
    except Exception as e:
        if os.environ.get('MCP_MODE') != '1':
            print(f"  ❌ Error leyendo manifiesto: {e}", file=sys.stderr)
        return False
    
    if os.environ.get('MCP_MODE') != '1':
        print("\n✅ Dataset verificado exitosamente!", file=sys.stderr)
    return True

def show_sample_data(dataset_dir: str, n_samples: int = 5):
    """Muestra datos de muestra del dataset"""
    
    import os
    if os.environ.get('MCP_MODE') != '1':
        print(f"\n📋 Muestra de {n_samples} segmentos:", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
    
    dataset_path = Path(dataset_dir)
    df = pd.read_pickle(dataset_path / "final" / "complete_dataset.pkl")
    
    # Tomar muestra aleatoria
    sample_df = df.sample(n=min(n_samples, len(df)))
    
    for idx, row in sample_df.iterrows():
        if os.environ.get('MCP_MODE') != '1':
            print(f"\n📄 Segmento {idx}:", file=sys.stderr)
            print(f"  📁 Archivo: {row['source_file']}", file=sys.stderr)
            print(f"  ⏱️  Tiempo: {row['start_time']:.1f}s - {row['end_time']:.1f}s ({row['duration']:.1f}s)", file=sys.stderr)
            print(f"  📝 Texto: {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}", file=sys.stderr)
        if 'confidence' in row:
            if os.environ.get('MCP_MODE') != '1':
                print(f"  🎯 Confianza: {row['confidence']:.2f}", file=sys.stderr)

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
        if os.environ.get('MCP_MODE') != '1':
            print(f"\n🎉 Dataset en {args.dataset_dir} está listo para usar!", file=sys.stderr)
        sys.exit(0)
    else:
        if os.environ.get('MCP_MODE') != '1':
            print(f"\n❌ Problemas encontrados en {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()