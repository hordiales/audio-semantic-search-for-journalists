#!/usr/bin/env python3
"""
Script para verificar qué información de modelos está disponible en un dataset existente.
Útil para determinar cómo se generaron los embeddings de un dataset.
"""

import json
from pathlib import Path
import sys

import pandas as pd


def check_dataset_models(dataset_dir: str) -> dict:
    """
    Verifica qué información de modelos está disponible en un dataset

    Args:
        dataset_dir: Directorio del dataset a verificar

    Returns:
        Diccionario con información encontrada
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Directorio no existe: {dataset_dir}")

    result = {
        "dataset_dir": str(dataset_path),
        "manifest_available": False,
        "manifest_models": {},
        "dataframe_models": {},
        "indices_metadata": {},
        "transcription_metadata": {}
    }

    # 1. Verificar manifest
    manifest_file = dataset_path / "final" / "dataset_manifest.json"
    if manifest_file.exists():
        result["manifest_available"] = True
        try:
            with open(manifest_file, encoding='utf-8') as f:
                manifest = json.load(f)

            # Extraer información de modelos del manifest
            if "models_used" in manifest:
                result["manifest_models"] = manifest["models_used"]
            elif "config" in manifest:
                # Manifest antiguo, extraer de config
                config = manifest["config"]
                result["manifest_models"] = {
                    "transcription": {
                        "model": config.get("whisper_model", "N/A"),
                        "language": config.get("language", "N/A")
                    },
                    "text_embeddings": {
                        "model": config.get("text_model", "N/A")
                    },
                    "audio_embeddings": {
                        "model": config.get("audio_model", "N/A")
                    }
                }
        except Exception as e:
            result["manifest_error"] = str(e)

    # 2. Verificar DataFrame
    dataset_file = dataset_path / "final" / "complete_dataset.pkl"
    if dataset_file.exists():
        try:
            df = pd.read_pickle(dataset_file)

            # Verificar columnas de modelos
            if 'embedding_model' in df.columns:
                result["dataframe_models"]["text_model"] = df['embedding_model'].iloc[0] if len(df) > 0 else None
            if 'embedding_dim' in df.columns:
                result["dataframe_models"]["text_dim"] = int(df['embedding_dim'].iloc[0]) if len(df) > 0 else None
            if 'audio_embedding_model' in df.columns:
                result["dataframe_models"]["audio_model"] = df['audio_embedding_model'].iloc[0] if len(df) > 0 else None
            if 'audio_embedding_dim' in df.columns:
                result["dataframe_models"]["audio_dim"] = int(df['audio_embedding_dim'].iloc[0]) if len(df) > 0 else None

            result["dataframe_available"] = True
            result["total_segments"] = len(df)
        except Exception as e:
            result["dataframe_error"] = str(e)

    # 3. Verificar metadata de índices
    indices_metadata_file = dataset_path / "indices" / "indices_metadata.json"
    if indices_metadata_file.exists():
        try:
            with open(indices_metadata_file, encoding='utf-8') as f:
                indices_metadata = json.load(f)
            result["indices_metadata"] = {
                "text_model": indices_metadata.get("text_model", "N/A"),
                "audio_model": indices_metadata.get("audio_model", "N/A"),
                "embedding_dimension": indices_metadata.get("embedding_dimension", "N/A")
            }
        except Exception as e:
            result["indices_error"] = str(e)

    # 4. Verificar transcripciones (puede tener info de Whisper)
    transcriptions_file = dataset_path / "transcriptions" / "all_transcriptions.json"
    if transcriptions_file.exists():
        try:
            with open(transcriptions_file, encoding='utf-8') as f:
                transcriptions = json.load(f)

            if "transcriptions" in transcriptions and len(transcriptions["transcriptions"]) > 0:
                first_transcription = transcriptions["transcriptions"][0]
                result["transcription_metadata"] = {
                    "whisper_model": first_transcription.get("whisper_model", "N/A"),
                    "language": first_transcription.get("language", "N/A")
                }
        except Exception as e:
            result["transcription_error"] = str(e)

    return result


def print_model_info(result: dict):
    """Imprime información de modelos de forma legible"""
    print("=" * 70)
    print("INFORMACIÓN DE MODELOS EN EL DATASET")
    print("=" * 70)
    print(f"📁 Dataset: {result['dataset_dir']}\n")

    # Información del manifest
    if result.get("manifest_available"):
        print("📋 MANIFEST (dataset_manifest.json):")
        if result.get("manifest_models"):
            models = result["manifest_models"]

            if "transcription" in models:
                trans = models["transcription"]
                print("  🤖 Transcripción:")
                print(f"     - Modelo Whisper: {trans.get('model', 'N/A')}")
                print(f"     - Idioma: {trans.get('language', 'N/A')}")
                print(f"     - Método segmentación: {trans.get('segmentation_method', 'N/A')}")

            if "text_embeddings" in models:
                text = models["text_embeddings"]
                print("  📝 Embeddings de Texto:")
                print(f"     - Modelo: {text.get('model', 'N/A')}")
                print(f"     - Dimensión: {text.get('embedding_dimension', 'N/A')}")

            if "audio_embeddings" in models:
                audio = models["audio_embeddings"]
                print("  🎵 Embeddings de Audio:")
                print(f"     - Modelo: {audio.get('model', 'N/A')}")
                print(f"     - Dimensión: {audio.get('embedding_dimension', 'N/A')}")
        else:
            print("  ⚠️  No se encontró información de modelos en el manifest")
        print()
    else:
        print("⚠️  No se encontró manifest (dataset_manifest.json)\n")

    # Información del DataFrame
    if result.get("dataframe_available"):
        print("💾 DATAFRAME (complete_dataset.pkl):")
        df_models = result.get("dataframe_models", {})
        if df_models:
            if "text_model" in df_models:
                print(f"  📝 Modelo texto: {df_models['text_model']}")
            if "text_dim" in df_models:
                print(f"  📊 Dimensión texto: {df_models['text_dim']}")
            if "audio_model" in df_models:
                print(f"  🎵 Modelo audio: {df_models['audio_model']}")
            if "audio_dim" in df_models:
                print(f"  📊 Dimensión audio: {df_models['audio_dim']}")
            if "total_segments" in result:
                print(f"  📊 Total segmentos: {result['total_segments']}")
        else:
            print("  ⚠️  No se encontraron columnas de modelos en el DataFrame")
        print()

    # Información de índices
    if result.get("indices_metadata"):
        print("🔍 ÍNDICES (indices_metadata.json):")
        indices = result["indices_metadata"]
        print(f"  📝 Modelo texto: {indices.get('text_model', 'N/A')}")
        print(f"  🎵 Modelo audio: {indices.get('audio_model', 'N/A')}")
        print(f"  📊 Dimensión: {indices.get('embedding_dimension', 'N/A')}")
        print()

    # Información de transcripciones
    if result.get("transcription_metadata"):
        print("📄 TRANSCRIPCIONES (all_transcriptions.json):")
        trans = result["transcription_metadata"]
        print(f"  🤖 Modelo Whisper: {trans.get('whisper_model', 'N/A')}")
        print(f"  🌐 Idioma: {trans.get('language', 'N/A')}")
        print()

    # Resumen
    print("=" * 70)
    print("RESUMEN:")
    print("=" * 70)

    has_complete_info = (
        result.get("manifest_available") and
        result.get("manifest_models") and
        "text_embeddings" in result.get("manifest_models", {}) and
        "audio_embeddings" in result.get("manifest_models", {})
    )

    if has_complete_info:
        print("✅ Información completa de modelos disponible en el manifest")
    elif result.get("manifest_available"):
        print("⚠️  Manifest disponible pero información de modelos incompleta")
    else:
        print("❌ No se encontró manifest con información de modelos")
        if result.get("dataframe_models"):
            print("   ℹ️  Se encontró información parcial en el DataFrame")
        if result.get("indices_metadata"):
            print("   ℹ️  Se encontró información parcial en índices")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verifica qué información de modelos está disponible en un dataset"
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="./dataset",
        help="Directorio del dataset a verificar (default: ./dataset)"
    )

    args = parser.parse_args()

    try:
        result = check_dataset_models(args.dataset_dir)
        print_model_info(result)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
