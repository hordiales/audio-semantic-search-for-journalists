#!/usr/bin/env python3
"""
Test CLAP con descarga completa (sin timeout)
"""

import os
import sys
import time
from pathlib import Path

# Configurar warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_clap_full():
    """Test CLAP con descarga completa"""
    print("⚡ TEST CLAP - DESCARGA COMPLETA")
    print("=" * 35)

    try:
        # Buscar archivo de audio
        audio_folder = Path("dataset/converted")
        audio_files = list(audio_folder.glob("*.wav"))

        if not audio_files:
            print("❌ No hay archivos de audio")
            return

        audio_path = str(audio_files[0])
        file_name = Path(audio_path).name
        print(f"📁 Usando: {file_name}")

        print("🔄 Importando CLAP...")
        import laion_clap

        print("🔄 Inicializando modelo CLAP...")
        print("💡 Primera ejecución: descargando modelo (~1-2 GB)...")
        start_time = time.time()

        model = laion_clap.CLAP_Module(enable_fusion=False)
        print("🔄 Cargando checkpoint (esto puede tardar varios minutos)...")

        model.load_ckpt()

        load_time = time.time() - start_time
        print(f"✅ CLAP cargado en {load_time:.1f}s")

        print("🔄 Generando embedding de audio...")
        embed_start = time.time()

        # Probar con diferentes métodos
        try:
            audio_embed = model.get_audio_embedding_from_filelist([audio_path], use_tensor=False)
            embedding_np = audio_embed[0]
        except Exception as e:
            print(f"⚠️ Método 1 falló: {e}")
            print("🔄 Probando método alternativo...")

            import librosa
            audio_data, sample_rate = librosa.load(audio_path, sr=48000)
            audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
            embedding_np = audio_embed[0]

        embed_time = time.time() - embed_start

        print(f"✅ Embedding generado:")
        print(f"   📏 Dimensión: {len(embedding_np)}")
        print(f"   ⏱️  Tiempo: {embed_time:.1f}s")
        print(f"   📊 Tipo: {type(embedding_np)}")
        print(f"   📊 Preview: {embedding_np[:5]}")

        # Test de embedding de texto
        print("🔄 Generando embedding de texto...")
        text_start = time.time()

        text_embed = model.get_text_embedding(["política argentina escándalo"])
        text_embedding_np = text_embed[0]

        text_time = time.time() - text_start

        print(f"✅ Embedding de texto generado:")
        print(f"   📏 Dimensión: {len(text_embedding_np)}")
        print(f"   ⏱️  Tiempo: {text_time:.1f}s")

        # Similitud audio-texto
        import numpy as np

        # Normalizar
        audio_norm = embedding_np / np.linalg.norm(embedding_np)
        text_norm = text_embedding_np / np.linalg.norm(text_embedding_np)

        # Similitud coseno
        similarity = np.dot(audio_norm, text_norm)

        print(f"🔄 Similitud audio-texto: {similarity:.4f}")

        # Múltiples queries de texto
        queries = [
            "política argentina",
            "escándalo parlamentario",
            "sesión cámara diputados",
            "música",
            "deporte fútbol"
        ]

        print("\n🔍 Similitudes con diferentes queries:")
        for query in queries:
            text_emb = model.get_text_embedding([query])[0]
            text_norm = text_emb / np.linalg.norm(text_emb)
            sim = np.dot(audio_norm, text_norm)
            print(f"   '{query}': {sim:.4f}")

        print("\n✅ CLAP funcionando correctamente!")
        print("💾 Modelo descargado y listo para uso futuro")

    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Ejecuta: pip install laion-clap")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clap_full()