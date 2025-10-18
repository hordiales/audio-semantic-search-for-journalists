#!/usr/bin/env python3
"""
Test directo de CLAP con timeout
"""

import os
import sys
import time
import signal
from pathlib import Path

# Configurar warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout")

def test_clap_with_timeout():
    """Test CLAP con timeout de 3*60 segundos"""
    print("⚡ TEST DIRECTO DE CLAP")
    print("=" * 25)

    # Configurar timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3*60)  # 3*60 segundos timeout

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
        start_time = time.time()

        model = laion_clap.CLAP_Module(enable_fusion=False)
        print("🔄 Cargando checkpoint...")

        model.load_ckpt()

        load_time = time.time() - start_time
        print(f"✅ CLAP cargado en {load_time:.1f}s")

        print("🔄 Generando embedding de audio...")
        embed_start = time.time()

        audio_embed = model.get_audio_embedding_from_filelist([audio_path], use_tensor=False)
        embedding_np = audio_embed[0]

        embed_time = time.time() - embed_start

        print(f"✅ Embedding generado:")
        print(f"   📏 Dimensión: {len(embedding_np)}")
        print(f"   ⏱️  Tiempo: {embed_time:.1f}s")
        print(f"   📊 Preview: {embedding_np[:5]}")

        # Test de embedding de texto
        print("🔄 Generando embedding de texto...")
        text_start = time.time()

        text_embed = model.get_text_embedding(["política argentina"])
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

        print("✅ CLAP funcionando correctamente!")

        signal.alarm(0)  # Cancelar timeout

    except TimeoutError:
        print("⏰ Timeout - CLAP tomó más de 60s")
        print("💡 CLAP puede ser lento en la primera carga")

    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Instala: pip install laion-clap")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        signal.alarm(0)  # Asegurar que se cancele el timeout

if __name__ == "__main__":
    test_clap_with_timeout()