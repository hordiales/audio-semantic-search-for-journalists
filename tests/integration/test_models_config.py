#!/usr/bin/env python3
"""
Script de prueba para el sistema de configuración de modelos
"""

import os
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import SRC_ROOT, ensure_sys_path

ensure_sys_path([SRC_ROOT])

def test_models_configuration():
    """Prueba el sistema de configuración de modelos"""
    print("🤖 Prueba del Sistema de Configuración de Modelos")
    print("=" * 60)

    try:
        from src.models_config import get_available_models_info, models_config_loader

        print("\n📋 Configuración actual:")
        models_config_loader.print_config_summary()

        print("\n🔍 Modelos disponibles:")
        available = get_available_models_info()
        for model_type, models in available.items():
            status = "✅" if models else "❌"
            model_list = ", ".join(models) if models else "Ninguno disponible"
            print(f"  {status} {model_type}: {model_list}")

        # Probar carga de transcriptor
        print("\n🎤 Probando AudioTranscriber:")
        try:
            from src.audio_transcription import AudioTranscriber
            transcriber = AudioTranscriber()
            print(f"  ✅ Whisper cargado: {transcriber.model_name} en {transcriber.device}")
        except Exception as e:
            print(f"  ❌ Error cargando Whisper: {e}")

        # Probar carga de embedding generator
        print("\n🔊 Probando AudioEmbeddingGenerator:")
        try:
            from src.audio_embeddings import get_audio_embedding_generator
            embedder = get_audio_embedding_generator()
            print(f"  ✅ Generador de embeddings cargado: {embedder.__class__.__name__}")
        except Exception as e:
            print(f"  ❌ Error cargando generador de embeddings: {e}")

        # Probar configuración de CLAP si está disponible
        print("\n🎵 Probando CLAP (opcional):")
        try:
            from src.clap_audio_embeddings import get_clap_embedding_generator
            clap_embedder = get_clap_embedding_generator()
            print(f"  ✅ CLAP disponible: {clap_embedder.config.model_name}")
        except Exception as e:
            print(f"  ℹ️  CLAP no disponible: {e}")
            print("     💡 Para instalar: pip install laion-clap")

        print("\n🎯 Recomendaciones:")
        print("  📝 Crea un archivo .env.models para personalizar configuración")
        print("  🚀 Usa CLAP para búsqueda semántica avanzada")
        print("  ⚡ Ajusta modelos según tus recursos de hardware")

        return True

    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        return False


def create_example_config():
    """Crea archivos de configuración de ejemplo"""
    print("\n📁 Creando archivos de configuración de ejemplo...")

    try:
        from src.models_config import ModelsConfigLoader

        loader = ModelsConfigLoader()

        # Crear template básico
        if not os.path.exists(".env.models"):
            loader.create_env_template(".env.models")
            print("  ✅ Creado: .env.models")
        else:
            print("  ℹ️  Ya existe: .env.models")

        # Crear configuración CLAP de ejemplo
        clap_config = """# Configuración CLAP para búsqueda semántica avanzada
DEFAULT_AUDIO_EMBEDDING_MODEL=clap_laion
DEFAULT_AUDIO_EVENT_DETECTION_MODEL=clap_laion

# CLAP específico
CLAP_MODEL_NAME=laion/clap-htsat-unfused
CLAP_DEVICE=auto
CLAP_ENABLE_FUSION=false

# Whisper optimizado
DEFAULT_SPEECH_TO_TEXT_MODEL=whisper_small
WHISPER_DEVICE=auto
WHISPER_LANGUAGE=es
"""

        with open(".env.models.clap", "w") as f:
            f.write(clap_config)
        print("  ✅ Creado: .env.models.clap (configuración CLAP)")

        print("\n💡 Para usar CLAP:")
        print("  1. pip install laion-clap")
        print("  2. cp .env.models.clap .env.models")
        print("  3. Reinicia el sistema")

        return True

    except Exception as e:
        print(f"❌ Error creando configuración: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Test del Sistema de Configuración de Modelos")
    print("=" * 50)

    # Ejecutar pruebas
    config_ok = test_models_configuration()
    example_ok = create_example_config()

    print("\n📊 Resumen:")
    print(f"  Sistema de configuración: {'✅' if config_ok else '❌'}")
    print(f"  Archivos de ejemplo: {'✅' if example_ok else '❌'}")

    if config_ok and example_ok:
        print("\n🎉 ¡Sistema de configuración listo!")
        print("📖 Lee MODELS_CONFIGURATION.md para más detalles")
        sys.exit(0)
    else:
        print("\n⚠️  Algunos componentes necesitan atención")
        sys.exit(1)
