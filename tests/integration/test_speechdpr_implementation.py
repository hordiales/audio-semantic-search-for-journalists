#!/usr/bin/env python3
"""
Script de prueba para la implementación de SpeechDPR.
Verifica que el modelo se pueda cargar correctamente y generar embeddings.
"""

import logging
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

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_speechdpr_availability():
    """Prueba si las dependencias de SpeechDPR están disponibles"""
    print("🔍 Verificando disponibilidad de SpeechDPR...")

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")

        import transformers
        print(f"✅ Transformers: {transformers.__version__}")

        # Verificar si CUDA está disponible
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) disponible")
        else:
            print("ℹ️  Solo CPU disponible")

        return True

    except ImportError as e:
        print(f"❌ Dependencias faltantes: {e}")
        print("💡 Instala con: pip install torch transformers")
        return False

def test_speechdpr_import():
    """Prueba importar el módulo SpeechDPR"""
    print("\n📦 Probando importación de SpeechDPR...")

    try:
        from speechdpr_audio_embeddings import (
            SPEECHDPR_AVAILABLE,
            SpeechDPRAudioEmbeddingGenerator,
            SpeechDPRConfig,
            get_speechdpr_embedding_generator,
        )

        print("✅ Módulo SpeechDPR importado correctamente")
        print(f"📊 SpeechDPR disponible: {SPEECHDPR_AVAILABLE}")

        return True, (SpeechDPRAudioEmbeddingGenerator, SpeechDPRConfig, get_speechdpr_embedding_generator)

    except ImportError as e:
        print(f"❌ Error importando SpeechDPR: {e}")
        return False, None

def test_config_integration():
    """Prueba la integración con models_config.py"""
    print("\n⚙️  Probando integración con configuración...")

    try:
        from models_config import (
            AudioEmbeddingModel,
            get_models_config,
        )

        # Verificar que SpeechDPR está en el enum
        assert AudioEmbeddingModel.SPEECHDPR.value == "speechdpr"
        print("✅ SpeechDPR integrado en AudioEmbeddingModel")

        # Cargar configuración
        config = get_models_config()
        print(f"✅ Configuración cargada: {type(config.speechdpr_config)}")

        # Verificar configuración de SpeechDPR
        speechdpr_config = config.speechdpr_config
        print(f"📐 Embedding dim: {speechdpr_config.embedding_dim}")
        print(f"🎤 Speech model: {speechdpr_config.speech_encoder_model}")
        print(f"📝 Text model: {speechdpr_config.text_encoder_model}")

        # Probar disponibilidad
        available = config.is_model_available(
            config.ModelType.AUDIO_EMBEDDING if hasattr(config, 'ModelType') else type('ModelType', (), {'AUDIO_EMBEDDING': 'audio_embedding'})(),
            "speechdpr"
        )
        print(f"📊 SpeechDPR disponible según config: {available}")

        return True

    except Exception as e:
        print(f"❌ Error en integración de configuración: {e}")
        return False

def test_speechdpr_initialization():
    """Prueba inicializar el generador SpeechDPR"""
    print("\n🚀 Probando inicialización de SpeechDPR...")

    try:
        from speechdpr_audio_embeddings import (
            SpeechDPRConfig,
            get_speechdpr_embedding_generator,
        )

        # Crear configuración de prueba (modelos más pequeños si es posible)
        test_config = SpeechDPRConfig()
        test_config.speech_encoder_model = "facebook/hubert-base-ls960"  # Modelo más pequeño
        test_config.max_audio_length = 10  # Duración más corta para pruebas

        print("📋 Configuración de prueba:")
        print(f"   Speech Encoder: {test_config.speech_encoder_model}")
        print(f"   Text Encoder: {test_config.text_encoder_model}")
        print(f"   Device: {test_config.device}")
        print(f"   Max Audio Length: {test_config.max_audio_length}s")

        # Intentar inicializar
        print("🔄 Inicializando generador SpeechDPR...")
        embedder = get_speechdpr_embedding_generator(test_config)

        print("✅ SpeechDPR inicializado correctamente")
        print(f"📐 Dimensión de embeddings: {embedder.embedding_dim}")
        print(f"🖥️  Device: {embedder.device}")

        return True, embedder

    except Exception as e:
        print(f"❌ Error inicializando SpeechDPR: {e}")
        return False, None

def test_text_embedding():
    """Prueba generar embedding de texto"""
    print("\n📝 Probando generación de embedding de texto...")

    try:
        from speechdpr_audio_embeddings import (
            SpeechDPRConfig,
            get_speechdpr_embedding_generator,
        )

        test_config = SpeechDPRConfig()
        test_config.speech_encoder_model = "facebook/hubert-base-ls960"

        embedder = get_speechdpr_embedding_generator(test_config)

        # Texto de prueba
        test_text = "discurso político sobre economía"
        print(f"🔤 Texto de prueba: '{test_text}'")

        # Generar embedding
        text_embedding = embedder.generate_text_embedding(test_text)

        print("✅ Embedding de texto generado")
        print(f"📊 Shape: {text_embedding.shape}")
        print(f"📈 Tipo: {type(text_embedding)}")
        print(f"🔢 Primeros valores: {text_embedding[:5]}")

        return True

    except Exception as e:
        print(f"❌ Error generando embedding de texto: {e}")
        return False

def test_models_config_summary():
    """Prueba el resumen de configuración con SpeechDPR"""
    print("\n📊 Probando resumen de configuración...")

    try:
        from models_config import ModelsConfigLoader

        loader = ModelsConfigLoader()
        config = loader.load_config()

        # Mostrar modelos disponibles
        available_models = config.get_available_models(
            type('ModelType', (), {'AUDIO_EMBEDDING': 'audio_embedding'})()
        )
        print(f"🎵 Modelos de audio embedding disponibles: {available_models}")

        # Validar configuración
        validation = config.validate_configuration()
        print(f"✅ Configuración válida: {validation['valid']}")

        if validation['warnings']:
            print("⚠️  Advertencias:")
            for warning in validation['warnings']:
                print(f"   - {warning}")

        if validation['recommended_changes']:
            print("💡 Recomendaciones:")
            for rec in validation['recommended_changes']:
                print(f"   - {rec}")

        return True

    except Exception as e:
        print(f"❌ Error en resumen de configuración: {e}")
        return False

def main():
    """Función principal que ejecuta todas las pruebas"""
    print("🎤 Pruebas de Implementación de SpeechDPR")
    print("=" * 50)

    # Lista de pruebas
    tests = [
        ("Disponibilidad de dependencias", test_speechdpr_availability),
        ("Importación del módulo", test_speechdpr_import),
        ("Integración con configuración", test_config_integration),
        ("Resumen de configuración", test_models_config_summary),
    ]

    # Pruebas que requieren dependencias
    advanced_tests = [
        ("Inicialización de SpeechDPR", test_speechdpr_initialization),
        ("Generación de embedding de texto", test_text_embedding),
    ]

    results = {}

    # Ejecutar pruebas básicas
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Ejecutando: {test_name}")
        print(f"{'='*50}")

        try:
            result = test_func()
            if isinstance(result, tuple):
                results[test_name] = result[0]
            else:
                results[test_name] = result
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {e}")
            results[test_name] = False

    # Solo ejecutar pruebas avanzadas si las dependencias están disponibles
    if results.get("Disponibilidad de dependencias", False):
        for test_name, test_func in advanced_tests:
            print(f"\n{'='*50}")
            print(f"Ejecutando: {test_name}")
            print(f"{'='*50}")

            try:
                result = test_func()
                if isinstance(result, tuple):
                    results[test_name] = result[0]
                else:
                    results[test_name] = result
            except Exception as e:
                print(f"❌ Error inesperado en {test_name}: {e}")
                results[test_name] = False
    else:
        print("\n⚠️  Saltando pruebas avanzadas debido a dependencias faltantes")

    # Resumen final
    print(f"\n{'='*50}")
    print("📊 RESUMEN DE PRUEBAS")
    print(f"{'='*50}")

    passed = 0
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n📈 Resultados: {passed}/{total} pruebas exitosas")

    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! SpeechDPR está listo para usar.")
    elif passed > 0:
        print("⚠️  Algunas pruebas fallaron. Verifica las dependencias e instalación.")
    else:
        print("❌ Todas las pruebas fallaron. Revisa la instalación.")

    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Pruebas interrumpidas por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
