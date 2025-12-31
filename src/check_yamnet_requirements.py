#!/usr/bin/env python3
"""
Verifica si el sistema puede ejecutar YAMNet real
"""

import importlib
import logging
import sys

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    logging.info(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and 8 <= version.minor <= 11:
        logging.info("✅ Versión de Python compatible con TensorFlow")
        return True
    logging.error("❌ Versión de Python no compatible (necesita 3.8-3.11)")
    return False

def check_tensorflow():
    """Verifica TensorFlow"""
    try:
        import tensorflow as tf
        logging.info(f"✅ TensorFlow: {tf.__version__}")

        # Verificar GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logging.info(f"🚀 GPU disponible: {len(gpus)} dispositivos")
            for i, gpu in enumerate(gpus):
                logging.info(f"   GPU {i}: {gpu.name}")
        else:
            logging.info("💻 Solo CPU disponible (más lento)")

        return True
    except ImportError:
        logging.error("❌ TensorFlow no instalado")
        return False

def check_tensorflow_hub():
    """Verifica TensorFlow Hub"""
    try:
        import tensorflow_hub as hub
        logging.info("✅ TensorFlow Hub disponible")
        return True
    except ImportError:
        logging.error("❌ TensorFlow Hub no instalado")
        return False

def check_audio_libraries():
    """Verifica librerías de audio"""
    libraries = {
        'librosa': 'Procesamiento de audio',
        'soundfile': 'Lectura/escritura de archivos de audio',
        'resampy': 'Remuestreo de audio (opcional)',
    }

    all_available = True
    for lib, description in libraries.items():
        try:
            importlib.import_module(lib)
            logging.info(f"✅ {lib}: {description}")
        except ImportError:
            logging.error(f"❌ {lib}: {description} - NO INSTALADO")
            all_available = False

    return all_available

def check_disk_space():
    """Verifica espacio en disco"""
    import shutil
    _total, _used, free = shutil.disk_usage('/')
    free_gb = free // (1024**3)
    logging.info(f"💾 Espacio libre: {free_gb} GB")

    if free_gb >= 2:
        logging.info("✅ Suficiente espacio para modelo YAMNet (~500MB)")
        return True
    logging.error("❌ Poco espacio en disco")
    return False

def check_internet():
    """Verifica conexión a internet"""
    try:
        import urllib.request
        urllib.request.urlopen('https://tfhub.dev', timeout=5)
        logging.info("✅ Conexión a TensorFlow Hub disponible")
        return True
    except:
        logging.error("❌ Sin conexión a TensorFlow Hub")
        return False

def estimate_processing_time():
    """Estima tiempo de procesamiento"""
    logging.info("\n⏱️  Estimación de Tiempos (31,954 segmentos):")
    logging.info("  💻 Solo CPU: ~8-12 horas")
    logging.info("  🚀 Con GPU: ~2-4 horas")
    logging.info("  📊 Dependiente de duración promedio de segmentos")

def print_installation_instructions():
    """Imprime instrucciones de instalación"""
    logging.info("\n🔧 Instrucciones de Instalación:")
    logging.info("""
# Instalar TensorFlow (CPU)
pip install tensorflow

# Instalar TensorFlow (GPU) - requiere CUDA
pip install tensorflow-gpu

# Instalar TensorFlow Hub
pip install tensorflow-hub

# Instalar librerías de audio
pip install librosa soundfile resampy

# Verificar instalación
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tensorflow_hub as hub; print('TF Hub OK')"
""")

def main():
    logging.info("🔍 Verificación de Requisitos para YAMNet Real")
    logging.info("=" * 50)

    all_good = True

    # Verificaciones
    all_good &= check_python_version()
    all_good &= check_tensorflow()
    all_good &= check_tensorflow_hub()
    all_good &= check_audio_libraries()
    all_good &= check_disk_space()
    all_good &= check_internet()

    estimate_processing_time()

    logging.info("\n📊 RESUMEN:")
    if all_good:
        logging.info("✅ Sistema listo para YAMNet real")
        logging.info("🚀 Puedes proceder con la generación de embeddings reales")
    else:
        logging.error("❌ Faltan requisitos para YAMNet real")
        print_installation_instructions()

    return all_good


if __name__ == "__main__":
    main()
