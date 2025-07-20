#!/usr/bin/env python3
"""
Verifica si el sistema puede ejecutar YAMNet real
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print("✅ Versión de Python compatible con TensorFlow")
        return True
    else:
        print("❌ Versión de Python no compatible (necesita 3.8-3.11)")
        return False

def check_tensorflow():
    """Verifica TensorFlow"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        # Verificar GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU disponible: {len(gpus)} dispositivos")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("💻 Solo CPU disponible (más lento)")
        
        return True
    except ImportError:
        print("❌ TensorFlow no instalado")
        return False

def check_tensorflow_hub():
    """Verifica TensorFlow Hub"""
    try:
        import tensorflow_hub as hub
        print(f"✅ TensorFlow Hub disponible")
        return True
    except ImportError:
        print("❌ TensorFlow Hub no instalado")
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
            print(f"✅ {lib}: {description}")
        except ImportError:
            print(f"❌ {lib}: {description} - NO INSTALADO")
            all_available = False
    
    return all_available

def check_disk_space():
    """Verifica espacio en disco"""
    import shutil
    total, used, free = shutil.disk_usage('/')
    free_gb = free // (1024**3)
    print(f"💾 Espacio libre: {free_gb} GB")
    
    if free_gb >= 2:
        print("✅ Suficiente espacio para modelo YAMNet (~500MB)")
        return True
    else:
        print("❌ Poco espacio en disco")
        return False

def check_internet():
    """Verifica conexión a internet"""
    try:
        import urllib.request
        urllib.request.urlopen('https://tfhub.dev', timeout=5)
        print("✅ Conexión a TensorFlow Hub disponible")
        return True
    except:
        print("❌ Sin conexión a TensorFlow Hub")
        return False

def estimate_processing_time():
    """Estima tiempo de procesamiento"""
    print(f"\n⏱️  Estimación de Tiempos (31,954 segmentos):")
    print(f"  💻 Solo CPU: ~8-12 horas")
    print(f"  🚀 Con GPU: ~2-4 horas")
    print(f"  📊 Dependiente de duración promedio de segmentos")

def print_installation_instructions():
    """Imprime instrucciones de instalación"""
    print(f"\n🔧 Instrucciones de Instalación:")
    print(f"""
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
    print("🔍 Verificación de Requisitos para YAMNet Real")
    print("=" * 50)
    
    all_good = True
    
    # Verificaciones
    all_good &= check_python_version()
    all_good &= check_tensorflow()
    all_good &= check_tensorflow_hub()
    all_good &= check_audio_libraries()
    all_good &= check_disk_space()
    all_good &= check_internet()
    
    estimate_processing_time()
    
    print(f"\n📊 RESUMEN:")
    if all_good:
        print("✅ Sistema listo para YAMNet real")
        print("🚀 Puedes proceder con la generación de embeddings reales")
    else:
        print("❌ Faltan requisitos para YAMNet real")
        print_installation_instructions()
    
    return all_good

if __name__ == "__main__":
    main()