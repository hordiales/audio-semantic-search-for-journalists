# Guía de Solución de Problemas

## 📑 Índice Rápido

- **[Problemas de Instalación](#-problemas-de-instalación)**
  - Error: openai-whisper build wheel failed
  - Error: No module named 'torch'
  - Error: ffmpeg not found
- **[Problemas de Ejecución](#-problemas-de-ejecución)**
  - ⚠️ **Error MPS/Whisper** (Apple Silicon)
  - Error: No module named 'triton'
  - Error: TensorFlow no disponible
  - Error: CUDA out of memory
  - Problemas con MPS (Apple Silicon)
- **[Problemas con Archivos](#-problemas-con-archivos)**
- **[Problemas de Búsqueda](#-problemas-de-búsqueda)**
- **[Problemas de Interfaz](#️-problemas-de-interfaz)**
- **[Debugging](#-debugging)**
- **[Optimización de Rendimiento](#-optimización-de-rendimiento)**

## 🚨 Problemas de Instalación

### Conflictos de dependencias

Si encuentras conflictos de dependencias, usa Poetry para gestión de dependencias:
```bash
poetry install
poetry shell
```


### Error: openai-whisper build wheel failed

**Síntomas:**
```
Getting requirements to build wheel did not run successfully
KeyError: '__version__'
```

**Soluciones:**

1. **Usar requirements actualizados:**
   ```bash
   pip install -r requirements-minimal.txt
   ```

2. **Instalar Whisper directamente:**
   ```bash
   pip install openai-whisper
   # O la versión más reciente
   pip install git+https://github.com/openai/whisper.git
   ```

3. **Si persiste el error (Python 3.13 u otra versión):**
   ```bash
   # Usar pyenv para instalar exactamente Python 3.11.13
   pyenv install 3.11.13
   pyenv local 3.11.13
   python --version  # Debe mostrar 3.11.13

   # O con conda
   conda create -n semantic-search python=3.11.13 -y
   conda activate semantic-search
   pip install -r requirements.txt
   ```

### Error: No module named 'torch'

**Solución:**
```bash
# Instalar PyTorch primero
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Luego el resto
pip install -r requirements-minimal.txt
```

### Error: ffmpeg not found

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
Descargar desde [ffmpeg.org](https://ffmpeg.org/download.html) y añadir al PATH.

## 🔧 Problemas de Ejecución

### Error: `NotImplementedError: Could not run 'aten::empty.memory_format' with arguments from the 'SparseMPS' backend`

**Problema**: Whisper falla al cargarse en MPS (Apple Silicon) debido a limitaciones del backend con operaciones de tensores dispersos.

**Síntomas:**
```
NotImplementedError: Could not run 'aten::empty.memory_format' with arguments from the 'SparseMPS' backend.
This could be because the operator doesn't exist for this backend...
```

**Solución automática**: El código detecta este error y automáticamente hace fallback a CPU. Verás un mensaje de advertencia:

```
⚠️  Error cargando modelo en MPS: ...
   Cambiando a CPU (MPS tiene limitaciones con algunas operaciones de Whisper)
Modelo Whisper 'base' cargado en cpu (fallback a CPU)
```

**Forzar CPU desde el inicio** (opcional):
```bash
export WHISPER_DEVICE=cpu
poetry run python src/simple_dataset_pipeline.py --input data/ --output ./dataset
```

**Más información**: Ver [docs/GPU_CONSIDERATIONS.md](GPU_CONSIDERATIONS.md) para detalles sobre GPU, MPS y compatibilidad.

### Error: `No module named 'triton'` en macOS

**Problema**: Triton no está disponible para macOS, pero el código funciona sin él.

**Síntomas:**
```
Unable to find installation candidates for triton (2.3.1)
6 wheel(s) were skipped as your project's environment does not support the identified abi tags
```

**Solución**: Esto es esperado. Si `poetry install` falla por triton:

```bash
# Crear venv (puede fallar en triton, pero crea el venv)
poetry install || true

# Instalar dependencias con pip (ignora triton)
poetry run pip install -r requirements.txt
```

**Nota**: Triton solo está disponible para Linux con CUDA. No es necesario para macOS.

### Error: TensorFlow no disponible (YAMNet)

**Síntomas:**
```
⚠️  TensorFlow no disponible. YAMNet no estará disponible.
   Para usar YAMNet: poetry install --extras yamnet
```

**Solución:**
```bash
# Instalar extras para YAMNet
poetry install --extras yamnet
```

### Error: CUDA out of memory

**Síntomas:**
```
RuntimeError: CUDA out of memory
```

**Soluciones:**

1. **Forzar uso de CPU:**
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   # O en Python
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = ''
   ```

2. **Usar modelos más pequeños:**
   ```bash
   # Usar modelo Whisper más pequeño
   poetry run python src/simple_dataset_pipeline.py \
       --input data/ \
       --output ./dataset \
       --whisper-model tiny  # En lugar de 'base' o 'medium'
   ```

3. **Procesar en lotes más pequeños:**
   ```python
   # En text_embeddings.py, reducir batch_size
   embeddings = self.model.encode(texts, batch_size=8)  # Reducir de 32
   ```

### Problemas con MPS (Apple Silicon)

**Verificar disponibilidad de MPS:**
```python
import torch
print(f"MPS disponible: {torch.backends.mps.is_available()}")
print(f"MPS construido: {torch.backends.mps.is_built()}")
```

**Forzar CPU en macOS:**
```bash
# Para Whisper
export WHISPER_DEVICE=cpu

# Para CLAP
export CLAP_DEVICE=cpu

# Para SpeechDPR
export SPEECHDPR_DEVICE=cpu
```

**Más información**: Ver [docs/GPU_CONSIDERATIONS.md](GPU_CONSIDERATIONS.md) para detalles completos sobre GPU, MPS, CUDA y rendimiento.


### Error: Cannot import name 'sentence_transformers'

**Solución:**
```bash
# Reinstalar sentence-transformers
pip uninstall sentence-transformers
pip install sentence-transformers>=2.2.0
```

## 📁 Problemas con Archivos

### Error: No such file or directory

**Verificar rutas:**
```python
import os
print("Directorio actual:", os.getcwd())
print("Archivos:", os.listdir('.'))
```

**Usar rutas absolutas:**
```python
import os
audio_file = os.path.abspath("ruta/a/audio.wav")
```

### Error: Invalid audio format

**Formatos soportados:**
- WAV (recomendado)
- MP3
- M4A
- FLAC

**Conversión con ffmpeg:**
```bash
# Convertir a WAV
ffmpeg -i input.mp3 output.wav

# Normalizar frecuencia de muestreo
ffmpeg -i input.wav -ar 16000 output_16k.wav
```

## 🔍 Problemas de Búsqueda

### Resultados irrelevantes

**Verificar indexación:**
```python
# Verificar que los índices existan
stats = search_engine.get_system_stats()
print(stats)
```

**Ajustar parámetros:**
```python
config = {
    'text_weight': 0.8,      # Más peso al texto
    'audio_weight': 0.2,     # Menos peso al audio
    'top_k_results': 10      # Más resultados
}
```

### Búsqueda muy lenta

**Optimizaciones:**

1. **Usar índices aproximados:**
   ```python
   # Para datasets grandes
   index_manager = VectorIndexManager(
       embedding_dim=384,
       index_type="IVF"  # En lugar de "L2"
   )
   ```

2. **Cache de embeddings:**
   ```python
   # Guardar embeddings procesados
   df.to_pickle('processed_embeddings.pkl')

   # Cargar en siguiente sesión
   df = pd.read_pickle('processed_embeddings.pkl')
   ```

## 🖥️ Problemas de Interfaz

### API REST no inicia

**Verificar puerto:**
```bash
# Verificar que el puerto 8000 esté disponible
lsof -i :8000

# Usar puerto diferente
python -m uvicorn api.main:app --port 8001
```

**Verificar firewall:**
```bash
# macOS: Permitir conexiones
# Windows: Verificar Windows Firewall
```

## 🐛 Debugging

### Habilitar logging detallado

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# En los módulos
logger = logging.getLogger(__name__)
logger.debug("Información de debug")
```

### Verificar memoria

```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memoria usada: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

### Test de componentes individuales

```python
# Test transcripción
from audio_transcription import AudioTranscriber
transcriber = AudioTranscriber('tiny')
# Test con archivo pequeño

# Test embeddings
from text_embeddings import TextEmbeddingGenerator
embedder = TextEmbeddingGenerator()
test_embedding = embedder.generate_query_embedding("test")
print(f"Embedding shape: {test_embedding.shape}")
```

## 📊 Optimización de Rendimiento

### Reducir uso de memoria

1. **Procesar archivos de uno en uno:**
   ```python
   for audio_file in audio_files:
       df = search_engine.process_audio_files([audio_file])
       # Procesar inmediatamente
   ```

2. **Limpiar variables grandes:**
   ```python
   import gc
   del large_variable
   gc.collect()
   ```

### Acelerar procesamiento

1. **Paralelización:**
   ```python
   from concurrent.futures import ProcessPoolExecutor

   with ProcessPoolExecutor(max_workers=4) as executor:
       results = executor.map(process_file, audio_files)
   ```

2. **Usar SSD para archivos temporales:**
   ```python
   import tempfile
   with tempfile.TemporaryDirectory(dir="/path/to/ssd") as temp_dir:
       # Procesamiento más rápido
   ```

## 🔄 Reinstalación Limpia

Si todos los problemas persisten:

```bash
# 1. Crear nuevo entorno con Python 3.11.13 exactamente
conda create -n semantic-search-clean python=3.11.13 -y
conda activate semantic-search-clean

# 2. Limpiar cache pip
pip cache purge

# 3. Instalar desde cero
pip install --no-cache-dir -r requirements-minimal.txt

# 4. Verificar
python install.py
```

## 📞 Obtener Ayuda

### Información del sistema para reportes

```python
# Ejecutar esto y incluir en el reporte
import sys
import platform
import torch
import numpy as np

print("Sistema:")
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Python: {sys.version}")
print(f"  NumPy: {np.__version__}")
print(f"  PyTorch: {torch.__version__}")

print("\nGPU/Dispositivos:")
print(f"  CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA versión: {torch.version.cuda}")
    print(f"  Dispositivos CUDA: {torch.cuda.device_count()}")
if hasattr(torch.backends, 'mps'):
    print(f"  MPS disponible: {torch.backends.mps.is_available()}")
    print(f"  MPS construido: {torch.backends.mps.is_built()}")

print("\nModelos:")
try:
    import whisper
    print(f"  Whisper: {whisper.__version__}")
except:
    print("  Whisper: No instalado")

try:
    import sentence_transformers
    print(f"  Sentence-transformers: {sentence_transformers.__version__}")
except:
    print("  Sentence-transformers: No instalado")

try:
    import tensorflow as tf
    print(f"  TensorFlow: {tf.__version__}")
except:
    print("  TensorFlow: No instalado")
```

### Donde reportar problemas

1. **Issues del proyecto**: GitHub Issues
2. **Documentación**: README.md y ARCHITECTURE.md
3. **Problemas de GPU/MPS**: Ver [docs/GPU_CONSIDERATIONS.md](GPU_CONSIDERATIONS.md)
4. **Ejemplos**: example_usage.py

### Información útil para reportes

- Output completo del error
- Información del sistema (código anterior)
- Pasos para reproducir el problema
- Archivos de audio utilizados (si es seguro compartir)
- Configuración utilizada
