# Consideraciones sobre GPU

Este documento explica cómo el proyecto utiliza aceleración por GPU y las diferencias entre plataformas.

## Resumen por Plataforma

| Plataforma | GPU | Backend | Triton | Estado |
|------------|-----|---------|--------|--------|
| Linux + NVIDIA | CUDA | cuDNN + Triton | ✅ Requerido | Óptimo |
| macOS Intel | CPU | - | ❌ No disponible | Funcional |
| macOS Apple Silicon | MPS | Metal | ❌ No disponible | Funcional |
| Windows + NVIDIA | CUDA | cuDNN | ⚠️ Experimental | Funcional |

## ¿Qué es Triton?

**Triton** es una librería de compilación de kernels GPU desarrollada por OpenAI.

### Funcionalidad
- Compila código Python a **kernels CUDA** optimizados para GPUs NVIDIA
- Permite escribir operaciones de deep learning eficientes sin código CUDA manual
- Usado internamente por PyTorch y Whisper para optimizar inferencia en GPU

### Dependencias que requieren Triton
```
openai-whisper requiere triton >=2.0.0
torch requiere triton 2.3.1 (en Linux)
```

### Limitaciones
- **Solo funciona con CUDA** (GPUs NVIDIA en Linux)
- No hay wheels disponibles para macOS o Windows
- Es una optimización, no un requisito funcional

## macOS (Apple Silicon)

### Backend: Metal Performance Shaders (MPS)

En Macs con chip Apple Silicon (M1, M2, M3, M4), PyTorch usa **MPS** en lugar de CUDA:

```python
import torch

# Verificar disponibilidad de MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Usando GPU Apple Silicon via MPS")
else:
    device = torch.device("cpu")
    print("Usando CPU")
```

### Problema con Poetry

`poetry install` puede fallar porque intenta instalar triton:

```
Unable to find installation candidates for triton (2.3.1)
6 wheel(s) were skipped as your project's environment does not support the identified abi tags
```

### Solución

Después de crear el virtualenv con poetry, instalar dependencias con pip:

```bash
# Crear venv (puede fallar en triton, pero crea el venv)
poetry install || true

# Instalar dependencias con pip (ignora triton)
poetry run pip install -r requirements.txt
```

### Rendimiento en macOS

| Modelo | CPU | MPS (Apple Silicon) | Notas |
|--------|-----|---------------------|-------|
| Whisper tiny | ~1x realtime | ~3x realtime | Bueno para pruebas |
| Whisper base | ~0.5x realtime | ~2x realtime | Recomendado |
| Whisper small | ~0.2x realtime | ~1x realtime | Buena calidad |
| Whisper medium | ~0.1x realtime | ~0.5x realtime | Alta calidad |
| Whisper large | Muy lento | ~0.3x realtime | Máxima calidad |

> **Nota**: "1x realtime" significa que 1 minuto de audio toma 1 minuto en procesar.

## Linux con GPU NVIDIA

### Requisitos
- CUDA Toolkit 11.8+ o 12.x
- cuDNN 8.x
- Driver NVIDIA compatible

### Instalación

```bash
# Instalación normal con poetry (triton se instala automáticamente)
poetry install

# Verificar CUDA
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Rendimiento en Linux + CUDA

| Modelo | CPU | CUDA (RTX 3080) | CUDA (A100) |
|--------|-----|-----------------|-------------|
| Whisper tiny | ~1x | ~10x realtime | ~20x realtime |
| Whisper base | ~0.5x | ~8x realtime | ~15x realtime |
| Whisper small | ~0.2x | ~5x realtime | ~10x realtime |
| Whisper medium | ~0.1x | ~3x realtime | ~6x realtime |
| Whisper large | Muy lento | ~2x realtime | ~4x realtime |

## Modelos de Audio y GPU

### Whisper (Transcripción)
- **Usa GPU**: Sí (CUDA o MPS)
- **Impacto**: Alto - transcripción 5-20x más rápida con GPU

### CLAP (Audio Embeddings)
- **Usa GPU**: Sí (CUDA o MPS)
- **Impacto**: Medio - embeddings más rápidos con GPU

### YAMNet (Audio Embeddings)
- **Usa GPU**: Sí (requiere TensorFlow)
- **Impacto**: Bajo - modelo pequeño, CPU es suficiente
- **Nota**: Opcional, requiere `poetry install --extras yamnet`

### Sentence Transformers (Text Embeddings)
- **Usa GPU**: Sí (CUDA o MPS)
- **Impacto**: Bajo - textos cortos se procesan rápido en CPU

## Variables de Entorno

```bash
# Forzar uso de CPU (ignorar GPU)
export CUDA_VISIBLE_DEVICES=""

# Seleccionar GPU específica (Linux multi-GPU)
export CUDA_VISIBLE_DEVICES="0"

# Habilitar MPS en macOS (habilitado por defecto)
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Troubleshooting

### "No module named 'triton'" en macOS
**Esperado.** Triton no está disponible para macOS. El código funciona sin él.

### "MPS backend not available"
```bash
# Verificar versión de macOS (requiere 12.3+)
sw_vers

# Verificar PyTorch con soporte MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

### GPU NVIDIA no detectada en Linux
```bash
# Verificar driver
nvidia-smi

# Verificar CUDA en PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Si falla, reinstalar PyTorch con CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memoria GPU insuficiente
```python
# Usar modelo más pequeño
whisper_model = "tiny"  # en lugar de "base" o "small"

# O forzar CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

## Referencias

- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [OpenAI Triton](https://github.com/openai/triton)
- [Whisper](https://github.com/openai/whisper)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
