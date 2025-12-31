# Requisitos de Versión de Python

Este proyecto requiere **Python 3.11.13** exactamente.

## ¿Por qué Python 3.11?

### Compatibilidad con Dependencias

| Dependencia | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
|-------------|-------------|-------------|-------------|-------------|
| laion-clap | ✅ | ✅ | ❌ | ❌ |
| TensorFlow 2.x | ✅ | ✅ | ⚠️ | ❌ |
| openai-whisper | ✅ | ✅ | ⚠️ | ⚠️ |
| torch/torchaudio | ✅ | ✅ | ✅ | ⚠️ |
| sentence-transformers | ✅ | ✅ | ✅ | ✅ |

### Dependencias Críticas

#### 1. laion-clap (Audio-Text Embeddings)
```
Soporta: Python 3.8 - 3.11
NO soporta: Python 3.12+
```
CLAP es fundamental para embeddings multimodales audio-texto. Sin compatibilidad con 3.12+, debemos usar 3.11.

#### 2. TensorFlow (para YAMNet)
```
TensorFlow 2.15: Python 3.9 - 3.11
TensorFlow 2.16+: Python 3.9 - 3.12 (breaking changes)
```
YAMNet requiere TensorFlow. La versión 2.16+ tiene cambios que pueden romper el código existente.

#### 3. openai-whisper
```
Recomendado: Python 3.9 - 3.11
Experimental: Python 3.12
```
Whisper y sus dependencias (especialmente triton) están mejor probadas en 3.11.

#### 4. triton (Compilador GPU)
```
Soporta: Python 3.10 - 3.11 (Linux only)
Issues conocidos: Python 3.12
```
Triton es dependencia de Whisper y torch para aceleración GPU.

## ¿Por qué 3.11.13 Exactamente?

**3.11.13** es la última versión de la serie 3.11, que incluye:
- ✅ Todos los security patches hasta la fecha
- ✅ Correcciones de bugs acumuladas
- ✅ Máxima estabilidad de la serie

### Versiones de Python 3.11

| Versión | Fecha | Estado |
|---------|-------|--------|
| 3.11.0 | Oct 2022 | Obsoleta |
| 3.11.9 | Apr 2024 | Obsoleta |
| **3.11.13** | Jun 2025 | ✅ **Actual** |

## Comparativa de Versiones

### Python 3.10
```
✅ Compatible con todas las dependencias
⚠️ ~10-25% más lento que 3.11
⚠️ Menos features de typing (no ParamSpec, etc.)
```

### Python 3.11 ✅ Recomendado
```
✅ Compatible con todas las dependencias
✅ 10-60% más rápido que 3.10
✅ Mejores mensajes de error
✅ Exception groups
✅ Typing improvements
```

### Python 3.12
```
❌ laion-clap NO compatible
⚠️ TensorFlow requiere 2.16+ (breaking changes)
⚠️ Algunos packages de ML aún no migrados
```

### Python 3.13
```
❌ Muy nuevo (Oct 2024)
❌ Muchas dependencias de ML no soportadas
❌ laion-clap NO compatible
```

## Instalación con pyenv

```bash
# Instalar Python 3.11.13
pyenv install 3.11.13

# Configurar para este proyecto
cd /path/to/project
pyenv local 3.11.13

# Verificar
python --version  # Python 3.11.13
```

## Configuración en pyproject.toml

```toml
[tool.poetry.dependencies]
python = "3.11.13"
```

> **Nota**: Usamos versión exacta (`3.11.13`) en lugar de rango (`^3.11`) para garantizar reproducibilidad.

## ¿Cuándo Podré Usar Python 3.12+?

Cuando estas dependencias actualicen su soporte:

1. **laion-clap**: Esperando PR de compatibilidad 3.12
2. **TensorFlow**: Ya soporta 3.12, pero con breaking changes
3. **triton**: Soporte experimental en 3.12

Estimación: **Q2-Q3 2025** (dependiendo de laion-clap)

## Alternativas si Necesitas Python 3.12+

Si absolutamente necesitas Python 3.12+:

1. **Reemplazar laion-clap** por otro modelo de embeddings audio-texto
2. **Usar solo text embeddings** (sentence-transformers funciona en 3.12+)
3. **Usar Docker** con Python 3.11 para el procesamiento de audio

## FAQ

### ¿Puedo usar Python 3.10?
Sí, pero 3.11 es 10-60% más rápido para código Python intensivo.

### ¿El proyecto funcionará si ignoro la versión?
Probablemente no. `poetry install` fallará al resolver laion-clap.

### ¿Cómo verifico mi versión?
```bash
python --version
# o
pyenv version
```

### ¿Puedo tener múltiples versiones de Python?
Sí, usa pyenv:
```bash
pyenv versions        # Ver instaladas
pyenv install 3.11.13 # Instalar
pyenv local 3.11.13   # Usar en este directorio
```
