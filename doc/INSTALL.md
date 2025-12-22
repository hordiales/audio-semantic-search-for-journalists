# Guía de Instalación

Esta guía cubre todos los métodos de instalación del sistema de búsqueda semántica de audio.

## ⚠️ Requisito Importante: Python 3.11.13

Este proyecto **requiere exactamente Python 3.11.13**. No se aceptan otras versiones.

Ver [REQUIREMENTS_PYTHON.md](REQUIREMENTS_PYTHON.md) para más detalles sobre por qué se requiere esta versión exacta.

## Método 1: Poetry (Recomendado)

### Prerequisitos

1. **Instalar pyenv** (gestor de versiones de Python)
   ```bash
   # macOS
   brew install pyenv
   
   # Linux
   curl https://pyenv.run | bash
   ```

2. **Configurar pyenv en el shell:**
   
   Añade estas líneas a tu `~/.zshrc` o `~/.bashrc`:
   ```bash
   export PYENV_ROOT="$HOME/.pyenv"
   export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init -)"
   ```
   
   Luego recarga tu shell:
   ```bash
   source ~/.zshrc  # o ~/.bashrc
   ```

3. **Instalar Python 3.11.13 con pyenv**
   ```bash
   pyenv install 3.11.13
   cd audio-semantic-search-for-journalists
   pyenv local 3.11.13
   python --version  # Debe mostrar Python 3.11.13
   ```

4. **Instalar Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   # Añadir Poetry al PATH según las instrucciones mostradas
   ```

### Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd audio-semantic-search-for-journalists

# 2. Instalar dependencias con Poetry
poetry install

# 3. Activar el entorno virtual
poetry shell

# 4. (Opcional) Instalar extras para YAMNet (requiere TensorFlow)
poetry install --extras yamnet
```

### Verificar Instalación

```bash
# Test rápido
poetry run python -c "import whisper, sentence_transformers; print('✅ Instalación OK')"

# Test completo
poetry run python tests/example_usage.py
```

### Comandos Útiles de Poetry

```bash
# Ver información del entorno
poetry env info

# Añadir nueva dependencia
poetry add nombre-paquete

# Actualizar dependencias
poetry update

# Ver dependencias instaladas
poetry show --tree
```

## Método 2: pip + venv

### Prerequisitos

1. **Configurar Python con pyenv**
   ```bash
   pyenv install 3.11.13
   pyenv local 3.11.13
   python --version  # Debe mostrar 3.11.13
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

### Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd audio-semantic-search-for-journalists

# 2. Actualizar pip
pip install --upgrade pip

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) Instalar TensorFlow para YAMNet
pip install tensorflow tensorflow-hub
```

### Verificar Instalación

```bash
# Test rápido
python -c "import whisper, sentence_transformers; print('✅ Instalación OK')"

# Test completo
python tests/example_usage.py
```

## Instalación de Herramientas del Sistema

### ffmpeg (Requerido)

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

**Verificar:**
```bash
ffmpeg -version
```

## Solución de Problemas

### Error: "Python version X.X.X is not supported"

**Solución:**
```bash
# Instalar Python 3.11.13 con pyenv
pyenv install 3.11.13
pyenv local 3.11.13

# Recrear entorno virtual
poetry env remove python3.11
poetry install
```

### Error: "ffmpeg not found"

Ver sección "Instalación de Herramientas del Sistema" arriba.

### Error: "Module not found"

```bash
# Con Poetry
poetry install

# Con pip
pip install -r requirements.txt
```

### Error: "CUDA out of memory"

```python
# Forzar uso de CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# O usar modelos más pequeños
config = {
    'whisper_model': 'tiny',  # En lugar de 'base' o 'medium'
    'use_mock_audio': True
}
```

Para más ayuda, consulta [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Información Adicional

- **Requisitos de Python**: Ver [REQUIREMENTS_PYTHON.md](REQUIREMENTS_PYTHON.md)
- **Changelog Poetry**: Ver [CHANGELOG_POETRY.md](CHANGELOG_POETRY.md)

## Próximos Pasos

Después de la instalación:

1. **Configurar variables de entorno**: Crear archivo `.env` (ver `src/config_loader.py`)
2. **Probar el sistema**: `poetry run python tests/example_usage.py`
3. **Leer documentación**: Ver [QUICK_START.md](QUICK_START.md) para comenzar
4. **Configurar dataset**: Ver [DATASET.md](DATASET.md) para procesar audio
