# Instalación del Sistema

## Método 1: Poetry (Recomendado)

### Prerequisitos

1. **Instalar pyenv** (gestor de versiones de Python)
   ```bash
   # macOS
   brew install pyenv
   
   # Linux
   curl https://pyenv.run | bash
   ```

2. **Instalar Python 3.11.13 con pyenv**
   ```bash
   pyenv install 3.11.13
   pyenv local 3.11.13
   python --version  # Debe mostrar 3.11.13
   ```

3. **Instalar Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   # Añadir Poetry al PATH (ver instrucciones al final de la instalación)
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

## Instalación de herramientas del sistema

### ffmpeg (requerido para procesamiento de audio)

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

### Verificar ffmpeg

```bash
ffmpeg -version
```

### Problemas de Instalación

Si encuentras errores, consulta [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para soluciones detalladas.