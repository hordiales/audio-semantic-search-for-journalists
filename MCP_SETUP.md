# Configuración del Servidor MCP para Búsqueda de Audio

Este documento explica cómo configurar el servidor MCP (Model Context Protocol) para acceder a las funcionalidades de búsqueda de audio desde Claude Desktop.


WARNING: puede tardar en cargar y que este disponible la tool (tiene carga asincrónica)

## Requisitos Previos

1. **Claude Desktop** instalado
2. **Python 3.8+**
3. **UV** instalado (recomendado) o **Poetry**
4. **Dataset de audio** procesado y disponible en el directorio `dataset/`

## Instalación de UV (Recomendado)

UV es un administrador de paquetes de Python ultrarrápido que reemplaza a pip y poetry:

```bash
# Instalar UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# O en macOS con Homebrew
brew install uv
```

## Configuración con UV (Método Recomendado)

### 1. Configurar el proyecto

```bash
# Desde el directorio raíz del proyecto
cd mcp_server

# Instalar dependencias con UV
uv sync

# Verificar instalación
uv run python --version
```

### 2. Probar el servidor

```bash
# Usar el script de inicio con UV
./start_mcp_uv.sh

# O manualmente
uv run python start_uv.py --dataset-dir ../dataset
```

### 3. Configuración de Claude Desktop con UV

Edita el archivo `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "audio-search": {
      "command": "/ruta/completa/al/proyecto/mcp_server/start_mcp_uv.sh",
      "args": [
        "--dataset-dir",
        "/ruta/completa/al/proyecto/dataset"
      ]
    }
  }
}
```

**Ejemplo completo con UV:**

```json
{
  "mcpServers": {
    "audio-search": {
      "command": "/Users/hordia/dev/master-thesis/audio-semantic-search-for-journalists/mcp_server/start_mcp_uv.sh",
      "args": [
        "--dataset-dir",
        "/Users/hordia/dev/master-thesis/audio-semantic-search-for-journalists/dataset"
      ]
    }
  }
}
```

## Configuración Alternativa con Poetry

Si prefieres usar Poetry en lugar de UV:

### 1. Instalar Poetry

```bash
# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# O en macOS con Homebrew
brew install poetry
```

### 2. Configurar el proyecto

```bash
# Desde el directorio mcp_server
cd mcp_server

# Instalar dependencias con Poetry
poetry install

# Activar el entorno virtual
poetry shell

# Probar el servidor
poetry run python server.py --dataset-dir ../dataset
```

### 3. Configuración de Claude Desktop con Poetry

```json
{
  "mcpServers": {
    "audio-search": {
      "command": "poetry",
      "args": [
        "run",
        "--directory",
        "/ruta/completa/al/proyecto/mcp_server",
        "python",
        "server.py",
        "--dataset-dir",
        "/ruta/completa/al/proyecto/dataset"
      ]
    }
  }
}
```

## Configuración Clásica (Python/pip)

Si no quieres usar UV ni Poetry:

### 1. Crear entorno virtual

```bash
# Desde el directorio mcp_server
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración de Claude Desktop

```json
{
  "mcpServers": {
    "audio-search": {
      "command": "/ruta/completa/al/proyecto/mcp_server/venv/bin/python",
      "args": [
        "/ruta/completa/al/proyecto/mcp_server/server.py",
        "--dataset-dir",
        "/ruta/completa/al/proyecto/dataset"
      ],
      "env": {
        "PYTHONPATH": "/ruta/completa/al/proyecto/src"
      }
    }
  }
}
```

## Ubicación del Archivo de Configuración

El archivo de configuración de Claude Desktop se encuentra en:

WARNING: puede tardar en cargar y que este disponible la tool (tiene carga asincrónica)

**macOS:**

```text
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**

```text
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**

```text
~/.config/Claude/claude_desktop_config.json
```

## Herramientas Disponibles

Una vez configurado, tendrás acceso a las siguientes herramientas desde Claude Desktop:

### 🔍 Búsquedas

- **`semantic_search`** - Búsqueda semántica de texto
  - Parámetros: `query` (texto), `k` (número de resultados)
  - Ejemplo: Buscar segmentos sobre "economía política"

- **`audio_search`** - Búsqueda por palabras clave de audio
  - Parámetros: `query` (término de audio), `k` (número de resultados)
  - Ejemplo: Buscar segmentos con "applause" o "music"

- **`hybrid_search`** - Búsqueda combinada texto + audio
  - Parámetros: `query`, `k`, `text_weight` (peso del texto 0.0-1.0)
  - Ejemplo: Búsqueda híbrida sobre "política" con peso 0.7 para texto

### 🎭 Análisis de Sentimientos

- **`sentiment_search`** - Búsqueda por sentimiento
  - Parámetros: `sentiment` (positive/negative/neutral/joy/anger/etc.), `k`
  - Ejemplo: Encontrar segmentos con sentimiento "positive"

- **`mood_search`** - Búsqueda con filtro de sentimiento
  - Parámetros: `query` (texto), `sentiment` (filtro), `k`
  - Ejemplo: Buscar "elecciones" con sentimiento "optimistic"

- **`analyze_sentiment`** - Analizar distribución de sentimientos por tema
  - Parámetros: `topic` (tema a analizar)
  - Ejemplo: Analizar sentimientos sobre "economía"

- **`list_sentiments`** - Listar sentimientos disponibles en el dataset
  - Sin parámetros

### 📊 Exploración de Datos

- **`browse_dataset`** - Explorar segmentos aleatorios
  - Parámetros: `count` (número de segmentos)
  - Ejemplo: Ver 10 segmentos aleatorios

- **`dataset_stats`** - Estadísticas del dataset
  - Sin parámetros
  - Muestra duración total, distribución de sentimientos, etc.

- **`find_text`** - Buscar texto específico en transcripciones
  - Parámetros: `text` (texto a buscar)
  - Ejemplo: Encontrar segmentos que contengan "inflación"

- **`get_similar`** - Encontrar segmentos similares
  - Parámetros: `index` (índice de referencia), `k`
  - Ejemplo: Encontrar segmentos similares al índice 150

### 🔧 Sistema

- **`get_capabilities`** - Ver capacidades del sistema
  - Sin parámetros
  - Muestra modelos cargados, índices disponibles, etc.

### 🎵 Reproducción de Audio

- **`play_audio_segment`** - Reproducir segmentos de audio del dataset
  - Parámetros: `source_file` (nombre del archivo), `start_time` (inicio en segundos), `end_time` (fin en segundos), `segment_index` (opcional)
  - Ejemplo: Reproducir un segmento específico encontrado en las búsquedas
  - Detecta automáticamente el reproductor de audio del sistema (ffplay en macOS, vlc/mplayer en Linux, wmplayer en Windows)

## Verificación de la Instalación

### Con UV

```bash
cd mcp_server
./start_mcp_uv.sh --dataset-dir ../dataset
```

### Con Poetry

```bash
cd mcp_server
poetry run python server.py --dataset-dir ../dataset
```

### Con Python/pip

```bash
cd mcp_server
source venv/bin/activate
python server.py --dataset-dir ../dataset
```

Si todo está correcto, deberías ver:

```text
✅ MCP Server initialized with dataset from: ../dataset
📊 Dataset loaded: 304 segments
```

## Ejemplos de Uso

### Búsqueda Semántica

```text
Usa la herramienta semantic_search para buscar segmentos sobre "crisis económica" y muestra los 5 mejores resultados.
```

### Análisis de Sentimientos

```text
Analiza los sentimientos en segmentos relacionados con "política" usando analyze_sentiment.
```

### Búsqueda Híbrida

```text
Haz una búsqueda híbrida de "manifestación" con peso 0.6 para texto y muestra los resultados.
```

### Exploración

```text
Muéstrame las estadísticas generales del dataset usando dataset_stats.
```

### Reproducción de Audio

```text
Busca segmentos sobre "política" y reproduce el primer resultado usando play_audio_segment.
```

```text
Reproduce el segmento del archivo "audio.wav" desde el segundo 10 al 15 usando play_audio_segment.
```

## Solución de Problemas

### Con UV

**Error: "uv: command not found"**

```bash
# Instalar UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# Reiniciar terminal o source ~/.bashrc
```

**Error: "Dependencies not installed"**

```bash
cd mcp_server
uv sync
```

### Con Poetry

**Error: "poetry: command not found"**

```bash
# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

**Error: "Dependencies not installed"**

```bash
cd mcp_server
poetry install
```

### Errores Generales

**Error: "Dataset no encontrado"**

- Verificar que el dataset existe en `dataset/final/complete_dataset.pkl`
- Verificar que la ruta en la configuración es absoluta y correcta

**Error: "Module not found"**

- Con UV: ejecutar `./start_mcp_uv.sh` para instalar automáticamente
- Con Poetry: ejecutar `poetry install` en el directorio mcp_server
- Verificar que las rutas en la configuración son correctas

**Error: "triton installation failed" (M1/M2 Mac)**

- Este error es común en Mac con Apple Silicon
- Solución: usar UV en lugar de Poetry (UV maneja mejor estas dependencias)
- O excluir triton del poetry.lock: `poetry lock --no-cache`

**Error: "No audio player found" (Reproducción de audio)**

- **macOS**: Instalar ffmpeg: `brew install ffmpeg`
- **Linux**: Instalar ffmpeg: `sudo apt install ffmpeg` (Ubuntu/Debian)
- **Windows**: Descargar ffmpeg desde https://ffmpeg.org/download.html
- Alternativamente, el sistema buscará reproductores nativos (afplay en macOS, vlc en Linux)

**Claude Desktop no reconoce las herramientas**

- Verificar que la configuración JSON es válida
- Reiniciar Claude Desktop completamente
- Verificar los logs del sistema para errores del servidor MCP
- Probar primero con: `./start_mcp_minimal.sh` para verificar que MCP funciona

### Verificar conexión MCP

```bash
# Con UV
cd mcp_server && ./start_mcp_uv.sh

# Con Poetry  
cd mcp_server && poetry run python server.py --dataset-dir ../dataset

# Con Python/pip
cd mcp_server && source venv/bin/activate && python server.py --dataset-dir ../dataset
```

## Ventajas de UV vs Poetry vs pip

### UV (Recomendado)

- ✅ **Más rápido**: 10-100x más rápido que pip/poetry
- ✅ **Gestión automática**: Maneja Python y dependencias
- ✅ **Compatible**: Funciona con pyproject.toml existente
- ✅ **Menos configuración**: No requiere activar entornos virtuales

### Poetry

- ✅ **Maduro**: Ecosistema estable y bien documentado
- ✅ **Gestión de dependencias**: Excelente resolución de dependencias
- ✅ **Publicación**: Facilita la publicación de paquetes

### Python/pip

- ✅ **Estándar**: Incluido con Python
- ✅ **Simple**: Configuración mínima
- ❌ **Manual**: Requiere gestión manual de entornos virtuales

## Personalización

Puedes modificar `mcp_server/server.py` para:

- Añadir nuevas herramientas
- Cambiar parámetros por defecto
- Personalizar el formato de respuestas
- Añadir filtros adicionales

¡Ya puedes usar las capacidades de búsqueda de audio directamente desde Claude Desktop con el administrador de paquetes de tu preferencia!