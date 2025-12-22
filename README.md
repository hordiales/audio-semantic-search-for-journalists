# Búsqueda Semántica en Audios con fines Periodísticos

Conjunto de aplicaciones para realizar búsqueda semántica multimodal (texto y audio) de contenido de audio hablado con enfoque en aplicaciones periodísticas. Permite la búsqueda analizando el texto y el análisis de sentimiento del mismo, pero también asi buscar en el audio por eventos de la ontología AudioSet (aplausos, gritos, música de fondo, etc)

## Características

- **Embeddings semánticos** de texto con sentence-transformers
- **Embeddings acústicos** con YAMNet según ontología de AudioSet
- **Indexación vectorial** con FAISS
- **Transcripción automática** con OpenAI Whisper
- **MCP server** para consultar desde LLMs
- [IN-PROGRESS] **Construcción del dataset orquestada** con Dagster
- **API Rest** con FastAPI para funcionar como servicio para otras aplicaciones

## Instalación

### Prerequisitos

- **Python 3.11.13** (usando pyenv)
- **Poetry** para gestión de dependencias
- **ffmpeg** para procesamiento de audio

### Instalación con Poetry (Recomendado)

```bash
# 1. Instalar pyenv (si no lo tienes)
# macOS: brew install pyenv
# Linux: https://github.com/pyenv/pyenv#installation

# 2. Instalar Python 3.11.13 con pyenv
pyenv install 3.11.13
pyenv local 3.11.13

# 3. Instalar Poetry (si no lo tienes)
curl -sSL https://install.python-poetry.org | python3 -

# 4. Clonar el repositorio
git clone <url-del-repositorio>
cd audio-semantic-search-for-journalists

# 5. Instalar dependencias con Poetry
poetry install

# 6. Activar el entorno virtual
poetry shell

# 7. (Opcional) Instalar extras para YAMNet
poetry install --extras yamnet
```

### Instalación Alternativa (pip)

Si prefieres usar pip en lugar de Poetry:

```bash
# 1. Configurar Python con pyenv
pyenv install 3.11.13
pyenv local 3.11.13

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) Instalar TensorFlow para YAMNet
pip install tensorflow tensorflow-hub
```

### Instalación de ffmpeg

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

IMPORTANTE: probado con python=3.11.13  
Más detalles y troubleshooting en [[INSTALL.md]]



## 📁 Estructura del proyecto

```
semantic-search-periodismo/
├── requirements.txt           # Dependencias del proyecto
├── audio_transcription.py     # Módulo de transcripción con Whisper
├── text_embeddings.py         # Generación de embeddings de texto
├── audio_embeddings.py        # Generación de embeddings de audio
├── vector_indexing.py         # Indexación vectorial con FAISS
├── semantic_search.py         # Motor de búsqueda principal
├── example_usage.py           # Ejemplos de uso
└── README.md                  # Este archivo
```

TODO: add diagrama de arquitectura

# Audioset ontology



## Uso

De ser necesario ajustar eventos en detect_audio_events.py
```python
        thresholds = {
            'laughter': 0.2,    # Reducido: risas en radio suelen ser más suaves
            'applause': 0.20,    # Muy reducido: era el más alto (0.4), ahora igual que música
            'music': 0.2,        # Mantener: funciona bien
            'singing': 0.25,     # Mantener: funciona bien
            'crowd': 0.18,       # Reducido: ruido de multitud suele ser de fondo
            'speech': 0.4,       # Mantener: debe ser bien detectado
            'cheering': 0.3,    # Reducido: vítores suelen mezclarse con otros sonidos
            'booing': 0.25       # Ligero ajuste: abucheos suelen ser más claros
        }
```
### Crear dataset/corpus

    Ubicar archivos de audio (mp3, ogg, wav, etc) en ./data

    Ejecutar pipeline:
        - Conversión a wav
        - SpeechToText tool
        - Cálculo de embeddings texto
        - Cálculo de embeddings audio
        - Análisis de sentimiento

Detalle de como construirlo en [[DATASET.md]]

Dataset de referencia [Europarl-ST](https://www.mllp.upv.es/europarl-st/) is a multilingual Spoken Language Translation corpus containing paired audio-text samples for SLT from and into 9 European languages, for a total of 72 different translation directions. This corpus has been compiled using the debates held in the European Parliament in the period between 2008 and 2012.
Nota: Este dataset ya contiene las transcripciones (evita el paso de speech2text)



    En ./dataset quedará la siguiente estructura

# Consulta (query) por línea de comando

    $ python src/query_client.py ./dataset --interactive

"""
Sistema híbrido de búsqueda de audio que combina:
1. Búsqueda por palabras clave (siempre funciona)
2. Búsqueda con embeddings YAMNet reales (si están disponibles)
"""


# Configuración

## Config entorno
Revisar módulo config_loader.py
y archivo .env para variables de entorno

dataset/search_config.json 


## Config de consulta

"""
Configuración de parámetros de búsqueda y filtros de score
"""

### Umbrales de score
    min_text_score: float = 0.3
    min_audio_score: float = 0.3
    min_hybrid_score: float = 0.3
    min_keyword_score: float = 0.3
    min_yamnet_score: float = 0.5


### Consulta

Modo interactivo por línea de comando: 
    $ python query_client.py ./dataset --interactive --load-real

# Referencias

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
    Audioset
- [FastAPI](https://fastapi.tiangolo.com/)

## Licencia

Este proyecto está bajo la licencia GPLv3. Ver `LICENSE` para más detalles.
