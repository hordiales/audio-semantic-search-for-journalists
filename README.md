# Búsqueda Semántica en Audios con fines Periodísticos

Conjunto de aplicaciones para realizar búsqueda semántica de contenido de audio hablado con enfoque en aplicaciones periodísticas. 

## Características

- **Embeddings semánticos** de texto con sentence-transformers
- **Embeddings acústicos** con YAMNet según ontología de AudioSet
- **Indexación vectorial** con FAISS
- [IN-PROGRESS] **Transcripción automática** con OpenAI Whisper

- [IN-PROGRESS] **Búsqueda multimodal** combinando texto y audio
- [IN-PROGRESS] **Interfaz web** con Streamlit
- [IN-PROGRESS] **Construcción del dataset orquestada** con Dagster
- [IN-PROGRESS] **API Rest** con FastAPI para funcionar como servicio para otras aplicaciones
- [IN-PROGRESS] **MCP server** para consultar desde LLMs

## Instalación

Recomendación: crear entorno virtual con conda, venv, poetry, etc

```bash
conda create -n AUDIOSEMANTIC
conda activate AUDIOSEMANTIC
```

``
```bash
# Clona el repositorio
git clone <url-del-repositorio>
cd semantic-search-periodismo

# Ejecuta el instalador rápido (app base)
./quick_install.sh
```
IMPORTANTE: probado con python=3.11.13
Más detalles y troubleshootting en [[INSTALL.md]]



## 📁 Estructura del proyecto

```
semantic-search-periodismo/
├── requirements.txt           # Dependencias del proyecto
├── audio_transcription.py     # Módulo de transcripción con Whisper
├── text_embeddings.py         # Generación de embeddings de texto
├── audio_embeddings.py        # Generación de embeddings de audio
├── vector_indexing.py         # Indexación vectorial con FAISS
├── semantic_search.py         # Motor de búsqueda principal
├── streamlit_app.py           # Interfaz web
├── example_usage.py           # Ejemplos de uso
└── README.md                  # Este archivo
```

TODO: add diagrama de arquitectura

# Audioset ontology



## Uso

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

    $ python query_client.py ./dataset --interactive

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
- [Streamlit](https://streamlit.io/)

## Licencia

Este proyecto está bajo la licencia GPLv3. Ver `LICENSE` para más detalles.
