# Búsqueda Semántica en Audios con fines Periodísticos

Sistema completo para realizar búsqueda semántica multimodal (texto y audio) de contenido de audio hablado con enfoque en aplicaciones periodísticas. Permite la búsqueda analizando el texto y el análisis de sentimiento del mismo, pero también buscar en el audio por eventos de la ontología AudioSet (aplausos, gritos, música de fondo, etc).

## 🎯 Características

- **Embeddings semánticos** de texto con sentence-transformers
- **Embeddings acústicos** con YAMNet según ontología de AudioSet
- **Múltiples modelos de audio**: YAMNet, CLAP, SpeechDPR
- **Indexación vectorial** con FAISS, Supabase, ChromaDB
- **Transcripción automática** con OpenAI Whisper
- **Análisis de sentimiento** integrado
- **MCP server** para consultar desde LLMs
- **API REST** con FastAPI para funcionar como servicio
- **CLI** para búsqueda interactiva

## 🚀 Inicio Rápido

### Prerequisitos

- **Python 3.11.13** (requerido exactamente) - usar pyenv
- **Poetry** para gestión de dependencias (recomendado)
- **ffmpeg** para procesamiento de audio

### Instalación Rápida

```bash
# 1. Instalar pyenv (si no lo tienes)
# macOS: brew install pyenv
# Linux: curl https://pyenv.run | bash

# 2. Instalar Python 3.11.13
pyenv install 3.11.13
pyenv local 3.11.13

# 3. Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 4. Clonar e instalar
git clone <url-del-repositorio>
cd audio-semantic-search-for-journalists
poetry install  # ⚠️ El venv se crea AUTOMÁTICAMENTE aquí
poetry shell    # Opcional: activar venv (o usar 'poetry run' sin activar)

# 5. (Opcional) Instalar extras para YAMNet
poetry install --extras yamnet
```

Para más detalles, ver [doc/INSTALLATION.md](doc/INSTALLATION.md).

**⚠️ IMPORTANTE**: Este proyecto requiere exactamente Python 3.11.13. Ver [doc/REQUIREMENTS_PYTHON.md](doc/REQUIREMENTS_PYTHON.md) para más información.

## 📖 Documentación

### Guías Principales

- **[Instalación](doc/INSTALLATION.md)** - Guía completa de instalación
- **[Inicio Rápido](doc/QUICK_START.md)** - Empezar en 5 minutos
- **[Arquitectura](doc/ARCHITECTURE_long.md)** - Diseño del sistema
- **[Dataset](doc/DATASET.md)** - Crear y procesar datasets
- **[Troubleshooting](doc/TROUBLESHOOTING.md)** - Solución de problemas

### Interfaces y APIs

- **[API REST](doc/API_README.md)** - Documentación de la API FastAPI
- **[MCP Server](doc/MCP_SETUP.md)** - Integración con LLMs
- **[Aplicaciones](doc/README_APPS.md)** - Guía de todas las interfaces

### Documentación Técnica

- **[Embeddings de Audio](doc/AUDIO_EMBEDDINGS_ARCHITECTURE.md)** - Arquitectura de embeddings
- **[Estrategia de Chunking](doc/ESTRATEGIA_CHUNKING.md)** - Segmentación de audio
- **[Evaluación de Modelos](doc/EMBEDDING_EVALUATION_SYSTEM.md)** - Framework de evaluación

## 💻 Uso

### CLI Interactivo

```bash
poetry run python src/query_client.py ./dataset --interactive
```

### API REST

```bash
# Opción 1: API principal (services/app/main.py)
cd services
poetry run python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Acceder a documentación Swagger
open http://localhost:8080/docs
# O ReDoc: http://localhost:8080/redoc
```

Ver [doc/API_FASTAPI.md](doc/API_FASTAPI.md) para más detalles y opciones.

### Uso Programático

```python
from src.semantic_search import SemanticSearchEngine

engine = SemanticSearchEngine()
results = engine.search("economía y inflación")
```

Ver [doc/QUICK_START.md](doc/QUICK_START.md) para más ejemplos.

## 📁 Estructura del Proyecto

```
audio-semantic-search-for-journalists/
├── src/                    # Código fuente principal
│   ├── audio_transcription.py
│   ├── text_embeddings.py
│   ├── audio_embeddings.py
│   ├── semantic_search.py
│   └── ...
├── benchmarks/             # Scripts de benchmarks y comparación
├── tools/                  # Herramientas y utilidades
│   ├── database/           # Scripts de bases de datos
│   └── setup/              # Scripts de configuración
├── examples/               # Ejemplos y demos
│   └── demos/              # Scripts de demostración
├── scripts/                # Scripts de utilidad general
│   ├── sql/                # Scripts SQL
│   └── shell/              # Scripts shell
├── doc/                    # Documentación
├── tests/                  # Tests
├── mcp_server/            # Servidor MCP
├── services/              # Servicios (GCP, etc.)
├── pyproject.toml         # Configuración Poetry
└── README.md              # Este archivo
```

## 🔧 Configuración

### Variables de Entorno

Crear archivo `.env` en la raíz:

```bash
# APIs opcionales
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Configuración de modelos
DEFAULT_WHISPER_MODEL=base
DEFAULT_AUDIO_EMBEDDING_MODEL=yamnet
USE_MOCK_AUDIO=false

# Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# Por defecto: DEBUG (para desarrollo)
# Para producción, cambiar a INFO
LOG_LEVEL=INFO
```

**Nota sobre LOG_LEVEL**:
- El valor por defecto es `DEBUG` para facilitar el desarrollo
- Para producción o cuando no necesites logs detallados, configura `LOG_LEVEL=INFO` en tu archivo `.env`
- Esto afecta a todos los scripts del proyecto, incluyendo `scripts/fix_ruff_errors.py`

Ver `src/config_loader.py` para todas las opciones.

## 🧪 Testing

```bash
# Ejecutar todos los tests
poetry run pytest

# Test específico
poetry run pytest tests/functional/test_audio_segment_extraction.py
```

## 📊 Modelos Soportados

### Embeddings de Audio
- **YAMNet**: Clasificación general de audio (1024 dim)
- **CLAP**: Búsqueda multimodal audio-texto (512 dim)
- **SpeechDPR**: Dense Passage Retrieval para speech (768 dim)

### Embeddings de Texto
- **Sentence Transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2

### Transcripción
- **OpenAI Whisper**: tiny, base, small, medium, large

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia GPLv3. Ver `LICENSE` para más detalles.

## 🔗 Referencias

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- [FastAPI](https://fastapi.tiangolo.com/)

## 📞 Soporte

- **Documentación**: Ver `doc/` para guías detalladas
- **Problemas**: Ver [doc/TROUBLESHOOTING.md](doc/TROUBLESHOOTING.md)
- **Issues**: Abrir un issue en el repositorio

---

**Versión**: 1.0.0
**Python**: 3.11.13 (requerido exactamente)
**Última actualización**: Enero 2025
