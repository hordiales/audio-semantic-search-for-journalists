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

Para más detalles, ver [docs/INSTALL.md](docs/INSTALL.md).

**⚠️ IMPORTANTE**: Este proyecto requiere exactamente Python 3.11.13.

## 📖 Documentación

### Guías Principales

- **[Instalación](docs/INSTALL.md)** - Guía completa de instalación
- **[Guía Completa: Dataset y Búsqueda](docs/GUIA_DATASET_Y_BUSQUEDA.md)** - 🆕 Paso a paso para generar dataset y usar CLI
- **[Dataset](docs/DATASET.md)** - Crear y procesar datasets
- **[README Principal de Docs](docs/README.md)** - Índice completo de documentación

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

**CLI de búsqueda local con FAISS (Recomendado):**
```bash
# Solo requiere dataset local - no necesita Supabase ni internet
poetry run python examples/demos/cli_audio_search.py ./dataset
```

**CLI alternativo:**
```bash
poetry run python src/query_client.py ./dataset --interactive
```

Ver [docs/GUIA_DATASET_Y_BUSQUEDA.md](docs/GUIA_DATASET_Y_BUSQUEDA.md) para guía completa paso a paso.

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

## 🔍 Pre-commit Hooks (Ruff)

El proyecto incluye pre-commit hooks que ejecutan ruff automáticamente antes de cada commit para mantener la calidad del código:

```bash
# 1. Instalar pre-commit (incluido en poetry install)
poetry install

# 2. Instalar los hooks de git
poetry run pre-commit install

# 3. (Opcional) Ejecutar manualmente en todos los archivos
poetry run pre-commit run --all-files
```

**Qué hace automáticamente:**
- ✅ Ejecuta ruff linting y corrige errores automáticamente
- ✅ Formatea el código con ruff
- ✅ Verifica archivos YAML, JSON, TOML
- ✅ Verifica que no se suban archivos grandes
- ✅ Elimina espacios en blanco al final de líneas

**Si un hook falla:**
- Ruff intenta corregir automáticamente los errores
- Si hay errores que no se pueden corregir automáticamente, el commit se bloquea
- Revisa los errores, corrígelos y vuelve a intentar el commit

Ver [docs/comandos-útiles.md](docs/comandos-útiles.md) para más detalles.

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

## 🔧 Troubleshooting

### Error: `NotImplementedError: Could not run 'aten::empty.memory_format' with arguments from the 'SparseMPS' backend`

**Problema**: Whisper falla al cargarse en MPS (Apple Silicon) debido a limitaciones del backend con operaciones de tensores dispersos.

**Solución automática**: El código detecta este error y automáticamente hace fallback a CPU. Verás un mensaje de advertencia:

```
⚠️  Error cargando modelo en MPS: ...
   Cambiando a CPU (MPS tiene limitaciones con algunas operaciones de Whisper)
```

**Forzar CPU desde el inicio** (opcional):
```bash
export WHISPER_DEVICE=cpu
poetry run python src/simple_dataset_pipeline.py --input data/ --output ./dataset
```

Para más detalles sobre GPU y MPS, ver [docs/GPU_CONSIDERATIONS.md](docs/GPU_CONSIDERATIONS.md).

### Error: `No module named 'triton'` en macOS

**Esperado.** Triton no está disponible para macOS. El código funciona sin él. Si `poetry install` falla por triton:

```bash
# Crear venv (puede fallar en triton, pero crea el venv)
poetry install || true

# Instalar dependencias con pip (ignora triton)
poetry run pip install -r requirements.txt
```

### Error: TensorFlow no disponible (YAMNet)

Si ves el mensaje `⚠️ TensorFlow no disponible. YAMNet no estará disponible.`:

```bash
# Instalar extras para YAMNet
poetry install --extras yamnet
```

### Problemas con Python 3.11.13

Este proyecto requiere exactamente Python 3.11.13. Si tienes otra versión:

```bash
# Con pyenv
pyenv install 3.11.13
pyenv local 3.11.13

# Verificar versión
python --version  # Debe mostrar 3.11.13
```

Para más problemas, ver [docs/GPU_CONSIDERATIONS.md](docs/GPU_CONSIDERATIONS.md) y [docs/INSTALL.md](docs/INSTALL.md).

## 📞 Soporte

- **Documentación**: Ver `docs/` para guías detalladas
- **Problemas de GPU/MPS**: Ver [docs/GPU_CONSIDERATIONS.md](docs/GPU_CONSIDERATIONS.md)
- **Problemas de instalación**: Ver [docs/INSTALL.md](docs/INSTALL.md)
- **Issues**: Abrir un issue en el repositorio

---

**Versión**: 1.0.0
**Python**: 3.11.13 (requerido exactamente)
**Última actualización**: Enero 2025
