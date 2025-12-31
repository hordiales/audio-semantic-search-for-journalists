# Servicio de Agente AI con LangChain

## Descripción

Este servicio proporciona una API REST basada en FastAPI que utiliza un agente LangChain para realizar búsquedas semánticas de contenido de audio. El agente interpreta consultas en lenguaje natural y utiliza herramientas especializadas para buscar segmentos de audio relevantes basándose en embeddings semánticos.

## Características

- **Agente Inteligente**: Utiliza LangChain con modelos de OpenAI (GPT-4o-mini por defecto) para interpretar consultas naturales
- **Búsqueda Semántica**: Integración con el sistema de búsqueda semántica existente basado en FAISS
- **API REST**: Endpoints FastAPI para consultas asíncronas
- **Herramientas Especializadas**: El agente tiene acceso a herramientas específicas para búsqueda de audio

## Arquitectura

El servicio está organizado en los siguientes componentes:

```
src/agent_service/
├── __init__.py          # Inicialización del paquete
├── main.py              # Aplicación FastAPI principal
├── agent.py             # Clase AudioAgent con el agente LangChain
├── search_engine.py     # Motor de búsqueda semántica (wrapper de LocalAudioSearch)
└── tools.py             # Herramientas LangChain para búsqueda de audio
```

### Componentes Principales

#### 1. AudioSearchEngine (`search_engine.py`)

Wrapper del motor de búsqueda semántica que replica la funcionalidad de `LocalAudioSearch` del CLI. Proporciona:

- Carga de datasets locales (PKL o CSV)
- Carga de embeddings pre-calculados
- Construcción y uso de índices FAISS
- Búsqueda semántica vectorial
- Generación de embeddings de texto en tiempo real (fallback)

#### 2. Tools (`tools.py`)

Herramientas LangChain que el agente puede usar:

- **buscar_audio**: Realiza búsqueda semántica de segmentos de audio
- **obtener_info_segmento**: Obtiene información detallada de un segmento específico

#### 3. AudioAgent (`agent.py`)

Clase principal que gestiona el agente LangChain:

- Inicialización del motor de búsqueda
- Configuración del modelo LLM (OpenAI)
- Creación del agente con herramientas
- Procesamiento de consultas (síncrono y asíncrono)

#### 4. FastAPI Application (`main.py`)

Aplicación web que expone:

- Endpoint de salud (`/health`)
- Endpoint de consulta (`/query`)
- Endpoint de consulta síncrona (`/query/sync`)
- Gestión del ciclo de vida de la aplicación

## Instalación

### Requisitos

1. Python 3.11.13
2. Dataset generado (ver [SIMPLE_DATASET_PIPELINE.md](../docs/SIMPLE_DATASET_PIPELINE.md))
3. API Key de OpenAI

### Dependencias

Las dependencias de LangChain se agregan automáticamente al `pyproject.toml`:

```toml
langchain = "^0.1.0"
langchain-openai = "^0.0.5"
langchain-core = "^0.1.0"
```

Instalar con Poetry:

```bash
poetry install
```

O con pip:

```bash
pip install langchain langchain-openai langchain-core
```

## Configuración

### Variables de Entorno

Configurar las siguientes variables de entorno:

```bash
# Ruta al dataset (requerido)
export DATASET_PATH=./dataset

# API Key de OpenAI (requerido)
export OPENAI_API_KEY=sk-...

# Modelo de OpenAI a usar (opcional, default: gpt-4o-mini)
export OPENAI_MODEL=gpt-4o-mini
```

### Archivo `.env`

También se puede usar un archivo `.env`:

```env
DATASET_PATH=./dataset
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Uso

### Iniciar el Servicio

#### Con uvicorn directamente:

```bash
poetry run uvicorn src.agent_service.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Con Python directamente:

```bash
poetry run python -m src.agent_service.main
```

#### Con uvicorn desde el directorio raíz:

```bash
cd /Volumes/MacMini2-Extra/audio-semantic-search-for-journalists
poetry run uvicorn src.agent_service.main:app --host 0.0.0.0 --port 8000
```

### Acceder a la Documentación

Una vez iniciado, acceder a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000/

## Endpoints

### GET `/`

Endpoint raíz con información del servicio.

**Respuesta:**
```json
{
  "service": "Audio Semantic Search Agent API",
  "version": "1.0.0",
  "description": "Servicio con agente LangChain para búsqueda semántica de audio"
}
```

### GET `/health`

Health check del servicio.

**Respuesta:**
```json
{
  "status": "healthy",
  "dataset_path": "./dataset",
  "model_name": "gpt-4o-mini",
  "agent_initialized": true
}
```

### POST `/query`

Ejecuta una consulta usando el agente LangChain.

**Request Body:**
```json
{
  "query": "Busca segmentos sobre política económica",
  "max_results": 5
}
```

**Parámetros:**
- `query` (string, requerido): Consulta en lenguaje natural
- `max_results` (integer, opcional): Número máximo de resultados (1-20, default: 5)

**Respuesta:**
```json
{
  "response": "Encontré 5 segmentos relevantes sobre política económica...",
  "query": "Busca segmentos sobre política económica"
}
```

**Ejemplos de consultas:**
- "Busca segmentos sobre política económica"
- "Encuentra audio donde se hable de tecnología"
- "Busca entrevistas relacionadas con ciencia"
- "¿Qué segmentos mencionan inteligencia artificial?"

### GET `/query/sync`

Versión síncrona del endpoint de consulta (para compatibilidad).

**Query Parameters:**
- `query` (string, requerido): Consulta en lenguaje natural
- `max_results` (integer, opcional): Número máximo de resultados (1-20, default: 5)

**Ejemplo:**
```bash
curl "http://localhost:8000/query/sync?query=Busca%20segmentos%20sobre%20tecnología&max_results=3"
```

## Ejemplos de Uso

### Python (con requests)

```python
import requests

# Consulta asíncrona
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "Busca segmentos sobre política económica",
        "max_results": 5
    }
)

result = response.json()
print(result["response"])
```

### cURL

```bash
# Consulta POST
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Busca segmentos sobre tecnología",
    "max_results": 5
  }'

# Consulta GET (síncrona)
curl "http://localhost:8000/query/sync?query=Busca%20segmentos%20sobre%20ciencia&max_results=3"
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'Busca segmentos sobre política económica',
    max_results: 5
  })
});

const result = await response.json();
console.log(result.response);
```

## Funcionalidades del Agente

El agente LangChain está programado para:

1. **Interpretar consultas naturales**: Entiende intenciones y reformula consultas si es necesario
2. **Usar herramientas de búsqueda**: Llama automáticamente a `buscar_audio` cuando detecta una consulta de búsqueda
3. **Formatear resultados**: Presenta los resultados de manera clara y organizada
4. **Proporcionar contexto**: Incluye información relevante como similitud, tiempos, archivos de origen
5. **Sugerir alternativas**: Si no encuentra resultados, sugiere reformulaciones

### Herramientas Disponibles

#### buscar_audio

Realiza búsqueda semántica de segmentos de audio.

**Parámetros:**
- `query`: Texto de búsqueda en lenguaje natural
- `k`: Número de resultados (default: 5)

**Retorna:**
Array JSON con resultados que incluyen:
- `segment_id`: ID del segmento
- `text`: Texto transcrito
- `similarity`: Similitud con la consulta (0-1)
- `similarity_percent`: Similitud en porcentaje
- `start_time`: Tiempo de inicio (segundos)
- `end_time`: Tiempo de fin (segundos)
- `duration`: Duración del segmento
- `original_file_name`: Nombre del archivo de audio
- `language`: Idioma detectado
- `confidence`: Confianza de la transcripción (si disponible)

#### obtener_info_segmento

Obtiene información detallada de un segmento específico.

**Parámetros:**
- `segment_id`: ID numérico del segmento

**Retorna:**
Objeto JSON con toda la información del segmento.

## Comparación con CLI

El servicio de agente proporciona las mismas funcionalidades que `cli_audio_search.py` pero con las siguientes mejoras:

| Funcionalidad | CLI | Servicio Agente |
|--------------|-----|-----------------|
| Búsqueda semántica | ✅ | ✅ |
| Interfaz interactiva | ✅ | ❌ (API REST) |
| Interpretación de consultas | Manual | Automática (LLM) |
| Formateo de resultados | Básico | Inteligente |
| Reproducción de audio | ✅ | ❌ (puede agregarse) |
| Extracción de segmentos | ✅ | ❌ (puede agregarse) |
| Uso programático | ❌ | ✅ |
| Integración con otras apps | ❌ | ✅ |

### Ventajas del Servicio Agente

1. **API REST**: Fácil integración con otras aplicaciones
2. **Interpretación inteligente**: El agente entiende consultas en lenguaje natural
3. **Formateo automático**: Resultados presentados de manera clara
4. **Escalabilidad**: Puede manejar múltiples consultas concurrentes
5. **Extensibilidad**: Fácil agregar nuevas herramientas y funcionalidades

## Extensión del Servicio

### Agregar Nuevas Herramientas

1. Crear función con decorador `@tool` en `tools.py`
2. Agregar la herramienta a la lista retornada por `get_tools()`
3. Actualizar el prompt del agente en `agent.py` si es necesario

**Ejemplo:**

```python
@tool
def nueva_herramienta(param: str) -> str:
    """Descripción de la herramienta para el agente"""
    # Implementación
    return resultado
```

### Modificar el Prompt del Agente

Editar el prompt en `agent.py`, método `initialize()`, variable `prompt`.

### Agregar Endpoints

Agregar nuevos endpoints en `main.py` usando decoradores FastAPI.

## Troubleshooting

### Error: "Dataset no encontrado"

**Solución**: Verificar que `DATASET_PATH` apunta a un directorio con dataset válido. Generar dataset con:

```bash
poetry run python src/simple_dataset_pipeline.py --input data/ --output ./dataset
```

### Error: "OPENAI_API_KEY no configurada"

**Solución**: Configurar la variable de entorno:

```bash
export OPENAI_API_KEY=sk-...
```

### Error: "Agente no inicializado"

**Solución**: Verificar logs del servidor. El agente se inicializa al arrancar. Revisar errores en la inicialización.

### Búsqueda lenta

**Posibles causas:**
- No hay embeddings pre-calculados (genera embeddings primero)
- No hay índice FAISS (se construye automáticamente si hay embeddings)
- Dataset muy grande (considera usar menos resultados)

**Solución**: Generar embeddings e índice FAISS durante la creación del dataset.

## Desarrollo

### Estructura de Código

El código sigue las convenciones del proyecto:

- Type hints en todas las funciones
- Docstrings en español
- Logging estructurado
- Manejo de errores apropiado

### Testing

Para probar el servicio:

```bash
# Health check
curl http://localhost:8000/health

# Consulta de prueba
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Busca segmentos sobre tecnología"}'
```

### Logs

Los logs se configuran en `main.py` y muestran:
- Inicialización del servicio
- Carga de dataset
- Consultas procesadas
- Errores y warnings

## Referencias

- [Documentación de LangChain](https://python.langchain.com/)
- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [CLI Original](../examples/demos/cli_audio_search.py)
- [Pipeline de Dataset](../docs/SIMPLE_DATASET_PIPELINE.md)

## Licencia

GPL-3.0 (igual que el proyecto principal)
