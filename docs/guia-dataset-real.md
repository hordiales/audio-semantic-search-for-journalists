# Crear un dataset con audios reales

Esta guía procesa un corpus real de WAV, OPUS u otros formatos de audio para
usarlo en búsqueda y evaluación. No utiliza el fixture creado por
`scripts/generate_test_dataset.py`: ese comando fabrica tres segmentos y
vectores sintéticos, sin leer archivos de audio.

## 1. Instalar el entorno

Desde la raíz del repositorio:

```bash
uv sync
brew install ffmpeg # macOS
cp .env.example .env
```

Para clasificar eventos AudioSet con YAMNet, instalar además:

```bash
uv sync --extra yamnet
```

## 2. Ubicar los audios de origen

Crear un directorio local y colocar allí los archivos. El directorio se lee
en un único nivel: mover los audios a esa carpeta, no a subdirectorios.

```bash
mkdir -p data/audio
cp /ruta/a/entrevista.wav data/audio/
cp /ruta/a/discurso.opus data/audio/
find data/audio -maxdepth 1 -type f
```

Se aceptan WAV, OPUS, MP3, M4A, FLAC, OGG, AAC y WMA. `ffmpeg` los normaliza
a WAV mono de 16 kHz antes de transcribirlos.

## 3. Configurar las rutas

Editar `.env`:

```dotenv
# Entrada: WAV/OPUS/etc. originales.
AUDIO_INPUT_DIR=./data/audio

# Salida: artefactos que producirá la ingesta.
DATASET_OUTPUT=./dataset-real

# Lectura: dataset que cargará el agente, la API y las evaluaciones.
DATASET_PATH=./dataset-real

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

Las tres variables no son equivalentes:

| Variable | Consumidor | Propósito |
|---|---|---|
| `AUDIO_INPUT_DIR` | Pipeline | Audios originales de entrada. |
| `DATASET_OUTPUT` | Pipeline | Directorio nuevo o regenerado por la ingesta. |
| `DATASET_PATH` | Agente/API/evaluación | Dataset ya procesado que se abre para buscar. |

`DATASET_PATH` **no está deprecado**. Normalmente debe apuntar al mismo valor
que `DATASET_OUTPUT` después de terminar una ingesta. Los flags `--input` y
`--output` tienen prioridad sobre las variables de entorno.

## 4. Elegir embeddings y clasificación acústica

El archivo [config/embeddings.toml](../config/embeddings.toml) define qué
modalidades se generan. La configuración inicial es:

```toml
[embeddings]
active = ["text", "clap"]
```

| Modalidad | Qué representa | Artefacto |
|---|---|---|
| `text` | Transcripción en el espacio MiniLM. | `embeddings/text/`, `text_index.faiss` |
| `clap` | Ventanas acústicas en un espacio audio↔texto compartido. | `embeddings/clap/`, `audio_index.faiss` |
| `gemini` | Audio nativo de Gemini Embedding 2; requiere `GEMINI_API_KEY`. | `embeddings/gemini/`, `gemini_audio_index.faiss` |
| `yamnet` | Etiquetas AudioSet como `Applause` o `Music`; no crea índice FAISS. | Columna `yamnet_top_classes` |

Por ejemplo, para producir los tres índices vectoriales:

```toml
[embeddings]
active = ["text", "clap", "gemini"]
```

Para añadir clasificación de eventos:

```toml
[embeddings]
active = ["text", "clap", "yamnet"]
```

Usar un archivo alternativo sin modificar el versionado:

```bash
cp config/embeddings.toml config/embeddings-local.toml
# editar config/embeddings-local.toml
```

## 5. Elegir la granularidad de los segmentos

Whisper siempre genera segmentos iniciales con timestamps. El pipeline puede
conservarlos o reagruparlos antes de calcular embeddings:

| Estrategia | Cuándo usarla | Parámetros relevantes |
|---|---|---|
| `whisper` | Punto de partida; conserva los segmentos nativos. | — |
| `fixed` | Necesitás unidades temporales homogéneas para retrieval. | `--chunk-duration`, `--chunk-overlap` |
| `sentence` | Entrevistas o discursos con frases breves y búsqueda textual precisa. | `--max-chunk-text-chars` |
| `paragraph` | Fragmentos más contextuales para temas largos. | `--max-chunk-text-chars` |

Ejemplos:

```bash
# Ventanas de transcripción de 30 s con 5 s de solapamiento.
--chunk-strategy fixed --chunk-duration 30 --chunk-overlap 5

# Agrupar oraciones hasta 500 caracteres.
--chunk-strategy sentence --max-chunk-text-chars 500
```

Esto es independiente de las ventanas acústicas. Para CLAP, Gemini y YAMNet,
cada segmento se corta en ventanas WAV de hasta 10 s con 2 s de solapamiento;
se puede ajustar con `--audio-window-duration` y `--audio-window-overlap`.
Un segmento largo representa el promedio normalizado de sus ventanas.

## 6. Procesar el corpus

Con `.env` configurado:

```bash
uv run python -m src.simple_dataset_pipeline \
  --whisper-model base \
  --language es \
  --chunk-strategy fixed \
  --chunk-duration 30 \
  --chunk-overlap 5 \
  --audio-window-duration 10 \
  --audio-window-overlap 2 \
  --batch-size 8
```

Para indicar rutas directamente:

```bash
uv run python -m src.simple_dataset_pipeline \
  --input ./data/audio \
  --output ./dataset-real \
  --embeddings-config ./config/embeddings-local.toml
```

Para una prueba estructural rápida, `--mock-audio` evita CLAP real, pero no es
válido para evaluar calidad acústica ni para producción.

## 7. Entender el dataset generado

Con `DATASET_OUTPUT=./dataset-real`, el resultado queda así:

```text
dataset-real/
├── converted/                         # WAV mono 16 kHz normalizados
├── transcriptions/
│   └── segments_metadata.csv           # texto y timestamps de Whisper/re-chunking
├── audio_segments/                     # recortes WAV usados por modelos acústicos
├── embeddings/
│   ├── text/segment_<id>.npy           # MiniLM, si está activo
│   ├── clap/segment_<id>.npy           # CLAP, si está activo
│   └── gemini/segment_<id>.npy         # Gemini, si está activo
├── indices/
│   ├── text_index.faiss
│   ├── audio_index.faiss
│   └── gemini_audio_index.faiss
└── final/
    ├── complete_dataset.pkl            # tabla completa, incluyendo vectores
    ├── dataset_metadata.csv            # tabla inspeccionable sin vectores
    └── dataset_manifest.json           # modelos, dimensiones, chunking y ventanas
```

Los índices sólo aparecen para modalidades activas. `dataset_manifest.json` es
la fuente de verdad de una ejecución: registra `active_embeddings`, el modelo y
dimensión de cada índice, la estrategia de chunking y las ventanas acústicas.

## 8. Validar y usar el dataset

```bash
cat "$DATASET_OUTPUT/final/dataset_manifest.json"

uv run python - <<'PY'
import os
import pandas as pd

path = os.environ["DATASET_PATH"]
df = pd.read_pickle(f"{path}/final/complete_dataset.pkl")
print(df[["segment_id", "original_file_name", "start_time", "end_time", "text"]].head())
print("IDs únicos:", df["segment_id"].is_unique)
PY
```

Cuando la validación sea correcta, iniciar el servicio con el mismo
`DATASET_PATH`:

```bash
uv run uvicorn src.fast_api_app:app --host 0.0.0.0 --port 8000
```

La documentación de especificación conserva detalles de diseño en
`spec/07-guia-preparar-procesar-dataset.md`; esta guía es el procedimiento
operativo para trabajar con archivos reales.
