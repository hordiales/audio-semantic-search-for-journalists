# 02 — Pipeline de Ingesta

## Resumen

El pipeline de ingesta transforma audio crudo en un dataset indexado listo para búsqueda semántica. Procesa archivos de
audio en batch, generando transcripciones, embeddings de texto y audio, clasificación opcional YAMNet, análisis de
sentimiento e índices vectoriales.

## Entrada y Salida

### Entrada

- Directorio con archivos de audio en cualquier formato soportado
- Formatos: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac`, `.wma`

### Salida

```text
dataset/
├── converted/                  # Audio convertido a WAV 16kHz mono
│   ├── archivo_001.wav
│   └── ...
├── transcriptions/             # Transcripciones por segmento
│   └── segments_metadata.csv   # Metadatos de todos los segmentos
├── embeddings/
│   ├── text/                   # MiniLM, si está activo
│   ├── clap/                   # CLAP, si está activo
│   └── gemini/                 # Gemini Embedding 2, si está activo
├── audio_segments/             # Ventanas WAV para CLAP, Gemini y/o YAMNet
├── segment_clips/              # Un Opus por segmento, para reproducción
│   ├── segment_0.opus
│   └── ...
├── indices/
│   ├── text_index.faiss        # Índice FAISS de texto
│   ├── audio_index.faiss       # Índice FAISS de CLAP
│   └── gemini_audio_index.faiss # Índice Gemini, si está activo
└── final/
    ├── complete_dataset.pkl    # DataFrame completo serializado
    ├── dataset_manifest.json   # Metadatos del dataset
    └── dataset_metadata.csv    # Vista sin embeddings (inspección)
```

## Etapas del Pipeline

### Configuración de modalidades

Las variables de entorno controlan qué se procesa: `EMBEDDINGS_ACTIVE`
(`text,clap[,gemini]`) y `CLASSIFIERS_ACTIVE` (`yamnet`), junto con
`TEXT_EMBEDDING_MODEL`, `CLAP_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_MODEL`,
`GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY`, `YAMNET_MODEL` y `YAMNET_TOP_K`.
Antes de cada corrida, `src/embedding_config.py` las vuelca a
`config/embeddings.toml` (`EMBEDDINGS_CONFIG_PATH`), que el pipeline consume.
**El TOML se regenera en cada ejecución: no se edita a mano.**

`text`, `clap` y `gemini` generan vectores e índices; `yamnet` sólo añade
`yamnet_top_classes` al dataset y el índice invertido
`indices/yamnet_inverted_index.json`. Las modalidades desactivadas no se
procesan y sus índices obsoletos se eliminan de `indices/` al reejecutar el
pipeline.

### Etapa 1: Conversión de Audio

**Responsabilidad**: Normalizar todos los audios a un formato estándar.

**Especificación**:

- Formato de salida: WAV, PCM 16-bit
- Sample rate: 16000 Hz
- Canales: Mono
- Herramienta: ffmpeg (vía subprocess o pydub)

**Interfaz**:

```python
def convert_audio(
    input_path: str,
    output_dir: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> str:
    """
    Convierte audio a formato WAV normalizado.

    Args:
        input_path: Ruta al archivo de audio original
        output_dir: Directorio de salida para archivos convertidos
        sample_rate: Sample rate objetivo (default: 16000)
        mono: Convertir a mono (default: True)

    Returns:
        Ruta al archivo WAV convertido
    """
```

**Criterios de aceptación**:

- Soporta todos los formatos listados
- Maneja errores de archivos corruptos gracefully
- Mantiene mapeo original_file → converted_file

### Etapa 2: Transcripción con Whisper

**Responsabilidad**: Generar transcripciones segmentadas con timestamps.

**Especificación**:

- Modelo: Whisper (configurable: tiny, base, small, medium, large)
- Default: `base` (balance calidad/velocidad)
- Idioma: auto-detección (predominantemente español)
- Salida: Segmentos con texto, timestamps, idioma, confianza

**Interfaz**:

```python
@dataclass
class TranscriptionSegment:
    segment_id: int
    text: str
    start_time: float  # Segundos
    end_time: float  # Segundos
    language: str  # Código ISO (es, en, ...)
    confidence: float  # 0.0 - 1.0
    original_file_name: str  # Nombre del archivo fuente


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
) -> list[TranscriptionSegment]:
    """
    Transcribe audio usando Whisper y retorna segmentos.

    Args:
        audio_path: Ruta al archivo WAV
        model_name: Modelo Whisper a usar
        language: Forzar idioma (None = auto-detección)

    Returns:
        Lista de segmentos transcritos
    """
```

**Criterios de aceptación**:

- Genera segmentos con timestamps precisos (±0.5s)
- Detecta idioma correctamente para español
- Soporta batch processing con checkpointing
- Puede reanudar si el proceso se interrumpe

### Etapa 3: Chunking Configurable (Opcional)

**Responsabilidad**: Re-segmentar los chunks de Whisper según estrategia configurable.

**Especificación**:

| Parámetro | Default | Descripción |
| :-------- | :------ | :---------- |
| `chunk_strategy` | `whisper` | Estrategia: whisper, fixed, sentence, paragraph |
| `chunk_duration_sec` | 30 | Duración máxima para estrategia `fixed` |
| `chunk_overlap_sec` | 5 | Solapamiento entre chunks consecutivos |
| `max_chunk_text_chars` | 500 | Máximo de caracteres por chunk de texto |

**Estrategias**:

1. **`whisper`** (default): Usa los segmentos nativos de Whisper sin modificar
2. **`fixed`**: Divide en ventanas de tiempo fijas con overlap configurable
3. **`sentence`**: Separa por oraciones completas usando NLP
4. **`paragraph`**: Separa por párrafos (útil para transcripciones estructuradas)

**Interfaz**:

```python
@dataclass
class ChunkingConfig:
    strategy: str = "whisper"  # whisper, fixed, sentence, paragraph
    duration_sec: float = 30.0
    overlap_sec: float = 5.0
    max_text_chars: int = 500


class ChunkingProcessor:
    def __init__(self, config: ChunkingConfig): ...

    def process_segments(self, segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
        """
        Aplica estrategia de chunking a segmentos de Whisper.

        Returns:
            Lista de segmentos re-chunked con nuevos IDs y timestamps
        """
```

**Impacto en métricas RAG**:

- Chunks de 20-40s optimizan balance entre contexto y precisión
- Overlap de 5s mejora continuidad semántica (MRR +10% esperado)
- Chunks más uniformes mejoran Precision@K y Recall@K

**Criterios de aceptación**:

- Preserva metadatos originales (archivo fuente, timestamps de audio)
- Genera nuevos `segment_id` secuenciales
- Mantiene continuidad temporal con overlap configurable
- Compatible con checkpointing del pipeline

### Etapa 4: Generación de Embeddings de Texto

**Responsabilidad**: Generar representaciones vectoriales de las transcripciones.

**Especificación**:

- Modelo: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensión del embedding: 384
- Normalización: L2 (para similitud coseno)

**Interfaz**:

```python
def generate_text_embeddings(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """
    Genera embeddings de texto para una lista de strings.

    Args:
        texts: Lista de textos a procesar
        model_name: Modelo Sentence Transformers
        batch_size: Tamaño del batch
        normalize: Normalizar vectores a norma L2 = 1

    Returns:
        Array numpy de shape (n_texts, 384)
    """
```

**Criterios de aceptación**:

- Genera embeddings para todos los segmentos
- Embeddings normalizados (norma L2 = 1)
- Persiste en formato `.npy` por segmento
- Soporta procesamiento por batches para memoria limitada

### Etapa 4: Generación de Embeddings de Audio (CLAP)

**Responsabilidad**: Generar embeddings acústicos en espacio compartido audio-texto.

**Especificación**:

- Modelo: `laion/clap-htsat-unfused` (LAION CLAP)
- Dimensión del embedding: 512
- Espacio compartido: Audio y texto en el mismo espacio vectorial
- Sample rate de entrada: 48000 Hz (internamente, CLAP resamplea)

**Interfaz**:

```python
@dataclass
class CLAPConfig:
    model_name: str = "laion/clap-htsat-unfused"
    device: str = "cpu"  # cpu, cuda, mps
    cache_dir: str = "/tmp/clap_cache"


class CLAPEmbedding:
    def __init__(self, config: CLAPConfig): ...

    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """Genera embedding de audio (512-dim, normalizado)."""

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Genera embedding de texto en espacio CLAP (512-dim, normalizado)."""

    @property
    def embedding_dim(self) -> int:
        """Retorna 512."""
```

**Criterios de aceptación**:

- Embeddings de 512 dimensiones normalizados
- Soporta audio de cualquier duración (segmenta internamente si necesario)
- El embedding de texto de CLAP permite búsqueda cross-modal
- Maneja GPU/CPU transparentemente

### Etapa 5: Clasificación de clases de audio (YAMNet, opcional)

**Responsabilidad**: identificar etiquetas interpretables de AudioSet por
segmento, como `Speech`, `Applause`, `Music` o `Siren`.

**Especificación**:

- Modelo: `https://tfhub.dev/google/yamnet/1` (TensorFlow Hub).
- Entrada: ventanas WAV mono de 16 kHz ya normalizadas por la etapa 1.
- Salida: las `top_k` clases con `class_id`, `class_name` y `score`.
- Agregación: score máximo entre las ventanas del segmento para conservar
  eventos acústicos breves.
- Activación: `CLASSIFIERS_ACTIVE=yamnet` en `.env` (el pipeline regenera
  `[classifiers].active` en el TOML); requiere `uv sync --extra yamnet`.

YAMNet no genera embeddings para FAISS ni reemplaza CLAP. Sus clases se
persisten en `yamnet_top_classes` y el agente las expone mediante
`obtener_clases_audio`.

### Etapa 6: Clips de reproducción por segmento

**Responsabilidad**: producir el audio que el usuario escucha al recibir un
resultado de búsqueda. Es un artefacto de servicio, no de indexación.

**Especificación**:

- Un archivo por segmento: `segment_clips/segment_{segment_id}.opus`.
- Rango exportado: `[start_time - ctx, end_time + ctx]`, recortado a los
  límites del archivo fuente. `ctx` se controla con `--clip-context`
  (o `SEGMENT_CLIP_CONTEXT_SEC`, default 5 s).
- Codec: Opus mono vía ffmpeg, bitrate `--clip-bitrate` (default `32k`),
  ≈14 MB por hora de corpus.
- Columnas agregadas al dataset: `clip_file_name`, `clip_start_time`,
  `clip_end_time`. Los timestamps del clip son más anchos que los del
  segmento: el consumidor necesita ambos pares para posicionar el player.
- Desactivable con `--no-segment-clips`.

**Por qué no se reutilizan las ventanas de `audio_segments/`**: esas ventanas se
solapan a propósito (`--audio-window-overlap`, default 2 s) para que CLAP y
YAMNet no pierdan eventos en los bordes. Reproducidas en secuencia repiten audio
y no coinciden con los timestamps del segmento. Sirven para embeddings, no para
escuchar.

**Incrementalidad**: sólo se generan clips de segmentos nuevos o modificados, y
se eliminan los de segmentos que ya no existen en el dataset. Cambiar el
contexto o el bitrate invalida la firma del pipeline y regenera todo.

**Publicación**: los clips viajan al bucket con el resto de la release
(`scripts/publish_dataset_release.py`), pero el runtime **no** los descarga: se
sirven on-demand con URLs firmadas. Ver `spec/08-frontend-busqueda-directa.md`.

### Etapa 7: Análisis de Sentimiento

**Responsabilidad**: Clasificar sentimiento del texto transcrito.

**Especificación**:

- Modelos (en orden de preferencia):
  1. `pysentimiento/robertuito-sentiment-analysis` (español)
  2. `finiteautomata/beto-sentiment-analysis` (español)
  3. `cardiffnlp/twitter-roberta-base-sentiment-latest` (fallback multilingüe)
- Salida: Scores positive, negative, neutral + dominant

**Interfaz**:

```python
@dataclass
class SentimentResult:
    positive: float  # 0.0 - 1.0
    negative: float  # 0.0 - 1.0
    neutral: float  # 0.0 - 1.0
    dominant: str  # "positive" | "negative" | "neutral"


def analyze_sentiment(text: str) -> SentimentResult:
    """Analiza sentimiento de un texto."""
```

**Criterios de aceptación**:

- Funciona para textos en español
- Scores suman ~1.0
- Batch processing eficiente

### Etapa 8: Indexación Vectorial (FAISS)

**Responsabilidad**: Construir índices vectoriales para búsqueda rápida.

**Especificación**:

- Tipo de índice: `IndexFlatIP` (producto interno = coseno con vectores normalizados)
- Índices separados: texto (384-dim) y audio (512-dim)
- Serialización: archivo `.faiss`

**Interfaz**:

```python
def build_faiss_index(
    embeddings: np.ndarray,
    output_path: str,
) -> faiss.IndexFlatIP:
    """
    Construye y persiste un índice FAISS.

    Args:
        embeddings: Array (n_vectors, dim), ya normalizado
        output_path: Ruta para serializar el índice

    Returns:
        Índice FAISS construido
    """


def search_faiss_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Busca los top-K vectores más similares.

    Returns:
        (similarities, indices) - arrays de shape (1, k)
    """
```

**Criterios de aceptación**:

- Índice construido y serializado correctamente
- Búsqueda retorna top-K con scores correctos
- Índice cargable desde disco sin reconstruir

## Orquestación del Pipeline

### Comando Principal

```bash
uv run python src/simple_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --whisper-model base \
    --batch-size 8 \
    --verbose
```

`--input` y `--output` pueden omitirse si `AUDIO_INPUT_DIR` y `DATASET_OUTPUT`
están definidos en `.env` (los flags tienen prioridad). Ver spec 07.

### Parámetros de Configuración

| Parámetro | Default | Descripción |
| :-------- | :------ | :---------- |
| `--input` | (requerido) | Directorio con archivos de audio |
| `--output` | (requerido) | Directorio de salida del dataset |
| `--whisper-model` | `base` | Modelo Whisper (tiny/base/small/medium/large) |
| `--chunk-strategy` | `whisper` | Estrategia: whisper/fixed/sentence/paragraph |
| `--chunk-duration` | `30` | Duración de chunks en segundos (para fixed) |
| `--chunk-overlap` | `5` | Solapamiento entre chunks en segundos |
| `--max-chunk-text-chars` | `500` | Límite de caracteres de sentence/paragraph |
| `--embeddings-config` | `EMBEDDINGS_CONFIG_PATH` | Ruta del TOML autogenerado con las modalidades y modelos activos |
| `--audio-window-duration` | `10` | Ventana WAV máxima por segmento |
| `--audio-window-overlap` | `2` | Solapamiento de ventanas acústicas |
| `--clip-context` | `5` | Contexto (s) antes y después del segmento en el clip |
| `--clip-bitrate` | `32k` | Bitrate del encoder Opus de los clips |
| `--no-segment-clips` | `false` | No generar clips de reproducción |
| `--batch-size` | `8` | Tamaño del batch para procesamiento |
| `--mock-audio` | `false` | Usar embeddings mock (testing) |
| `--verbose` | `false` | Logging detallado |

### Reejecución incremental

La ingesta es **incremental**: el pipeline conserva un estado en
`final/ingestion_state.json` y sólo reprocesa lo que cambió.

- **Fingerprint por archivo**: cada audio fuente se identifica por `sha256` +
  tamaño (`_file_fingerprint`), de modo que reemplazar un archivo conservando
  el nombre se detecta y se reprocesa ese archivo.
- **Firma del pipeline**: `_pipeline_signature` registra los parámetros que
  invalidan artefactos previos (modelo Whisper, idioma, estrategia de chunking,
  ventanas acústicas, clips, `mock_audio` y la configuración de
  embeddings/clasificadores). Cambiar cualquiera de ellos regenera lo afectado;
  agregar clips o cambiar contexto/bitrate reprocesa todo el corpus.
- **Limpieza**: se eliminan clips huérfanos de segmentos que ya no existen y
  los índices de modalidades desactivadas.
- **Log de corrida**: cada ejecución registra parámetros y resultados para
  auditoría en `final/process_run.json` (`_write_process_run_log`).

No existe un comando `resume_pipeline.py`: basta reejecutar el mismo comando y
el estado incremental decide qué trabajo falta.

### Publicación del release

`scripts/publish_dataset_release.py` sube el dataset procesado a GCS como un
release versionado (`gs://<bucket>/releases/<fecha>`), incluyendo índices,
manifiesto y `segment_clips/`. El runtime lo descarga con
`DATASET_GCS_URI` + `DATASET_PATH` (`src/dataset_storage.py`, snapshot atómico),
excepto los clips, que se sirven on-demand con URLs firmadas V4
(`src/segment_clip_storage.py`). Ver spec 05 y spec 08 §6.

## Configuración de Recursos

### Recomendaciones por Hardware

| Hardware | workers | batch_size | whisper_model | RAM estimada |
| ---------- | --------- | ------------ | --------------- | ------------- |
| 8GB RAM, 4 cores | 2 | 4 | tiny/base | ~4-6GB |
| 16GB RAM, 8 cores | 6 | 8 | base | ~8-12GB |
| 32GB RAM, 16 cores | 12 | 16 | medium | ~16-24GB |
| GPU disponible | 4 | 8 | medium | ~8GB VRAM |

### Tiempos Estimados

Para un corpus de ~1000 archivos de audio (duración media 5 min):

- Conversión: ~10 min
- Transcripción (base): ~60-90 min (CPU), ~15 min (GPU)
- Chunking: ~2-5 min (depende de estrategia)
- Embeddings texto: ~5 min
- Embeddings CLAP: ~30 min (CPU), ~10 min (GPU)
- Sentimiento: ~10 min
- Indexación: ~1 min

## Formato del Dataset Final

### `complete_dataset.pkl`

DataFrame pandas serializado con pickle. Columnas:

| Columna | Tipo | Descripción |
| --------- | ------ | ------------- |
| `segment_id` | int | ID único |
| `text` | str | Transcripción |
| `start_time` | float | Inicio en segundos |
| `end_time` | float | Fin en segundos |
| `original_file_name` | str | Archivo de audio original |
| `language` | str | Idioma detectado |
| `confidence` | float | Confianza transcripción |
| `text_embedding` | np.ndarray | Vector MiniLM, si `text` está activo |
| `clap_embedding` | np.ndarray | Vector CLAP, si `clap` está activo |
| `gemini_embedding` | np.ndarray | Vector Gemini, si `gemini` está activo |
| `audio_window_count` | int | Número de ventanas WAV que representan el segmento |
| `clip_file_name` | str | Clip de reproducción (`segment_{id}.opus`), si están habilitados |
| `clip_start_time` | float | Inicio del clip: `start_time` menos el contexto |
| `clip_end_time` | float | Fin del clip: `end_time` más el contexto |
| `yamnet_top_classes` | list[dict] | Clases AudioSet por segmento (opcional) |
| `sentiment_positive` | float | Score positivo |
| `sentiment_negative` | float | Score negativo |
| `sentiment_neutral` | float | Score neutral |
| `dominant_sentiment` | str | Sentimiento dominante |

### `dataset_manifest.json`

```json
{
    "version": "1.0",
    "created_at": "2025-01-15T10:30:00",
    "total_segments": 5000,
    "total_audio_files": 1447,
    "whisper_model": "base",
    "chunking": {"strategy": "fixed", "duration_sec": 30, "overlap_sec": 5},
    "active_embeddings": ["text", "clap"],
    "active_classifiers": ["yamnet"],
    "embeddings": {
        "text": {"model": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384, "index": "text_index.faiss"},
        "clap": {"model": "laion/clap-htsat-unfused", "dimension": 512, "index": "audio_index.faiss"}
    },
    "classifiers": {
        "yamnet": {"model": "https://tfhub.dev/google/yamnet/1", "top_k": 5}
    },
    "languages": {"es": 4800, "en": 200},
    "audio_window_duration_sec": 10,
    "audio_window_overlap_sec": 2,
    "segment_clips": {
        "enabled": true,
        "directory": "segment_clips",
        "format": "opus",
        "codec": "libopus",
        "bitrate": "32k",
        "context_sec": 5.0,
        "naming": "segment_{segment_id}.opus",
        "count": 5000,
        "total_bytes": 186000000,
        "served": "on-demand"
    }
}
```
