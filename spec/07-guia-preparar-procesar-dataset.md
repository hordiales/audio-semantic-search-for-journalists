# Guía paso a paso: preparar y procesar un dataset de audio

**Objetivo:** construir un dataset indexado a partir de archivos de audio para usarlo con el agente de búsqueda. El
resultado incluye transcripciones, embeddings de texto y audio, metadatos e índices FAISS.

## 1. Elegir el tipo de dataset

### Fixture sintético: integración local

El repositorio ya incluye un dataset sintético en `dataset/` . Tiene tres segmentos ficticios e índices FAISS generados
con vectores sintéticos. Sirve para verificar que el agente, las tools y la API cargan un dataset correctamente.

No sirve para medir relevancia ni para producir resultados periodísticos reales.

Para regenerarlo **sólo como fixture sintético** (no lee WAV/OPUS de origen ni
usa `AUDIO_INPUT_DIR`/`DATASET_OUTPUT`):

```bash
uv run python scripts/generate_test_dataset.py --output dataset
```

### Corpus real: evaluación y producción

Para obtener resultados reales, es necesario procesar archivos de audio. El pipeline transcribe los audios, genera
embeddings y construye los índices de búsqueda.

El procedimiento operativo actualizado para WAV/OPUS reales está en
[`docs/guia-dataset-real.md`](../docs/guia-dataset-real.md); esta spec conserva
el contexto de diseño y el fixture de integración.

## 2. Preparar el entorno

Desde la raíz del repositorio, instalar las dependencias bloqueadas:

```bash
uv sync
```

Instalar `ffmpeg`, necesario para normalizar los formatos de audio:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Comprobar las instalaciones:

```bash
uv run python --version
ffmpeg -version
```

El proyecto requiere Python 3.11.

### Activar clasificación de eventos con YAMNet (opcional)

YAMNet identifica clases estandarizadas de AudioSet por ventana de audio (por
ejemplo, `Speech`, `Applause` o `Music`). No sustituye CLAP: CLAP recupera por
similitud texto↔audio y YAMNet añade etiquetas interpretables a cada segmento.

Instalar la dependencia adicional en la máquina que ejecutará la ingesta:

```bash
uv sync --extra yamnet
```

Activar el clasificador en `.env` antes de procesar:

```bash
# .env
EMBEDDINGS_ACTIVE=text,clap
CLASSIFIERS_ACTIVE=yamnet
YAMNET_TOP_K=5
```

El pipeline vuelca estas variables a `config/embeddings.toml` en cada corrida
(`EMBEDDINGS_CONFIG_PATH`); el TOML es un artefacto generado y no debe editarse
a mano.

Las clases se calculan durante la ingesta sobre ventanas WAV mono de 16 kHz y
se agregan por score máximo. Esto evita perder eventos breves. TensorFlow se
instala sólo con el extra porque aumenta de forma importante el entorno local.

## 3. Preparar los archivos de audio

Crear un directorio local para los audios fuente:

```bash
mkdir -p data/audio
```

Copiar allí los archivos que se procesarán. El pipeline acepta formatos que `ffmpeg` pueda leer, por ejemplo `.mp3` ,
`.wav` , `.m4a` , `.flac` y `.ogg` .

```bash
find data/audio -type f
```

Antes de procesar un corpus grande:

- Confirmar que se cuenta con derechos para almacenar y procesar los audios.
- Empezar con 3 a 10 archivos cortos para validar configuración, tiempos y calidad de la transcripción.
- Evitar versionar audios originales, claves o datasets sensibles.

La ubicación del directorio con los audios fuente se puede fijar en `.env` mediante `AUDIO_INPUT_DIR` (ver
`.env.example` ), en vez de pasarla en cada invocación:

```bash
# .env
AUDIO_INPUT_DIR=./data/audio
```

Si se pasa el flag `--input` al ejecutar el pipeline, este tiene prioridad sobre `AUDIO_INPUT_DIR` . Si no se define
ninguno de los dos, el comando falla pidiendo `--input` .

## 4. Elegir el directorio de salida

El directorio de salida contiene artefactos generados. Por defecto se usa `dataset/`:

```bash
DATASET_OUTPUT=dataset
```

Si el directorio contiene el fixture sintético, usar otro directorio para conservar ambos conjuntos:

```bash
DATASET_OUTPUT=dataset-real
```

Al igual que `AUDIO_INPUT_DIR` , `DATASET_OUTPUT` puede definirse en `.env` para no repetirlo en cada corrida. El flag
`--output` , si se pasa, sobrescribe el valor de `DATASET_OUTPUT` ; si no se define ninguno de los dos, el comando falla
pidiendo `--output` .

## 5. Ejecutar el pipeline real

Ejecutar el pipeline con Whisper base y español. Si `AUDIO_INPUT_DIR` y `DATASET_OUTPUT` ya están definidos en `.env` ,
los flags `--input` /`--output` pueden omitirse; se muestran aquí de forma explícita:

```bash
uv run python src/simple_dataset_pipeline.py \
  --input data/audio \
  --output "$DATASET_OUTPUT" \
  --whisper-model base \
  --language es \
  --batch-size 8
```

Opciones relevantes:

| Opción | Valor sugerido | Uso |
| --- | ---: | --- |
| `--input` | `data/audio` | Directorio con audios fuente. Si se omite, se usa `AUDIO_INPUT_DIR` del `.env`; si tampoco está definida, el comando falla. |
| `--output` | `dataset-real` | Directorio donde se escribirá el dataset procesado. Si se omite, se usa `DATASET_OUTPUT` del `.env`; si tampoco está definida, el comando falla. |
| `--whisper-model` | `base` | Modelo de transcripción; usar `tiny` para una prueba rápida, o `small`/`medium` para más precisión y mayor coste. |
| `--language` | `es` | Fuerza español cuando el corpus es principalmente hispanohablante. Omitirlo para detección automática. |
| `--batch-size` | `8` | Cantidad de textos procesados en cada lote; reducirlo si falta memoria. |
| `--mock-audio` | — | Solo para desarrollo: evita CLAP real y crea embeddings acústicos aleatorios. No usar para evaluación o producción. |
| `--verbose` | — | Habilita logs detallados para diagnosticar errores. |

Para una validación rápida sin cargar CLAP:

```bash
uv run python src/simple_dataset_pipeline.py \
  --input data/audio \
  --output dataset-prueba \
  --whisper-model tiny \
  --language es \
  --mock-audio \
  --verbose
```

## 6. Qué procesa el pipeline

El pipeline ejecuta estas etapas:

1. Convierte los audios a WAV mono de 16 kHz mediante `ffmpeg`.
2. Transcribe cada archivo con Whisper y produce segmentos con timestamps.
3. Genera embeddings semánticos del texto con Sentence Transformers.
4. Genera embeddings texto-audio con CLAP.
5. Si YAMNet está activo, clasifica eventos acústicos y guarda las mejores
   clases por segmento.
6. Calcula sentimiento de la transcripción.
7. Construye índices FAISS para texto y audio.
8. Guarda el dataset final y un manifiesto de configuración.

## 7. Verificar la salida

La estructura esperada es:

```text
dataset-real/
├── converted/                         # Audio normalizado
├── transcriptions/
│   └── segments_metadata.csv
├── embeddings/
│   ├── text_embeddings/
│   └── audio_embeddings/
├── indices/
│   ├── text_index.faiss
│   └── audio_index.faiss
└── final/
    ├── complete_dataset.pkl
    ├── dataset_metadata.csv
    └── dataset_manifest.json
```

Comprobar que los archivos esenciales existen:

```bash
find "$DATASET_OUTPUT" -maxdepth 3 -type f | sort
```

Inspeccionar el manifiesto:

```bash
cat "$DATASET_OUTPUT/final/dataset_manifest.json"
```

Inspeccionar los primeros segmentos y campos:

```bash
uv run python - <<'PY'
import os
import pandas as pd

dataset_path = os.environ.get("DATASET_OUTPUT", "dataset-real")
df = pd.read_pickle(f"{dataset_path}/final/complete_dataset.pkl")
print(f"segmentos: {len(df)}")
print(df[["segment_id", "original_file_name", "start_time", "end_time", "text"]].head())
print(df["segment_id"].is_unique)
PY
```

El último valor debe ser `True` . Si hay IDs duplicados, no usar el dataset para evaluación o producción hasta corregir
la identidad de segmentos.

## 8. Conectar el dataset al agente

Configurar la ruta en `.env`:

```bash
DATASET_PATH=./dataset-real
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

Ejecutar una consulta local:

```bash
agents-cli run "Busca menciones sobre inflación y cita archivo y timestamps"
```

Para eventos acústicos:

```bash
agents-cli run "Busca aplausos durante un discurso e indica el momento del audio"
```

Para inspeccionar clases YAMNet de un resultado, pedir al agente que identifique
las clases del `segment_id` encontrado. La tool `obtener_clases_audio` devuelve
etiquetas AudioSet en inglés y su score de clasificación; no son porcentajes de
similitud de CLAP.

## 9. Preparar el dataset de evaluación

Después de comprobar que el agente funciona con el corpus real:

1. Seleccionar como mínimo 30 consultas representativas.
2. Incluir contenido textual, entrevistas, discursos, podcasts y eventos acústicos.
3. Anotar los `segment_id` relevantes para cada consulta.
4. Redactar una respuesta de referencia y contextos relevantes para RAGAS.
5. Ejecutar `agents-cli eval generate` y `agents-cli eval grade` con los casos de `tests/eval/`.

No evaluar calidad semántica con el fixture sintético: sus embeddings no representan el contenido de los audios.

## 10. Problemas frecuentes

| Problema | Comprobación / solución |
| --- | --- |
| `ffmpeg not found` | Instalar `ffmpeg` y verificar con `ffmpeg -version`. |
| Sin archivos convertidos | Confirmar que `data/audio` contiene archivos soportados y legibles. |
| `--input`/`--output` requerido | Definir `AUDIO_INPUT_DIR`/`DATASET_OUTPUT` en `.env`, o pasar los flags explícitamente en cada corrida. |
| Memoria insuficiente | Reducir `--batch-size`, usar un modelo Whisper menor o procesar menos archivos. |
| CLAP falla o tarda demasiado | Probar primero `--mock-audio`; para producción, revisar la instalación de PyTorch/CLAP y la disponibilidad de GPU. |
| YAMNet no está disponible | Ejecutar `uv sync --extra yamnet`, definir `CLASSIFIERS_ACTIVE=yamnet` en `.env` y reprocesar el corpus. |
| El agente no encuentra dataset | Verificar `DATASET_PATH` y la presencia de `final/complete_dataset.pkl` e `indices/*.faiss`. |
| Resultados poco relevantes | Revisar transcripciones, tamaño del corpus, anotaciones y elegir un modelo Whisper mayor antes de modificar umbrales de evaluación. |

## Resultado esperado

Al finalizar, `DATASET_PATH` apunta a un directorio que contiene el dataset, sus metadatos y ambos índices FAISS. Ese
directorio es la entrada para el agente local, la API, la evaluación y el snapshot de producción.
