
#  Scripts disponibles para diferentes casos:

  1. Pipeline completo: simple_dataset_pipeline.py
  2. Reanudar desde paso específico: resume_pipeline.py
  3. Solo análisis de sentimientos: cli_sentiment_search.py
  4. Verificar dataset: verify_dataset.py

##  Tools:
  * Transcripciones: Modelo Whisper https://github.com/openai/whisper
    * Robust Speech Recognition via Large-Scale Weak Supervision
  * 


# Estructura final del dataset:

  ./dataset/
  ├── converted/          # Audio convertido
  ├── transcriptions/     # Transcripciones
  ├── embeddings/        # Embeddings de texto
  ├── indices/           # Índices FAISS
  └── final/
      ├── complete_dataset.pkl    # Dataset completo con sentimientos
      └── dataset_manifest.json   # Metadatos

  Para dataset grande (>1000 archivos):

  # Procesar por lotes
  python simple_dataset_pipeline.py -i data/ -o ./dataset --batch_size 50

## RECOMENDACIÓN PRÁCTICA

  1. Para empezar (primeros tests):
  python simple_dataset_pipeline.py \
    --input data/ \
    --output ./test_dataset \
    --whisper-model tiny \
    --mock-audio

  2. Para dataset real pequeño/mediano:
  python simple_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --whisper-model base

  3. Para dataset grande en producción:
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 8 \
    --batch-size 8 \
    --whisper-model base \
    --resume \
    --verbose


# IMPACTO DEL BATCH SIZE Y PROCESAMIENTO EN PARALELO

  ## INFLUENCIA DEL BATCH SIZE

  1. Uso de memoria:
  # Batch pequeño - Menos memoria, más lento
  --batch-size 2    # 2GB-4GB RAM
  --batch-size 4    # 4GB-8GB RAM  
  --batch-size 8    # 8GB-16GB RAM
  --batch-size 16   # 16GB+ RAM

  2. Velocidad de procesamiento:
  - Batch pequeño (2-4): Más lento pero estable
  - Batch mediano (8-16): Balance óptimo
  - Batch grande (32+): Más rápido pero puede causar OOM

  3. Para diferentes componentes:

  | Componente   | Batch Size Recomendado | Memoria Requerida |
  |--------------|------------------------|-------------------|
  | Whisper      | 4-8                    | 4-8GB             |
  | Embeddings   | 16-32                  | 2-4GB             |
  | YAMNet       | 8-16                   | 4-8GB             |
  | Sentimientos | 32-64                  | 1-2GB             |

  ⚡ PROCESAMIENTO EN PARALELO

  1. Pipeline completo con paralelización:
  # Máximo rendimiento
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 8 \
    --batch-size 8 \
    --whisper-model base \
    --verbose

  2. Configuración por número de CPU cores:
  # Para 4 cores
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 4 \
    --batch-size 4

  # Para 8 cores  
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 8 \
    --batch-size 8

  # Para 16 cores
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 12 \
    --batch-size 16

  3. Paralelización específica por etapa:

  Transcripción con Whisper:
  # Múltiples workers para Whisper
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 6 \
    --batch-size 6 \
    --whisper-model base

  Embeddings en paralelo:
  # Para embeddings de texto
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 8 \
    --batch-size 32 \
    --text-model sentence-transformers/all-MiniLM-L6-v2

  YAMNet batch processing:
  # Para audio embeddings
  python yamnet_batch_processor.py \
    --dataset ./dataset \
    --batch-size 16 \
    --workers 4

  🎯 CONFIGURACIONES OPTIMIZADAS

  1. Para máquina con 16GB RAM, 8 cores:
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 6 \
    --batch-size 8 \
    --whisper-model base \
    --text-model sentence-transformers/all-MiniLM-L6-v2 \
    --verbose

  2. Para máquina con 32GB RAM, 16 cores:
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 12 \
    --batch-size 16 \
    --whisper-model medium \
    --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --verbose

  3. Para desarrollo/testing (máquina pequeña):
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 2 \
    --batch-size 2 \
    --whisper-model tiny \
    --mock-audio \
    --verbose

  📊 MONITOREO DEL RENDIMIENTO

  1. Verificar uso de recursos:
  # En otra terminal mientras corre el pipeline
  htop
  # o
  nvidia-smi  # Si usas GPU

  2. Pipeline con métricas:
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 8 \
    --batch-size 8 \
    --verbose 2>&1 | tee pipeline.log

  🔧 PARALELIZACIÓN POR ETAPAS

  1. Solo transcripción en paralelo:
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --workers 8 \
    --batch-size 4 \
    --skip-embeddings

  2. Solo embeddings en paralelo:
  # Después de tener transcripciones
  python text_embeddings.py \
    --dataframe ./dataset/transcriptions/segments_metadata.csv \
    --output ./dataset/embeddings/ \
    --batch-size 32 \
    --workers 4

  3. YAMNet en paralelo:
  python yamnet_batch_processor.py \
    --dataset ./dataset \
    --batch-size 10 \
    --checkpoint-every 100

  ⚠️ CONSIDERACIONES IMPORTANTES

  1. Balance CPU vs GPU:
  - Whisper usa GPU si está disponible
  - YAMNet usa GPU intensivamente
  - Embeddings de texto usan principalmente CPU

  2. Gestión de memoria:
  # Si tienes problemas de memoria
  --batch-size 2 --workers 2

  # Si tienes mucha memoria
  --batch-size 32 --workers 12

  3. Resumir procesamiento:
  # Si el proceso se interrumpe
  python run_dataset_pipeline.py \
    --input data/ \
    --output ./dataset \
    --resume \
    --workers 8 \
    --batch-size 8


# ANÁLISIS DE SENTIMIENTOS

Se analiza el sentimiento del texto
  # Intentar usar un modelo en español si está disponible
  spanish_models = [
      "pysentimiento/robertuito-sentiment-analysis",
      "finiteautomata/beto-sentiment-analysis", 
      "cardiffnlp/twitter-roberta-base-sentiment-latest"
  ]
            

 $ python add_sentiment_to_dataset.py dataset

    - 4 columnas de sentimiento:
      - sentiment_positive
      - sentiment_negative
      - sentiment_neutral
      - dominant_sentiment

  1. Verificar el estado:
  python check_dataset_sentiment.py ./dataset --sample

  2. Usar el modo interactivo:
  python query_client.py ./dataset --interactive --load-real

  3. Comandos disponibles en modo interactivo:
  - sentiment feliz - Buscar por sentimiento específico
  - mood política optimista - Buscar con filtro emocional
  - analyze elecciones - Analizar estado de ánimo de un tema
  - sentiments - Ver todos los sentimientos disponibles
  - stats - Ver estadísticas incluindo sentimientos

  4. Búsquedas directas:
  # Búsqueda por sentimiento
  python cli_sentiment_search.py --load-real --sentiment feliz

  # Búsqueda combinada
  python cli_sentiment_search.py --load-real --query "economía" --filter optimista

  # Análisis de mood
  python cli_sentiment_search.py --load-real --analyze política
