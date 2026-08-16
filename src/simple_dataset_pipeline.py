"""Pipeline de ingesta: audio → dataset indexado para búsqueda semántica."""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.audio_conversion import convert_directory
from src.audio_segmenting import build_audio_windows, extract_wav_window
from src.audio_transcription import ChunkingConfig, ChunkingProcessor, transcribe_directory
from src.clap_audio_embeddings import CLAPConfig, CLAPEmbedding
from src.embedding_config import load_embedding_config
from src.gemini_multimodal_embeddings import GeminiEmbeddingConfig, GeminiMultimodalEmbedding
from src.sentiment_analysis import SentimentAnalyzer
from src.text_embeddings import TextEmbeddingModel
from src.vector_indexing import build_faiss_index
from src.yamnet_audio_classifier import (
    YAMNetAudioClassifier,
    YAMNetConfig,
    aggregate_yamnet_classes,
)

logger = logging.getLogger(__name__)


def _pool_audio_embeddings(window_embeddings: list[np.ndarray], dimension: int) -> np.ndarray:
    if not window_embeddings:
        return np.zeros(dimension, dtype=np.float32)
    pooled = np.mean(window_embeddings, axis=0)
    norm = np.linalg.norm(pooled)
    return (pooled / norm if norm > 0 else pooled).astype(np.float32)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(
    input_dir: str,
    output_dir: str,
    whisper_model: str = "base",
    batch_size: int = 8,
    language: str | None = None,
    chunk_strategy: str = "whisper",
    chunk_duration_sec: float = 30.0,
    chunk_overlap_sec: float = 5.0,
    max_chunk_text_chars: int = 500,
    embeddings_config_path: str = "config/embeddings.toml",
    audio_window_duration_sec: float = 10.0,
    audio_window_overlap_sec: float = 2.0,
    mock_audio: bool = False,
    verbose: bool = False,
):
    """
    Ejecuta el pipeline completo de ingesta.

    Etapas:
    1. Conversión de audio a WAV 16kHz mono
    2. Transcripción con Whisper
    3. Generación de embeddings de texto
    4. Generación de embeddings de audio (CLAP)
    5. Clasificación opcional de eventos acústicos con YAMNet
    6. Análisis de sentimiento
    7. Indexación vectorial FAISS
    8. Serialización del dataset final
    """
    if audio_window_duration_sec <= 0:
        raise ValueError("audio_window_duration_sec must be greater than zero")
    if not 0 <= audio_window_overlap_sec < audio_window_duration_sec:
        raise ValueError(
            "audio_window_overlap_sec must be non-negative and smaller than "
            "audio_window_duration_sec"
        )

    setup_logging(verbose)
    embedding_config = load_embedding_config(embeddings_config_path)
    logger.info("Active embeddings: %s", ", ".join(sorted(embedding_config.active)))

    output = Path(output_dir)
    converted_dir = output / "converted"
    transcriptions_dir = output / "transcriptions"
    embeddings_dir = output / "embeddings"
    audio_segments_dir = output / "audio_segments"
    indices_dir = output / "indices"
    final_dir = output / "final"

    for d in [
        converted_dir, transcriptions_dir, embeddings_dir,
        audio_segments_dir, indices_dir, final_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Etapa 1: Conversión ---
    logger.info("=" * 60)
    logger.info("ETAPA 1: Conversión de audio")
    logger.info("=" * 60)
    converted_files = convert_directory(input_dir, str(converted_dir))
    if not converted_files:
        logger.error("No audio files converted. Aborting.")
        sys.exit(1)

    # --- Etapa 2: Transcripción ---
    logger.info("=" * 60)
    logger.info("ETAPA 2: Transcripción con Whisper (model=%s)", whisper_model)
    logger.info("=" * 60)
    segments = transcribe_directory(
        str(converted_dir), model_name=whisper_model, language=language
    )
    if not segments:
        logger.error("No segments transcribed. Aborting.")
        sys.exit(1)

    segments = ChunkingProcessor(ChunkingConfig(
        strategy=chunk_strategy,
        duration_sec=chunk_duration_sec,
        overlap_sec=chunk_overlap_sec,
        max_text_chars=max_chunk_text_chars,
    )).process_segments(segments)
    logger.info("Total segments after %s chunking: %d", chunk_strategy, len(segments))

    # Build DataFrame from segments
    df = pd.DataFrame([asdict(s) for s in segments])

    # Save transcription metadata
    csv_path = transcriptions_dir / "segments_metadata.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved transcription metadata to %s", csv_path)

    texts = df["text"].tolist()
    embeddings: dict[str, np.ndarray] = {}
    if embedding_config.is_active("text"):
        logger.info("Generating text embeddings (%s)", embedding_config.text_model)
        text_model = TextEmbeddingModel(embedding_config.text_model)
        embeddings["text"] = text_model.generate_embeddings(texts, batch_size=batch_size)

    audio_windows: dict[int, list[Path]] = {}
    if embedding_config.active & {"clap", "gemini", "yamnet"}:
        for _, row in df.iterrows():
            source_path = converted_dir / row["original_file_name"]
            if not source_path.exists():
                audio_windows[int(row["segment_id"])] = []
                continue
            import soundfile as sf

            windows = build_audio_windows(
                row["start_time"], row["end_time"], sf.info(source_path).duration,
                window_duration_sec=audio_window_duration_sec,
                overlap_sec=audio_window_overlap_sec,
            )
            paths = []
            for window_number, (window_start, window_end) in enumerate(windows):
                clip_path = audio_segments_dir / f"segment_{row['segment_id']}_window_{window_number}.wav"
                extract_wav_window(source_path, clip_path, window_start, window_end)
                paths.append(clip_path)
            audio_windows[int(row["segment_id"])] = paths
        df["audio_window_count"] = [len(audio_windows[int(segment_id)]) for segment_id in df["segment_id"]]

    if embedding_config.is_active("clap") and mock_audio:
        logger.warning("Using MOCK audio embeddings (testing mode)")
        clap_embeddings = np.random.randn(len(df), 512).astype(np.float32)
        clap_embeddings /= np.linalg.norm(clap_embeddings, axis=1, keepdims=True)
    elif embedding_config.is_active("clap"):
        logger.info("Generating CLAP audio embeddings (%s)", embedding_config.clap_model)
        clap = CLAPEmbedding(CLAPConfig(model_name=embedding_config.clap_model))
        clap_embeddings = []
        for _, row in df.iterrows():
            window_embeddings = [clap.generate_embedding(str(path)) for path in audio_windows[int(row["segment_id"])]]
            clap_embeddings.append(_pool_audio_embeddings(window_embeddings, 512))
        clap_embeddings = np.asarray(clap_embeddings, dtype=np.float32)
    if embedding_config.is_active("clap"):
        embeddings["clap"] = clap_embeddings

    if embedding_config.is_active("gemini"):
        logger.info("Generating Gemini native audio embeddings (%s)", embedding_config.gemini_model)
        gemini = GeminiMultimodalEmbedding(GeminiEmbeddingConfig(
            model_name=embedding_config.gemini_model,
            output_dimensionality=embedding_config.gemini_output_dimensionality,
        ))
        gemini_embeddings = []
        for _, row in df.iterrows():
            window_embeddings = [gemini.generate_audio_embedding(path) for path in audio_windows[int(row["segment_id"])]]
            gemini_embeddings.append(_pool_audio_embeddings(window_embeddings, embedding_config.gemini_output_dimensionality))
        embeddings["gemini"] = np.asarray(gemini_embeddings, dtype=np.float32)

    if embedding_config.is_active("yamnet"):
        logger.info("Classifying acoustic events with YAMNet (%s)", embedding_config.yamnet_model)
        yamnet = YAMNetAudioClassifier(
            YAMNetConfig(
                model_url=embedding_config.yamnet_model,
                top_k=embedding_config.yamnet_top_k,
            )
        )
        df["yamnet_top_classes"] = [
            aggregate_yamnet_classes(
                [yamnet.classify(path) for path in audio_windows[int(segment_id)]],
                top_k=embedding_config.yamnet_top_k,
            )
            for segment_id in df["segment_id"]
        ]

    for name, values in embeddings.items():
        embedding_dir = embeddings_dir / name
        embedding_dir.mkdir(exist_ok=True)
        for segment_id, embedding in zip(df["segment_id"], values, strict=True):
            np.save(embedding_dir / f"segment_{segment_id}.npy", embedding)
        df[f"{name}_embedding"] = list(values)

    # --- Etapa 6: Análisis de sentimiento ---
    logger.info("=" * 60)
    logger.info("ETAPA 6: Análisis de sentimiento")
    logger.info("=" * 60)
    sentiment_analyzer = SentimentAnalyzer()
    sentiments = sentiment_analyzer.analyze_batch(texts, batch_size=batch_size)

    df["sentiment_positive"] = [s.positive for s in sentiments]
    df["sentiment_negative"] = [s.negative for s in sentiments]
    df["sentiment_neutral"] = [s.neutral for s in sentiments]
    df["dominant_sentiment"] = [s.dominant for s in sentiments]
    logger.info("Sentiment analysis complete")

    # --- Etapa 7: Indexación vectorial ---
    logger.info("=" * 60)
    logger.info("ETAPA 7: Indexación FAISS")
    logger.info("=" * 60)
    index_files = {
        "text": "text_index.faiss",
        "clap": "audio_index.faiss",
        "gemini": "gemini_audio_index.faiss",
    }
    for name, filename in index_files.items():
        stale_index = indices_dir / filename
        if name not in embedding_config.active and stale_index.exists():
            stale_index.unlink()
            logger.info("Removed stale disabled index: %s", stale_index)
    for name, values in embeddings.items():
        build_faiss_index(values, str(indices_dir / index_files[name]))

    # --- Etapa 8: Dataset final ---
    logger.info("=" * 60)
    logger.info("ETAPA 8: Serialización del dataset final")
    logger.info("=" * 60)

    # Save complete dataset
    pkl_path = final_dir / "complete_dataset.pkl"
    df.to_pickle(pkl_path)
    logger.info("Saved complete dataset: %s", pkl_path)

    # Save metadata CSV (without embeddings for inspection)
    meta_cols = [c for c in df.columns if "embedding" not in c]
    metadata_df = df[meta_cols].copy()
    if "yamnet_top_classes" in metadata_df:
        metadata_df["yamnet_top_classes"] = metadata_df["yamnet_top_classes"].map(
            lambda classes: json.dumps(classes, ensure_ascii=False)
        )
    metadata_df.to_csv(final_dir / "dataset_metadata.csv", index=False)

    # Save manifest
    manifest = {
        "version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "total_segments": len(df),
        "total_audio_files": df["original_file_name"].nunique(),
        "whisper_model": whisper_model,
        "chunking": {
            "strategy": chunk_strategy,
            "duration_sec": chunk_duration_sec,
            "overlap_sec": chunk_overlap_sec,
            "max_text_chars": max_chunk_text_chars,
        },
        "active_embeddings": sorted(embedding_config.active),
        "embeddings": {name: {"model": getattr(embedding_config, f"{name}_model"), "dimension": int(values.shape[1]), "index": index_files[name]} for name, values in embeddings.items()},
        "classifiers": (
            {
                "yamnet": {
                    "model": embedding_config.yamnet_model,
                    "top_k": embedding_config.yamnet_top_k,
                    "window_aggregation": "max_score",
                }
            }
            if embedding_config.is_active("yamnet")
            else {}
        ),
        "audio_window_duration_sec": audio_window_duration_sec,
        "audio_window_overlap_sec": audio_window_overlap_sec,
        "languages": df["language"].value_counts().to_dict(),
    }
    manifest_path = final_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Saved manifest: %s", manifest_path)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Dataset: %s", output_dir)
    logger.info("Segments: %d | Files: %d", manifest["total_segments"], manifest["total_audio_files"])
    logger.info("=" * 60)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Pipeline de ingesta de audio")
    parser.add_argument(
        "--input",
        default=os.getenv("AUDIO_INPUT_DIR"),
        required=os.getenv("AUDIO_INPUT_DIR") is None,
        help="Directorio con archivos de audio (default: variable de entorno AUDIO_INPUT_DIR)",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("DATASET_OUTPUT"),
        required=os.getenv("DATASET_OUTPUT") is None,
        help="Directorio de salida del dataset (default: variable de entorno DATASET_OUTPUT)",
    )
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", default=None, help="Forzar idioma (ej: es, en)")
    parser.add_argument("--chunk-strategy", default="whisper", choices=["whisper", "fixed", "sentence", "paragraph"])
    parser.add_argument("--chunk-duration", type=float, default=30.0)
    parser.add_argument("--chunk-overlap", type=float, default=5.0)
    parser.add_argument("--max-chunk-text-chars", type=int, default=500)
    parser.add_argument("--embeddings-config", default="config/embeddings.toml",
                        help="TOML con los embeddings activos durante la ingesta")
    parser.add_argument("--audio-window-duration", type=float, default=10.0,
                        help="Duración máxima (s) de cada ventana acústica")
    parser.add_argument("--audio-window-overlap", type=float, default=2.0,
                        help="Solapamiento (s) entre ventanas de un segmento largo")
    parser.add_argument("--mock-audio", action="store_true", help="Usar embeddings mock para CLAP")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        language=args.language,
        chunk_strategy=args.chunk_strategy,
        chunk_duration_sec=args.chunk_duration,
        chunk_overlap_sec=args.chunk_overlap,
        max_chunk_text_chars=args.max_chunk_text_chars,
        embeddings_config_path=args.embeddings_config,
        audio_window_duration_sec=args.audio_window_duration,
        audio_window_overlap_sec=args.audio_window_overlap,
        mock_audio=args.mock_audio,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
