"""Pipeline de ingesta: audio → dataset indexado para búsqueda semántica."""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio_conversion import convert_directory
from src.audio_transcription import transcribe_directory
from src.clap_audio_embeddings import CLAPConfig, CLAPEmbedding
from src.sentiment_analysis import SentimentAnalyzer
from src.text_embeddings import TextEmbeddingModel
from src.vector_indexing import build_faiss_index

logger = logging.getLogger(__name__)


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
    5. Análisis de sentimiento
    6. Indexación vectorial FAISS
    7. Serialización del dataset final
    """
    setup_logging(verbose)

    output = Path(output_dir)
    converted_dir = output / "converted"
    transcriptions_dir = output / "transcriptions"
    text_emb_dir = output / "embeddings" / "text_embeddings"
    audio_emb_dir = output / "embeddings" / "audio_embeddings"
    indices_dir = output / "indices"
    final_dir = output / "final"

    for d in [converted_dir, transcriptions_dir, text_emb_dir, audio_emb_dir, indices_dir, final_dir]:
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

    logger.info("Total segments: %d", len(segments))

    # Build DataFrame from segments
    df = pd.DataFrame([asdict(s) for s in segments])

    # Save transcription metadata
    csv_path = transcriptions_dir / "segments_metadata.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved transcription metadata to %s", csv_path)

    # --- Etapa 3: Embeddings de texto ---
    logger.info("=" * 60)
    logger.info("ETAPA 3: Embeddings de texto (Sentence Transformers)")
    logger.info("=" * 60)
    text_model = TextEmbeddingModel()
    texts = df["text"].tolist()
    text_embeddings = text_model.generate_embeddings(texts, batch_size=batch_size)

    # Save individual embeddings
    for i, emb in enumerate(text_embeddings):
        np.save(text_emb_dir / f"segment_{i}_embedding.npy", emb)

    df["text_embedding"] = list(text_embeddings)
    logger.info("Generated %d text embeddings (dim=%d)", len(text_embeddings), text_embeddings.shape[1])

    # --- Etapa 4: Embeddings de audio (CLAP) ---
    logger.info("=" * 60)
    logger.info("ETAPA 4: Embeddings de audio (CLAP)")
    logger.info("=" * 60)

    if mock_audio:
        logger.warning("Using MOCK audio embeddings (testing mode)")
        audio_embeddings = np.random.randn(len(df), 512).astype(np.float32)
        norms = np.linalg.norm(audio_embeddings, axis=1, keepdims=True)
        audio_embeddings = audio_embeddings / norms
    else:
        clap = CLAPEmbedding(CLAPConfig())
        audio_paths = [str(converted_dir / row["original_file_name"]) for _, row in df.iterrows()]
        # For CLAP we need the actual segment audio - use full file as approximation
        # In production, segment audio extraction would be needed
        unique_files = list(set(audio_paths))
        file_to_embedding = {}
        for fpath in unique_files:
            if Path(fpath).exists():
                file_to_embedding[fpath] = clap.generate_embedding(fpath)
            else:
                file_to_embedding[fpath] = np.zeros(512, dtype=np.float32)

        audio_embeddings = np.array(
            [file_to_embedding.get(p, np.zeros(512, dtype=np.float32)) for p in audio_paths],
            dtype=np.float32,
        )

    for i, emb in enumerate(audio_embeddings):
        np.save(audio_emb_dir / f"segment_{i}_clap.npy", emb)

    df["audio_embedding"] = list(audio_embeddings)
    logger.info("Generated %d audio embeddings (dim=512)", len(audio_embeddings))

    # --- Etapa 5: Análisis de sentimiento ---
    logger.info("=" * 60)
    logger.info("ETAPA 5: Análisis de sentimiento")
    logger.info("=" * 60)
    sentiment_analyzer = SentimentAnalyzer()
    sentiments = sentiment_analyzer.analyze_batch(texts, batch_size=batch_size)

    df["sentiment_positive"] = [s.positive for s in sentiments]
    df["sentiment_negative"] = [s.negative for s in sentiments]
    df["sentiment_neutral"] = [s.neutral for s in sentiments]
    df["dominant_sentiment"] = [s.dominant for s in sentiments]
    logger.info("Sentiment analysis complete")

    # --- Etapa 6: Indexación vectorial ---
    logger.info("=" * 60)
    logger.info("ETAPA 6: Indexación FAISS")
    logger.info("=" * 60)
    text_index_path = str(indices_dir / "text_index.faiss")
    audio_index_path = str(indices_dir / "audio_index.faiss")

    build_faiss_index(text_embeddings, text_index_path)
    build_faiss_index(audio_embeddings, audio_index_path)

    # --- Etapa 7: Dataset final ---
    logger.info("=" * 60)
    logger.info("ETAPA 7: Serialización del dataset final")
    logger.info("=" * 60)

    # Save complete dataset
    pkl_path = final_dir / "complete_dataset.pkl"
    df.to_pickle(pkl_path)
    logger.info("Saved complete dataset: %s", pkl_path)

    # Save metadata CSV (without embeddings for inspection)
    meta_cols = [c for c in df.columns if "embedding" not in c]
    df[meta_cols].to_csv(final_dir / "dataset_metadata.csv", index=False)

    # Save manifest
    manifest = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_segments": len(df),
        "total_audio_files": df["original_file_name"].nunique(),
        "whisper_model": whisper_model,
        "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "text_embedding_dim": 384,
        "audio_embedding_model": "laion/clap-htsat-unfused",
        "audio_embedding_dim": 512,
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
    parser = argparse.ArgumentParser(description="Pipeline de ingesta de audio")
    parser.add_argument("--input", required=True, help="Directorio con archivos de audio")
    parser.add_argument("--output", required=True, help="Directorio de salida del dataset")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", default=None, help="Forzar idioma (ej: es, en)")
    parser.add_argument("--mock-audio", action="store_true", help="Usar embeddings mock para CLAP")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        language=args.language,
        mock_audio=args.mock_audio,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
