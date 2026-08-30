"""Generate a minimal synthetic dataset compatible with AudioSearchEngine."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.vector_indexing import build_faiss_index


def _normalized_vectors(count: int, dimension: int, seed: int) -> np.ndarray:
    vectors = np.random.default_rng(seed).normal(size=(count, dimension)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def generate(output_dir: Path) -> None:
    """Create a reproducible structural fixture without source audio files."""
    segments = [
        {
            "segment_id": 1001,
            "text": "El ministro afirmó que la inflación interanual bajó durante enero.",
            "start_time": 65.0,
            "end_time": 83.0,
            "original_file_name": "entrevista_economia_enero.wav",
            "language": "es",
            "confidence": 0.96,
            "sentiment_positive": 0.12,
            "sentiment_negative": 0.34,
            "sentiment_neutral": 0.54,
            "dominant_sentiment": "neutral",
        },
        {
            "segment_id": 1002,
            "text": "El público aplaude al finalizar el discurso presidencial.",
            "start_time": 312.5,
            "end_time": 326.0,
            "original_file_name": "discurso_presidencial.wav",
            "language": "es",
            "confidence": 0.93,
            "sentiment_positive": 0.72,
            "sentiment_negative": 0.04,
            "sentiment_neutral": 0.24,
            "dominant_sentiment": "positive",
        },
        {
            "segment_id": 1003,
            "text": "La entrevistada explicó medidas frente al cambio climático.",
            "start_time": 142.0,
            "end_time": 166.0,
            "original_file_name": "podcast_ambiente.wav",
            "language": "es",
            "confidence": 0.95,
            "sentiment_positive": 0.41,
            "sentiment_negative": 0.18,
            "sentiment_neutral": 0.41,
            "dominant_sentiment": "neutral",
        },
    ]

    output_dir = output_dir.resolve()
    indices_dir = output_dir / "indices"
    final_dir = output_dir / "final"
    embeddings_dir = output_dir / "embeddings"
    indices_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(segments)
    text_embeddings = _normalized_vectors(len(dataframe), 384, seed=100)
    audio_embeddings = _normalized_vectors(len(dataframe), 512, seed=200)
    dataframe["text_embedding"] = list(text_embeddings)
    dataframe["audio_embedding"] = list(audio_embeddings)

    dataframe.to_pickle(final_dir / "complete_dataset.pkl")
    dataframe.drop(columns=["text_embedding", "audio_embedding"]).to_csv(
        final_dir / "dataset_metadata.csv", index=False
    )
    build_faiss_index(text_embeddings, str(indices_dir / "text_index.faiss"))
    build_faiss_index(audio_embeddings, str(indices_dir / "audio_index.faiss"))

    manifest = {
        "version": "test-fixture-1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "synthetic": True,
        "total_segments": len(dataframe),
        "total_audio_files": int(dataframe["original_file_name"].nunique()),
        "text_embedding_dim": 384,
        "audio_embedding_dim": 512,
        "notice": "Vectores sintéticos: válido para probar contratos y carga de índices, no calidad semántica.",
    }
    (final_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Dataset sintético creado en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    generate(parser.parse_args().output)
