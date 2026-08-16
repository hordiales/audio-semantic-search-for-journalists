"""Gemini Embedding 2 wrapper for native multimodal audio embeddings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeminiEmbeddingConfig:
    model_name: str = "gemini-embedding-2"
    output_dimensionality: int = 1536
    api_key: str | None = None


def _normalize(values: list[float]) -> np.ndarray:
    embedding = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


class GeminiMultimodalEmbedding:
    """Embed text queries and WAV windows into Gemini's shared vector space."""

    def __init__(self, config: GeminiEmbeddingConfig | None = None):
        self.config = config or GeminiEmbeddingConfig()
        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini comparison benchmark")

        from google import genai

        self._client = genai.Client(api_key=api_key)

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Embed a text query using Gemini's asymmetric search format."""
        return self._embed_content(f"task: search result | query: {query}")

    def generate_audio_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Embed one WAV/MP3 window without a transcription intermediary."""
        from google.genai import types

        path = Path(audio_path)
        mime_type = {".wav": "audio/wav", ".mp3": "audio/mpeg"}.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError(f"Unsupported Gemini audio format: {path.suffix}")
        return self._embed_content(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))

    def _embed_content(self, content) -> np.ndarray:
        from google.genai import types

        result = self._client.models.embed_content(
            model=self.config.model_name,
            contents=content,
            config=types.EmbedContentConfig(
                output_dimensionality=self.config.output_dimensionality
            ),
        )
        if not result.embeddings:
            raise RuntimeError("Gemini returned no embedding")
        return _normalize(result.embeddings[0].values)
