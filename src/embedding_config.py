"""Versioned configuration for embeddings produced during ingestion."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

VALID_EMBEDDINGS = frozenset({"text", "clap", "gemini", "yamnet"})


@dataclass(frozen=True)
class EmbeddingConfig:
    active: frozenset[str]
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clap_model: str = "laion/clap-htsat-unfused"
    gemini_model: str = "gemini-embedding-2"
    gemini_output_dimensionality: int = 1536
    yamnet_model: str = "https://tfhub.dev/google/yamnet/1"
    yamnet_top_k: int = 5

    def is_active(self, embedding_name: str) -> bool:
        return embedding_name in self.active


def load_embedding_config(path: str | Path) -> EmbeddingConfig:
    """Load and validate the explicit list of embeddings to generate."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Embedding configuration not found: {config_path}")
    raw = tomllib.loads(config_path.read_text())
    embeddings = raw.get("embeddings", {})
    active = frozenset(embeddings.get("active", []))
    unsupported = active - VALID_EMBEDDINGS
    if unsupported:
        raise ValueError(f"Unsupported embeddings in {config_path}: {sorted(unsupported)}")
    if not active:
        raise ValueError("embeddings.active must enable at least one embedding")

    text = embeddings.get("text", {})
    clap = embeddings.get("clap", {})
    gemini = embeddings.get("gemini", {})
    yamnet = embeddings.get("yamnet", {})
    dimensions = gemini.get("output_dimensionality", 1536)
    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError("embeddings.gemini.output_dimensionality must be a positive integer")
    yamnet_top_k = yamnet.get("top_k", 5)
    if not isinstance(yamnet_top_k, int) or yamnet_top_k <= 0:
        raise ValueError("embeddings.yamnet.top_k must be a positive integer")
    return EmbeddingConfig(
        active=active,
        text_model=text.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        clap_model=clap.get("model", "laion/clap-htsat-unfused"),
        gemini_model=gemini.get("model", "gemini-embedding-2"),
        gemini_output_dimensionality=dimensions,
        yamnet_model=yamnet.get("model", "https://tfhub.dev/google/yamnet/1"),
        yamnet_top_k=yamnet_top_k,
    )
