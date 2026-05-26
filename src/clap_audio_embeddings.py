"""CLAP audio embeddings module for cross-modal audio-text search."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CLAP_EMBEDDING_DIM = 512


@dataclass
class CLAPConfig:
    model_name: str = "laion/clap-htsat-unfused"
    device: str = "cpu"
    cache_dir: str = "/tmp/clap_cache"


class CLAPEmbedding:
    """CLAP model wrapper for generating audio and text embeddings in shared space."""

    def __init__(self, config: CLAPConfig | None = None):
        self.config = config or CLAPConfig()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Lazy-load the CLAP model."""
        import laion_clap

        logger.info("Loading CLAP model: %s (device: %s)", self.config.model_name, self.config.device)
        self._model = laion_clap.CLAP_Module(enable_fusion=False)
        self._model.load_ckpt()
        logger.info("CLAP model loaded successfully")

    @property
    def embedding_dim(self) -> int:
        return CLAP_EMBEDDING_DIM

    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """
        Genera embedding de audio (512-dim, normalizado).

        Args:
            audio_path: Path to audio file

        Returns:
            Normalized numpy array of shape (512,)
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        embedding = self.model.get_audio_embedding_from_filelist(
            [str(audio_file)], use_tensor=False
        )
        embedding = embedding[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding de texto en espacio CLAP (512-dim, normalizado).

        Args:
            text: Text query for cross-modal search

        Returns:
            Normalized numpy array of shape (512,)
        """
        embedding = self.model.get_text_embedding([text], use_tensor=False)
        embedding = embedding[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def generate_batch_audio_embeddings(
        self, audio_paths: list[str], batch_size: int = 8
    ) -> np.ndarray:
        """
        Generate embeddings for multiple audio files.

        Args:
            audio_paths: List of audio file paths
            batch_size: Processing batch size

        Returns:
            Array of shape (n_files, 512)
        """
        all_embeddings = []

        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i : i + batch_size]
            logger.info(
                "CLAP batch %d/%d (%d files)",
                i // batch_size + 1,
                (len(audio_paths) + batch_size - 1) // batch_size,
                len(batch),
            )
            embeddings = self.model.get_audio_embedding_from_filelist(
                batch, use_tensor=False
            )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            all_embeddings.append(embeddings)

        result = np.vstack(all_embeddings).astype(np.float32)
        logger.info("Generated %d CLAP audio embeddings", len(result))
        return result
