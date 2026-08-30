"""Text embeddings module using Sentence Transformers."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class TextEmbeddingModel:
    """Wrapper around Sentence Transformers for text embedding generation."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info("Loading text embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def generate_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate a single text embedding."""
        embedding = self.model.encode(text, normalize_embeddings=normalize)
        return np.array(embedding, dtype=np.float32)

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Genera embeddings de texto para una lista de strings.

        Args:
            texts: Lista de textos a procesar
            batch_size: Tamaño del batch
            normalize: Normalizar vectores a norma L2 = 1

        Returns:
            Array numpy de shape (n_texts, 384)
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        logger.info("Generating embeddings for %d texts (batch_size=%d)", len(texts), batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(embeddings, dtype=np.float32)


def generate_text_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
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
    model = TextEmbeddingModel(model_name)
    return model.generate_embeddings(texts, batch_size=batch_size, normalize=normalize)
