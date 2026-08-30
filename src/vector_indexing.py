"""Vector indexing module using FAISS."""

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


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
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

    n_vectors, dim = embeddings.shape
    logger.info("Building FAISS IndexFlatIP: %d vectors, %d dimensions", n_vectors, dim)

    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_file))

    logger.info("FAISS index saved to: %s (%d vectors)", output_path, index.ntotal)
    return index


def load_faiss_index(index_path: str) -> faiss.IndexFlatIP:
    """
    Load a FAISS index from disk.

    Args:
        index_path: Path to the .faiss file

    Returns:
        Loaded FAISS index
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(index_path)
    logger.info("Loaded FAISS index from %s (%d vectors)", index_path, index.ntotal)
    return index


def search_faiss_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Busca los top-K vectores más similares.

    Args:
        index: FAISS index
        query_embedding: Query vector (1D or 2D)
        k: Number of results

    Returns:
        (similarities, indices) - arrays of shape (1, k)
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    query_embedding = query_embedding.astype(np.float32)

    k = min(k, index.ntotal)
    similarities, indices = index.search(query_embedding, k)

    return similarities, indices
