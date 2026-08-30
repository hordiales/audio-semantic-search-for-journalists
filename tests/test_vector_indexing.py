"""Tests for the vector indexing module."""

import tempfile
from pathlib import Path

import numpy as np

from src.vector_indexing import build_faiss_index, load_faiss_index, search_faiss_index


def test_build_and_search_index():
    """Test building an index and searching it."""
    dim = 384
    n_vectors = 100

    embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = str(Path(tmpdir) / "test_index.faiss")
        index = build_faiss_index(embeddings, index_path)

        assert index.ntotal == n_vectors
        assert Path(index_path).exists()

        loaded_index = load_faiss_index(index_path)
        assert loaded_index.ntotal == n_vectors

        query = embeddings[0:1]
        similarities, indices = search_faiss_index(loaded_index, query, k=5)

        assert similarities.shape == (1, 5)
        assert indices.shape == (1, 5)
        assert indices[0][0] == 0
        assert similarities[0][0] > 0.99


def test_search_with_1d_query():
    """Test searching with a 1D query vector."""
    dim = 512
    n_vectors = 50

    embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = str(Path(tmpdir) / "test.faiss")
        index = build_faiss_index(embeddings, index_path)

        query = embeddings[5]  # 1D
        similarities, indices = search_faiss_index(index, query, k=3)

        assert similarities.shape == (1, 3)
        assert indices[0][0] == 5
