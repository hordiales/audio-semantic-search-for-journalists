import numpy as np

from src.gemini_multimodal_embeddings import _normalize


def test_normalize_returns_unit_vector():
    assert np.allclose(_normalize([3.0, 4.0]), np.array([0.6, 0.8], dtype=np.float32))


def test_normalize_preserves_zero_vector():
    assert np.array_equal(_normalize([0.0, 0.0]), np.array([0.0, 0.0], dtype=np.float32))
