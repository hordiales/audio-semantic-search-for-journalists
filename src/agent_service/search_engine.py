"""Audio search engine with FAISS-based semantic search."""

import json
import logging
from pathlib import Path

import faiss
import pandas as pd

from src.clap_audio_embeddings import CLAPEmbedding
from src.text_embeddings import TextEmbeddingModel
from src.vector_indexing import load_faiss_index, search_faiss_index

logger = logging.getLogger(__name__)


class AudioSearchEngine:
    """Motor de búsqueda semántica sobre el corpus de audio indexado."""

    def __init__(self, dataset_path: str):
        """
        Carga dataset, embeddings e índices FAISS.

        Args:
            dataset_path: Path to the processed dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self._df: pd.DataFrame | None = None
        self._text_index: faiss.IndexFlatIP | None = None
        self._audio_index: faiss.IndexFlatIP | None = None
        self._text_model: TextEmbeddingModel | None = None
        self._clap_model: CLAPEmbedding | None = None
        self._active_embeddings: set[str] | None = None

        self._load_dataset()
        self._load_indices()

    def _load_dataset(self):
        """Load the complete dataset."""
        pkl_path = self.dataset_path / "final" / "complete_dataset.pkl"
        if pkl_path.exists():
            self._df = pd.read_pickle(pkl_path)
            logger.info("Loaded dataset: %d segments from %s", len(self._df), pkl_path)
            manifest_path = self.dataset_path / "final" / "dataset_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                active_embeddings = manifest.get("active_embeddings")
                if active_embeddings is not None:
                    self._active_embeddings = set(active_embeddings)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {pkl_path}. Run the ingestion pipeline first."
            )

    def _load_indices(self):
        """Load FAISS indices."""
        indices_dir = self.dataset_path / "indices"

        text_index_path = indices_dir / "text_index.faiss"
        if self._is_embedding_active("text") and text_index_path.exists():
            self._text_index = load_faiss_index(str(text_index_path))
            logger.info("Loaded text index: %d vectors", self._text_index.ntotal)

        audio_index_path = indices_dir / "audio_index.faiss"
        if self._is_embedding_active("clap") and audio_index_path.exists():
            self._audio_index = load_faiss_index(str(audio_index_path))
            logger.info("Loaded audio index: %d vectors", self._audio_index.ntotal)

    def _is_embedding_active(self, embedding_name: str) -> bool:
        """Respect manifest configuration while retaining legacy dataset support."""
        return self._active_embeddings is None or embedding_name in self._active_embeddings

    @property
    def text_model(self) -> TextEmbeddingModel:
        if self._text_model is None:
            self._text_model = TextEmbeddingModel()
        return self._text_model

    @property
    def clap_model(self) -> CLAPEmbedding:
        if self._clap_model is None:
            self._clap_model = CLAPEmbedding()
        return self._clap_model

    @property
    def total_segments(self) -> int:
        return len(self._df) if self._df is not None else 0

    def search_semantic(self, query_text: str, k: int = 5) -> list[dict]:
        """
        Búsqueda semántica por texto (transcripciones).

        Args:
            query_text: Natural language query
            k: Number of results to return

        Returns:
            List of dicts with segment info and similarity scores
        """
        if self._text_index is None:
            raise RuntimeError("Text index not loaded")

        query_embedding = self.text_model.generate_embedding(query_text, normalize=True)
        similarities, indices = search_faiss_index(self._text_index, query_embedding, k=k)

        results = []
        for sim, idx in zip(similarities[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._df):
                continue
            row = self._df.iloc[idx]
            results.append({
                "segment": self._row_to_segment_dict(row),
                "similarity": float(sim),
                "distance": float(1.0 - sim),
            })

        return results

    def search_audio_by_text(self, query_text: str, k: int = 5) -> list[dict]:
        """
        Búsqueda cross-modal texto→audio usando CLAP.

        Args:
            query_text: Text query (e.g., "applause", "music")
            k: Number of results

        Returns:
            List of dicts with segment info and similarity scores
        """
        if self._audio_index is None:
            raise RuntimeError("Audio index not loaded")

        query_embedding = self.clap_model.generate_text_embedding(query_text)
        similarities, indices = search_faiss_index(self._audio_index, query_embedding, k=k)

        results = []
        for sim, idx in zip(similarities[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._df):
                continue
            row = self._df.iloc[idx]
            results.append({
                "segment": self._row_to_segment_dict(row),
                "similarity": float(sim),
                "distance": float(1.0 - sim),
            })

        return results

    def get_segment_info(self, segment_id: int) -> dict | None:
        """
        Get full information for a specific segment.

        Args:
            segment_id: The segment ID

        Returns:
            Dict with segment info or None if not found
        """
        if self._df is None:
            return None

        matches = self._df[self._df["segment_id"] == segment_id]
        if matches.empty:
            return None

        row = matches.iloc[0]
        return self._row_to_segment_dict(row, include_sentiment=True)

    def _row_to_segment_dict(self, row: pd.Series, include_sentiment: bool = False) -> dict:
        """Convert a DataFrame row to a segment dict."""
        result = {
            "segment_id": int(row.get("segment_id", 0)),
            "text": str(row.get("text", "")),
            "start_time": float(row.get("start_time", 0.0)),
            "end_time": float(row.get("end_time", 0.0)),
            "original_file_name": str(row.get("original_file_name", "")),
            "language": str(row.get("language", "unknown")),
            "confidence": float(row.get("confidence", 0.0)),
        }

        if include_sentiment:
            result.update({
                "sentiment_positive": float(row.get("sentiment_positive", 0.0)),
                "sentiment_negative": float(row.get("sentiment_negative", 0.0)),
                "sentiment_neutral": float(row.get("sentiment_neutral", 0.0)),
                "dominant_sentiment": str(row.get("dominant_sentiment", "neutral")),
            })

        return result
