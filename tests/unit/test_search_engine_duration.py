from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agent_service.search_engine import (
    AudioSearchEngine,
    _configured_min_segment_duration,
)


def _engine(rows: list[dict]) -> AudioSearchEngine:
    engine = AudioSearchEngine.__new__(AudioSearchEngine)
    engine._df = pd.DataFrame(rows)
    engine._text_index = SimpleNamespace(ntotal=len(rows))
    engine._text_model = SimpleNamespace(
        generate_embedding=lambda query, normalize=True: np.array([[1.0]])
    )
    engine._clip_store = SimpleNamespace(reference=lambda _: None)
    engine._min_segment_duration_seconds = 5.0
    return engine


def _row(segment_id: int, start: float, end: float) -> dict:
    return {
        "segment_id": segment_id,
        "text": f"segmento {segment_id}",
        "start_time": start,
        "end_time": end,
        "original_file_name": "audio.wav",
        "language": "es",
        "confidence": 0.9,
    }


def test_minimum_duration_defaults_to_five_seconds_and_is_configurable(monkeypatch):
    monkeypatch.delenv("MIN_SEGMENT_DURATION_SECONDS", raising=False)
    assert _configured_min_segment_duration() == 5.0

    monkeypatch.setenv("MIN_SEGMENT_DURATION_SECONDS", "7.5")
    assert _configured_min_segment_duration() == 7.5


def test_minimum_duration_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("MIN_SEGMENT_DURATION_SECONDS", "-1")
    with pytest.raises(ValueError, match="MIN_SEGMENT_DURATION_SECONDS"):
        _configured_min_segment_duration()


def test_semantic_search_excludes_segments_at_or_below_threshold(monkeypatch):
    rows = [
        _row(1, 0.0, 5.0),
        _row(2, 10.0, 16.0),
        _row(3, 20.0, 30.0),
    ]
    rows[1]["yamnet_top_classes"] = [
        {"class_id": "/m/04rlf", "class_name": "Music", "score": 0.84}
    ]
    engine = _engine(rows)
    observed_k: list[int] = []

    def fake_search(index, embedding, k):
        observed_k.append(k)
        return np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]])

    monkeypatch.setattr("src.agent_service.search_engine.search_faiss_index", fake_search)

    results = engine.search_semantic("consulta", k=2)

    assert observed_k == [3]
    assert [item["segment"]["segment_id"] for item in results] == [2, 3]
    assert results[0]["segment"]["yamnet_audio_classes"] == [
        {"class_id": "/m/04rlf", "class_name": "Music", "score": 0.84}
    ]
