"""Contracts for YAMNet class retrieval and CLAP result enrichment."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agent_service.search_engine import AudioSearchEngine
from src.yamnet_inverted_index import build_yamnet_inverted_index


def _engine(
    rows: list[dict],
    active_embeddings: set[str] | None = None,
    active_classifiers: set[str] | None = None,
) -> AudioSearchEngine:
    engine = AudioSearchEngine.__new__(AudioSearchEngine)
    engine._df = pd.DataFrame(rows)
    engine._active_embeddings = active_embeddings
    engine._active_classifiers = active_classifiers
    engine._audio_index = None
    engine._clap_model = None
    engine._clip_store = SimpleNamespace(reference=lambda _: None)
    engine._segment_positions = {
        int(segment_id): position for position, segment_id in enumerate(engine._df["segment_id"])
    }
    engine._yamnet_inverted_index = build_yamnet_inverted_index(rows)["postings"]
    return engine


def _row(segment_id: int, classes: list[tuple[str, float]]) -> dict:
    return {
        "segment_id": segment_id,
        "text": f"segmento {segment_id}",
        "start_time": float(segment_id),
        "end_time": float(segment_id + 10),
        "original_file_name": "audio.wav",
        "language": "es",
        "confidence": 0.9,
        "yamnet_top_classes": [
            {"class_id": f"/m/{position}", "class_name": name, "score": score}
            for position, (name, score) in enumerate(classes)
        ],
    }


def test_yamnet_search_prefers_segments_covering_more_requested_classes(monkeypatch):
    monkeypatch.setenv("QUERY_LANGUAGE", "en")
    engine = _engine(
        [
            _row(1, [("Speech", 0.9), ("Applause", 0.8)]),
            _row(2, [("Speech", 0.95)]),
        ],
        active_classifiers={"yamnet"},
    )

    results = engine.search_audio_by_classes("applause during speech", k=5)

    assert [item["segment"]["segment_id"] for item in results] == [1, 2]
    assert results[0]["similarity"] == pytest.approx(0.85)
    assert [
        item["class_name"] for item in results[0]["segment"]["yamnet_matched_classes"]
    ] == ["Speech", "Applause"]
    assert results[0]["segment"]["yamnet_audio_classes"][0]["class_name"] == "Speech"


def test_yamnet_search_is_only_advertised_when_classes_are_active():
    rows = [_row(1, [("Music", 0.8)])]
    assert _engine(rows, active_classifiers={"yamnet"}).yamnet_available is True
    assert _engine(rows, active_embeddings={"text", "clap"}).yamnet_available is False


def test_yamnet_search_uses_postings_without_scanning_all_dataset_rows(monkeypatch):
    monkeypatch.setenv("QUERY_LANGUAGE", "en")
    engine = _engine(
        [
            _row(1, [("Speech", 0.9), ("Applause", 0.8)]),
            _row(2, [("Music", 0.95)]),
        ],
        active_classifiers={"yamnet"},
    )
    monkeypatch.setattr(
        engine._df,
        "iterrows",
        lambda: (_ for _ in ()).throw(AssertionError("must not scan the dataset")),
    )

    results = engine.search_audio_by_classes("applause during speech", k=5)

    assert [item["segment"]["segment_id"] for item in results] == [1]


def test_clap_results_include_yamnet_classes(monkeypatch):
    engine = _engine(
        [_row(1, [("Music", 0.8)])],
        active_embeddings={"clap"},
        active_classifiers={"yamnet"},
    )
    engine._audio_index = object()
    engine._clap_model = SimpleNamespace(
        generate_text_embedding=lambda query, source_language=None: np.array([[1.0]])
    )
    monkeypatch.setattr(
        "src.agent_service.search_engine.search_faiss_index",
        lambda index, embedding, k: (np.array([[0.7]]), np.array([[0]])),
    )

    results = engine.search_audio_by_text("music", k=1, source_language="en")

    assert results[0]["segment"]["yamnet_audio_classes"] == [
        {"class_id": "/m/0", "class_name": "Music", "score": 0.8}
    ]
