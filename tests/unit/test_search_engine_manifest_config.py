import json

import pandas as pd

from src.agent_service.search_engine import AudioSearchEngine


def _load_engine_manifest(tmp_path, manifest: dict) -> AudioSearchEngine:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    pd.DataFrame(
        [{"segment_id": 1, "start_time": 0.0, "end_time": 8.0, "text": "hola"}]
    ).to_pickle(final_dir / "complete_dataset.pkl")
    (final_dir / "dataset_manifest.json").write_text(json.dumps(manifest))

    engine = AudioSearchEngine.__new__(AudioSearchEngine)
    engine.dataset_path = tmp_path
    engine._active_embeddings = None
    engine._active_classifiers = None
    engine._dataset_version = None
    engine._segment_positions = {}
    engine._load_dataset()
    return engine


def test_new_manifest_separates_embeddings_from_classifiers(tmp_path):
    engine = _load_engine_manifest(
        tmp_path,
        {
            "active_embeddings": ["text", "clap"],
            "active_classifiers": ["yamnet"],
        },
    )

    assert engine._is_embedding_active("clap") is True
    assert engine._is_embedding_active("yamnet") is False
    assert engine._is_classifier_active("yamnet") is True


def test_legacy_manifest_keeps_yamnet_available(tmp_path):
    engine = _load_engine_manifest(
        tmp_path,
        {"active_embeddings": ["text", "clap", "yamnet"]},
    )

    assert engine._is_classifier_active("yamnet") is True


def test_manifest_can_infer_classifiers_from_legacy_classifier_metadata(tmp_path):
    engine = _load_engine_manifest(
        tmp_path,
        {
            "active_embeddings": ["text", "clap"],
            "classifiers": {"yamnet": {"top_k": 5}},
        },
    )

    assert engine._is_classifier_active("yamnet") is True
