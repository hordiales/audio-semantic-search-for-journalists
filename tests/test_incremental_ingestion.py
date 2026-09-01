from src.embedding_config import EmbeddingConfig
from src.simple_dataset_pipeline import _file_fingerprint, _pipeline_signature


def test_file_fingerprint_changes_when_same_named_source_changes(tmp_path):
    source = tmp_path / "interview.wav"
    source.write_bytes(b"first recording")
    initial = _file_fingerprint(source)

    source.write_bytes(b"replacement recording")

    assert _file_fingerprint(source) != initial


def test_pipeline_signature_invalidates_when_chunking_or_embedding_changes():
    config = EmbeddingConfig(active_embeddings=frozenset({"text", "clap"}))
    common = {
        "whisper_model": "base",
        "language": "es",
        "chunk_strategy": "fixed",
        "chunk_duration_sec": 30.0,
        "chunk_overlap_sec": 5.0,
        "max_chunk_text_chars": 500,
        "audio_window_duration_sec": 10.0,
        "audio_window_overlap_sec": 2.0,
        "mock_audio": False,
        "embedding_config": config,
    }

    signature = _pipeline_signature(**common)
    changed = _pipeline_signature(**{**common, "chunk_duration_sec": 45.0})

    assert signature != changed
