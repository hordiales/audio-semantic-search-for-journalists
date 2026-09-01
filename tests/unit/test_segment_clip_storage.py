from __future__ import annotations

import pytest

from src.segment_clip_storage import SegmentClipStore, resolve_clips_gcs_uri
from src.segment_clips import clip_file_name


def test_clips_uri_derives_from_dataset_release(monkeypatch):
    monkeypatch.delenv("SEGMENT_CLIPS_GCS_URI", raising=False)
    monkeypatch.setenv("DATASET_GCS_URI", "gs://your-test-bucket/releases/v1")

    assert resolve_clips_gcs_uri() == "gs://your-test-bucket/releases/v1/segment_clips"


def test_explicit_clips_uri_wins(monkeypatch):
    monkeypatch.setenv("DATASET_GCS_URI", "gs://your-test-bucket/releases/v1")
    monkeypatch.setenv("SEGMENT_CLIPS_GCS_URI", "gs://other-bucket/clips/")

    assert resolve_clips_gcs_uri() == "gs://other-bucket/clips"


def test_no_gcs_configuration_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("SEGMENT_CLIPS_GCS_URI", raising=False)
    monkeypatch.delenv("DATASET_GCS_URI", raising=False)

    assert resolve_clips_gcs_uri() == ""


def test_local_store_resolves_existing_clip(tmp_path, monkeypatch):
    monkeypatch.delenv("SEGMENT_CLIPS_GCS_URI", raising=False)
    monkeypatch.delenv("DATASET_GCS_URI", raising=False)
    clips_dir = tmp_path / "segment_clips"
    clips_dir.mkdir()
    (clips_dir / clip_file_name(3)).write_bytes(b"clip")
    store = SegmentClipStore(dataset_path=tmp_path)

    reference = store.reference(3)

    assert store.uses_gcs is False
    assert reference is not None
    assert reference.location == "local"
    assert reference.url.endswith("segment_3.opus")
    assert store.reference(99) is None


def test_gcs_store_signs_once_and_reuses_the_url(monkeypatch):
    signed_calls = []

    class _FakeBlob:
        def __init__(self, name):
            self.name = name

        def generate_signed_url(self, **kwargs):
            signed_calls.append((self.name, kwargs))
            return (
                f"https://signed.example/{self.name}?X-Goog-Algorithm=GOOG4-RSA-SHA256"
                "&X-Goog-Signature=signature"
            )

    class _FakeBucket:
        def blob(self, name):
            return _FakeBlob(name)

    store = SegmentClipStore(
        dataset_path="/tmp/unused",
        clips_gcs_uri="gs://your-test-bucket/releases/v1/segment_clips",
        ttl_seconds=900,
    )
    monkeypatch.setattr(store, "_get_bucket", lambda: _FakeBucket())
    monkeypatch.setattr(store, "_signing_kwargs", dict)

    first = store.reference(12)
    second = store.reference(12)

    assert store.uses_gcs is True
    assert (
        first.url
        == second.url
        == "https://signed.example/releases/v1/segment_clips/segment_12.opus?"
        "X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Signature=signature"
    )
    assert first.location == "gcs"
    assert len(signed_calls) == 1, "cached URLs must avoid one IAM signBlob call per result"
    assert signed_calls[0][1]["response_type"] == "audio/ogg"


def test_gcs_store_rejects_plain_object_url(monkeypatch):
    class _FakeBlob:
        def generate_signed_url(self, **kwargs):
            return "https://storage.googleapis.com/your-test-bucket/segment_12.opus"

    class _FakeBucket:
        def blob(self, name):
            return _FakeBlob()

    store = SegmentClipStore(clips_gcs_uri="gs://your-test-bucket/releases/v1/segment_clips")
    monkeypatch.setattr(store, "_get_bucket", lambda: _FakeBucket())
    monkeypatch.setattr(store, "_signing_kwargs", dict)

    with pytest.raises(RuntimeError, match="did not produce a V4 signed URL"):
        store.reference(12)


def test_gcs_store_rejects_non_gcs_uri():
    with pytest.raises(ValueError, match="must be a GCS prefix"):
        SegmentClipStore(clips_gcs_uri="https://example.com/clips")
