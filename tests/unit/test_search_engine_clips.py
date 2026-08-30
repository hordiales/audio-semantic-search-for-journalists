from types import SimpleNamespace

import pandas as pd

from src.agent_service.search_engine import AudioSearchEngine


def test_segment_result_adds_a_signed_clip_url_without_reading_audio():
    engine = object.__new__(AudioSearchEngine)
    calls: list[int] = []

    class _ClipStore:
        def reference(self, segment_id: int):
            calls.append(segment_id)
            return SimpleNamespace(url="https://signed.example/segment_9.opus")

    engine._clip_store = _ClipStore()
    segment = engine._row_to_segment_dict(
        pd.Series(
            {
                "segment_id": 9,
                "text": "Aplausos.",
                "start_time": 10.0,
                "end_time": 14.0,
                "original_file_name": "discurso.wav",
                "clip_file_name": "segment_9.opus",
                "clip_start_time": 5.0,
                "clip_end_time": 19.0,
            }
        )
    )

    assert calls == [9]
    assert segment["clip_url"] == "https://signed.example/segment_9.opus"
    assert segment["clip_start_time"] == 5.0
    assert segment["clip_end_time"] == 19.0


def test_segment_result_keeps_search_available_when_clip_signing_fails():
    engine = object.__new__(AudioSearchEngine)

    class _UnavailableClipStore:
        def reference(self, segment_id: int):
            raise RuntimeError("signBlob unavailable")

    engine._clip_store = _UnavailableClipStore()
    segment = engine._row_to_segment_dict(
        pd.Series(
            {
                "segment_id": 10,
                "text": "Música.",
                "start_time": 1.0,
                "end_time": 3.0,
                "original_file_name": "cortina.wav",
                "clip_file_name": "segment_10.opus",
            }
        )
    )

    assert segment["segment_id"] == 10
    assert "clip_url" not in segment
