from src.audio_transcription import ChunkingConfig, ChunkingProcessor, TranscriptionSegment


def _segment(segment_id: int, text: str, start: float, end: float) -> TranscriptionSegment:
    return TranscriptionSegment(segment_id, text, start, end, "es", 0.9, "source.wav")


def test_fixed_chunking_uses_overlapping_time_windows_and_sequential_ids():
    chunks = ChunkingProcessor(
        ChunkingConfig(strategy="fixed", duration_sec=10, overlap_sec=2)
    ).process_segments(
        [
            _segment(9, "uno", 0, 8),
            _segment(10, "dos", 8, 16),
            _segment(11, "tres", 16, 20),
        ]
    )

    assert [(chunk.start_time, chunk.end_time) for chunk in chunks] == [(0, 10), (8, 18), (16, 20)]
    assert [chunk.segment_id for chunk in chunks] == [0, 1, 2]


def test_sentence_chunking_preserves_time_and_splits_sentences():
    chunks = ChunkingProcessor(
        ChunkingConfig(strategy="sentence", max_text_chars=20)
    ).process_segments(
        [
            _segment(4, "Primera frase. Segunda frase.", 0, 10),
        ]
    )

    assert [chunk.text for chunk in chunks] == ["Primera frase.", "Segunda frase."]
    assert chunks[0].start_time == 0
    assert chunks[-1].end_time == 10
