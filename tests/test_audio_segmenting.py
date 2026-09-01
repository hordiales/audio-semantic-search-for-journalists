"""Tests for timestamp-aligned WAV extraction."""

import numpy as np
import soundfile as sf

from src.audio_segmenting import build_audio_windows, extract_wav_window


def test_short_segment_gets_context_window():
    windows = build_audio_windows(8.0, 10.0, audio_duration=30.0, window_duration_sec=10.0)

    assert windows == [(4.0, 14.0)]


def test_long_segment_is_split_into_overlapping_windows():
    windows = build_audio_windows(
        0.0, 23.0, audio_duration=30.0, window_duration_sec=10.0, overlap_sec=2.0
    )

    assert windows == [(0.0, 10.0), (8.0, 18.0), (16.0, 23.0)]


def test_extract_wav_window_uses_requested_timestamps(tmp_path):
    source = tmp_path / "source.wav"
    target = tmp_path / "clip.wav"
    sample_rate = 100
    sf.write(source, np.arange(1_000, dtype=np.float32) / 1_000, sample_rate)

    start, end = extract_wav_window(source, target, 2.0, 4.5)
    clip, clip_rate = sf.read(target)

    assert (start, end) == (2.0, 4.5)
    assert clip_rate == sample_rate
    assert len(clip) == 250
