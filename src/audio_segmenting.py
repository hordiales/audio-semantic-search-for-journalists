"""Utilities for extracting timestamp-aligned audio windows from WAV files."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf


def build_audio_windows(
    start_time: float,
    end_time: float,
    audio_duration: float,
    window_duration_sec: float = 10.0,
    overlap_sec: float = 2.0,
) -> list[tuple[float, float]]:
    """Return real-audio windows covering a transcription segment.

    Short transcription segments get one window centred on the segment, adding
    acoustic context. Longer segments are split into overlapping windows. All
    boundaries are clamped to the source audio duration.
    """
    if window_duration_sec <= 0:
        raise ValueError("window_duration_sec must be greater than zero")
    if not 0 <= overlap_sec < window_duration_sec:
        raise ValueError("overlap_sec must be non-negative and smaller than window_duration_sec")
    if audio_duration <= 0:
        return []

    start = min(max(float(start_time), 0.0), audio_duration)
    end = min(max(float(end_time), start), audio_duration)
    if end <= start:
        return []

    segment_duration = end - start
    if segment_duration <= window_duration_sec:
        centre = (start + end) / 2
        window_start = max(0.0, centre - window_duration_sec / 2)
        window_end = min(audio_duration, window_start + window_duration_sec)
        window_start = max(0.0, window_end - window_duration_sec)
        return [(window_start, window_end)]

    step = window_duration_sec - overlap_sec
    windows: list[tuple[float, float]] = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + window_duration_sec, end)
        windows.append((window_start, window_end))
        if window_end == end:
            break
        window_start += step
    return windows


def extract_wav_window(
    source_path: str | Path,
    output_path: str | Path,
    start_time: float,
    end_time: float,
) -> tuple[float, float]:
    """Write a WAV excerpt and return the clipped timestamps actually used."""
    source = Path(source_path)
    target = Path(output_path)
    with sf.SoundFile(source) as audio:
        sample_rate = audio.samplerate
        total_frames = len(audio)
        start_frame = min(max(round(start_time * sample_rate), 0), total_frames)
        end_frame = min(max(round(end_time * sample_rate), start_frame), total_frames)
        audio.seek(start_frame)
        frames = audio.read(end_frame - start_frame, dtype="float32")
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, frames, sample_rate, subtype="PCM_16")

    return start_frame / sample_rate, end_frame / sample_rate
