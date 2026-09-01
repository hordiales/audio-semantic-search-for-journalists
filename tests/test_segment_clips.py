import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from src.segment_clips import (
    CLIP_EXTENSION,
    build_clip_bounds,
    clip_file_name,
    export_segment_clip,
    prune_orphan_clips,
    segment_id_from_clip_name,
)

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_clip_bounds_add_context_on_both_sides():
    assert build_clip_bounds(50.0, 80.0, audio_duration=600.0, context_sec=5.0) == (45.0, 85.0)


def test_clip_bounds_clamp_to_audio_boundaries():
    assert build_clip_bounds(2.0, 8.0, audio_duration=10.0, context_sec=5.0) == (0.0, 10.0)


def test_clip_bounds_handle_segment_beyond_audio_duration():
    start, end = build_clip_bounds(120.0, 150.0, audio_duration=100.0, context_sec=5.0)
    assert start <= end <= 100.0


def test_clip_bounds_reject_negative_context():
    with pytest.raises(ValueError, match="non-negative"):
        build_clip_bounds(0.0, 5.0, audio_duration=10.0, context_sec=-1.0)


def test_clip_names_round_trip():
    assert clip_file_name(42) == f"segment_42{CLIP_EXTENSION}"
    assert segment_id_from_clip_name(clip_file_name(42)) == 42


def test_prune_orphan_clips_keeps_only_current_segments(tmp_path):
    for segment_id in (1, 2, 3):
        (tmp_path / clip_file_name(segment_id)).write_bytes(b"clip")
    (tmp_path / "notes.txt").write_text("unrelated")

    removed = prune_orphan_clips(tmp_path, valid_segment_ids={2})

    assert removed == 2
    assert {path.name for path in tmp_path.glob(f"*{CLIP_EXTENSION}")} == {clip_file_name(2)}
    assert (tmp_path / "notes.txt").exists()


@requires_ffmpeg
def test_export_segment_clip_encodes_requested_window(tmp_path):
    source = tmp_path / "interview.wav"
    sample_rate = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(30 * sample_rate) / sample_rate).astype(np.float32)
    sf.write(source, tone, sample_rate)

    clip = export_segment_clip(source, tmp_path / clip_file_name(7), 10.0, 20.0)

    assert clip.exists()
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_name",
            "-of",
            "default=nw=1:nk=1",
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert "opus" in probe
    assert 9.5 < float(probe[-1]) < 10.5
    # Opus must stay far below the ~320 KB the same window needs as 16 kHz PCM.
    assert clip.stat().st_size < 100_000


@requires_ffmpeg
def test_export_segment_clip_rejects_empty_range(tmp_path):
    source = tmp_path / "interview.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000)

    with pytest.raises(ValueError, match="Empty clip range"):
        export_segment_clip(source, tmp_path / "out.opus", 5.0, 5.0)
