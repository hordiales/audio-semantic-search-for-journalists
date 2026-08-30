"""Playback clips: one compressed audio excerpt per transcription segment.

These clips are a serving artifact, not an ingestion artifact. The overlapping
windows produced by :mod:`src.audio_segmenting` exist to feed CLAP/YAMNet and
deliberately repeat audio across window boundaries, so they are unusable for
listening. Here every segment gets exactly one Opus file covering its own
timestamps plus a configurable amount of surrounding context.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

CLIP_DIR_NAME = "segment_clips"
CLIP_EXTENSION = ".opus"
DEFAULT_CLIP_CONTEXT_SEC = 5.0
DEFAULT_CLIP_BITRATE = "32k"


def clip_file_name(segment_id: int) -> str:
    """Return the stable clip file name for a segment."""
    return f"segment_{int(segment_id)}{CLIP_EXTENSION}"


def segment_id_from_clip_name(name: str) -> int:
    """Inverse of :func:`clip_file_name`; raises ValueError on foreign names."""
    stem = Path(name).stem
    return int(stem.removeprefix("segment_"))


def build_clip_bounds(
    start_time: float,
    end_time: float,
    audio_duration: float,
    context_sec: float = DEFAULT_CLIP_CONTEXT_SEC,
) -> tuple[float, float]:
    """Return the padded ``[start, end]`` window to export for one segment.

    The padding lets a journalist hear what comes immediately before and after
    the match, which is what usually decides whether a hit is usable. Bounds are
    clamped to the source audio, so a segment at the very start or end of a file
    simply gets less context instead of an invalid range.
    """
    if context_sec < 0:
        raise ValueError("context_sec must be non-negative")
    if audio_duration <= 0:
        return (0.0, 0.0)

    start = min(max(float(start_time), 0.0), audio_duration)
    end = min(max(float(end_time), start), audio_duration)
    return (max(0.0, start - context_sec), min(audio_duration, end + context_sec))


def export_segment_clip(
    source_path: str | Path,
    output_path: str | Path,
    start_time: float,
    end_time: float,
    bitrate: str = DEFAULT_CLIP_BITRATE,
) -> Path:
    """Encode ``[start_time, end_time]`` of a WAV file to a mono Opus clip.

    Opus keeps a 30-second excerpt around 100 KB instead of the ~1 MB the raw
    16 kHz WAV would need, which is what makes on-demand streaming from the
    bucket viable, and every current browser decodes it natively.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    duration = float(end_time) - float(start_time)
    if duration <= 0:
        raise ValueError(f"Empty clip range for {source.name}: {start_time}-{end_time}")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss",
        f"{float(start_time):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-vbr",
        "on",
        "-y",
        str(target),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ffmpeg failed to export clip for {source.name}: {error.stderr[:500]}"
        ) from error
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg (macOS)") from error

    return target


def prune_orphan_clips(clips_dir: str | Path, valid_segment_ids: set[int]) -> int:
    """Delete clips whose segment no longer exists; return how many were removed."""
    directory = Path(clips_dir)
    if not directory.is_dir():
        return 0

    removed = 0
    for artifact in directory.glob(f"segment_*{CLIP_EXTENSION}"):
        try:
            segment_id = segment_id_from_clip_name(artifact.name)
        except ValueError:
            logger.warning("Ignoring unexpected file in %s: %s", directory, artifact.name)
            continue
        if segment_id not in valid_segment_ids:
            artifact.unlink()
            removed += 1
    return removed
