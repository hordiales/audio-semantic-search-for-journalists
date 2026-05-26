"""Audio transcription module using OpenAI Whisper."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import whisper

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    segment_id: int
    text: str
    start_time: float
    end_time: float
    language: str
    confidence: float
    original_file_name: str


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
) -> list[TranscriptionSegment]:
    """
    Transcribe audio usando Whisper y retorna segmentos.

    Args:
        audio_path: Ruta al archivo WAV
        model_name: Modelo Whisper a usar (tiny/base/small/medium/large)
        language: Forzar idioma (None = auto-detección)

    Returns:
        Lista de segmentos transcritos
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Loading Whisper model '%s'...", model_name)
    model = whisper.load_model(model_name)

    logger.info("Transcribing: %s", audio_file.name)
    options = {}
    if language:
        options["language"] = language

    result = model.transcribe(str(audio_file), **options)

    detected_language = result.get("language", "unknown")
    segments = []

    for i, seg in enumerate(result.get("segments", [])):
        text = seg.get("text", "").strip()
        if not text:
            continue

        confidence = _compute_confidence(seg)

        segments.append(
            TranscriptionSegment(
                segment_id=i,
                text=text,
                start_time=seg.get("start", 0.0),
                end_time=seg.get("end", 0.0),
                language=detected_language,
                confidence=confidence,
                original_file_name=audio_file.name,
            )
        )

    logger.info(
        "Transcribed %d segments from %s (language: %s)",
        len(segments),
        audio_file.name,
        detected_language,
    )
    return segments


def _compute_confidence(segment: dict) -> float:
    """Compute confidence score from Whisper segment data."""
    avg_logprob = segment.get("avg_logprob", -1.0)
    no_speech_prob = segment.get("no_speech_prob", 0.0)
    import math

    prob = math.exp(avg_logprob) if avg_logprob > -10 else 0.0
    confidence = prob * (1.0 - no_speech_prob)
    return max(0.0, min(1.0, confidence))


def transcribe_directory(
    audio_dir: str,
    model_name: str = "base",
    language: str | None = None,
    start_id: int = 0,
) -> list[TranscriptionSegment]:
    """
    Transcribe all WAV files in a directory.

    Args:
        audio_dir: Directory with WAV files
        model_name: Whisper model name
        language: Force language or None for auto-detect
        start_id: Starting segment_id for global uniqueness

    Returns:
        All transcription segments with globally unique IDs
    """
    audio_path = Path(audio_dir)
    wav_files = sorted(audio_path.glob("*.wav"))

    if not wav_files:
        logger.warning("No WAV files found in %s", audio_dir)
        return []

    logger.info("Loading Whisper model '%s' once for batch...", model_name)
    model = whisper.load_model(model_name)

    all_segments = []
    current_id = start_id

    for wav_file in wav_files:
        logger.info("Transcribing: %s", wav_file.name)
        options = {}
        if language:
            options["language"] = language

        result = model.transcribe(str(wav_file), **options)
        detected_language = result.get("language", "unknown")

        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if not text:
                continue

            confidence = _compute_confidence(seg)

            all_segments.append(
                TranscriptionSegment(
                    segment_id=current_id,
                    text=text,
                    start_time=seg.get("start", 0.0),
                    end_time=seg.get("end", 0.0),
                    language=detected_language,
                    confidence=confidence,
                    original_file_name=wav_file.name,
                )
            )
            current_id += 1

    logger.info("Total segments transcribed: %d from %d files", len(all_segments), len(wav_files))
    return all_segments
