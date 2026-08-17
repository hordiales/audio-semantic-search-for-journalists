"""Audio transcription module using OpenAI Whisper."""

import logging
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ChunkingConfig:
    """How Whisper segments are transformed before embedding generation."""

    strategy: str = "whisper"
    duration_sec: float = 30.0
    overlap_sec: float = 5.0
    max_text_chars: int = 500


class ChunkingProcessor:
    """Create globally identified chunks while preserving audio provenance."""

    _VALID_STRATEGIES = frozenset({"whisper", "fixed", "sentence", "paragraph"})

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        if self.config.strategy not in self._VALID_STRATEGIES:
            raise ValueError(f"Unsupported chunk strategy: {self.config.strategy}")
        if self.config.duration_sec <= 0:
            raise ValueError("duration_sec must be greater than zero")
        if not 0 <= self.config.overlap_sec < self.config.duration_sec:
            raise ValueError("overlap_sec must be non-negative and less than duration_sec")
        if self.config.max_text_chars <= 0:
            raise ValueError("max_text_chars must be greater than zero")

    def process_segments(self, segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
        """Return chunks with sequential IDs, grouped independently per source file."""
        by_file: dict[str, list[TranscriptionSegment]] = {}
        for segment in segments:
            by_file.setdefault(segment.original_file_name, []).append(segment)

        chunks: list[TranscriptionSegment] = []
        for file_segments in by_file.values():
            ordered = sorted(file_segments, key=lambda item: item.start_time)
            if self.config.strategy == "whisper":
                chunks.extend(ordered)
            elif self.config.strategy == "fixed":
                chunks.extend(self._fixed_chunks(ordered))
            elif self.config.strategy == "sentence":
                chunks.extend(self._sentence_chunks(ordered))
            else:
                chunks.extend(self._paragraph_chunks(ordered))

        return [
            TranscriptionSegment(segment_id=index, **{
                "text": chunk.text,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "language": chunk.language,
                "confidence": chunk.confidence,
                "original_file_name": chunk.original_file_name,
            })
            for index, chunk in enumerate(chunks)
        ]

    def _fixed_chunks(self, segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
        if not segments:
            return []
        chunks = []
        start = segments[0].start_time
        final_end = segments[-1].end_time
        step = self.config.duration_sec - self.config.overlap_sec
        while start < final_end:
            end = min(start + self.config.duration_sec, final_end)
            overlapping = [segment for segment in segments if segment.end_time > start and segment.start_time < end]
            if overlapping:
                chunks.append(self._combine(overlapping, start, end))
            if end == final_end:
                break
            start += step
        return chunks

    def _sentence_chunks(self, segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
        chunks = []
        for segment in segments:
            sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", segment.text) if item.strip()]
            if len(sentences) <= 1:
                chunks.append(segment)
                continue
            total_chars = sum(len(sentence) for sentence in sentences)
            cursor = segment.start_time
            for sentence in sentences:
                duration = (segment.end_time - segment.start_time) * len(sentence) / total_chars
                chunks.append(TranscriptionSegment(
                    segment_id=0, text=sentence, start_time=cursor,
                    end_time=min(cursor + duration, segment.end_time),
                    language=segment.language, confidence=segment.confidence,
                    original_file_name=segment.original_file_name,
                ))
                cursor += duration
        return self._merge_by_text_length(chunks)

    def _paragraph_chunks(self, segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
        return self._merge_by_text_length(segments)

    def _merge_by_text_length(self, segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
        if not segments:
            return []
        chunks: list[TranscriptionSegment] = []
        current: list[TranscriptionSegment] = []
        current_length = 0
        for segment in segments:
            additional = len(segment.text) + (1 if current else 0)
            if current and current_length + additional > self.config.max_text_chars:
                chunks.append(self._combine(current, current[0].start_time, current[-1].end_time))
                current, current_length = [], 0
            current.append(segment)
            current_length += additional
        if current:
            chunks.append(self._combine(current, current[0].start_time, current[-1].end_time))
        return chunks

    @staticmethod
    def _combine(
        segments: list[TranscriptionSegment], start_time: float, end_time: float
    ) -> TranscriptionSegment:
        return TranscriptionSegment(
            segment_id=0,
            text=" ".join(segment.text for segment in segments),
            start_time=start_time,
            end_time=end_time,
            language=segments[0].language,
            confidence=sum(segment.confidence for segment in segments) / len(segments),
            original_file_name=segments[0].original_file_name,
        )


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
    return transcribe_files(sorted(audio_path.glob("*.wav")), model_name, language, start_id)


def transcribe_files(
    wav_files: list[str | Path],
    model_name: str = "base",
    language: str | None = None,
    start_id: int = 0,
) -> list[TranscriptionSegment]:
    """Transcribe a selected list of WAV files with one shared Whisper model."""
    wav_paths = [Path(wav_file) for wav_file in wav_files]

    if not wav_paths:
        logger.warning("No WAV files selected for transcription")
        return []

    logger.info("Loading Whisper model '%s' once for batch...", model_name)
    model = whisper.load_model(model_name)

    all_segments = []
    current_id = start_id

    for wav_file in wav_paths:
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

    logger.info("Total segments transcribed: %d from %d files", len(all_segments), len(wav_paths))
    return all_segments
