"""Audio conversion module: normalizes audio files to WAV 16kHz mono."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


def convert_audio(
    input_path: str,
    output_dir: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> str:
    """
    Convierte audio a formato WAV normalizado.

    Args:
        input_path: Ruta al archivo de audio original
        output_dir: Directorio de salida para archivos convertidos
        sample_rate: Sample rate objetivo (default: 16000)
        mono: Convertir a mono (default: True)

    Returns:
        Ruta al archivo WAV convertido

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el formato no es soportado
        RuntimeError: Si ffmpeg falla
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    suffix = input_file.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{suffix}'. Supported: {SUPPORTED_FORMATS}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{input_file.stem}.wav"

    if output_file.exists():
        logger.info("Skipping already converted: %s", output_file)
        return str(output_file)

    cmd = [
        "ffmpeg",
        "-i", str(input_file),
        "-ar", str(sample_rate),
        "-ac", "1" if mono else "2",
        "-sample_fmt", "s16",
        "-y",
        str(output_file),
    ]

    logger.info("Converting: %s -> %s", input_file.name, output_file.name)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed for {input_file.name}: {e.stderr[:500]}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg (macOS)"
        )

    logger.debug("ffmpeg stdout: %s", result.stdout[:200])
    return str(output_file)


def convert_directory(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> list[str]:
    """
    Convierte todos los archivos de audio de un directorio.

    Returns:
        Lista de rutas a archivos WAV convertidos
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    audio_files = [
        f for f in sorted(input_path.iterdir())
        if f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not audio_files:
        logger.warning("No audio files found in %s", input_dir)
        return []

    logger.info("Found %d audio files to convert", len(audio_files))

    converted = []
    for audio_file in audio_files:
        try:
            result = convert_audio(str(audio_file), output_dir, sample_rate, mono)
            converted.append(result)
        except (RuntimeError, ValueError) as e:
            logger.error("Failed to convert %s: %s", audio_file.name, e)

    logger.info("Successfully converted %d/%d files", len(converted), len(audio_files))
    return converted
