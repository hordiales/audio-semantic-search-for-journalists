"""Publica un dataset procesado como release versionada en Cloud Storage.

Sube tres artefactos al prefijo de la release:

- ``final/`` e ``indices/``: los descarga el runtime al arrancar
  (``src/dataset_storage.resolve_dataset_path``).
- ``segment_clips/``: se queda en el bucket y se sirve on-demand vía URLs
  firmadas (``src/segment_clip_storage``). El staging local lo omite a
  propósito, así el arranque en frío no crece con las horas de audio.

Uso:
    uv run python scripts/publish_dataset_release.py \
        --dataset ./dataset --bucket mi-bucket-audio-search
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from src.segment_clips import CLIP_DIR_NAME, CLIP_EXTENSION

logger = logging.getLogger("publish_dataset_release")

STAGED_DIRECTORIES = ("final", "indices")
REQUIRED_FILE = Path("final/complete_dataset.pkl")


def default_release_name(dataset_dir: Path) -> str:
    """Fecha + huella del dataset, para que cada contenido tenga su propia URI."""
    digest = hashlib.sha256()
    digest.update((dataset_dir / REQUIRED_FILE).read_bytes())
    for index in sorted((dataset_dir / "indices").glob("*.faiss")):
        digest.update(index.read_bytes())
    return f"{datetime.now(UTC):%Y-%m-%d}-{digest.hexdigest()[:7]}"


def rsync(source: Path, destination: str, dry_run: bool) -> None:
    command = ["gcloud", "storage", "rsync", "--recursive", str(source), destination]
    if dry_run:
        command.append("--dry-run")
    logger.info("%s -> %s", source, destination)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("gcloud CLI no encontrado. Instalá Google Cloud SDK.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Falló la sincronización de {source}: exit {error.returncode}"
        ) from error


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("DATASET_OUTPUT") or os.getenv("DATASET_PATH", "./dataset"),
        help="Directorio del dataset procesado (default: DATASET_OUTPUT o DATASET_PATH)",
    )
    parser.add_argument(
        "--bucket", required=True, help="Bucket destino, con o sin el prefijo gs://"
    )
    parser.add_argument(
        "--release", default=None, help="Nombre de la release (default: fecha + hash del dataset)"
    )
    parser.add_argument(
        "--prefix", default="releases", help="Prefijo de releases dentro del bucket"
    )
    parser.add_argument("--skip-clips", action="store_true", help="No subir segment_clips/")
    parser.add_argument(
        "--dry-run", action="store_true", help="Mostrar qué se subiría sin escribir nada"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    if not (dataset_dir / REQUIRED_FILE).is_file():
        logger.error(
            "No se encontró %s en %s. Corré el pipeline de ingesta primero.",
            REQUIRED_FILE,
            dataset_dir,
        )
        return 1

    bucket = args.bucket.removeprefix("gs://").strip("/")
    release = args.release or default_release_name(dataset_dir)
    release_uri = f"gs://{bucket}/{args.prefix}/{release}"

    for name in STAGED_DIRECTORIES:
        source = dataset_dir / name
        if not source.is_dir():
            logger.error("Falta el directorio %s en el dataset", source)
            return 1
        rsync(source, f"{release_uri}/{name}", args.dry_run)

    clips_dir = dataset_dir / CLIP_DIR_NAME
    clips = sorted(clips_dir.glob(f"segment_*{CLIP_EXTENSION}")) if clips_dir.is_dir() else []
    if args.skip_clips:
        logger.warning("Clips omitidos por --skip-clips: el front no podrá reproducir audio")
    elif not clips:
        logger.warning(
            "No hay clips en %s. Reprocesá el dataset sin --no-segment-clips para habilitar la reproducción.",
            clips_dir,
        )
    else:
        total_mb = sum(clip.stat().st_size for clip in clips) / 1024 / 1024
        logger.info("Subiendo %d clips de reproducción (%.1f MB)", len(clips), total_mb)
        rsync(clips_dir, f"{release_uri}/{CLIP_DIR_NAME}", args.dry_run)

    manifest_path = dataset_dir / "final" / "dataset_manifest.json"
    segments = (
        json.loads(manifest_path.read_text()).get("total_segments")
        if manifest_path.is_file()
        else None
    )

    print()
    print(f"Release publicada: {release_uri}")
    if segments is not None:
        print(f"Segmentos: {segments} | Clips: {len(clips)}")
    print()
    print("Configuración para el deploy del agente:")
    print(f"  DATASET_GCS_URI={release_uri}")
    print(f"  DATASET_PATH=/tmp/audio-search-dataset/{release}")
    print(f"  # SEGMENT_CLIPS_GCS_URI se deriva sola como {release_uri}/{CLIP_DIR_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
