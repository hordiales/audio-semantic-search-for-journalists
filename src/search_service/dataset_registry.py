"""Multi-dataset registry with lazy loading and LRU eviction for the search service."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import TypedDict
from urllib.parse import urlparse

from google.cloud import storage

from src.agent_service.search_engine import AudioSearchEngine
from src.clap_audio_embeddings import CLAPEmbedding
from src.dataset_storage import stage_release
from src.segment_clips import CLIP_DIR_NAME
from src.text_embeddings import TextEmbeddingModel

logger = logging.getLogger(__name__)

_CATALOG_URI_ENV = "DATASET_CATALOG_GCS_URI"
_DEFAULT_DATASET_ID_ENV = "DEFAULT_DATASET_ID"
_CACHE_MAX_ENTRIES_ENV = "DATASET_CACHE_MAX_ENTRIES"
_GCS_URI_ENV = "DATASET_GCS_URI"
_LOCAL_DATASET_PATH_ENV = "DATASET_PATH"
_EPHEMERAL_ROOT = Path("/tmp").resolve()


class DatasetInfo(TypedDict):
    id: str
    label: str
    language: str
    active_indexes: list[str]
    dataset_version: str | None


class DatasetRegistry:
    """Maintain a catalog of datasets and lazily load AudioSearchEngine instances.

    The registry reads a JSON catalog from ``DATASET_CATALOG_GCS_URI`` when
    configured. Each catalog entry maps a stable ``id`` to a versioned GCS
    release URI. Engines are staged on demand under ``/tmp`` and evicted using
    an LRU policy controlled by ``DATASET_CACHE_MAX_ENTRIES``.

    When no catalog is configured the registry falls back to the legacy single
    dataset mode driven by ``DATASET_GCS_URI`` / ``DATASET_PATH``.
    """

    def __init__(self) -> None:
        self._catalog_uri = os.getenv(_CATALOG_URI_ENV, "").strip() or None
        self._default_dataset_id = os.getenv(_DEFAULT_DATASET_ID_ENV, "").strip() or None
        self._max_cache = max(1, int(os.getenv(_CACHE_MAX_ENTRIES_ENV, "2")))
        self._fallback_uri = os.getenv(_GCS_URI_ENV, "").strip() or None
        self._fallback_path = os.getenv(_LOCAL_DATASET_PATH_ENV, "./dataset").strip()
        self._catalog: dict[str, dict] = {}
        self._engines: OrderedDict[str, AudioSearchEngine] = OrderedDict()
        self._shared_text_model: TextEmbeddingModel | None = None
        self._shared_clap_model: CLAPEmbedding | None = None
        self._lock = Lock()
        self._load_catalog()

    def _load_catalog(self) -> None:
        if self._catalog_uri:
            data = self._fetch_catalog(self._catalog_uri)
            version = data.get("version")
            if version != 1:
                raise ValueError(f"Unsupported dataset catalog version: {version}")
            for entry in data.get("datasets", []):
                dataset_id = entry.get("id")
                if dataset_id and entry.get("enabled", True):
                    self._catalog[dataset_id] = entry
            if not self._default_dataset_id and self._catalog:
                self._default_dataset_id = next(iter(self._catalog))

        if not self._catalog:
            dataset_id = self._default_dataset_id or "default"
            self._catalog[dataset_id] = self._single_dataset_entry(dataset_id)

        if not self._default_dataset_id:
            self._default_dataset_id = next(iter(self._catalog))

    def _single_dataset_entry(self, dataset_id: str) -> dict:
        entry = {
            "id": dataset_id,
            "label": "Default dataset",
            "language": "es",
            "enabled": True,
        }
        if self._fallback_uri:
            entry["release_uri"] = self._fallback_uri
        return entry

    @staticmethod
    def _fetch_catalog(catalog_uri: str) -> dict:
        parsed = urlparse(catalog_uri)
        if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError(f"Invalid GCS catalog URI: {catalog_uri}")
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        text = blob.download_as_text()
        return json.loads(text)

    def default_dataset_id(self) -> str:
        return self._default_dataset_id or "default"

    def catalog_entry(self, dataset_id: str) -> dict | None:
        return self._catalog.get(dataset_id)

    def list_datasets(self) -> list[DatasetInfo]:
        """Return public metadata for all configured datasets."""
        result: list[DatasetInfo] = []
        for dataset_id, entry in self._catalog.items():
            engine = self._engines.get(dataset_id)
            result.append(
                DatasetInfo(
                    id=dataset_id,
                    label=entry.get("label") or dataset_id,
                    language=entry.get("language") or "es",
                    active_indexes=engine.active_indexes if engine else [],
                    dataset_version=engine.dataset_version if engine else None,
                )
            )
        return result

    def get_engine(self, dataset_id: str | None) -> AudioSearchEngine:
        """Return a loaded engine for the given dataset, staging it if necessary."""
        dataset_id = dataset_id or self._default_dataset_id
        if not dataset_id:
            raise ValueError("No dataset_id provided and no default dataset configured")
        if dataset_id not in self._catalog:
            available = ", ".join(sorted(self._catalog))
            raise ValueError(f"Dataset '{dataset_id}' not found. Available: {available}")

        with self._lock:
            engine = self._engines.pop(dataset_id, None)
            if engine is not None:
                self._engines[dataset_id] = engine
                return engine

            engine = self._load_engine(dataset_id)
            self._engines[dataset_id] = engine
            self._evict_if_needed()
            return engine

    def _load_engine(self, dataset_id: str) -> AudioSearchEngine:
        entry = self._catalog[dataset_id]
        release_uri = entry.get("release_uri") or self._fallback_uri
        if release_uri:
            local_path = (
                _EPHEMERAL_ROOT
                / "audio-search-datasets"
                / dataset_id
                / Path(release_uri).name
            )
            stage_release(release_uri, local_path)
        else:
            local_path = Path(self._fallback_path)

        clips_uri = f"{release_uri}/{CLIP_DIR_NAME}" if release_uri else None
        engine = AudioSearchEngine(
            str(local_path),
            text_model=self._shared_text_model,
            clap_model=self._shared_clap_model,
            clips_gcs_uri=clips_uri,
        )

        # Share the first loaded model instances with subsequent engines so the
        # CLAP checkpoint (~1.7 GB) and text encoder are only loaded once per
        # container.
        if self._shared_text_model is None:
            self._shared_text_model = engine.text_model
        if self._shared_clap_model is None:
            self._shared_clap_model = engine.clap_model

        logger.info(
            "Loaded dataset '%s' (%d segments, indexes: %s)",
            dataset_id,
            engine.total_segments,
            ", ".join(engine.active_indexes),
        )
        return engine

    def _evict_if_needed(self) -> None:
        while len(self._engines) > self._max_cache:
            oldest_id, oldest_engine = self._engines.popitem(last=False)
            logger.info("Evicting dataset '%s' from cache", oldest_id)
            del oldest_engine
            path = _EPHEMERAL_ROOT / "audio-search-datasets" / oldest_id
            shutil.rmtree(path, ignore_errors=True)
