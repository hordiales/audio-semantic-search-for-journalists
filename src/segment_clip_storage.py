"""On-demand access to per-segment playback clips stored in Cloud Storage.

Unlike the FAISS indices and the dataset pickle, playback clips are *not*
staged to local disk at startup: the corpus grows linearly with audio hours and
only the handful of segments a journalist actually clicks needs to be fetched.
Clips are therefore left in the bucket and exposed to the browser as short-lived
V4 signed URLs, so audio bytes never transit the agent service.

Local development keeps working without a bucket: when no GCS URI is configured
the store resolves clips to their path inside ``DATASET_PATH``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

from src.segment_clips import CLIP_DIR_NAME, clip_file_name

logger = logging.getLogger(__name__)

_CLIPS_GCS_URI_ENV = "SEGMENT_CLIPS_GCS_URI"
_DATASET_GCS_URI_ENV = "DATASET_GCS_URI"
_TTL_ENV = "SEGMENT_CLIP_URL_TTL_SECONDS"
_DEFAULT_TTL_SECONDS = 900
# Reuse a cached URL only while it still has enough life left for playback.
_URL_REUSE_MARGIN_SECONDS = 60


def resolve_clips_gcs_uri() -> str:
    """Return the configured GCS prefix holding the playback clips, if any.

    Defaults to ``<DATASET_GCS_URI>/segment_clips`` so a release published as a
    single prefix needs no extra configuration.
    """
    explicit = os.getenv(_CLIPS_GCS_URI_ENV, "").strip()
    if explicit:
        return explicit.rstrip("/")
    dataset_uri = os.getenv(_DATASET_GCS_URI_ENV, "").strip().rstrip("/")
    return f"{dataset_uri}/{CLIP_DIR_NAME}" if dataset_uri else ""


@dataclass(frozen=True)
class ClipReference:
    """Where a segment's audio lives and how long the reference stays valid."""

    segment_id: int
    file_name: str
    url: str
    expires_at: float | None
    location: str  # "gcs" | "local"


class SegmentClipStore:
    """Resolve segment ids to playable audio URLs without downloading the corpus."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        clips_gcs_uri: str | None = None,
        ttl_seconds: int | None = None,
    ):
        self._dataset_path = Path(dataset_path or os.getenv("DATASET_PATH", "./dataset"))
        self._clips_uri = (
            clips_gcs_uri if clips_gcs_uri is not None else resolve_clips_gcs_uri()
        ).rstrip("/")
        self._ttl_seconds = int(ttl_seconds or os.getenv(_TTL_ENV, _DEFAULT_TTL_SECONDS))
        self._bucket_name, self._prefix = (
            _parse_gcs_uri(self._clips_uri) if self._clips_uri else ("", "")
        )
        self._bucket = None
        self._credentials = None
        self._cache: dict[str, ClipReference] = {}
        self._lock = Lock()

    @property
    def uses_gcs(self) -> bool:
        return bool(self._clips_uri)

    @property
    def local_dir(self) -> Path:
        return self._dataset_path / CLIP_DIR_NAME

    def local_path(self, segment_id: int) -> Path:
        """Path of a clip inside the local dataset directory (may not exist)."""
        return self.local_dir / clip_file_name(segment_id)

    def reference(self, segment_id: int) -> ClipReference | None:
        """Return a playable reference for a segment, or ``None`` if unavailable."""
        name = clip_file_name(segment_id)
        if not self.uses_gcs:
            path = self.local_path(segment_id)
            if not path.is_file():
                return None
            return ClipReference(segment_id, name, path.as_uri(), None, "local")

        with self._lock:
            cached = self._cache.get(name)
            if (
                cached
                and cached.expires_at
                and cached.expires_at - time.time() > _URL_REUSE_MARGIN_SECONDS
            ):
                return cached

        url, expires_at = self._signed_url(name)
        reference = ClipReference(segment_id, name, url, expires_at, "gcs")
        with self._lock:
            self._cache[name] = reference
        return reference

    def _signed_url(self, name: str) -> tuple[str, float]:
        from datetime import timedelta

        blob = self._get_bucket().blob(f"{self._prefix}/{name}" if self._prefix else name)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=self._ttl_seconds),
            method="GET",
            response_type="audio/ogg",
            **self._signing_kwargs(),
        )
        # A private bucket only accepts a V4 URL carrying these query
        # parameters. Do not silently return a plain object URL: browsers would
        # reach GCS as anonymous users and show an opaque AccessDenied XML page.
        query = parse_qs(urlparse(url).query)
        if (
            query.get("X-Goog-Algorithm") != ["GOOG4-RSA-SHA256"]
            or not query.get("X-Goog-Signature")
        ):
            raise RuntimeError(
                "Clip URL signing did not produce a V4 signed URL; refusing to expose "
                f"the private object {name}."
            )
        return url, time.time() + self._ttl_seconds

    def _get_bucket(self):
        if self._bucket is None:
            from google.cloud import storage

            self._bucket = storage.Client().bucket(self._bucket_name)
        return self._bucket

    def _signing_kwargs(self) -> dict:
        """Return the extra arguments V4 signing needs on keyless runtimes.

        Cloud Run and Agent Runtime hold no private key, so signing is delegated
        to the IAM ``signBlob`` API, which requires the runtime service account
        to hold ``roles/iam.serviceAccountTokenCreator`` on itself. Credentials
        backed by a key file sign locally and need none of this.
        """
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account

        credentials = self._get_credentials()
        if isinstance(credentials, service_account.Credentials):
            return {}

        if not credentials.valid:
            credentials.refresh(Request())
        email = getattr(credentials, "service_account_email", None)
        if not email or email == "default":
            raise RuntimeError(
                "Cannot sign clip URLs: the active credentials have no service account "
                f"identity. Deploy with a service account, or leave {_CLIPS_GCS_URI_ENV} "
                "and DATASET_GCS_URI unset to serve clips from the local dataset."
            )
        return {"service_account_email": email, "access_token": credentials.token}

    def _get_credentials(self):
        if self._credentials is None:
            import google.auth

            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        return self._credentials


def _parse_gcs_uri(source_uri: str) -> tuple[str, str]:
    parsed = urlparse(source_uri)
    prefix = parsed.path.lstrip("/").rstrip("/")
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(
            f"{_CLIPS_GCS_URI_ENV} must be a GCS prefix such as "
            "gs://audio-search-datasets/releases/2026-08-26/segment_clips"
        )
    return parsed.netloc, prefix
