"""Client the agent uses to reach the retrieval service over HTTP.

Mirrors the method names of ``AudioSearchEngine`` so ``tools.py`` can hold
either one without caring which. Everything here is intentionally free of
PyTorch, FAISS and pandas: this module is what allows the agent image to be
built with ``--no-group ml``.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

_URL_ENV = "SEARCH_SERVICE_URL"
_AUTH_ENV = "SEARCH_SERVICE_AUTH"
_TIMEOUT_ENV = "SEARCH_SERVICE_TIMEOUT"
# Cloud Run ID tokens live for an hour. Refreshing well before that avoids
# racing the expiry without minting a token on every single tool call.
_TOKEN_TTL_SEC = 1800.0
_DEFAULT_TIMEOUT_SEC = 30.0


class SearchServiceError(RuntimeError):
    """Raised when the retrieval service is unreachable or returns an error."""


def search_service_url() -> str | None:
    """Return the configured retrieval service URL, if any.

    Absence is meaningful rather than an error: it selects the in-process engine
    used for local development and the ingestion pipeline.
    """
    url = os.environ.get(_URL_ENV, "").strip()
    return url.rstrip("/") or None


class SearchServiceClient:
    """Calls the Cloud Run retrieval service with an IAM identity token."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        resolved = (base_url or search_service_url() or "").rstrip("/")
        if not resolved:
            raise SearchServiceError(f"{_URL_ENV} is not set.")
        self.base_url = resolved
        if timeout is None:
            timeout = float(os.environ.get(_TIMEOUT_ENV, _DEFAULT_TIMEOUT_SEC))
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        # "none" is for a service exposed without IAM, e.g. a local uvicorn.
        self._use_auth = os.environ.get(_AUTH_ENV, "iam").lower() != "none"
        self._token: str | None = None
        self._token_expires_at = 0.0

    def _auth_headers(self) -> dict[str, str]:
        if not self._use_auth:
            return {}
        now = time.monotonic()
        if self._token is None or now >= self._token_expires_at:
            # Imported lazily so that unauthenticated local runs do not require
            # ambient Google credentials to exist.
            from google.auth.transport.requests import Request
            from google.oauth2 import id_token

            try:
                self._token = id_token.fetch_id_token(Request(), self.base_url)
            except Exception as error:  # noqa: BLE001 - reported as a tool error
                raise SearchServiceError(
                    f"Could not mint an identity token for {self.base_url}: {error}"
                ) from error
            self._token_expires_at = now + _TOKEN_TTL_SEC
        return {"Authorization": f"Bearer {self._token}"}

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self._client.request(method, path, headers=self._auth_headers(), **kwargs)
        except httpx.HTTPError as error:
            raise SearchServiceError(
                f"Retrieval service unreachable at {self.base_url}: {error}"
            ) from error
        if response.status_code == 404:
            return response
        if response.status_code >= 400:
            # 503 here almost always means the service is still loading the CLAP
            # checkpoint, which is worth distinguishing in the agent's answer.
            raise SearchServiceError(
                f"Retrieval service returned {response.status_code}: {response.text[:200]}"
            )
        return response

    def _search(self, path: str, query: str, k: int) -> list[dict]:
        response = self._request("POST", path, json={"query": query, "k": k})
        return response.json().get("results", [])

    def search_semantic(self, query_text: str, k: int = 5) -> list[dict]:
        return self._search("/search/semantic", query_text, k)

    def search_audio_by_text(self, query_text: str, k: int = 5) -> list[dict]:
        return self._search("/search/audio", query_text, k)

    def search_audio_by_classes(self, query_text: str, k: int = 5) -> list[dict]:
        return self._search("/search/yamnet", query_text, k)

    def get_segment_info(self, segment_id: int) -> dict | None:
        response = self._request("GET", f"/segments/{segment_id}")
        if response.status_code == 404:
            return None
        return response.json().get("segment")

    def get_audio_classes(self, segment_id: int) -> list[dict] | None:
        response = self._request("GET", f"/segments/{segment_id}/audio-classes")
        if response.status_code == 404:
            return None
        return response.json().get("classes", [])
