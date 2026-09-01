from __future__ import annotations

import httpx
import pytest

from src.agent_service.search_client import (
    SearchServiceClient,
    SearchServiceError,
    search_service_url,
)


def _client(handler, monkeypatch, **env) -> SearchServiceClient:
    """Build a client whose transport is a local handler instead of the network."""
    monkeypatch.setenv("SEARCH_SERVICE_AUTH", env.pop("auth", "none"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    client = SearchServiceClient("https://search.example.com")
    client._client = httpx.Client(
        base_url="https://search.example.com", transport=httpx.MockTransport(handler)
    )
    return client


def test_search_service_url_treats_blank_as_unset(monkeypatch):
    monkeypatch.setenv("SEARCH_SERVICE_URL", "   ")
    assert search_service_url() is None
    monkeypatch.setenv("SEARCH_SERVICE_URL", "https://search.example.com/")
    assert search_service_url() == "https://search.example.com"


def test_missing_url_is_reported_as_configuration_error(monkeypatch):
    monkeypatch.delenv("SEARCH_SERVICE_URL", raising=False)
    with pytest.raises(SearchServiceError, match="SEARCH_SERVICE_URL"):
        SearchServiceClient()


def test_text_clap_and_yamnet_searches_hit_their_own_endpoints(monkeypatch):
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"results": [{"segment": {"segment_id": 7}}]})

    client = _client(handler, monkeypatch)
    assert client.search_semantic("aplausos", k=3)[0]["segment"]["segment_id"] == 7
    client.search_audio_by_text("música", k=2)
    client.search_audio_by_classes("aplausos", k=4)

    assert seen == [
        ("/search/semantic", {"query": "aplausos", "k": 3}),
        ("/search/audio", {"query": "música", "k": 2}),
        ("/search/yamnet", {"query": "aplausos", "k": 4}),
    ]


def test_segment_lookups_map_404_to_none_instead_of_raising(monkeypatch):
    client = _client(lambda request: httpx.Response(404, json={"detail": "nope"}), monkeypatch)
    assert client.get_segment_info(42) is None
    assert client.get_audio_classes(42) is None


def test_audio_classes_distinguishes_empty_list_from_missing_segment(monkeypatch):
    client = _client(lambda request: httpx.Response(200, json={"classes": []}), monkeypatch)
    # An ingested-without-YAMNet segment must not look like a missing segment,
    # because the agent reports those two cases differently.
    assert client.get_audio_classes(1) == []


def test_server_error_becomes_search_service_error(monkeypatch):
    client = _client(lambda request: httpx.Response(503, text="warming up"), monkeypatch)
    with pytest.raises(SearchServiceError, match="503"):
        client.search_semantic("aplausos")


def test_transport_failure_names_the_unreachable_service(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client(handler, monkeypatch)
    with pytest.raises(SearchServiceError, match="unreachable at https://search.example.com"):
        client.search_semantic("aplausos")


def test_identity_token_is_minted_once_and_reused(monkeypatch):
    from google.oauth2 import id_token

    calls: list[str] = []

    def fake_fetch_id_token(request, audience):
        calls.append(audience)
        return "fake-token"

    monkeypatch.setattr(id_token, "fetch_id_token", fake_fetch_id_token)

    seen_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json={"results": []})

    client = _client(handler, monkeypatch, auth="iam")
    client.search_semantic("uno")
    client.search_semantic("dos")

    assert seen_headers == ["Bearer fake-token", "Bearer fake-token"]
    assert calls == ["https://search.example.com"]
