import shutil
from pathlib import Path

import pytest

from src.search_service import dataset_registry as registry_module
from src.search_service.dataset_registry import DatasetRegistry


class _FakeEngine:
    def __init__(self, dataset_path, text_model=None, clap_model=None, clips_gcs_uri=None):
        self.dataset_path = dataset_path
        self._text_model = text_model
        self._clap_model = clap_model
        self.total_segments = 10
        self.active_indexes = ["text"]
        self.dataset_version = "v1"

    @property
    def text_model(self):
        if self._text_model is None:
            self._text_model = object()
        return self._text_model

    @property
    def clap_model(self):
        if self._clap_model is None:
            self._clap_model = object()
        return self._clap_model


@pytest.fixture
def fake_registry(monkeypatch, tmp_path):
    def fake_stage_release(source_uri, destination):
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "final").mkdir(exist_ok=True)
        (dest / "final" / "complete_dataset.pkl").write_bytes(b"dataset")
        return str(dest)

    catalog = {
        "version": 1,
        "datasets": [
            {
                "id": "news",
                "label": "Noticias 2026",
                "language": "es",
                "release_uri": "gs://bucket/releases/news-2026",
                "enabled": True,
            },
            {
                "id": "culture",
                "label": "Cultura",
                "language": "es",
                "release_uri": "gs://bucket/releases/culture-2026",
                "enabled": True,
            },
            {
                "id": "disabled",
                "label": "Deshabilitado",
                "release_uri": "gs://bucket/releases/disabled",
                "enabled": False,
            },
        ],
    }

    monkeypatch.setattr(DatasetRegistry, "_fetch_catalog", staticmethod(lambda uri: catalog))
    monkeypatch.setattr(registry_module, "stage_release", fake_stage_release)
    monkeypatch.setattr(registry_module, "AudioSearchEngine", _FakeEngine)
    monkeypatch.setenv("DATASET_CATALOG_GCS_URI", "gs://config/catalog.json")
    monkeypatch.setenv("DEFAULT_DATASET_ID", "culture")
    monkeypatch.setenv("DATASET_CACHE_MAX_ENTRIES", "1")

    registry = DatasetRegistry()
    yield registry
    # Clean up any staged directories created during tests.
    for dataset_id in catalog["datasets"]:
        if dataset_id["enabled"]:
            shutil.rmtree(Path("/tmp") / "audio-search-datasets" / dataset_id["id"], ignore_errors=True)


def test_registry_lists_enabled_datasets(fake_registry):
    datasets = fake_registry.list_datasets()
    ids = [dataset["id"] for dataset in datasets]

    assert "news" in ids
    assert "culture" in ids
    assert "disabled" not in ids
    assert datasets[ids.index("culture")]["label"] == "Cultura"


def test_registry_returns_configured_default(fake_registry):
    assert fake_registry.default_dataset_id() == "culture"


def test_registry_loads_requested_engine(fake_registry):
    engine = fake_registry.get_engine("news")

    assert engine.total_segments == 10
    assert Path(engine.dataset_path).name == "news-2026"


def test_registry_rejects_unknown_dataset(fake_registry):
    with pytest.raises(ValueError, match="not found"):
        fake_registry.get_engine("missing")


def test_registry_uses_default_when_dataset_id_is_none(fake_registry):
    engine = fake_registry.get_engine(None)

    assert Path(engine.dataset_path).name == "culture-2026"


def test_registry_caches_engines(fake_registry):
    engine1 = fake_registry.get_engine("news")
    engine2 = fake_registry.get_engine("news")

    assert engine1 is engine2


def test_registry_evicts_oldest_engine_when_cache_is_full(fake_registry):
    engine1 = fake_registry.get_engine("news")
    engine2 = fake_registry.get_engine("culture")

    assert fake_registry.get_engine("news") is not engine1
    assert fake_registry.get_engine("culture") is not engine2


def test_registry_without_catalog_uses_single_dataset_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("DATASET_CATALOG_GCS_URI", raising=False)
    monkeypatch.setenv("DATASET_GCS_URI", "gs://bucket/releases/default")
    monkeypatch.setenv("DATASET_PATH", str(tmp_path))
    monkeypatch.setattr(registry_module, "stage_release", lambda source, dest: str(tmp_path))
    monkeypatch.setattr(registry_module, "AudioSearchEngine", _FakeEngine)

    registry = DatasetRegistry()

    datasets = registry.list_datasets()
    assert len(datasets) == 1
    assert datasets[0]["id"] == "default"
    assert registry.default_dataset_id() == "default"
