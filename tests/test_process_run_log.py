import json

from src.embedding_config import EmbeddingConfig
from src.simple_dataset_pipeline import _write_process_run_log


def test_process_run_log_records_effective_parameters_and_command(tmp_path):
    log_path = tmp_path / "process_run.json"
    _write_process_run_log(
        log_path,
        parameters={"whisper_model": "base", "chunk_strategy": "fixed"},
        embedding_config=EmbeddingConfig(active_embeddings=frozenset({"text", "clap"})),
        command_argv=["python", "-m", "src.simple_dataset_pipeline", "--chunk-strategy", "fixed"],
    )

    log = json.loads(log_path.read_text())

    assert log["parameters"]["chunk_strategy"] == "fixed"
    assert log["embedding_configuration"]["active_embeddings"] == ["clap", "text"]
    assert log["embedding_configuration"]["active_classifiers"] == []
    assert log["command_argv"][-2:] == ["--chunk-strategy", "fixed"]
    assert log["completed_at"]
