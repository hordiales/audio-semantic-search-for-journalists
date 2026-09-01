import pytest
from evaluation.evaluation_config import EvaluationSettings


def test_evaluation_settings_default_to_ragas(monkeypatch):
    monkeypatch.delenv("EVALUATION_FRAMEWORK", raising=False)
    monkeypatch.delenv("EVALUATION_JUDGE_TEMPERATURE", raising=False)

    settings = EvaluationSettings.from_environment()

    assert settings.framework == "ragas"
    assert settings.judge_temperature == 0


def test_evaluation_settings_support_deepeval(monkeypatch):
    monkeypatch.setenv("EVALUATION_FRAMEWORK", "deepeval")
    monkeypatch.setenv("EVALUATION_JUDGE_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EVALUATION_JUDGE_TEMPERATURE", "0")

    settings = EvaluationSettings.from_environment()

    assert settings.framework == "deepeval"


def test_evaluation_settings_reject_non_deterministic_judge(monkeypatch):
    monkeypatch.setenv("EVALUATION_JUDGE_TEMPERATURE", "0.2")

    with pytest.raises(ValueError, match="must be 0"):
        EvaluationSettings.from_environment()
