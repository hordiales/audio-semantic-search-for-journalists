"""Configuration shared by the interchangeable agent-evaluation backends."""

from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED_EVALUATION_FRAMEWORKS = frozenset({"ragas", "deepeval"})


@dataclass(frozen=True)
class EvaluationSettings:
    """Environment-controlled evaluation settings recorded with every run."""

    framework: str = "ragas"
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.0

    @classmethod
    def from_environment(cls) -> EvaluationSettings:
        framework = os.getenv("EVALUATION_FRAMEWORK", "ragas").strip().lower()
        if framework not in SUPPORTED_EVALUATION_FRAMEWORKS:
            supported = ", ".join(sorted(SUPPORTED_EVALUATION_FRAMEWORKS))
            raise ValueError(f"EVALUATION_FRAMEWORK must be one of: {supported}")
        try:
            judge_temperature = float(os.getenv("EVALUATION_JUDGE_TEMPERATURE", "0"))
        except ValueError as error:
            raise ValueError("EVALUATION_JUDGE_TEMPERATURE must be numeric") from error
        if judge_temperature != 0:
            raise ValueError("EVALUATION_JUDGE_TEMPERATURE must be 0 for reproducible grading")
        return cls(
            framework=framework,
            judge_model=os.getenv("EVALUATION_JUDGE_MODEL", "gpt-4o-mini"),
            judge_temperature=judge_temperature,
        )
