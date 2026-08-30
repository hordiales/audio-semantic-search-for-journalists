"""Configuration shared by the interchangeable agent-evaluation backends."""

from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED_EVALUATION_FRAMEWORKS = frozenset({"ragas", "deepeval"})
SUPPORTED_DATASET_SOURCES = frozenset({"manual", "synthetic", "hybrid"})


@dataclass(frozen=True)
class EvaluationSettings:
    """Environment-controlled evaluation settings recorded with every run."""

    framework: str = "ragas"
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    dataset_source: str = "manual"
    dataset_path: str = "evaluation/test_datasets/ragas_eval_dataset_unlabeled.json"
    dataset_size: int = 20
    dataset_max_segments: int = 0
    manual_dataset_path: str = "evaluation/test_datasets/ragas_eval_dataset_unlabeled.json"
    synthetic_dataset_path: str = "evaluation/test_datasets/synthetic/ragas_synthetic_questions.json"
    hybrid_dataset_path: str = "evaluation/test_datasets/synthetic/ragas_hybrid_questions.json"
    openai_model: str = "gpt-4o-mini"
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def from_environment(cls) -> EvaluationSettings:
        framework = os.getenv("EVALUATION_FRAMEWORK", "ragas").strip().lower()
        if framework not in SUPPORTED_EVALUATION_FRAMEWORKS:
            supported = ", ".join(sorted(SUPPORTED_EVALUATION_FRAMEWORKS))
            raise ValueError(f"EVALUATION_FRAMEWORK must be one of: {supported}")

        dataset_source = os.getenv("EVALUATION_DATASET_SOURCE", cls.dataset_source).strip().lower()
        if dataset_source not in SUPPORTED_DATASET_SOURCES:
            supported = ", ".join(sorted(SUPPORTED_DATASET_SOURCES))
            raise ValueError(f"EVALUATION_DATASET_SOURCE must be one of: {supported}")

        try:
            judge_temperature = float(os.getenv("EVALUATION_JUDGE_TEMPERATURE", "0"))
        except ValueError as error:
            raise ValueError("EVALUATION_JUDGE_TEMPERATURE must be numeric") from error
        if judge_temperature != 0:
            raise ValueError("EVALUATION_JUDGE_TEMPERATURE must be 0 for reproducible grading")

        try:
            dataset_size = int(os.getenv("EVALUATION_DATASET_SIZE", str(cls.dataset_size)))
            dataset_max_segments = int(os.getenv("EVALUATION_DATASET_MAX_SEGMENTS", str(cls.dataset_max_segments)))
        except ValueError as error:
            raise ValueError("EVALUATION_DATASET_SIZE and EVALUATION_DATASET_MAX_SEGMENTS must be integers") from error

        manual_dataset_path = os.getenv("EVALUATION_MANUAL_DATASET_PATH", cls.manual_dataset_path)
        synthetic_dataset_path = os.getenv("EVALUATION_SYNTHETIC_DATASET_PATH", cls.synthetic_dataset_path)
        hybrid_dataset_path = os.getenv("EVALUATION_HYBRID_DATASET_PATH", cls.hybrid_dataset_path)
        dataset_path = os.getenv("EVALUATION_DATASET_PATH")
        if dataset_path is None:
            if dataset_source == "synthetic":
                dataset_path = synthetic_dataset_path
            elif dataset_source == "hybrid":
                dataset_path = hybrid_dataset_path
            else:
                dataset_path = manual_dataset_path

        return cls(
            framework=framework,
            judge_model=os.getenv("EVALUATION_JUDGE_MODEL", cls.judge_model),
            judge_temperature=judge_temperature,
            dataset_source=dataset_source,
            dataset_path=dataset_path,
            dataset_size=dataset_size,
            dataset_max_segments=dataset_max_segments,
            manual_dataset_path=manual_dataset_path,
            synthetic_dataset_path=synthetic_dataset_path,
            hybrid_dataset_path=hybrid_dataset_path,
            openai_model=os.getenv("OPENAI_MODEL", cls.openai_model),
            text_embedding_model=os.getenv("TEXT_EMBEDDING_MODEL", cls.text_embedding_model),
        )
