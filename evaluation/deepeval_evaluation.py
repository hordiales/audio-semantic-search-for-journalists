"""DeepEval backend using the same `(question, contexts, answer)` contract as RAGAS."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from evaluation.evaluation_config import EvaluationSettings
from evaluation.ragas_evaluation import load_eval_dataset, query_agent_with_context

logger = logging.getLogger(__name__)


def _metric_result(metric: object, test_case: object) -> dict:
    metric.measure(test_case)
    return {
        "score": float(metric.score),
        "reason": str(getattr(metric, "reason", "")),
    }


async def run_deepeval_evaluation(
    eval_dataset_path: str,
    agent_service_url: str = "http://localhost:8000",
    output_path: str | None = None,
) -> dict:
    """Score the agent with DeepEval's equivalent RAG metrics and one common schema."""
    settings = EvaluationSettings.from_environment()
    if settings.framework != "deepeval":
        raise ValueError("EVALUATION_FRAMEWORK must be 'deepeval' for this command")
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError as error:
        raise RuntimeError(
            "DeepEval is not installed. Run `uv sync --extra eval-deepeval`."
        ) from error

    samples = load_eval_dataset(eval_dataset_path)
    per_question: list[dict] = []
    metric_totals = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }
    for index, sample in enumerate(samples, start=1):
        logger.info("Evaluating %d/%d: %s", index, len(samples), sample["question"][:50])
        answer, contexts, retrieved_segments = await query_agent_with_context(
            agent_service_url, sample["question"]
        )
        test_case = LLMTestCase(
            input=sample["question"],
            actual_output=answer,
            expected_output=sample.get("ground_truth", ""),
            retrieval_context=contexts,
        )
        metrics = {
            "faithfulness": FaithfulnessMetric(
                threshold=0.7, model=settings.judge_model, include_reason=True
            ),
            "answer_relevancy": AnswerRelevancyMetric(
                threshold=0.7, model=settings.judge_model, include_reason=True
            ),
            "context_precision": ContextualPrecisionMetric(
                threshold=0.6, model=settings.judge_model, include_reason=True
            ),
            "context_recall": ContextualRecallMetric(
                threshold=0.6, model=settings.judge_model, include_reason=True
            ),
        }
        scores = {name: _metric_result(metric, test_case) for name, metric in metrics.items()}
        for name, result in scores.items():
            metric_totals[name].append(result["score"])
        per_question.append(
            {
                "question": sample["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": sample.get("ground_truth", ""),
                "retrieved_segments": retrieved_segments,
                "metrics": scores,
            }
        )

    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "framework": "deepeval",
        "metrics": {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in metric_totals.items()
        },
        "per_question": per_question,
        "config": {
            "agent_url": agent_service_url,
            "dataset_size": len(samples),
            "judge_model": settings.judge_model,
            "judge_temperature": settings.judge_temperature,
        },
    }
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return output
