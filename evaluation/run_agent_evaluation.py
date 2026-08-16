"""Run the selected RAG evaluator without changing datasets or result contracts."""

from __future__ import annotations

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from evaluation.evaluation_config import EvaluationSettings


async def run_selected_evaluation(dataset: str, agent_url: str, output: str) -> dict:
    """Dispatch to the backend selected by ``EVALUATION_FRAMEWORK``."""
    settings = EvaluationSettings.from_environment()
    if settings.framework == "ragas":
        from evaluation.ragas_evaluation import run_ragas_evaluation

        return await run_ragas_evaluation(dataset, agent_url, output)
    from evaluation.deepeval_evaluation import run_deepeval_evaluation

    return await run_deepeval_evaluation(dataset, agent_url, output)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluación RAG con RAGAS o DeepEval")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--agent-url", default="http://localhost:8000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_selected_evaluation(args.dataset, args.agent_url, args.output))
    print(f"{result['framework'].upper()} EVALUATION RESULTS")
    for name, score in result["metrics"].items():
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
