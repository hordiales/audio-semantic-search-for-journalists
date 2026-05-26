"""Evaluación del agente con RAGAS (Retrieval Augmented Generation Assessment)."""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class ContextCaptureHandler:
    """Captures tool outputs for RAGAS evaluation."""

    def __init__(self):
        self.captured_contexts: list[str] = []

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Captura el output de cada tool call."""
        try:
            results = json.loads(output)
            if isinstance(results, list):
                for r in results:
                    if "text" in r:
                        self.captured_contexts.append(r["text"])
        except (json.JSONDecodeError, TypeError):
            pass


def load_eval_dataset(path: str) -> list[dict]:
    """Load evaluation dataset from JSON."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    return data.get("samples", data.get("questions", []))


async def query_agent_with_context(
    service_url: str, question: str
) -> tuple[str, list[str]]:
    """
    Query the agent service and capture response + contexts.

    Uses the /query endpoint and parses tool outputs from the response.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{service_url}/query",
            json={"query": question, "max_results": 5},
        )
        response.raise_for_status()
        data = response.json()

    answer = data.get("response", "")
    contexts = data.get("contexts", [])

    return answer, contexts


async def run_ragas_evaluation(
    eval_dataset_path: str,
    agent_service_url: str = "http://localhost:8000",
    output_path: str | None = None,
) -> dict:
    """
    Ejecuta evaluación RAGAS contra el servicio del agente.

    Args:
        eval_dataset_path: Ruta al JSON con preguntas y ground truth
        agent_service_url: URL del servicio del agente
        output_path: Optional path to save results

    Returns:
        Dict con scores por métrica
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    eval_samples = load_eval_dataset(eval_dataset_path)
    logger.info("Loaded %d evaluation samples", len(eval_samples))

    results = []
    for i, sample in enumerate(eval_samples):
        logger.info("Evaluating %d/%d: %s", i + 1, len(eval_samples), sample["question"][:50])

        try:
            answer, contexts = await query_agent_with_context(
                agent_service_url, sample["question"]
            )
        except Exception as e:
            logger.error("Failed to query agent: %s", e)
            answer = f"Error: {e}"
            contexts = []

        if not contexts:
            contexts = sample.get("ground_truth_contexts", [""])

        results.append({
            "question": sample["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": sample.get("ground_truth", ""),
        })

    dataset = Dataset.from_list(results)

    logger.info("Running RAGAS evaluation...")
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
    )

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))},
        "per_question": results,
        "config": {
            "agent_url": agent_service_url,
            "dataset_size": len(eval_samples),
        },
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        logger.info("Results saved to %s", output_path)

    return output


def main():
    import asyncio

    parser = argparse.ArgumentParser(description="Evaluación RAGAS del agente")
    parser.add_argument("--dataset", required=True, help="JSON con preguntas de evaluación")
    parser.add_argument("--agent-url", default="http://localhost:8000")
    parser.add_argument("--output", required=True, help="Path para resultados JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = asyncio.run(
        run_ragas_evaluation(args.dataset, args.agent_url, args.output)
    )

    print(f"\n{'='*50}")
    print("RAGAS EVALUATION RESULTS")
    print(f"{'='*50}")
    for metric, score in results.get("metrics", {}).items():
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()
