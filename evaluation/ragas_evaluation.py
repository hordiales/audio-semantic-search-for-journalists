"""Evaluación del agente con RAGAS (Retrieval Augmented Generation Assessment)."""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evaluation.evaluation_config import EvaluationSettings

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
) -> tuple[str, list[str], list[dict]]:
    """
    Query the agent service and capture response + contexts.

    Uses the evaluation endpoint, which returns evidence emitted by retrieval tools.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{service_url}/evaluate/query",
            json={"query": question, "max_results": 5},
        )
        response.raise_for_status()
        data = response.json()

    answer = data.get("response", "")
    contexts = data.get("contexts", [])

    return answer, contexts, data.get("retrieved_segments", [])


def _select_ragas_metrics(has_ground_truth: bool) -> tuple[list, list[str]]:
    """
    Elige el set de métricas RAGAS según si el dataset trae `ground_truth`.

    RAGAS no requiere, por definición, un dataset etiquetado: Faithfulness,
    Answer Relevancy y Context Utilization se calculan exclusivamente a
    partir de (pregunta, contexto recuperado, respuesta generada), sin
    comparar contra una referencia humana. Solo se agregan Context Precision,
    Context Recall y Answer Correctness cuando *todas* las muestras traen
    `ground_truth`, porque esas tres sí lo requieren para funcionar
    (comparan la respuesta/contexto recuperado contra la referencia).
    """
    from ragas.metrics import (
        ContextUtilization,
        answer_relevancy,
        faithfulness,
    )

    # Reference-free: no necesitan ground_truth ni ground_truth_contexts.
    metrics: list = [faithfulness, answer_relevancy, ContextUtilization()]
    metric_names = ["faithfulness", "answer_relevancy", "context_utilization"]

    if has_ground_truth:
        from ragas.metrics import answer_correctness, context_precision, context_recall

        metrics += [context_precision, context_recall, answer_correctness]
        metric_names += ["context_precision", "context_recall", "answer_correctness"]

    return metrics, metric_names


async def run_ragas_evaluation(
    eval_dataset_path: str,
    agent_service_url: str = "http://localhost:8000",
    output_path: str | None = None,
) -> dict:
    """
    Ejecuta evaluación RAGAS contra el servicio del agente.

    No requiere un dataset etiquetado: si las muestras no incluyen
    `ground_truth`, se calculan únicamente las métricas reference-free
    (Faithfulness, Answer Relevancy, Context Utilization). Si todas las
    muestras incluyen `ground_truth`, se agregan también Context Precision,
    Context Recall y Answer Correctness.

    Args:
        eval_dataset_path: Ruta al JSON con preguntas (y opcionalmente ground truth)
        agent_service_url: URL del servicio del agente
        output_path: Optional path to save results

    Returns:
        Dict con scores por métrica
    """
    settings = EvaluationSettings.from_environment()
    if settings.framework != "ragas":
        raise ValueError("EVALUATION_FRAMEWORK must be 'ragas' for this command")
    from datasets import Dataset
    from ragas import evaluate

    eval_samples = load_eval_dataset(eval_dataset_path)
    logger.info("Loaded %d evaluation samples", len(eval_samples))

    results = []
    for i, sample in enumerate(eval_samples):
        logger.info("Evaluating %d/%d: %s", i + 1, len(eval_samples), sample["question"][:50])

        try:
            answer, contexts, retrieved_segments = await query_agent_with_context(
                agent_service_url, sample["question"]
            )
        except Exception as e:
            logger.error("Failed to query agent: %s", e)
            answer = f"Error: {e}"
            contexts = []
            retrieved_segments = []

        results.append({
            "question": sample["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": sample.get("ground_truth", ""),
            "retrieved_segments": retrieved_segments,
        })

    has_ground_truth = all(r["ground_truth"] for r in results)
    metrics, metric_names = _select_ragas_metrics(has_ground_truth)
    logger.info(
        "Dataset %s ground_truth; usando métricas: %s",
        "incluye" if has_ground_truth else "no incluye",
        ", ".join(metric_names),
    )

    dataset = Dataset.from_list(results)

    logger.info("Running RAGAS evaluation...")
    ragas_result = evaluate(dataset, metrics=metrics)
    score_df = ragas_result.to_pandas()

    metrics_scores = {}
    for name in metric_names:
        if name in score_df.columns:
            values = score_df[name].dropna()
            if not values.empty:
                metrics_scores[name] = float(values.mean())

    # Add per-question metric scores back into the raw results for inspection
    for name in metric_names:
        if name in score_df.columns:
            for row, value in zip(results, score_df[name].to_list()):
                row[name] = None if (isinstance(value, float) and value != value) else value

    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "framework": "ragas",
        "dataset_source": settings.dataset_source,
        "dataset_path": str(eval_dataset_path),
        "metrics": metrics_scores,
        "metrics_used": metric_names,
        "reference_free": not has_ground_truth,
        "per_question": results,
        "config": {
            "agent_url": agent_service_url,
            "dataset_size": len(eval_samples),
            "judge_model": settings.judge_model,
            "judge_temperature": settings.judge_temperature,
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

    from dotenv import load_dotenv

    load_dotenv()

    settings = EvaluationSettings.from_environment()
    if settings.framework != "ragas":
        raise ValueError("EVALUATION_FRAMEWORK must be 'ragas' for this command")

    parser = argparse.ArgumentParser(description="Evaluación RAGAS del agente")
    parser.add_argument(
        "--dataset",
        default=settings.dataset_path,
        help="JSON con preguntas de evaluación (por defecto: EVALUATION_DATASET_SOURCE)",
    )
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
