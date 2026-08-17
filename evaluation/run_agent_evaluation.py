"""Run the selected RAG evaluator without changing datasets or result contracts."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from evaluation.evaluation_config import EvaluationSettings

logger = logging.getLogger(__name__)


def _load_samples(path: str) -> list[dict]:
    """Carga un dataset de evaluación JSON en formato lista o dict envuelto."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("samples", data.get("questions", []))


def _build_hybrid_dataset(settings: EvaluationSettings) -> str:
    """Combina el dataset manual y el sintético, eliminando preguntas duplicadas."""
    manual = _load_samples(settings.manual_dataset_path)
    synthetic = _load_samples(settings.synthetic_dataset_path)

    seen: dict[str, int] = {}
    merged: list[dict] = []
    for sample in manual + synthetic:
        q = (sample.get("question") or "").strip().lower()
        if not q:
            continue
        has_ref = bool(sample.get("ground_truth"))
        if q in seen:
            idx = seen[q]
            if has_ref and not merged[idx].get("ground_truth"):
                merged[idx] = sample
            continue
        seen[q] = len(merged)
        merged.append(sample)

    out = Path(settings.hybrid_dataset_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Dataset híbrido construido: %d preguntas (%d manual + %d sintético) en %s",
        len(merged),
        len(manual),
        len(synthetic),
        out,
    )
    return str(out)


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
    settings = EvaluationSettings.from_environment()
    parser = argparse.ArgumentParser(description="Evaluación RAG con RAGAS o DeepEval")
    parser.add_argument(
        "--dataset",
        default=settings.dataset_path,
        help="Dataset de evaluación (por defecto: EVALUATION_DATASET_SOURCE)",
    )
    parser.add_argument("--agent-url", default="http://localhost:8000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = args.dataset
    if settings.dataset_source == "hybrid":
        dataset = _build_hybrid_dataset(settings)

    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_selected_evaluation(dataset, args.agent_url, args.output))
    print(f"{result['framework'].upper()} EVALUATION RESULTS")
    for name, score in result["metrics"].items():
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
