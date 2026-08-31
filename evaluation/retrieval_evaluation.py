"""Evaluación del retrieval (RAG aislado) con métricas de Information Retrieval."""

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at: dict[int, float]
    recall_at: dict[int, float]
    f1_at: dict[int, float]
    mrr: float
    ndcg_at: dict[int, float]
    fp_cost_at: dict[int, float]


def precision_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Proporción de resultados relevantes en top-K."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for x in top_k if x in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Proporción de relevantes encontrados en top-K."""
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for x in top_k if x in relevant_ids)
    return hits / len(relevant_ids)


def f1_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Media armónica de Precision@K y Recall@K."""
    precision = precision_at_k(ranked_ids, relevant_ids, k)
    recall = recall_at_k(ranked_ids, relevant_ids, k)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def mean_reciprocal_rank(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """Inverso del rank del primer resultado relevante."""
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain (binario)."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    rel = np.array([1.0 if x in relevant_ids else 0.0 for x in top_k])
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(rel)))
    dcg = float(np.sum(rel * discounts))
    ideal_rel = np.zeros(len(rel), dtype=float)
    ideal_rel[: min(len(relevant_ids), len(rel))] = 1.0
    idcg = float(np.sum(ideal_rel * discounts))
    if idcg <= 1e-12:
        return 0.0
    return dcg / idcg


def fp_cost_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Costo de falsos positivos con descuento logarítmico por posición."""
    return float(
        sum(
            1.0 / np.log2(rank + 1)
            for rank, result_id in enumerate(ranked_ids[:k], start=1)
            if result_id not in relevant_ids
        )
    )


def compute_retrieval_metrics(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """Calcula todas las métricas de retrieval para una query."""
    if k_values is None:
        k_values = [1, 5, 10]

    return RetrievalMetrics(
        precision_at={k: precision_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        recall_at={k: recall_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        f1_at={k: f1_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        mrr=mean_reciprocal_rank(ranked_ids, relevant_ids),
        ndcg_at={k: ndcg_at_k(ranked_ids, relevant_ids, k) for k in k_values},
        fp_cost_at={k: fp_cost_at_k(ranked_ids, relevant_ids, k) for k in k_values},
    )


def evaluate_retrieval(
    dataset_path: str,
    search_engine,
    k_values: list[int] | None = None,
    max_k: int = 10,
) -> dict:
    """
    Evalúa el retrieval sobre el dataset de evaluación.

    Args:
        dataset_path: Ruta al JSON con queries y relevant_ids
        search_engine: Instancia de AudioSearchEngine
        k_values: Valores de K para Precision@K, Recall@K, NDCG@K
        max_k: Máximo K para la búsqueda

    Returns:
        Dict con métricas agregadas y por query
    """
    if k_values is None:
        k_values = [1, 5, 10]

    queries = json.loads(Path(dataset_path).read_text())["queries"]

    all_metrics = []
    per_query_results = []

    for q in queries:
        results = search_engine.search_semantic(q["query_text"], k=max_k)
        ranked_ids = [r["segment"]["segment_id"] for r in results]
        relevant_ids = set(q["relevant_segment_ids"])

        metrics = compute_retrieval_metrics(ranked_ids, relevant_ids, k_values)
        all_metrics.append(metrics)

        per_query_results.append(
            {
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "category": q.get("category", "unknown"),
                "metrics": {
                    "precision_at": metrics.precision_at,
                    "recall_at": metrics.recall_at,
                    "f1_at": metrics.f1_at,
                    "mrr": metrics.mrr,
                    "ndcg_at": metrics.ndcg_at,
                    "fp_cost_at": metrics.fp_cost_at,
                },
            }
        )

    aggregated = {
        "precision_at": {
            k: float(np.mean([m.precision_at[k] for m in all_metrics])) for k in k_values
        },
        "recall_at": {k: float(np.mean([m.recall_at[k] for m in all_metrics])) for k in k_values},
        "f1_at": {k: float(np.mean([m.f1_at[k] for m in all_metrics])) for k in k_values},
        "mrr": float(np.mean([m.mrr for m in all_metrics])),
        "ndcg_at": {k: float(np.mean([m.ndcg_at[k] for m in all_metrics])) for k in k_values},
        "fp_cost_at": {k: float(np.mean([m.fp_cost_at[k] for m in all_metrics])) for k in k_values},
    }

    # By category
    by_category = {}
    for r in per_query_results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r["metrics"])

    category_aggregated = {}
    for cat, metrics_list in by_category.items():
        category_aggregated[cat] = {
            "mrr": float(np.mean([m["mrr"] for m in metrics_list])),
            "precision_at_5": float(np.mean([m["precision_at"][5] for m in metrics_list])),
        }

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "aggregated": aggregated,
        "by_category": category_aggregated,
        "per_query": per_query_results,
        "config": {
            "k_values": k_values,
            "max_k": max_k,
            "total_queries": len(queries),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluación de retrieval")
    parser.add_argument("--dataset", required=True, help="JSON con queries de evaluación")
    parser.add_argument("--dataset-path", required=True, help="Path al dataset procesado")
    parser.add_argument("--output", required=True, help="Path para resultados JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from src.agent_service.search_engine import AudioSearchEngine

    engine = AudioSearchEngine(args.dataset_path)
    results = evaluate_retrieval(args.dataset, engine)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", args.output)

    print(f"\n{'=' * 50}")
    print("RETRIEVAL EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"Queries: {results['config']['total_queries']}")
    print(f"MRR: {results['aggregated']['mrr']:.4f}")
    for k in results["config"]["k_values"]:
        print(f"Precision@{k}: {results['aggregated']['precision_at'][k]:.4f}")
        print(f"Recall@{k}: {results['aggregated']['recall_at'][k]:.4f}")
        print(f"F1@{k}: {results['aggregated']['f1_at'][k]:.4f}")
        print(f"NDCG@{k}: {results['aggregated']['ndcg_at'][k]:.4f}")
        print(f"FP_Cost@{k}: {results['aggregated']['fp_cost_at'][k]:.4f}")


if __name__ == "__main__":
    main()
