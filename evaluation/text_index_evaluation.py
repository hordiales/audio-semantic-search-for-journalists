"""Evaluate the text FAISS index with RAGAS-generated reference questions.

RAGAS provides questions and reference contexts, but the current serialized
test set does not preserve source segment IDs. This module maps every reference
context back to the current corpus by normalized exact text containment and
then evaluates the deterministic retrieval ranking with classic IR metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.ragas_evaluation import load_eval_dataset
from evaluation.retrieval_evaluation import RetrievalMetrics, compute_retrieval_metrics

DEFAULT_K_VALUES = [1, 5, 10, 20]


def normalize_for_alignment(value: object) -> str:
    """Normalize ASR text and RAGAS contexts for deterministic containment."""
    text = str(value).replace("<1-hop>", " ").replace("<2-hop>", " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def align_reference_contexts(
    samples: list[dict],
    corpus: pd.DataFrame,
    *,
    minimum_normalized_chars: int = 12,
) -> list[dict]:
    """Attach corpus segment IDs found verbatim inside each reference context."""
    normalized_segments = [
        (int(row["segment_id"]), normalize_for_alignment(row["text"]))
        for _, row in corpus.iterrows()
    ]
    aligned: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        contexts = [
            normalize_for_alignment(context)
            for context in sample.get("ground_truth_contexts", [])
            if str(context).strip()
        ]
        relevant_ids = sorted(
            segment_id
            for segment_id, segment_text in normalized_segments
            if len(segment_text) >= minimum_normalized_chars
            and any(segment_text in context for context in contexts)
        )
        aligned.append(
            {
                **sample,
                "query_id": sample.get("query_id", f"ragas_{index:03d}"),
                "relevant_segment_ids": relevant_ids,
                "alignment": {
                    "method": "normalized_exact_containment",
                    "minimum_normalized_chars": minimum_normalized_chars,
                    "reference_context_count": len(contexts),
                    "matched_segment_count": len(relevant_ids),
                },
            }
        )
    return aligned


def describe_corpus(dataset_path: str, corpus: pd.DataFrame) -> dict:
    """Return the compact corpus composition recorded with every evaluation."""
    durations = corpus["end_time"].astype(float) - corpus["start_time"].astype(float)
    manifest_path = Path(dataset_path) / "final" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    per_file = []
    for file_name, group in corpus.groupby("original_file_name"):
        per_file.append(
            {
                "original_file_name": str(file_name),
                "segments": int(len(group)),
                "start_time": float(group["start_time"].min()),
                "end_time": float(group["end_time"].max()),
            }
        )
    return {
        "total_segments": int(len(corpus)),
        "total_audio_files": int(corpus["original_file_name"].nunique()),
        "languages": {str(k): int(v) for k, v in corpus["language"].value_counts().items()},
        "sentiments": {
            str(k): int(v) for k, v in corpus["dominant_sentiment"].value_counts().items()
        },
        "segment_duration_seconds": {
            "minimum": float(durations.min()),
            "median": float(durations.median()),
            "mean": float(durations.mean()),
            "maximum": float(durations.max()),
            "strictly_greater_than_5_seconds": int((durations > 5.0).sum()),
        },
        "audio_span_seconds": float(corpus.groupby("original_file_name")["end_time"].max().sum()),
        "per_file": per_file,
        "active_embeddings": manifest.get("active_embeddings", []),
        "embeddings": manifest.get("embeddings", {}),
        "active_classifiers": sorted((manifest.get("classifiers") or {}).keys()),
    }


def _aggregate_metrics(metrics: list[RetrievalMetrics], k_values: list[int]) -> dict:
    if not metrics:
        return {}
    return {
        "precision_at": {
            str(k): float(np.mean([item.precision_at[k] for item in metrics])) for k in k_values
        },
        "recall_at": {
            str(k): float(np.mean([item.recall_at[k] for item in metrics])) for k in k_values
        },
        "f1_at": {str(k): float(np.mean([item.f1_at[k] for item in metrics])) for k in k_values},
        "mrr": float(np.mean([item.mrr for item in metrics])),
        "ndcg_at": {
            str(k): float(np.mean([item.ndcg_at[k] for item in metrics])) for k in k_values
        },
        "fp_cost_at": {
            str(k): float(np.mean([item.fp_cost_at[k] for item in metrics])) for k in k_values
        },
    }


def evaluate_text_index(
    eval_dataset_path: str,
    dataset_path: str,
    *,
    k_values: list[int] | None = None,
    minimum_segment_duration: float = 0.0,
    search_engine=None,
) -> dict:
    """Run deterministic text retrieval and return aggregate plus per-query evidence."""
    k_values = k_values or DEFAULT_K_VALUES
    max_k = max(k_values)
    corpus_path = Path(dataset_path) / "final" / "complete_dataset.pkl"
    corpus = pd.read_pickle(corpus_path)
    samples = align_reference_contexts(load_eval_dataset(eval_dataset_path), corpus)
    unaligned = [sample["query_id"] for sample in samples if not sample["relevant_segment_ids"]]
    if unaligned:
        raise ValueError(f"Reference contexts did not align to corpus segments: {unaligned}")

    if search_engine is None:
        from src.agent_service.search_engine import AudioSearchEngine

        search_engine = AudioSearchEngine(dataset_path)
    search_engine._min_segment_duration_seconds = minimum_segment_duration

    duration_by_id = {
        int(row["segment_id"]): float(row["end_time"] - row["start_time"])
        for _, row in corpus.iterrows()
    }
    all_metrics: list[RetrievalMetrics] = []
    metrics_by_synthesizer: dict[str, list[RetrievalMetrics]] = defaultdict(list)
    per_query = []
    for sample in samples:
        results = search_engine.search_semantic(sample["question"], k=max_k)
        ranked_ids = [int(result["segment"]["segment_id"]) for result in results]
        relevant_ids = set(sample["relevant_segment_ids"])
        metrics = compute_retrieval_metrics(ranked_ids, relevant_ids, k_values)
        all_metrics.append(metrics)
        synthesizer = sample.get("synthesizer_name", "unknown")
        metrics_by_synthesizer[synthesizer].append(metrics)
        per_query.append(
            {
                "query_id": sample["query_id"],
                "question": sample["question"],
                "synthesizer_name": synthesizer,
                "relevant_segment_ids": sorted(relevant_ids),
                "relevant_segment_count": len(relevant_ids),
                "searchable_relevant_segment_count": sum(
                    duration_by_id[segment_id] > minimum_segment_duration
                    for segment_id in relevant_ids
                ),
                "alignment": sample["alignment"],
                "ranked_results": [
                    {
                        "rank": rank,
                        "segment_id": int(result["segment"]["segment_id"]),
                        "similarity": float(result["similarity"]),
                        "relevant": int(result["segment"]["segment_id"]) in relevant_ids,
                        "text": result["segment"].get("text", ""),
                        "original_file_name": result["segment"].get("original_file_name", ""),
                        "start_time": result["segment"].get("start_time"),
                        "end_time": result["segment"].get("end_time"),
                    }
                    for rank, result in enumerate(results, start=1)
                ],
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

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "evaluation_type": "text_index_retrieval_with_ragas_generated_questions",
        "dataset_path": str(dataset_path),
        "questions_path": str(eval_dataset_path),
        "corpus_composition": describe_corpus(dataset_path, corpus),
        "question_composition": {
            "total": len(samples),
            "by_synthesizer": {
                name: len(items)
                for name, items in sorted(
                    (name, [s for s in samples if s.get("synthesizer_name", "unknown") == name])
                    for name in {s.get("synthesizer_name", "unknown") for s in samples}
                )
            },
            "language": "es",
            "ground_truth_source": "RAGAS reference_contexts aligned to corpus segments",
        },
        "config": {
            "k_values": k_values,
            "maximum_k": max_k,
            "minimum_segment_duration_seconds": minimum_segment_duration,
            "text_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "aggregated": _aggregate_metrics(all_metrics, k_values),
        "by_synthesizer": {
            name: _aggregate_metrics(items, k_values)
            for name, items in metrics_by_synthesizer.items()
        },
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Preguntas sintéticas generadas con RAGAS")
    parser.add_argument("--dataset-path", required=True, help="Corpus procesado con índice textual")
    parser.add_argument("--output", required=True, help="Archivo JSON de resultados")
    parser.add_argument(
        "--minimum-segment-duration",
        type=float,
        default=0.0,
        help="Filtro de serving; 0 evalúa el índice puro y 5 reproduce el default del servicio",
    )
    args = parser.parse_args()

    report = evaluate_text_index(
        args.dataset,
        args.dataset_path,
        minimum_segment_duration=args.minimum_segment_duration,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Queries: {report['question_composition']['total']}")
    print(f"MRR: {report['aggregated']['mrr']:.4f}")
    for k in report["config"]["k_values"]:
        print(
            f"K={k}: Precision={report['aggregated']['precision_at'][str(k)]:.4f} "
            f"Recall={report['aggregated']['recall_at'][str(k)]:.4f} "
            f"F1={report['aggregated']['f1_at'][str(k)]:.4f} "
            f"nDCG={report['aggregated']['ndcg_at'][str(k)]:.4f} "
            f"FP_Cost={report['aggregated']['fp_cost_at'][str(k)]:.4f}"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
