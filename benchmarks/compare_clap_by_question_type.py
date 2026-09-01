"""Compara el retrieval CLAP según el tipo de pregunta: conceptual vs. acústica.

Usa el dataset balanceado generado por `evaluation/generate_balanced_questions.py`
y mide, directamente sobre el índice CLAP del corpus, cuánto mejor (o distinto)
recupera cuando la query menciona un evento o propiedad sonora.

A diferencia de la evaluación del agente, este benchmark no genera respuestas
ni usa RAGAS: solo consulta texto→audio sobre el índice FAISS existente y
computa métricas de Information Retrieval contra el segmento de origen de cada
pregunta. Es más rápido y barato, y aísla el retrieval de los errores del LLM.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Load project environment variables before importing modules that read them.
load_dotenv()

# Inicializar PyTorch antes que FAISS/pandas para evitar conflictos con el
# backend de torchlibrosa en Apple Silicon.
import torch  # noqa: E402

torch.set_num_threads(1)

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from evaluation.retrieval_evaluation import compute_retrieval_metrics  # noqa: E402

from src.clap_audio_embeddings import CLAPEmbedding  # noqa: E402
from src.vector_indexing import load_faiss_index, search_faiss_index  # noqa: E402

K_VALUES = [1, 5, 10]


def load_corpus_and_index(dataset_path: str):
    """Load dataframe, segment-id map and CLAP audio index."""
    dataset_dir = Path(dataset_path)
    df = pd.read_pickle(dataset_dir / "final" / "complete_dataset.pkl")
    index = load_faiss_index(str(dataset_dir / "indices" / "audio_index.faiss"))
    segment_to_position = {int(row["segment_id"]): idx for idx, row in df.iterrows()}
    return df, index, segment_to_position


def evaluate_question_type(
    samples: list[dict], clap: CLAPEmbedding, index: faiss.IndexFlatIP, segment_to_position: dict
) -> dict:
    per_question = []
    for sample in samples:
        relevant_segment_id = int(sample["segment_id"])
        relevant_position = segment_to_position.get(relevant_segment_id)
        if relevant_position is None:
            continue

        query_embedding = clap.generate_text_embedding(sample["question"])
        _, indices = search_faiss_index(index, query_embedding, k=max(K_VALUES))
        ranked_ids = [int(pos) for pos in indices[0] if pos >= 0]
        ranked_segment_ids = [
            list(segment_to_position.keys())[list(segment_to_position.values()).index(pos)]
            for pos in ranked_ids
        ]

        metrics = compute_retrieval_metrics(ranked_segment_ids, {relevant_segment_id}, K_VALUES)

        per_question.append(
            {
                "question": sample["question"],
                "type": sample.get("type", "unknown"),
                "relevant_segment_id": relevant_segment_id,
                "ranked_segment_ids": ranked_segment_ids,
                "metrics": {
                    "precision_at": metrics.precision_at,
                    "recall_at": metrics.recall_at,
                    "mrr": metrics.mrr,
                    "ndcg_at": metrics.ndcg_at,
                },
            }
        )

    if not per_question:
        raise ValueError("No hay preguntas válidas para evaluar")

    aggregated = {
        "precision_at": {
            k: float(np.mean([q["metrics"]["precision_at"][k] for q in per_question]))
            for k in K_VALUES
        },
        "recall_at": {
            k: float(np.mean([q["metrics"]["recall_at"][k] for q in per_question]))
            for k in K_VALUES
        },
        "mrr": float(np.mean([q["metrics"]["mrr"] for q in per_question])),
        "ndcg_at": {
            k: float(np.mean([q["metrics"]["ndcg_at"][k] for q in per_question])) for k in K_VALUES
        },
    }
    return {"aggregated": aggregated, "per_question": per_question, "n": len(per_question)}


def compare_clap_by_type(dataset_path: str, questions_path: str) -> dict:
    samples = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("samples", samples.get("questions", []))

    by_type: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_type[sample.get("type", "unknown")].append(sample)

    df, index, segment_to_position = load_corpus_and_index(dataset_path)
    clap = CLAPEmbedding()

    results: dict[str, dict] = {}
    for qtype, subset in by_type.items():
        results[qtype] = evaluate_question_type(subset, clap, index, segment_to_position)

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_path": dataset_path,
        "questions_path": questions_path,
        "total_segments": len(df),
        "by_type": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="Dataset procesado con índice CLAP")
    parser.add_argument(
        "--questions", required=True, help="JSON de preguntas balanceadas (con campo 'type')"
    )
    parser.add_argument("--output", required=True, help="Ruta del JSON de resultados")
    args = parser.parse_args()

    report = compare_clap_by_type(args.dataset_path, args.questions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("CLAP retrieval por tipo de pregunta")
    print(f"{'=' * 60}")
    for qtype, result in report["by_type"].items():
        print(f"\n--- {qtype.upper()} ({result['n']} preguntas) ---")
        agg = result["aggregated"]
        for k in K_VALUES:
            print(f"  Recall@{k}:     {agg['recall_at'][k]:.4f}")
            print(f"  Precision@{k}:  {agg['precision_at'][k]:.4f}")
        print(f"  MRR:          {agg['mrr']:.4f}")
        print(f"  NDCG@10:      {agg['ndcg_at'][10]:.4f}")
    print(f"\nReporte guardado en: {output_path}")


if __name__ == "__main__":
    main()
