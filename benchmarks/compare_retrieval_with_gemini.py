"""Compare current MiniLM/CLAP retrieval with Gemini Embedding 2 audio retrieval."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from evaluation.retrieval_evaluation import compute_retrieval_metrics

from src.agent_service.search_engine import AudioSearchEngine
from src.gemini_multimodal_embeddings import GeminiEmbeddingConfig, GeminiMultimodalEmbedding


def _window_paths(audio_segments_dir: Path, segment_id: int) -> list[Path]:
    return sorted(
        audio_segments_dir.glob(f"segment_{segment_id}_window_*.wav"),
        key=lambda path: int(path.stem.rsplit("_", maxsplit=1)[1]),
    )


def _pool_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    pooled = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(pooled)
    return (pooled / norm if norm > 0 else pooled).astype(np.float32)


def build_gemini_audio_index(
    dataframe: pd.DataFrame,
    audio_segments_dir: Path,
    embedder: GeminiMultimodalEmbedding,
) -> faiss.IndexFlatIP:
    """Create an in-memory Gemini index aligned one-to-one with dataset rows."""
    embeddings = []
    for _, row in dataframe.iterrows():
        paths = _window_paths(audio_segments_dir, int(row["segment_id"]))
        if not paths:
            raise FileNotFoundError(
                f"No audio windows found for segment {row['segment_id']} in {audio_segments_dir}. "
                "Re-run the ingestion pipeline with timestamp-aligned audio windows."
            )
        embeddings.append(_pool_embeddings([embedder.generate_audio_embedding(path) for path in paths]))

    vectors = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _aggregate(entries: list[dict], k_values: list[int]) -> dict:
    return {
        "precision_at": {k: float(np.mean([x["metrics"].precision_at[k] for x in entries])) for k in k_values},
        "recall_at": {k: float(np.mean([x["metrics"].recall_at[k] for x in entries])) for k in k_values},
        "mrr": float(np.mean([x["metrics"].mrr for x in entries])),
        "ndcg_at": {k: float(np.mean([x["metrics"].ndcg_at[k] for x in entries])) for k in k_values},
    }


def compare_retrieval(dataset_path: str, queries_path: str, output_dimensionality: int = 1536, max_k: int = 10) -> dict:
    """Evaluate MiniLM text, CLAP audio, and Gemini native-audio rankings fairly."""
    dataset = Path(dataset_path)
    dataframe = pd.read_pickle(dataset / "final" / "complete_dataset.pkl")
    queries = json.loads(Path(queries_path).read_text())["queries"]
    labeled_queries = [query for query in queries if query.get("relevant_segment_ids")]
    if not labeled_queries:
        raise ValueError("The comparison requires queries with relevant_segment_ids annotations")

    current = AudioSearchEngine(str(dataset))
    gemini = GeminiMultimodalEmbedding(GeminiEmbeddingConfig(output_dimensionality=output_dimensionality))
    gemini_index = build_gemini_audio_index(dataframe, dataset / "audio_segments", gemini)
    k_values = [1, 5, 10]
    per_system: dict[str, list[dict]] = defaultdict(list)
    for query in labeled_queries:
        text = query["query_text"]
        rankings = {
            "minilm_text": [x["segment"]["segment_id"] for x in current.search_semantic(text, k=max_k)],
            "clap_audio": [x["segment"]["segment_id"] for x in current.search_audio_by_text(text, k=max_k)],
        }
        _, indices = gemini_index.search(gemini.generate_query_embedding(text).reshape(1, -1), min(max_k, gemini_index.ntotal))
        rankings["gemini_embedding_2_audio"] = [int(dataframe.iloc[i]["segment_id"]) for i in indices[0] if i >= 0]
        relevant = set(query["relevant_segment_ids"])
        for system, ranking in rankings.items():
            per_system[system].append({"query_id": query["query_id"], "query_text": text, "category": query.get("category", "unknown"), "ranked_segment_ids": ranking, "metrics": compute_retrieval_metrics(ranking, relevant, k_values)})

    systems = {}
    for system, entries in per_system.items():
        systems[system] = {"aggregated": _aggregate(entries, k_values), "per_query": [{**{key: value for key, value in entry.items() if key != "metrics"}, "metrics": {"precision_at": entry["metrics"].precision_at, "recall_at": entry["metrics"].recall_at, "mrr": entry["metrics"].mrr, "ndcg_at": entry["metrics"].ndcg_at}} for entry in entries]}
    return {"created_at": datetime.now(UTC).isoformat(), "configuration": {"gemini_model": "gemini-embedding-2", "gemini_output_dimensionality": output_dimensionality, "max_k": max_k, "evaluated_queries": len(labeled_queries)}, "systems": systems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gemini-dimensions", type=int, default=1536)
    parser.add_argument("--max-k", type=int, default=10)
    args = parser.parse_args()
    report = compare_retrieval(args.dataset_path, args.queries, args.gemini_dimensions, args.max_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Comparison report written to {output}")


if __name__ == "__main__":
    main()
