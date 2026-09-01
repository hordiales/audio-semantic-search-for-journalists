"""Prepare a stratified CLAP candidate set for blind human relevance review."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def load_question_catalog(path: str | Path) -> tuple[dict, list[dict]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return {}, data
    questions = data.get("questions", data.get("samples", []))
    if not isinstance(questions, list):
        raise ValueError("Question catalog must contain a questions or samples array")
    return {key: value for key, value in data.items() if key != "questions"}, questions


def stratified_questions(questions: list[dict], maximum: int) -> list[dict]:
    """Select questions round-robin by category while preserving source order."""
    if maximum <= 0 or maximum >= len(questions):
        return questions
    by_category: dict[str, deque] = defaultdict(deque)
    for question in questions:
        by_category[str(question.get("category", "unknown"))].append(question)
    selected: list[dict] = []
    categories = sorted(by_category)
    while len(selected) < maximum:
        added = False
        for category in categories:
            if by_category[category] and len(selected) < maximum:
                selected.append(by_category[category].popleft())
                added = True
        if not added:
            break
    return selected


def prepare_review_set(
    questions_path: str,
    dataset_path: str,
    *,
    top_k: int = 5,
    boundary_rank: int = 25,
    boundary_count: int = 2,
    negative_count: int = 1,
    candidate_pool_size: int = 100,
    maximum_questions: int = 0,
    minimum_segment_duration: float = 0.0,
    search_engine=None,
) -> dict:
    """Retrieve CLAP candidates without treating model output as ground truth."""
    metadata, all_questions = load_question_catalog(questions_path)
    questions = stratified_questions(all_questions, maximum_questions)
    if search_engine is None:
        # CLAP/torchlibrosa must initialize PyTorch before FAISS on Apple Silicon.
        # This is the same ordering used by benchmarks/compare_clap_by_question_type.py.
        import torch

        torch.set_num_threads(1)
        from src.agent_service.search_engine import AudioSearchEngine

        search_engine = AudioSearchEngine(dataset_path)
    search_engine._min_segment_duration_seconds = minimum_segment_duration

    cases = []
    for question in questions:
        query_en = question.get("clap_query_en") or question["question"]
        boundary_endpoint = boundary_rank + boundary_count - 1 if boundary_count > 0 else 0
        required_pool = max(top_k, boundary_endpoint, negative_count)
        pool_size = max(candidate_pool_size, required_pool)
        results = search_engine.search_audio_by_text(query_en, k=pool_size, source_language="en")
        selected_ranks = list(range(1, min(top_k, len(results)) + 1))
        selected_ranks.extend(
            range(boundary_rank, min(boundary_rank + boundary_count, len(results) + 1))
        )
        if negative_count > 0:
            selected_ranks.extend(
                range(max(1, len(results) - negative_count + 1), len(results) + 1)
            )
        selected_ranks = list(dict.fromkeys(selected_ranks))
        candidates = []
        for rank in selected_ranks:
            result = results[rank - 1]
            segment = dict(result["segment"])
            segment_id = int(segment["segment_id"])
            segment["clip_url"] = f"/api/audio/{segment_id}"
            if rank <= top_k:
                stratum = "top_k"
            elif boundary_rank <= rank < boundary_rank + boundary_count:
                stratum = "boundary"
            else:
                stratum = "negative_control"
            candidates.append(
                {
                    "rank": rank,
                    "stratum": stratum,
                    "similarity": float(result["similarity"]),
                    "segment": segment,
                }
            )
        cases.append(
            {
                "eval_case_id": question["eval_case_id"],
                "category": question.get("category", "unknown"),
                "difficulty": question.get("difficulty", "unknown"),
                "question": question["question"],
                "clap_query_en": query_en,
                "target_description": question.get("ground_truth", ""),
                "candidates": candidates,
            }
        )

    category_counts = Counter(case["category"] for case in cases)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "review_protocol": "human_relevance_grading_0_to_3",
        "ground_truth_warning": (
            "CLAP rankings are candidates only. Human judgments exported by the review "
            "interface form the gold subset."
        ),
        "questions_path": str(questions_path),
        "dataset_path": str(dataset_path),
        "source_catalog": metadata,
        "configuration": {
            "query_field": "clap_query_en",
            "source_language": "en",
            "top_k": top_k,
            "boundary_rank": boundary_rank,
            "boundary_count": boundary_count,
            "negative_count": negative_count,
            "candidate_pool_size": candidate_pool_size,
            "minimum_segment_duration_seconds": minimum_segment_duration,
            "yamnet_available": bool(getattr(search_engine, "yamnet_available", False)),
        },
        "sample_composition": {
            "questions": len(cases),
            "candidate_judgments": sum(len(case["candidates"]) for case in cases),
            "by_category": dict(sorted(category_counts.items())),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Corpus procesado; si se omite, se usa DATASET_PATH del archivo .env",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--boundary-rank", type=int, default=25)
    parser.add_argument("--boundary-count", type=int, default=2)
    parser.add_argument("--negative-count", type=int, default=1)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument(
        "--maximum-questions",
        type=int,
        default=0,
        help="0 uses every question; a positive value selects categories round-robin",
    )
    parser.add_argument("--minimum-segment-duration", type=float, default=0.0)
    args = parser.parse_args()
    dataset_path = args.dataset_path or os.getenv("DATASET_PATH")
    if not dataset_path:
        parser.error("--dataset-path is required when DATASET_PATH is not defined in .env")

    review_set = prepare_review_set(
        args.questions,
        dataset_path,
        top_k=args.top_k,
        boundary_rank=args.boundary_rank,
        boundary_count=args.boundary_count,
        negative_count=args.negative_count,
        candidate_pool_size=args.candidate_pool_size,
        maximum_questions=args.maximum_questions,
        minimum_segment_duration=args.minimum_segment_duration,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(review_set, indent=2, ensure_ascii=False))
    print(json.dumps(review_set["sample_composition"], indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
