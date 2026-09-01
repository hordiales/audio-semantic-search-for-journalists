"""Contracts for evaluating text retrieval with RAGAS-generated questions."""

import json

import pandas as pd
from evaluation.text_index_evaluation import (
    align_reference_contexts,
    evaluate_text_index,
    normalize_for_alignment,
)


def test_normalize_for_alignment_removes_hop_markers_and_accents():
    assert normalize_for_alignment("<1-hop>\n\n¿Qué pasó?") == "que paso"


def test_align_reference_contexts_maps_context_back_to_segments():
    corpus = pd.DataFrame(
        [
            {"segment_id": 1, "text": "El ministro habló sobre inflación."},
            {"segment_id": 2, "text": "Luego respondió otras preguntas."},
        ]
    )
    samples = [
        {
            "question": "¿Qué dijo el ministro?",
            "ground_truth_contexts": [
                "Antes de la conferencia, el ministro habló sobre inflación. Luego se retiró."
            ],
        }
    ]

    aligned = align_reference_contexts(samples, corpus)

    assert aligned[0]["relevant_segment_ids"] == [1]
    assert aligned[0]["alignment"]["matched_segment_count"] == 1


class _FakeSearchEngine:
    def __init__(self):
        self._min_segment_duration_seconds = None

    def search_semantic(self, query_text: str, k: int):
        del query_text
        rows = [
            {
                "segment": {
                    "segment_id": 1,
                    "text": "El ministro habló sobre inflación.",
                    "original_file_name": "noticia.wav",
                    "start_time": 0.0,
                    "end_time": 6.0,
                },
                "similarity": 0.9,
            },
            {
                "segment": {
                    "segment_id": 2,
                    "text": "Luego respondió otras preguntas.",
                    "original_file_name": "noticia.wav",
                    "start_time": 6.0,
                    "end_time": 12.0,
                },
                "similarity": 0.5,
            },
        ]
        return rows[:k]


def test_evaluate_text_index_records_rankings_and_metrics(tmp_path):
    dataset_path = tmp_path / "dataset"
    final_path = dataset_path / "final"
    final_path.mkdir(parents=True)
    corpus = pd.DataFrame(
        [
            {
                "segment_id": 1,
                "text": "El ministro habló sobre inflación.",
                "start_time": 0.0,
                "end_time": 6.0,
                "original_file_name": "noticia.wav",
                "language": "es",
                "dominant_sentiment": "neutral",
            },
            {
                "segment_id": 2,
                "text": "Luego respondió otras preguntas.",
                "start_time": 6.0,
                "end_time": 12.0,
                "original_file_name": "noticia.wav",
                "language": "es",
                "dominant_sentiment": "neutral",
            },
        ]
    )
    corpus.to_pickle(final_path / "complete_dataset.pkl")
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "question": "¿Qué dijo el ministro?",
                    "ground_truth_contexts": [
                        "Durante la entrevista, el ministro habló sobre inflación."
                    ],
                    "synthesizer_name": "single_hop",
                }
            ]
        )
    )

    report = evaluate_text_index(
        str(questions_path),
        str(dataset_path),
        k_values=[1, 2],
        search_engine=_FakeSearchEngine(),
    )

    assert report["aggregated"]["mrr"] == 1.0
    assert report["aggregated"]["recall_at"]["1"] == 1.0
    assert report["per_query"][0]["ranked_results"][0]["relevant"] is True
    assert report["question_composition"]["by_synthesizer"] == {"single_hop": 1}
