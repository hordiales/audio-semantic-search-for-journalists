"""Tests for the CLAP human-review candidate preparation."""

import json

from evaluation.prepare_acoustic_human_review import (
    prepare_review_set,
    stratified_questions,
)


def _question(case_id: str, category: str) -> dict:
    return {
        "eval_case_id": case_id,
        "category": category,
        "difficulty": "easy",
        "question": f"Pregunta {case_id}",
        "clap_query_en": f"English query {case_id}",
        "ground_truth": f"Target {case_id}",
    }


def test_stratified_questions_round_robins_categories():
    questions = [
        _question("a1", "a"),
        _question("a2", "a"),
        _question("b1", "b"),
        _question("b2", "b"),
    ]

    selected = stratified_questions(questions, 3)

    assert [question["eval_case_id"] for question in selected] == ["a1", "b1", "a2"]


class _FakeAcousticEngine:
    yamnet_available = False

    def __init__(self):
        self._min_segment_duration_seconds = None
        self.calls = []

    def search_audio_by_text(self, query_text: str, k: int, source_language: str):
        self.calls.append((query_text, k, source_language))
        return [
            {
                "similarity": 0.75,
                "segment": {
                    "segment_id": 42,
                    "text": "Transcripción",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "original_file_name": "audio.wav",
                },
            }
        ]


def test_prepare_review_set_uses_english_query_and_marks_candidates(tmp_path):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps({"questions": [_question("case", "voice")]}))
    engine = _FakeAcousticEngine()

    review_set = prepare_review_set(
        str(questions_path),
        str(tmp_path),
        top_k=3,
        boundary_count=0,
        negative_count=0,
        candidate_pool_size=3,
        search_engine=engine,
    )

    assert engine.calls == [("English query case", 3, "en")]
    assert review_set["configuration"]["yamnet_available"] is False
    assert review_set["cases"][0]["candidates"][0]["segment"]["clip_url"] == ("/api/audio/42")
