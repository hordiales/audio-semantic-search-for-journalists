"""API contracts for the local acoustic human-review interface."""

import json

from evaluation.human_review.app import create_app
from fastapi.testclient import TestClient


def _review_set() -> dict:
    return {
        "configuration": {"yamnet_available": False},
        "sample_composition": {"questions": 1, "candidate_judgments": 1},
        "cases": [
            {
                "eval_case_id": "case_1",
                "category": "voice",
                "question": "¿Hay gritos?",
                "clap_query_en": "people shouting",
                "target_description": "Gritos",
                "candidates": [
                    {
                        "rank": 1,
                        "similarity": 0.5,
                        "segment": {
                            "segment_id": 7,
                            "clip_url": "/api/audio/7",
                            "text": "texto",
                        },
                    }
                ],
            }
        ],
    }


def test_review_api_persists_and_exports_annotations(tmp_path):
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(_review_set()))
    annotations_path = tmp_path / "annotations.json"
    app = create_app(
        review_set_path=review_path,
        dataset_path=tmp_path,
        annotations_path=annotations_path,
    )
    client = TestClient(app)

    response = client.get("/api/review-set")
    assert response.status_code == 200
    assert response.json()["saved_annotations"] == []

    payload = {
        "eval_case_id": "case_1",
        "segment_id": 7,
        "relevance": 3,
        "event_present": True,
        "confidence": 4,
        "notes": "Evento claro",
        "reviewer": "tester",
    }
    response = client.post("/api/annotations", json=payload)
    assert response.status_code == 200
    assert json.loads(annotations_path.read_text())["annotations"][0]["relevance"] == 3
    assert client.get("/api/export").status_code == 200


def test_review_api_rejects_unknown_candidate(tmp_path):
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(_review_set()))
    app = create_app(
        review_set_path=review_path,
        dataset_path=tmp_path,
        annotations_path=tmp_path / "annotations.json",
    )
    client = TestClient(app)

    response = client.post(
        "/api/annotations",
        json={
            "eval_case_id": "case_1",
            "segment_id": 99,
            "relevance": 0,
            "confidence": 3,
        },
    )

    assert response.status_code == 404
