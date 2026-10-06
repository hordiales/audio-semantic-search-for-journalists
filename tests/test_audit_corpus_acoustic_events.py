"""Tests for the YAMNet corpus audit helpers that do not require TensorFlow."""

from evaluation.audit_corpus_acoustic_events import (
    QUERY_EVENT_CLASSES,
    _human_evidence,
    verdict,
)


def test_verdict_thresholds():
    assert verdict(0.9) == "strong_evidence"
    assert verdict(0.3) == "weak_evidence"
    assert verdict(0.05) == "no_evidence"


def test_human_evidence_counts_event_presence_per_query():
    review_set = {
        "cases": [
            {
                "eval_case_id": "q1",
                "candidates": [{"segment": {"segment_id": 1}}, {"segment": {"segment_id": 2}}],
            }
        ]
    }
    annotations = {
        "annotations": [
            {"eval_case_id": "q1", "segment_id": 1, "event_present": True},
            {"eval_case_id": "q1", "segment_id": 2, "event_present": None},
        ]
    }

    assert _human_evidence(review_set, annotations)["q1"] == {
        "candidates": 2,
        "event_present_true": 1,
        "event_present_false": 0,
        "event_present_unknown": 1,
    }


def test_mapping_covers_every_catalog_query():
    assert len(QUERY_EVENT_CLASSES) == 65
