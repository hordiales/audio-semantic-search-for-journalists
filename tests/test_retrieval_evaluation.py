"""Tests for retrieval evaluation metrics."""

from evaluation.retrieval_evaluation import (
    compute_retrieval_metrics,
    f1_at_k,
    fp_cost_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 3, 5}

    assert precision_at_k(ranked, relevant, 1) == 1.0
    assert precision_at_k(ranked, relevant, 3) == 2.0 / 3
    assert precision_at_k(ranked, relevant, 5) == 3.0 / 5


def test_recall_at_k():
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 3, 5, 7}

    assert recall_at_k(ranked, relevant, 1) == 1.0 / 4
    assert recall_at_k(ranked, relevant, 5) == 3.0 / 4


def test_f1_at_k():
    ranked = [1, 2, 3, 4]
    relevant = {1, 3, 5, 7}

    assert f1_at_k(ranked, relevant, 4) == 0.5
    assert f1_at_k(ranked, {99}, 4) == 0.0


def test_mrr():
    assert mean_reciprocal_rank([1, 2, 3], {1}) == 1.0
    assert mean_reciprocal_rank([1, 2, 3], {2}) == 0.5
    assert mean_reciprocal_rank([1, 2, 3], {3}) == 1.0 / 3
    assert mean_reciprocal_rank([1, 2, 3], {99}) == 0.0


def test_ndcg_at_k():
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 3}

    score = ndcg_at_k(ranked, relevant, 5)
    assert 0.0 < score <= 1.0

    perfect = ndcg_at_k([1, 3, 2, 4, 5], {1, 3}, 5)
    assert perfect >= score

    incomplete = ndcg_at_k([1, 9, 8], {1, 2, 3}, 3)
    assert incomplete < 1.0


def test_fp_cost_at_k_penalizes_early_false_positives_more():
    relevant = {1, 2}

    early_false_positive = fp_cost_at_k([9, 1, 2], relevant, 3)
    late_false_positive = fp_cost_at_k([1, 2, 9], relevant, 3)

    assert early_false_positive > late_false_positive


def test_compute_retrieval_metrics():
    ranked = [10, 20, 30, 40, 50]
    relevant = {10, 30}

    metrics = compute_retrieval_metrics(ranked, relevant, k_values=[1, 3, 5])

    assert metrics.precision_at[1] == 1.0
    assert metrics.mrr == 1.0
    assert metrics.recall_at[5] == 1.0
    assert metrics.f1_at[5] > 0.0
    assert metrics.fp_cost_at[5] > 0.0


def test_empty_relevant_set():
    ranked = [1, 2, 3]
    relevant = set()

    assert recall_at_k(ranked, relevant, 3) == 0.0
    assert mean_reciprocal_rank(ranked, relevant) == 0.0
