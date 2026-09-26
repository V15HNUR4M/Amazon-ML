"""
test_threshold.py — Unit tests for threshold search, multi-match, and singleton handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import entity_f05, macro_f05
from src.prediction import predict_matches
from src.threshold import evaluate_at_threshold, find_optimal_threshold


# ---------------------------------------------------------------------------
# Test Multi-Match and Singleton Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_multi_match():
    """Verify that multiple candidates above threshold are all predicted for one entity."""
    pairs = pd.DataFrame(
        [
            {"s1_id": "S1_1", "candidate_id": "S2_100"},
            {"s1_id": "S1_1", "candidate_id": "S2_101"},
            {"s1_id": "S1_1", "candidate_id": "S3_500"},
            {"s1_id": "S1_1", "candidate_id": "S2_900"},
        ]
    )
    proba = np.array([0.97, 0.95, 0.91, 0.21])

    # With threshold 0.90, all 3 top candidates should be selected
    preds = predict_matches(pairs, proba, threshold=0.90)
    assert preds["S1_1"] == {"S2_100", "S2_101", "S3_500"}

    # With threshold 0.96, only the single top candidate should be selected
    preds_strict = predict_matches(pairs, proba, threshold=0.96)
    assert preds_strict["S1_1"] == {"S2_100"}


def test_predict_matches_singleton_rejection():
    """Verify that when no candidate meets threshold, the entity maps to an empty set."""
    pairs = pd.DataFrame(
        [
            {"s1_id": "S1_SINGLETON", "candidate_id": "S2_1"},
            {"s1_id": "S1_SINGLETON", "candidate_id": "S3_2"},
        ]
    )
    proba = np.array([0.25, 0.35])

    # Threshold 0.50 should reject both candidates
    preds = predict_matches(pairs, proba, threshold=0.50)
    assert "S1_SINGLETON" in preds
    assert preds["S1_SINGLETON"] == set()


def test_predict_matches_preserves_all_entities():
    """Verify that entities with zero candidates appear as empty sets when s1_entities is given."""
    pairs = pd.DataFrame(
        [
            {"s1_id": "S1_1", "candidate_id": "S2_1"},
        ]
    )
    proba = np.array([0.80])
    all_s1 = ["S1_1", "S1_NO_CANDS_1", "S1_NO_CANDS_2"]

    preds = predict_matches(pairs, proba, threshold=0.50, s1_entities=all_s1)
    assert preds["S1_1"] == {"S2_1"}
    assert preds["S1_NO_CANDS_1"] == set()
    assert preds["S1_NO_CANDS_2"] == set()


# ---------------------------------------------------------------------------
# Test Evaluation at Threshold
# ---------------------------------------------------------------------------


def test_evaluate_at_threshold_singleton_scoring():
    """Verify singleton evaluation rules:
    - true empty, pred empty: 1.0
    - true empty, pred non-empty: 0.0
    """
    pairs = pd.DataFrame(
        [
            {"s1_id": "S1_MATCH", "candidate_id": "S2_10", "label": 1},
            {"s1_id": "S1_SING_OK", "candidate_id": "S2_20", "label": 0},
            {"s1_id": "S1_SING_BAD", "candidate_id": "S2_30", "label": 0},
        ]
    )
    # Probabilities:
    # S1_MATCH: 0.90 (true=S2_10, pred=S2_10) -> score 1.0
    # S1_SING_OK: 0.20 (true=empty, pred=empty at 0.5) -> score 1.0
    # S1_SING_BAD: 0.70 (true=empty, pred=S2_30 at 0.5) -> score 0.0
    proba = np.array([0.90, 0.20, 0.70])

    gt = {
        "S1_MATCH": {"S2_10"},
        "S1_SING_OK": set(),
        "S1_SING_BAD": set(),
    }

    metrics = evaluate_at_threshold(pairs, proba, threshold=0.50, ground_truth=gt)

    # 2 correct (1.0 each), 1 wrong (0.0) -> macro F0.5 = 2/3
    assert pytest.approx(metrics["macro_f05"], abs=1e-4) == 2.0 / 3.0
    # 1 out of 2 singletons correct -> 50%
    assert pytest.approx(metrics["singleton_accuracy"], abs=1e-4) == 0.50
    # Non-singleton score: 1.0
    assert pytest.approx(metrics["non_singleton_macro_f05"], abs=1e-4) == 1.0


# ---------------------------------------------------------------------------
# Test Threshold Optimization
# ---------------------------------------------------------------------------


def test_find_optimal_threshold_selects_best_macro_f05():
    """Verify that find_optimal_threshold finds the threshold maximizing macro F0.5."""
    pairs = pd.DataFrame(
        [
            # S1_1: true match with proba 0.85, false alarm with proba 0.72
            {"s1_id": "S1_1", "candidate_id": "S2_TRUE", "label": 1},
            {"s1_id": "S1_1", "candidate_id": "S2_DIST", "label": 0},
            # S1_2: singleton with candidate at 0.60
            {"s1_id": "S1_2", "candidate_id": "S2_DIST2", "label": 0},
        ]
    )
    proba = np.array([0.85, 0.72, 0.60])
    gt = {
        "S1_1": {"S2_TRUE"},
        "S1_2": set(),
    }

    # At threshold 0.50:
    # S1_1 pred = {S2_TRUE, S2_DIST} -> precision 0.5, recall 1.0 -> F0.5 ~ 0.555
    # S1_2 pred = {S2_DIST2} -> score 0.0
    # Mean ~ 0.278
    #
    # At threshold 0.75:
    # S1_1 pred = {S2_TRUE} -> score 1.0
    # S1_2 pred = set() -> score 1.0
    # Mean = 1.0

    best_thresh, best_score, history = find_optimal_threshold(
        pairs, proba, ground_truth=gt, threshold_grid=[0.50, 0.70, 0.75, 0.90]
    )

    assert best_thresh == 0.75
    assert pytest.approx(best_score, abs=1e-4) == 1.0
    assert history[0.75] > history[0.50]
