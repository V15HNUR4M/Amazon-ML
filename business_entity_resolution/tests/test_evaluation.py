"""
test_evaluation.py — Behavioral tests for evaluation.py metrics.

Covers:
- entity_precision
- entity_recall
- entity_f05 (including singletons and multi-match)
- macro_f05
- ground_truth_to_dict conversion
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import (
    BETA,
    entity_f05,
    entity_precision,
    entity_recall,
    ground_truth_to_dict,
    macro_f05,
)


def test_beta_constant():
    assert BETA == 0.5


def test_entity_precision():
    assert entity_precision(set(), set()) == 1.0
    assert entity_precision(set(), {"S2-1"}) == 0.0
    assert entity_precision({"S2-1"}, set()) == 0.0
    assert entity_precision({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0
    assert entity_precision({"S2-1"}, {"S2-1", "S2-2"}) == 0.5


def test_entity_recall():
    assert entity_recall(set(), set()) == 1.0
    assert entity_recall(set(), {"S2-1"}) == 0.0
    assert entity_recall({"S2-1"}, set()) == 0.0
    assert entity_recall({"S2-1", "S3-2"}, {"S2-1"}) == 0.5
    assert entity_recall({"S2-1"}, {"S2-1", "S2-2"}) == 1.0


def test_entity_f05_singletons():
    # Both empty -> correct singleton prediction scores 1.0
    assert entity_f05(set(), set()) == 1.0
    # True singleton predicted non-empty -> 0.0
    assert entity_f05(set(), {"S2-1"}) == 0.0
    # True non-empty predicted singleton -> 0.0
    assert entity_f05({"S2-1"}, set()) == 0.0


def test_entity_f05_exact_and_partial():
    # Perfect match
    assert entity_f05({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0

    # Partial: True = {S2-1, S3-2}, Pred = {S2-1, S2-99}
    # P = 1/2 = 0.5, R = 1/2 = 0.5
    # F0.5 = 1.25 * 0.25 / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    assert entity_f05({"S2-1", "S3-2"}, {"S2-1", "S2-99"}) == 0.5


def test_macro_f05():
    gt = {
        "S1-1": {"S2-1"},
        "S1-2": set(),  # singleton
    }
    preds = {
        "S1-1": {"S2-1"},  # 1.0
        "S1-2": set(),     # 1.0
    }
    assert macro_f05(gt, preds) == 1.0

    preds_imperfect = {
        "S1-1": {"S2-1"},       # 1.0
        "S1-2": {"S2-false"},   # 0.0
    }
    assert macro_f05(gt, preds_imperfect) == 0.5


def test_ground_truth_to_dict():
    df = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-10, S3-20", "", "nan"],
        }
    )
    res = ground_truth_to_dict(df)
    assert res["S1-1"] == {"S2-10", "S3-20"}
    assert res["S1-2"] == set()
    assert res["S1-3"] == set()
