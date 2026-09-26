"""
test_model.py — Unit tests for ML matching models and entity-level splitting.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import (
    DEFAULT_FEATURE_NAMES,
    LightGBMMatcher,
    LogisticRegressionMatcher,
    MatcherBase,
    entity_train_val_split,
)


@pytest.fixture
def synthetic_pairs_and_features():
    """Create synthetic candidate pairs with 21 features and labels."""
    np.random.seed(42)
    n_samples = 120

    # 20 distinct S1 entities, each having 6 candidate pairs
    s1_ids = [f"S1_{i:03d}" for i in range(20) for _ in range(6)]
    cand_ids = [f"S2_{i:04d}" for i in range(n_samples)]

    # Make true matches correlated with high feature values
    labels = np.array([1 if i % 3 == 0 else 0 for i in range(n_samples)])

    feat_data = {}
    for col in DEFAULT_FEATURE_NAMES:
        if "match" in col:
            # Binary-like feature
            feat_data[col] = labels * 0.8 + np.random.uniform(0, 0.2, n_samples)
            feat_data[col] = (feat_data[col] > 0.5).astype(float)
        else:
            # Continuous feature
            feat_data[col] = labels * 0.7 + np.random.uniform(0, 0.3, n_samples)

    df = pd.DataFrame(feat_data)
    df["s1_id"] = s1_ids
    df["candidate_id"] = cand_ids
    df["label"] = labels

    # Build ground truth dict
    gt_dict = {}
    for s1 in set(s1_ids):
        pos_cands = set(df[(df["s1_id"] == s1) & (df["label"] == 1)]["candidate_id"])
        gt_dict[s1] = pos_cands

    # Add 2 singleton entities with no matches
    gt_dict["S1_SINGLETON_1"] = set()
    gt_dict["S1_SINGLETON_2"] = set()

    return df, gt_dict


# ---------------------------------------------------------------------------
# Test Entity Train / Validation Split
# ---------------------------------------------------------------------------


def test_entity_train_val_split_zero_leakage(synthetic_pairs_and_features):
    """Verify that entity_train_val_split guarantees 0 overlap in S1 entities."""
    pairs_df, gt_dict = synthetic_pairs_and_features

    train_pairs, val_pairs, train_gt, val_gt = entity_train_val_split(
        pairs_df, gt_dict, val_size=0.25, random_state=42
    )

    train_s1 = set(train_pairs["s1_id"].unique()) | set(train_gt.keys())
    val_s1 = set(val_pairs["s1_id"].unique()) | set(val_gt.keys())

    # Strict assertion: intersection MUST be empty
    overlap = train_s1 & val_s1
    assert len(overlap) == 0, f"Found leaking S1 entities: {overlap}"

    # Verify all entities accounted for
    all_original_s1 = set(gt_dict.keys()) | set(pairs_df["s1_id"].unique())
    assert train_s1 | val_s1 == all_original_s1

    # Verify split size is approximately 75/25
    total = len(all_original_s1)
    assert len(val_s1) == pytest.approx(int(total * 0.25), abs=1)


def test_entity_train_val_split_reproducibility(synthetic_pairs_and_features):
    """Verify entity_train_val_split with identical seed produces identical splits."""
    pairs_df, gt_dict = synthetic_pairs_and_features

    t1, v1, tgt1, vgt1 = entity_train_val_split(pairs_df, gt_dict, val_size=0.20, random_state=123)
    t2, v2, tgt2, vgt2 = entity_train_val_split(pairs_df, gt_dict, val_size=0.20, random_state=123)

    assert set(t1["s1_id"]) == set(t2["s1_id"])
    assert set(v1["s1_id"]) == set(v2["s1_id"])
    assert tgt1 == tgt2
    assert vgt1 == vgt2


# ---------------------------------------------------------------------------
# Test Logistic Regression Matcher
# ---------------------------------------------------------------------------


def test_logistic_regression_matcher_train_and_predict(synthetic_pairs_and_features):
    """Test LogisticRegressionMatcher fitting, prediction, and feature importance."""
    pairs_df, _ = synthetic_pairs_and_features
    X = pairs_df[DEFAULT_FEATURE_NAMES]
    y = pairs_df["label"]

    lr = LogisticRegressionMatcher(random_state=42)
    lr.fit(X, y)

    assert lr.is_fitted
    proba = lr.predict_proba(X)
    assert len(proba) == len(X)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    # Positive pairs should have higher mean probability than negatives
    pos_proba = proba[y == 1]
    neg_proba = proba[y == 0]
    assert np.mean(pos_proba) > np.mean(neg_proba)

    # Check feature importances
    fi = lr.get_feature_importances()
    assert len(fi) == 21
    assert set(fi.keys()) == set(DEFAULT_FEATURE_NAMES)
    assert pytest.approx(sum(fi.values()), abs=1e-5) == 1.0


def test_logistic_regression_save_load(synthetic_pairs_and_features):
    """Test persistence and roundtrip loading for LogisticRegressionMatcher."""
    pairs_df, _ = synthetic_pairs_and_features
    X = pairs_df[DEFAULT_FEATURE_NAMES]
    y = pairs_df["label"]

    lr = LogisticRegressionMatcher(random_state=42)
    lr.fit(X, y)
    orig_proba = lr.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "lr_model.joblib"
        lr.save(save_path)
        assert save_path.exists()

        loaded_lr = LogisticRegressionMatcher.load(save_path)
        loaded_proba = loaded_lr.predict_proba(X)

        np.testing.assert_allclose(orig_proba, loaded_proba, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test LightGBM Matcher
# ---------------------------------------------------------------------------


def test_lightgbm_matcher_train_and_predict(synthetic_pairs_and_features):
    """Test LightGBMMatcher fitting, prediction, and feature importance."""
    pairs_df, _ = synthetic_pairs_and_features
    X = pairs_df[DEFAULT_FEATURE_NAMES]
    y = pairs_df["label"]

    lgb = LightGBMMatcher(n_estimators=30, random_state=42)
    lgb.fit(X, y)

    assert lgb.is_fitted
    proba = lgb.predict_proba(X)
    assert len(proba) == len(X)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    pos_proba = proba[y == 1]
    neg_proba = proba[y == 0]
    assert np.mean(pos_proba) > np.mean(neg_proba)

    fi = lgb.get_feature_importances()
    assert len(fi) == 21
    assert set(fi.keys()) == set(DEFAULT_FEATURE_NAMES)
    assert pytest.approx(sum(fi.values()), abs=1e-5) == 1.0


def test_lightgbm_save_load(synthetic_pairs_and_features):
    """Test persistence and roundtrip loading for LightGBMMatcher."""
    pairs_df, _ = synthetic_pairs_and_features
    X = pairs_df[DEFAULT_FEATURE_NAMES]
    y = pairs_df["label"]

    lgb = LightGBMMatcher(n_estimators=30, random_state=42)
    lgb.fit(X, y)
    orig_proba = lgb.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "lgb_model.joblib"
        lgb.save(save_path)
        assert save_path.exists()

        loaded_lgb = LightGBMMatcher.load(save_path)
        loaded_proba = loaded_lgb.predict_proba(X)

        np.testing.assert_allclose(orig_proba, loaded_proba, rtol=1e-5)


def test_matcher_extracts_features_from_dataframe_with_extra_columns(synthetic_pairs_and_features):
    """Verify matcher handles DataFrames containing extra non-feature columns."""
    pairs_df, _ = synthetic_pairs_and_features
    # pairs_df contains s1_id, candidate_id, label, and the 21 features
    lgb = LightGBMMatcher(n_estimators=10, random_state=42)
    lgb.fit(pairs_df, pairs_df["label"])
    proba = lgb.predict_proba(pairs_df)
    assert len(proba) == len(pairs_df)
