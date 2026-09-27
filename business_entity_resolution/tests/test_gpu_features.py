"""
Unit tests for GPU-accelerated and Turn 7 accuracy features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.gpu.gpu_features import (
    BASELINE_FEATURE_NAMES,
    TURN7_FEATURE_NAMES,
    TokenIDFStore,
    char_ngram_jaccard,
    compute_gpu_features,
    fast_levenshtein_distance,
    fast_normalized_levenshtein,
)


def test_fast_levenshtein() -> None:
    assert fast_levenshtein_distance("kitten", "sitting") == 3
    assert fast_levenshtein_distance("same", "same") == 0
    assert fast_levenshtein_distance("", "test") == 4
    assert fast_normalized_levenshtein("apple", "apple") == 1.0
    assert 0.0 <= fast_normalized_levenshtein("apple", "banana") <= 1.0


def test_char_ngram_jaccard() -> None:
    assert char_ngram_jaccard("acmecorp", "acmecorp") == 1.0
    assert char_ngram_jaccard("acmecorp", "acmeindustries") > 0.0
    assert char_ngram_jaccard("", "something") == 0.0


def test_compute_gpu_features_baseline_shape() -> None:
    pairs = pd.DataFrame([
        {
            "s1_id": "S1-1",
            "candidate_id": "S2-1",
            "s1_name_norm": "acme supply company",
            "cand_name_norm": "acme supply corp",
            "s1_name_compact": "acmesupplycompany",
            "cand_name_compact": "acmesupplycorp",
            "s1_name_no_suffix": "acme supply",
            "cand_name_no_suffix": "acme supply",
            "s1_address_norm": "100 main st",
            "cand_address_norm": "100 main street",
            "s1_address_is_missing": False,
            "cand_address_is_missing": False,
            "s1_country_norm": "US",
            "cand_country_norm": "US",
            "s1_script": "latin",
            "cand_script": "latin",
            "cand_source": "S2",
        }
    ])
    feats_21 = compute_gpu_features(pairs, include_turn7_features=False)
    assert feats_21.shape == (1, 21)
    assert list(feats_21.columns) == BASELINE_FEATURE_NAMES


def test_compute_gpu_features_turn7_shape() -> None:
    pairs = pd.DataFrame([
        {
            "s1_id": "S1-1",
            "candidate_id": "S2-1",
            "s1_name_norm": "acme supply company",
            "cand_name_norm": "acme supply corp",
            "s1_name_compact": "acmesupplycompany",
            "cand_name_compact": "acmesupplycorp",
            "s1_name_no_suffix": "acme supply",
            "cand_name_no_suffix": "acme supply",
            "s1_address_norm": "100 main st",
            "cand_address_norm": "104 main street",  # conflicting numbers!
            "s1_address_is_missing": False,
            "cand_address_is_missing": False,
            "s1_country_norm": "US",
            "cand_country_norm": "US",
            "s1_script": "latin",
            "cand_script": "latin",
            "cand_source": "S2",
        }
    ])
    feats_34 = compute_gpu_features(pairs, include_turn7_features=True)
    assert feats_34.shape == (1, 34)
    assert list(feats_34.columns) == TURN7_FEATURE_NAMES
    # Verify street number conflict was flagged
    assert feats_34.iloc[0]["address_number_conflict"] == 1.0


def test_token_idf_store() -> None:
    store = TokenIDFStore()
    df = pd.DataFrame([
        {"business_name": "acme global supplies"},
        {"business_name": "acme global logistics"},
        {"business_name": "rarebrand international"},
    ])
    store.fit(df)
    assert "acme" in store.idf_table
    assert "rarebrand" in store.idf_table
    # 'rarebrand' is rarer than 'acme', so IDF should be higher
    assert store.idf_table["rarebrand"] > store.idf_table["acme"]


def test_compute_gpu_features_cpu_gpu_parity() -> None:
    """Verify that CPU and GPU feature extraction produce identical numerical features."""
    pairs = pd.DataFrame([
        {
            "s1_id": f"S1-{i}",
            "candidate_id": f"S2-{i}",
            "s1_name_norm": f"acme corporation {i}",
            "cand_name_norm": f"acme corp {i}",
            "s1_name_compact": f"acmecorporation{i}",
            "cand_name_compact": f"acmecorp{i}",
            "s1_name_no_suffix": f"acme {i}",
            "cand_name_no_suffix": f"acme {i}",
            "s1_address_norm": f"{100 + i} main st suite {i}",
            "cand_address_norm": f"{100 + i} main street",
            "s1_address_is_missing": False,
            "cand_address_is_missing": False,
            "s1_country_norm": "US",
            "cand_country_norm": "US",
            "s1_script": "latin",
            "cand_script": "latin",
            "cand_source": "S2",
        }
        for i in range(20)
    ])
    cpu_feats = compute_gpu_features(pairs, include_turn7_features=True, device="cpu")
    gpu_feats = compute_gpu_features(pairs, include_turn7_features=True, device="gpu")
    max_diff = np.max(np.abs(cpu_feats.values - gpu_feats.values))
    assert max_diff < 1e-4

