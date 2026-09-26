"""
test_pair_builder.py — Tests for candidate-pair construction and training pairs.

Covers:
- Positive pair labelling (label = 1)
- Hard negative labelling (label = 0)
- Multi-match entities (multiple positives for a single S1)
- Singleton entities (0 positives, all candidates labelled 0)
- Group preservation (s1_id preserved for entity-level train-validation splitting)
- Joined preprocessed fields
- Negative subsampling ratio
- Test pair construction (unlabelled)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.pair_builder import build_test_pairs, build_train_pairs


@pytest.fixture
def sample_data():
    s1 = pd.DataFrame(
        {
            "entity_id": ["S1-1", "S1-2", "S1-3"],
            "business_name": [
                "Orelee's Barbershop LLC",
                "Singleton Ventures",
                "Multi Match Corp",
            ],
            "business_address": [
                "1795 Westchester Drive, High Point, NC",
                "100 Broadway, New York, NY",
                "500 Market St, San Francisco, CA",
            ],
            "country": ["US", "US", "US"],
        }
    )

    s2 = pd.DataFrame(
        {
            "entity_id": ["S2-10", "S2-30", "S2-99"],
            "business_name": [
                "Orelee's Barbershop",
                "Multi Match Corp S2",
                "Unrelated S2",
            ],
            "business_address": [
                "1795 Westchester Drive",
                "500 Market St",
                "777 Somewhere St",
            ],
            "country": ["US", "US", "US"],
        }
    )

    s3 = pd.DataFrame(
        {
            "entity_id": ["S3-31", "S3-99"],
            "business_name": [
                "Multi Match S3",
                "Unrelated S3",
            ],
            "business_address": [
                "500 Market St Suite 100",
                "888 Anywhere St",
            ],
            "country": ["US", "US"],
        }
    )

    candidates = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2", "S1-3", "S1-3", "S1-3"],
            "candidate_id": ["S2-10", "S2-99", "S3-99", "S2-30", "S3-31", "S2-99"],
        }
    )

    gt = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            # S1-1 has 1 match, S1-2 is singleton, S1-3 has 2 matches (multi-match)
            "matched_entity_ids": ["S2-10", "", "S2-30,S3-31"],
        }
    )

    return s1, s2, s3, candidates, gt


def test_build_train_pairs_labels(sample_data):
    s1, s2, s3, cands, gt = sample_data
    pairs = build_train_pairs(s1, s2, s3, cands, gt)

    assert "label" in pairs.columns
    assert "s1_id" in pairs.columns
    assert "candidate_id" in pairs.columns

    # Map (s1_id, candidate_id) -> label
    pair_labels = dict(zip(zip(pairs["s1_id"], pairs["candidate_id"]), pairs["label"]))

    # S1-1: S2-10 is positive, S2-99 is negative
    assert pair_labels[("S1-1", "S2-10")] == 1
    assert pair_labels[("S1-1", "S2-99")] == 0

    # S1-2: singleton -> S3-99 is negative
    assert pair_labels[("S1-2", "S3-99")] == 0

    # S1-3: multi-match -> S2-30 and S3-31 are positives, S2-99 is negative
    assert pair_labels[("S1-3", "S2-30")] == 1
    assert pair_labels[("S1-3", "S3-31")] == 1
    assert pair_labels[("S1-3", "S2-99")] == 0


def test_build_train_pairs_group_preservation(sample_data):
    s1, s2, s3, cands, gt = sample_data
    pairs = build_train_pairs(s1, s2, s3, cands, gt)

    # s1_id must be present on every single row to enable GroupKFold in Turn 5
    assert not pairs["s1_id"].isna().any()
    assert set(pairs["s1_id"].unique()) == {"S1-1", "S1-2", "S1-3"}


def test_build_train_pairs_joined_fields(sample_data):
    s1, s2, s3, cands, gt = sample_data
    pairs = build_train_pairs(s1, s2, s3, cands, gt)

    expected_cols = [
        "s1_name_norm",
        "cand_name_norm",
        "s1_address_norm",
        "cand_address_norm",
        "s1_country_norm",
        "cand_country_norm",
        "cand_source",
    ]
    for col in expected_cols:
        assert col in pairs.columns


def test_build_train_pairs_negative_sampling(sample_data):
    s1, s2, s3, cands, gt = sample_data

    # Add 10 dummy negative candidates for S1-1
    extra_negs = pd.DataFrame(
        {
            "s1_id": ["S1-1"] * 10,
            "candidate_id": [f"S2-dummy-{i}" for i in range(10)],
        }
    )
    s2_extra = s2.copy()
    for i in range(10):
        s2_extra = pd.concat(
            [
                s2_extra,
                pd.DataFrame(
                    [{"entity_id": f"S2-dummy-{i}", "business_name": "Dummy", "business_address": "None", "country": "US"}]
                ),
            ],
            ignore_index=True,
        )

    all_cands = pd.concat([cands, extra_negs], ignore_index=True)

    # Subsample negatives to 2x positives
    sampled_pairs = build_train_pairs(
        s1, s2_extra, s3, all_cands, gt, sample_neg_ratio=2.0
    )

    s1_1_negs = sampled_pairs[(sampled_pairs["s1_id"] == "S1-1") & (sampled_pairs["label"] == 0)]
    assert len(s1_1_negs) <= 2


def test_build_test_pairs(sample_data):
    s1, s2, s3, cands, gt = sample_data
    test_pairs = build_test_pairs(s1, s2, s3, cands)

    assert "label" not in test_pairs.columns
    assert "s1_id" in test_pairs.columns
    assert "candidate_id" in test_pairs.columns
    assert "cand_source" in test_pairs.columns
    assert len(test_pairs) == len(cands)
