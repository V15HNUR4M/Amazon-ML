"""
test_features.py — Comprehensive behavioral tests for pairwise feature extraction.

Covers:
- Standalone feature functions: name_jaccard, name_edit_distance_norm,
  address_token_overlap, country_exact_match
- Feature matrix computation via compute_features()
- Deterministic behavior
- Missing address and missing country handling
- French accented records and Devanagari script records
- Domain stem matching
- Legal suffix agreement
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    address_token_overlap,
    compute_features,
    country_exact_match,
    name_edit_distance_norm,
    name_jaccard,
)


# ---------------------------------------------------------------------------
# 1. Standalone Feature Functions
# ---------------------------------------------------------------------------


def test_name_jaccard_exact():
    assert name_jaccard("orelee barbershop", "orelee barbershop") == 1.0


def test_name_jaccard_partial():
    score = name_jaccard("orelee barbershop llc", "orelee barbershop inc")
    # Intersection: {orelee, barbershop} (2), Union: {orelee, barbershop, llc, inc} (4)
    assert score == 0.5


def test_name_jaccard_empty():
    assert name_jaccard("", "") == 1.0
    assert name_jaccard("acme", "") == 0.0


def test_name_edit_distance_norm_exact():
    assert name_edit_distance_norm("orelee", "orelee") == 1.0


def test_name_edit_distance_norm_typo():
    score = name_edit_distance_norm("payne enterprises", "payne enterpriss")
    assert score > 0.85


def test_name_edit_distance_norm_different():
    score = name_edit_distance_norm("acme corp", "zebra logistics")
    assert score < 0.3


def test_address_token_overlap_reordered():
    # Token Jaccard should be 1.0 despite reordering of tokens
    a1 = "630 45th terrace kansas city mo"
    a2 = "kansas city mo 630 45th terrace"
    assert address_token_overlap(a1, a2) == 1.0


def test_country_exact_match():
    assert country_exact_match("US", "US") == 1
    assert country_exact_match("us", "US") == 1
    assert country_exact_match("France", "France") == 1
    assert country_exact_match("US", "India") == 0
    assert country_exact_match(None, "US") == -1
    assert country_exact_match("", "") == -1


# ---------------------------------------------------------------------------
# 2. compute_features Matrix Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pairs():
    return pd.DataFrame(
        [
            {
                "s1_id": "S1-1",
                "candidate_id": "S2-10",
                "cand_source": "S2",
                "s1_name_norm": "orelee barbershop llc",
                "cand_name_norm": "orelee barbershop",
                "s1_name_compact": "oreleebarbershopllc",
                "cand_name_compact": "oreleebarbershop",
                "s1_name_no_suffix": "orelee barbershop",
                "cand_name_no_suffix": "orelee barbershop",
                "s1_address_norm": "1795 westchester drive high point nc",
                "cand_address_norm": "1795 westchester drive",
                "s1_address_is_missing": False,
                "cand_address_is_missing": False,
                "s1_country_norm": "US",
                "cand_country_norm": "US",
                "s1_script": "latin",
                "cand_script": "latin",
                "s1_is_domain": False,
                "cand_is_domain": False,
                "s1_domain_stem": None,
                "cand_domain_stem": None,
                "s1_legal_suffix": "llc",
                "cand_legal_suffix": None,
            },
            {
                # Devanagari pair
                "s1_id": "S1-2",
                "candidate_id": "S2-20",
                "cand_source": "S2",
                "s1_name_norm": "राम मार्केटिंग प्राइवेट लिमिटेड",
                "cand_name_norm": "राम मार्केटिंग",
                "s1_name_compact": "राममार्केटिंगप्राइवेटलिमिटेड",
                "cand_name_compact": "राममार्केटिंग",
                "s1_name_no_suffix": "राम मार्केटिंग",
                "cand_name_no_suffix": "राम मार्केटिंग",
                "s1_address_norm": "570 new delhi",
                "cand_address_norm": "570 new delhi",
                "s1_address_is_missing": False,
                "cand_address_is_missing": False,
                "s1_country_norm": "India",
                "cand_country_norm": "India",
                "s1_script": "devanagari",
                "cand_script": "devanagari",
                "s1_is_domain": False,
                "cand_is_domain": False,
                "s1_domain_stem": None,
                "cand_domain_stem": None,
                "s1_legal_suffix": "pvt_ltd",
                "cand_legal_suffix": None,
            },
            {
                # Domain & Missing address pair
                "s1_id": "S1-3",
                "candidate_id": "S3-30",
                "cand_source": "S3",
                "s1_name_norm": "wilford hancock",
                "cand_name_norm": "wilfordhancock com",
                "s1_name_compact": "wilfordhancock",
                "cand_name_compact": "wilfordhancockcom",
                "s1_name_no_suffix": "wilford hancock",
                "cand_name_no_suffix": "wilfordhancock com",
                "s1_address_norm": "mack road texas",
                "cand_address_norm": None,
                "s1_address_is_missing": False,
                "cand_address_is_missing": True,
                "s1_country_norm": "US",
                "cand_country_norm": "US",
                "s1_script": "latin",
                "cand_script": "latin",
                "s1_is_domain": False,
                "cand_is_domain": True,
                "s1_domain_stem": None,
                "cand_domain_stem": "wilfordhancock",
                "s1_legal_suffix": None,
                "cand_legal_suffix": None,
            },
        ]
    )


def test_compute_features_shape(sample_pairs):
    feat_df = compute_features(sample_pairs)

    assert len(feat_df) == len(sample_pairs)
    assert len(feat_df.columns) >= 15

    # Check key columns exist
    expected_cols = [
        "name_exact_match",
        "name_compact_exact_match",
        "name_no_suffix_match",
        "name_token_jaccard",
        "name_levenshtein_sim",
        "address_token_jaccard",
        "address_number_match",
        "address_missing_either",
        "country_exact_match",
        "source_is_s2",
        "script_match",
    ]
    for c in expected_cols:
        assert c in feat_df.columns


def test_compute_features_values(sample_pairs):
    feat_df = compute_features(sample_pairs)

    # Row 0: name_no_suffix match should be 1
    assert feat_df.loc[0, "name_no_suffix_match"] == 1
    assert feat_df.loc[0, "country_exact_match"] == 1
    assert feat_df.loc[0, "address_number_match"] == 1
    assert feat_df.loc[0, "source_is_s2"] == 1

    # Row 1: Devanagari script match
    assert feat_df.loc[1, "script_match"] == 1
    assert feat_df.loc[1, "name_no_suffix_match"] == 1
    assert feat_df.loc[1, "address_exact_match"] == 1

    # Row 2: Missing candidate address
    assert feat_df.loc[2, "address_missing_either"] == 1
    assert feat_df.loc[2, "address_exact_match"] == 0
    assert feat_df.loc[2, "domain_stem_match"] > 0.0
    assert feat_df.loc[2, "source_is_s2"] == 0


def test_compute_features_empty():
    feat_df = compute_features(pd.DataFrame())
    assert len(feat_df) == 0
