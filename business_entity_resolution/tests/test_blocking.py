"""
test_blocking.py — Comprehensive behavioral tests for multi-strategy blocking.

Covers:
- Blocker A (Exact name, compact name, name without suffix, domain stem)
- Blocker B (Selective token blocks, token bigrams, stopword filtering)
- Blocker C (Address number + street/city tokens)
- Blocker D (Composite name prefix/token + address number)
- Candidate deduplication
- Posting list explosion control (max_candidates_per_block)
- Top-k candidate cap per Source 1 entity
- Country openness (France records match without rejection)
- Missing address resilience (missing address matches via name blockers)
- Unicode & Indic script candidate retrieval
- Candidate recall and reduction ratio calculation
- Detailed diagnostics metrics
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.blocking import (
    BlockingConfig,
    BlockingIndex,
    blocking_recall,
    blocking_reduction_ratio,
    build_candidates,
    detailed_blocking_diagnostics,
    generate_record_blocking_keys,
)


@pytest.fixture
def sample_sources():
    s1 = pd.DataFrame(
        {
            "entity_id": ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5"],
            "business_name": [
                "Orelee's Barbershop LLC",
                "রাম मार्केटिंग प्राइवेट लिमिटेड",
                "wilfordhancock.com",
                "Marina École France Sarl",
                "Singleton Corp",
            ],
            "business_address": [
                "1795 Westchester Drive, High Point, NC",
                "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
                "Mack Rd, Haltom City, Texas",
                "175 Boulevard du Président Franklin Roosevelt, Bordeaux",
                "100 Unknown St, New York, NY",
            ],
            "country": ["US", "India", "US", "France", "US"],
        }
    )

    s2 = pd.DataFrame(
        {
            "entity_id": ["S2-10", "S2-20", "S2-40"],
            "business_name": [
                "Orelee's Barbershop",
                "राम मार्केटिंग",
                "Marina École France",
            ],
            "business_address": [
                "1795 Westchester Drive, High Point, NC",
                "KH NO. -570/13, NEW DELHI, Delhi",
                "Bordeaux, 175 Boulevard du Président Franklin Roosevelt",
            ],
            "country": ["US", "India", "France"],
        }
    )

    s3 = pd.DataFrame(
        {
            "entity_id": ["S3-30", "S3-100"],
            "business_name": [
                "wilfordhancock.com",
                "Unrelated Business LLC",
            ],
            "business_address": [
                None,  # Missing address
                "999 Random Blvd, Austin, TX",
            ],
            "country": ["US", "US"],
        }
    )

    gt = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5"],
            "matched_entity_ids": ["S2-10", "S2-20", "S3-30", "S2-40", ""],
        }
    )

    return s1, s2, s3, gt


# ---------------------------------------------------------------------------
# 1. Key Generation Tests
# ---------------------------------------------------------------------------


def test_generate_blocking_keys_blocker_a():
    row = {
        "entity_id": "S1-1",
        "business_name": "Orelee's Barbershop LLC",
        "business_address": "1795 Westchester Drive",
        "country": "US",
    }
    cfg = BlockingConfig()
    keys = generate_record_blocking_keys(row, cfg)

    assert any("oreleesbarbershop" in k for k in keys["A"])
    assert any("orelee s barbershop" in k for k in keys["A"])


def test_generate_blocking_keys_domain():
    row = {
        "entity_id": "S3-30",
        "business_name": "wilfordhancock.com",
        "business_address": None,
        "country": "US",
    }
    cfg = BlockingConfig()
    keys = generate_record_blocking_keys(row, cfg)

    assert "a_domain:wilfordhancock" in keys["A"]


def test_generate_blocking_keys_stopwords_filtered():
    row = {
        "entity_id": "S1-99",
        "business_name": "The Company Group LLC",
        "business_address": "123 Main St",
        "country": "US",
    }
    cfg = BlockingConfig()
    keys = generate_record_blocking_keys(row, cfg)

    # Generic stopwords must not generate single-token blocks
    assert "b_tok:the" not in keys["B"]
    assert "b_tok:company" not in keys["B"]
    assert "b_tok:llc" not in keys["B"]
    assert "b_tok:group" not in keys["B"]


def test_generate_blocking_keys_composite_blocker_d():
    row = {
        "entity_id": "S1-1",
        "business_name": "Orelee's Barbershop",
        "business_address": "1795 Westchester Drive",
        "country": "US",
    }
    cfg = BlockingConfig()
    keys = generate_record_blocking_keys(row, cfg)

    assert any("d_name_num:orelee_1795" in k or "1795" in k for k in keys["D"])


# ---------------------------------------------------------------------------
# 2. Candidate Retrieval Tests
# ---------------------------------------------------------------------------


def test_build_candidates_basic(sample_sources):
    s1, s2, s3, gt = sample_sources
    candidates = build_candidates(s1, s2, s3)

    assert "s1_id" in candidates.columns
    assert "candidate_id" in candidates.columns
    assert len(candidates) > 0

    # True matches must all be retrieved
    c_pairs = set(zip(candidates["s1_id"], candidates["candidate_id"]))
    assert ("S1-1", "S2-10") in c_pairs
    assert ("S1-2", "S2-20") in c_pairs
    assert ("S1-3", "S3-30") in c_pairs
    assert ("S1-4", "S2-40") in c_pairs


def test_candidate_deduplication(sample_sources):
    s1, s2, s3, gt = sample_sources
    candidates = build_candidates(s1, s2, s3)

    # Pairs must be completely unique
    duplicates = candidates.duplicated(subset=["s1_id", "candidate_id"]).sum()
    assert duplicates == 0


def test_no_self_matches(sample_sources):
    s1, s2, s3, gt = sample_sources
    candidates = build_candidates(s1, s2, s3)

    # S1 entity must never be paired with another S1 entity
    assert not candidates["candidate_id"].str.startswith("S1").any()


def test_missing_address_candidate_retrieval(sample_sources):
    s1, s2, s3, gt = sample_sources
    candidates = build_candidates(s1, s2, s3)

    # S1-3 matches S3-30 which has None as address
    c_pairs = set(zip(candidates["s1_id"], candidates["candidate_id"]))
    assert ("S1-3", "S3-30") in c_pairs


def test_france_country_candidate_retrieval(sample_sources):
    s1, s2, s3, gt = sample_sources
    candidates = build_candidates(s1, s2, s3)

    # French entity S1-4 matches French candidate S2-40
    c_pairs = set(zip(candidates["s1_id"], candidates["candidate_id"]))
    assert ("S1-4", "S2-40") in c_pairs


def test_blocker_ablation_toggles(sample_sources):
    s1, s2, s3, gt = sample_sources

    # Test Blocker A only
    cfg_a = BlockingConfig(
        enable_blocker_a=True,
        enable_blocker_b=False,
        enable_blocker_c=False,
        enable_blocker_d=False,
    )
    cands_a = build_candidates(s1, s2, s3, config=cfg_a)
    pairs_a = set(zip(cands_a["s1_id"], cands_a["candidate_id"]))
    assert ("S1-1", "S2-10") in pairs_a

    # Test Blocker B only
    cfg_b = BlockingConfig(
        enable_blocker_a=False,
        enable_blocker_b=True,
        enable_blocker_c=False,
        enable_blocker_d=False,
    )
    cands_b = build_candidates(s1, s2, s3, config=cfg_b)
    assert len(cands_b) > 0


def test_candidate_explosion_capped():
    # Construct 150 candidate records sharing the same name
    large_s2 = pd.DataFrame(
        {
            "entity_id": [f"S2-{i}" for i in range(150)],
            "business_name": ["Apex Logistics Inc"] * 150,
            "business_address": ["100 Main St"] * 150,
            "country": ["US"] * 150,
        }
    )
    s1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "business_name": ["Apex Logistics Inc"],
            "business_address": ["100 Main St"],
            "country": ["US"],
        }
    )

    cfg = BlockingConfig(max_candidates_per_block=30, top_k_per_s1=25)
    candidates = build_candidates(s1, large_s2, pd.DataFrame(), config=cfg)

    # Must be bounded by top_k_per_s1
    assert len(candidates) <= 25


# ---------------------------------------------------------------------------
# 3. Quality Metrics Tests
# ---------------------------------------------------------------------------


def test_blocking_recall_perfect(sample_sources):
    s1, s2, s3, gt = sample_sources
    candidates = build_candidates(s1, s2, s3)

    rec = blocking_recall(candidates, gt)
    assert rec == 1.0


def test_blocking_recall_zero(sample_sources):
    s1, s2, s3, gt = sample_sources
    empty_cands = pd.DataFrame(columns=["s1_id", "candidate_id"])
    rec = blocking_recall(empty_cands, gt)
    assert rec == 0.0


def test_blocking_reduction_ratio():
    cands = pd.DataFrame({"s1_id": ["S1-1"] * 10, "candidate_id": [f"S2-{i}" for i in range(10)]})
    rr = blocking_reduction_ratio(cands, source1=100, source2=1000, source3=1000)
    # Total pairs = 100 * 2000 = 200,000. 10 candidates -> RR = 1 - 10/200,000 = 0.99995
    assert rr > 0.9999


def test_detailed_blocking_diagnostics(sample_sources):
    s1, s2, s3, gt = sample_sources
    candidates = build_candidates(s1, s2, s3)

    diag = detailed_blocking_diagnostics(
        candidates,
        gt,
        source1=s1,
        source2_count=len(s2),
        source3_count=len(s3),
    )

    assert diag["overall_candidate_recall"] == 1.0
    assert diag["s2_candidate_recall"] == 1.0
    assert diag["s3_candidate_recall"] == 1.0
    assert diag["singleton_count"] == 1
    assert diag["total_candidates"] == len(candidates)
    assert diag["candidate_reduction_ratio"] is not None
    assert diag["candidates_per_s1_max"] >= 1
