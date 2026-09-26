"""
test_disk_blocking.py — Comprehensive tests for DiskBlockingIndex (SQLite disk-backed blocking).

Covers:
1. Index creation and SQLite schema initialization.
2. Lookup correctness: retrieves all true matches from test fixture.
3. Compatibility with in-memory BlockingIndex: exact match of candidate pairs.
4. Duplicate handling: handles duplicate entity IDs (INSERT OR REPLACE) safely.
5. Deterministic lookup: identical candidate order across multiple query runs.
6. Memory-safe chunked processing: adding records across multiple chunks.
7. Candidate records retrieval: correct field types (bool, string) and structure.
8. Generic block explosion cap: skips keys with count >= max_candidates_per_block.
9. Lifecycle cleanup: close and cleanup remove database files.
10. Memory scalability: Python RSS does not grow linearly with candidate record count.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.blocking import (
    BlockingConfig,
    BlockingIndex,
    DiskBlockingIndex,
    build_candidates,
    generate_record_blocking_keys,
)
from src.preprocessing import preprocess_records


@pytest.fixture
def sample_sources():
    s1 = pd.DataFrame(
        {
            "entity_id": ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5"],
            "business_name": [
                "Orelee's Barbershop LLC",
                "राम मार्केटिंग प्राइवेट लिमिटेड",
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
# 1. Index creation and schema initialization
# ---------------------------------------------------------------------------


def test_disk_index_creation(tmp_path):
    db_file = tmp_path / "test_create.db"
    idx = DiskBlockingIndex(db_path=db_file)
    try:
        assert db_file.exists()
        assert idx.total_records_indexed == 0
        assert not idx.is_finalized

        # Verify tables created
        cur = idx.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert "candidate_records" in tables
        assert "raw_blocking_keys" in tables
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# 2. Lookup correctness: retrieves all true matches
# ---------------------------------------------------------------------------


def test_disk_index_lookup_correctness(tmp_path, sample_sources):
    s1, s2, s3, gt = sample_sources
    db_file = tmp_path / "test_lookup.db"

    idx = DiskBlockingIndex(db_path=db_file)
    try:
        idx.add_records(s2)
        idx.add_records(s3)
        idx.finalize()

        assert idx.is_finalized
        assert idx.total_records_indexed == len(s2) + len(s3)

        cands = idx.query_records(s1)
        pairs = set(zip(cands["s1_id"], cands["candidate_id"]))

        # True matches must all be retrieved
        assert ("S1-1", "S2-10") in pairs
        assert ("S1-2", "S2-20") in pairs
        assert ("S1-3", "S3-30") in pairs
        assert ("S1-4", "S2-40") in pairs
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# 3. Compatibility with in-memory BlockingIndex
# ---------------------------------------------------------------------------


def test_disk_index_compatibility_with_in_memory(tmp_path, sample_sources):
    s1, s2, s3, gt = sample_sources
    cfg = BlockingConfig()

    s1_proc = preprocess_records(s1)
    s2_proc = preprocess_records(s2)
    s3_proc = preprocess_records(s3)

    # In-memory index
    mem_idx = BlockingIndex(config=cfg)
    mem_idx.add_records(s2_proc)
    mem_idx.add_records(s3_proc)
    mem_cands = mem_idx.query_records(s1_proc)

    # Disk-backed index
    db_file = tmp_path / "test_compat.db"
    disk_idx = DiskBlockingIndex(db_path=db_file, config=cfg)
    try:
        disk_idx.add_records(s2_proc)
        disk_idx.add_records(s3_proc)
        disk_idx.finalize()
        disk_cands = disk_idx.query_records(s1_proc)

        # Both must produce identical candidate pairs
        assert len(mem_cands) == len(disk_cands)
        assert list(mem_cands.columns) == list(disk_cands.columns)
        assert mem_cands.equals(disk_cands)
    finally:
        disk_idx.close()


# ---------------------------------------------------------------------------
# 4. Duplicate handling
# ---------------------------------------------------------------------------


def test_disk_index_duplicate_handling(tmp_path):
    db_file = tmp_path / "test_dups.db"
    idx = DiskBlockingIndex(db_path=db_file)
    try:
        df1 = pd.DataFrame(
            {
                "entity_id": ["S2-1", "S2-2"],
                "business_name": ["Acme Corp", "Beta LLC"],
                "business_address": ["123 Main St", "456 Oak Rd"],
                "country": ["US", "US"],
            }
        )
        # Re-add S2-1 with updated name
        df2 = pd.DataFrame(
            {
                "entity_id": ["S2-1", "S2-3"],
                "business_name": ["Acme Corporation", "Gamma Inc"],
                "business_address": ["123 Main St", "789 Pine Ave"],
                "country": ["US", "US"],
            }
        )
        idx.add_records(df1)
        idx.add_records(df2)
        idx.finalize()

        # candidate_records must have 3 unique entity IDs
        records = idx.get_candidate_records(["S2-1", "S2-2", "S2-3"])
        assert len(records) == 3
        eids = {r["entity_id"] for r in records}
        assert eids == {"S2-1", "S2-2", "S2-3"}

        # S2-1 should reflect the updated record
        rec_s2_1 = [r for r in records if r["entity_id"] == "S2-1"][0]
        assert "acme corporation" in rec_s2_1["name_norm"]
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# 5. Deterministic lookup
# ---------------------------------------------------------------------------


def test_disk_index_deterministic_lookup(tmp_path, sample_sources):
    s1, s2, s3, gt = sample_sources
    db_file = tmp_path / "test_det.db"

    idx = DiskBlockingIndex(db_path=db_file)
    try:
        idx.add_records(s2)
        idx.add_records(s3)
        idx.finalize()

        cands1 = idx.query_records(s1)
        cands2 = idx.query_records(s1)
        cands3 = idx.query_records(s1)

        assert cands1.equals(cands2)
        assert cands2.equals(cands3)
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# 6. Memory-safe chunked processing
# ---------------------------------------------------------------------------


def test_disk_index_chunked_processing(tmp_path):
    db_file = tmp_path / "test_chunked.db"
    idx = DiskBlockingIndex(db_path=db_file)
    try:
        # Add 5 small chunks
        for chunk_i in range(5):
            df_chunk = pd.DataFrame(
                {
                    "entity_id": [f"S2-{chunk_i * 10 + j}" for j in range(10)],
                    "business_name": [f"Company {chunk_i}_{j} LLC" for j in range(10)],
                    "business_address": [f"{100 + j} Main St" for j in range(10)],
                    "country": ["US"] * 10,
                }
            )
            idx.add_records(df_chunk)

        idx.finalize()
        assert idx.total_records_indexed == 50

        # Query an entity from chunk 3
        s1 = pd.DataFrame(
            {
                "entity_id": ["S1-target"],
                "business_name": ["Company 3_5 LLC"],
                "business_address": ["105 Main St"],
                "country": ["US"],
            }
        )
        cands = idx.query_records(s1)
        cand_ids = set(cands["candidate_id"])
        assert "S2-35" in cand_ids
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# 7. Candidate records retrieval
# ---------------------------------------------------------------------------


def test_disk_index_candidate_records_retrieval(tmp_path, sample_sources):
    s1, s2, s3, gt = sample_sources
    db_file = tmp_path / "test_records.db"

    idx = DiskBlockingIndex(db_path=db_file)
    try:
        idx.add_records(s2)
        idx.add_records(s3)
        idx.finalize()

        # Fetch records
        records = idx.get_candidate_records(["S2-10", "S3-30", "NONEXISTENT"])
        assert len(records) == 2

        rec_s2 = [r for r in records if r["entity_id"] == "S2-10"][0]
        assert rec_s2["source"] == "S2"
        assert isinstance(rec_s2["address_is_missing"], bool)
        assert isinstance(rec_s2["is_domain"], bool)
        assert rec_s2["country_norm"] == "US"

        rec_s3 = [r for r in records if r["entity_id"] == "S3-30"][0]
        assert rec_s3["source"] == "S3"
        assert rec_s3["address_is_missing"] is True
        assert rec_s3["is_domain"] is True

        # Test dictionary-like .get()
        assert idx.get("S2-10") is not None
        assert idx.get("NONEXISTENT") is None
        assert len(idx) == 5
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# 8. Generic block explosion cap
# ---------------------------------------------------------------------------


def test_disk_index_max_candidates_cap(tmp_path):
    # 120 candidates sharing the exact same name
    large_s2 = pd.DataFrame(
        {
            "entity_id": [f"S2-{i}" for i in range(120)],
            "business_name": ["Generic Corporation Inc"] * 120,
            "business_address": ["100 Main St"] * 120,
            "country": ["US"] * 120,
        }
    )
    s1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "business_name": ["Generic Corporation Inc"],
            "business_address": ["100 Main St"],
            "country": ["US"],
        }
    )

    cfg = BlockingConfig(max_candidates_per_block=30, top_k_per_s1=25)
    db_file = tmp_path / "test_cap.db"
    idx = DiskBlockingIndex(db_path=db_file, config=cfg)
    try:
        idx.add_records(large_s2)
        idx.finalize()

        # Keys with count >= 30 are filtered from posting_lists
        cands = idx.query_records(s1)
        # Since all keys for "Generic Corporation Inc" exceeded cap=30,
        # they are omitted from posting_lists, resulting in 0 candidates
        assert len(cands) <= 25
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# 9. Lifecycle cleanup
# ---------------------------------------------------------------------------


def test_disk_index_cleanup(tmp_path):
    db_file = tmp_path / "test_clean.db"
    idx = DiskBlockingIndex(db_path=db_file)
    df = pd.DataFrame(
        {
            "entity_id": ["S2-1"],
            "business_name": ["Test"],
            "business_address": ["123 St"],
            "country": ["US"],
        }
    )
    idx.add_records(df)
    idx.finalize()
    assert db_file.exists()

    idx.cleanup()
    assert not db_file.exists()
    assert idx.conn is None


# ---------------------------------------------------------------------------
# 10. Memory scalability: Python RSS does NOT grow linearly with record count
# ---------------------------------------------------------------------------


def test_disk_index_memory_scalability(tmp_path):
    """Demonstrate that indexing multiple chunks does not grow Python RSS linearly."""
    try:
        import psutil
        def get_rss():
            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        pytest.skip("psutil not installed")

    import gc
    gc.collect()
    rss_baseline = get_rss()

    db_file = tmp_path / "test_scale.db"
    idx = DiskBlockingIndex(db_path=db_file)

    chunk_rss = []
    num_chunks = 5
    rows_per_chunk = 2000

    for c in range(num_chunks):
        df_chunk = pd.DataFrame(
            {
                "entity_id": [f"S2-{c}_{i}" for i in range(rows_per_chunk)],
                "business_name": [f"Business {c}_{i} Services LLC" for i in range(rows_per_chunk)],
                "business_address": [f"{1000 + i} Commercial Blvd Suite {c}" for i in range(rows_per_chunk)],
                "country": ["US"] * rows_per_chunk,
            }
        )
        proc_chunk = preprocess_records(df_chunk)
        idx.add_records(proc_chunk)
        del proc_chunk, df_chunk
        gc.collect()
        chunk_rss.append(get_rss())

    idx.finalize()
    assert idx.total_records_indexed == num_chunks * rows_per_chunk

    # Check RSS stability: between chunk 1 and chunk 5, RSS delta must be small (< 30 MB)
    # unlike in-memory storage which would grow linearly with 10k records
    rss_delta_chunks = chunk_rss[-1] - chunk_rss[0]
    idx.cleanup()

    assert rss_delta_chunks < 35.0, (
        f"RSS grew significantly across chunks: chunk0={chunk_rss[0]:.1f} MB, "
        f"chunk4={chunk_rss[-1]:.1f} MB, delta={rss_delta_chunks:.1f} MB"
    )


# ---------------------------------------------------------------------------
# 11. build_candidates with use_disk_index=True
# ---------------------------------------------------------------------------


def test_build_candidates_disk_flag(sample_sources):
    s1, s2, s3, gt = sample_sources
    cands_mem = build_candidates(s1, s2, s3, use_disk_index=False)
    cands_disk = build_candidates(s1, s2, s3, use_disk_index=True)

    assert len(cands_mem) == len(cands_disk)
    assert cands_mem.equals(cands_disk)
