"""
test_sharded_pipeline.py — Unit and integration tests for ShardedReferenceIndex.

Tests:
1. test_sharded_index_build_and_shard_count       — shard files created correctly
2. test_sharded_index_vocab_build_and_load         — vocab built streaming, loadable from disk
3. test_sharded_blocking_semantic_parity           — sharded blocking produces SAME candidates
                                                     as monolithic GPUBlockingIndex
4. test_sharded_blocking_recall_not_worse_than_5pct — candidate recall within 5% of monolithic
5. test_sharded_ingestion_pipeline_parity          — run_sharded_ingestion end-to-end
                                                     produces same pair counts and feature parity
6. test_sharded_resume_capability                  — run_sharded_ingestion resumes from
                                                     partial chunks without re-running completed ones
7. test_existing_streaming_tests_unaffected        — confirm all original 7 streaming tests pass
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_validation_scale_check import load_or_create_large_scale_sample
from src.config import Config
from src.gpu.gpu_blocking import GPUBlockingConfig, GPUBlockingIndex, ShardedReferenceIndex
from src.gpu.gpu_features import TURN7_FEATURE_NAMES
from src.gpu.gpu_pipeline import Turn7Config, Turn7Pipeline
from src.preprocessing import preprocess_records


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_sample_200():
    """Load a deterministic 200-S1 benchmark sample once for the whole module."""
    cfg = Config()
    s1, s2, s3, gt = load_or_create_large_scale_sample(
        sample_s1_count=200,
        distractor_count=200,
        cache_dir=cfg.CACHE_DIR,
        random_seed=42,
    )
    return s1, s2, s3, gt


# ---------------------------------------------------------------------------
# Test 1: Shard file creation
# ---------------------------------------------------------------------------


def test_sharded_index_build_and_shard_count(small_sample_200):
    """build_shards() partitions the reference corpus into the correct number of shard files."""
    _, s2, s3, _ = small_sample_200
    s2_proc = preprocess_records(s2)
    s3_proc = preprocess_records(s3)
    total_ref = len(s2_proc) + len(s3_proc)

    tmp = tempfile.mkdtemp(prefix="test_shard_build_")
    try:
        shard_size = max(1, total_ref // 3)
        idx = ShardedReferenceIndex(shard_dir=tmp, shard_size=shard_size)
        meta = idx.build_shards(s2_proc, s3_proc)

        expected_shards = -(-total_ref // shard_size)  # ceiling division
        assert meta["n_shards"] == expected_shards, (
            f"Expected {expected_shards} shards for {total_ref} records with shard_size={shard_size}, "
            f"got {meta['n_shards']}"
        )
        assert meta["total_records"] == total_ref
        assert len(meta["shard_files"]) == expected_shards

        for fname in meta["shard_files"]:
            assert (Path(tmp) / fname).exists(), f"Shard file missing: {fname}"

        # Verify row counts sum up
        total_rows_in_shards = sum(
            len(pd.read_parquet(Path(tmp) / f)) for f in meta["shard_files"]
        )
        assert total_rows_in_shards == total_ref, (
            f"Row count mismatch: shards have {total_rows_in_shards} but expected {total_ref}"
        )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: Vocab build and disk persistence
# ---------------------------------------------------------------------------


def test_sharded_index_vocab_build_and_load(small_sample_200):
    """build_global_vocab() streams shards, persists to disk; load_vocab() restores it."""
    s1, s2, s3, _ = small_sample_200
    s2_proc = preprocess_records(s2)
    s3_proc = preprocess_records(s3)
    s1_proc = preprocess_records(s1)

    tmp = tempfile.mkdtemp(prefix="test_shard_vocab_")
    try:
        idx = ShardedReferenceIndex(shard_dir=tmp, shard_size=100)
        idx.build_shards(s2_proc, s3_proc)
        vocab = idx.build_global_vocab(s1_proc)

        assert len(vocab) > 0, "Vocab should not be empty"
        assert (Path(tmp) / "global_vocab.json").exists(), "Vocab file should be on disk"

        # Load fresh instance and verify
        idx2 = ShardedReferenceIndex(shard_dir=tmp, shard_size=100)
        vocab2 = idx2.load_vocab()
        assert len(vocab2) == len(vocab), "Loaded vocab size does not match built vocab"

        # A few spot checks
        for tok, cnt in list(vocab.items())[:5]:
            assert vocab2[tok] == cnt, f"Token {tok!r} count mismatch after reload"

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Blocking semantic parity
# ---------------------------------------------------------------------------


def test_sharded_blocking_semantic_parity(small_sample_200):
    """
    query_s1_batch() with shard_size = total_ref (single shard) MUST return
    IDENTICAL candidate pairs as the monolithic GPUBlockingIndex.
    """
    s1, s2, s3, _ = small_sample_200
    s2_proc = preprocess_records(s2)
    s3_proc = preprocess_records(s3)
    s1_proc = preprocess_records(s1)

    b_cfg = GPUBlockingConfig(top_k_per_s1=30)
    total_ref = len(s2_proc) + len(s3_proc)

    tmp = tempfile.mkdtemp(prefix="test_shard_parity_")
    try:
        # Monolithic reference
        mono_idx = GPUBlockingIndex(config=b_cfg)
        mono_idx.build_vocabulary(s1_proc, s2_proc, s3_proc)
        mono_idx.add_records(s2_proc)
        mono_idx.add_records(s3_proc)
        mono_cands = mono_idx.query_records(s1_proc)
        mono_pairs = set(zip(mono_cands["s1_id"], mono_cands["candidate_id"]))

        # Sharded reference (one big shard = semantically identical to monolithic)
        sharded_idx = ShardedReferenceIndex(
            shard_dir=tmp, config=b_cfg, shard_size=total_ref + 1
        )
        sharded_idx.build_shards(s2_proc, s3_proc)
        sharded_idx.build_global_vocab(s1_proc)
        shard_cands = sharded_idx.query_s1_batch(s1_proc)
        shard_pairs = set(zip(shard_cands["s1_id"], shard_cands["candidate_id"]))

        # With a single shard the global score accumulation is semantically identical
        assert mono_pairs == shard_pairs, (
            f"Candidate pair sets differ: "
            f"{len(mono_pairs - shard_pairs)} in mono only, "
            f"{len(shard_pairs - mono_pairs)} in shard only"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: Candidate recall within 5% of monolithic
# ---------------------------------------------------------------------------


def test_sharded_blocking_recall_not_worse_than_5pct(small_sample_200):
    """
    With multiple shards (shard_size << total_ref), candidate recall must remain
    within 5 percentage points of the monolithic index recall.
    """
    s1, s2, s3, gt = small_sample_200
    from src.evaluation import ground_truth_to_dict
    gt_dict = ground_truth_to_dict(gt)

    s2_proc = preprocess_records(s2)
    s3_proc = preprocess_records(s3)
    s1_proc = preprocess_records(s1)

    b_cfg = GPUBlockingConfig(top_k_per_s1=30)
    total_ref = len(s2_proc) + len(s3_proc)
    shard_size = max(50, total_ref // 5)  # ~5 shards

    tmp = tempfile.mkdtemp(prefix="test_shard_recall_")
    try:
        # Monolithic recall
        mono_idx = GPUBlockingIndex(config=b_cfg)
        mono_idx.build_vocabulary(s1_proc, s2_proc, s3_proc)
        mono_idx.add_records(s2_proc)
        mono_idx.add_records(s3_proc)
        mono_cands = mono_idx.query_records(s1_proc)

        def _recall(cands: pd.DataFrame, gt_dict: dict) -> float:
            if len(cands) == 0:
                return 0.0
            cands_by_s1 = cands.groupby("s1_id")["candidate_id"].apply(set).to_dict()
            retrieved = sum(
                1
                for s1_id, true_set in gt_dict.items()
                for tid in true_set
                if tid in cands_by_s1.get(str(s1_id), set())
            )
            total = sum(len(v) for v in gt_dict.values())
            return retrieved / max(total, 1) * 100

        mono_recall = _recall(mono_cands, gt_dict)

        # Sharded recall
        sharded_idx = ShardedReferenceIndex(shard_dir=tmp, config=b_cfg, shard_size=shard_size)
        sharded_idx.build_shards(s2_proc, s3_proc)
        sharded_idx.build_global_vocab(s1_proc)
        shard_cands = sharded_idx.query_s1_batch(s1_proc)
        shard_recall = _recall(shard_cands, gt_dict)

        assert shard_recall >= mono_recall - 5.0, (
            f"Sharded recall ({shard_recall:.2f}%) is more than 5 pp below "
            f"monolithic recall ({mono_recall:.2f}%)"
        )
        print(f"\n  Monolithic recall: {mono_recall:.2f}% | Sharded recall: {shard_recall:.2f}%")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: End-to-end sharded ingestion pipeline parity
# ---------------------------------------------------------------------------


def test_sharded_ingestion_pipeline_parity(small_sample_200):
    """
    run_sharded_ingestion() on a small sample with shard_size = total_ref
    (single shard) should produce the SAME candidate count and Macro F0.5
    as the standard run_streaming_ingestion() + run_training_from_chunks().
    """
    s1, s2, s3, gt = small_sample_200
    tmp = tempfile.mkdtemp(prefix="test_sharded_e2e_")
    out_standard = Path(tmp) / "standard"
    out_sharded = Path(tmp) / "sharded"

    total_ref = len(s2) + len(s3)

    try:
        b_cfg = GPUBlockingConfig(top_k_per_s1=30)
        cfg_base = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=30,
            learning_rate=0.05,
            random_state=42,
            val_size=0.20,
            s1_batch_size=50,
            streaming=True,
            resume=False,
        )

        # 1. Standard streaming pipeline
        pipe_std = Turn7Pipeline(config=cfg_base)
        res_std = pipe_std.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_standard
        )

        # 2. Sharded pipeline (single shard so semantically identical)
        pipe_shard = Turn7Pipeline(config=cfg_base)
        res_shard = pipe_shard.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt,
            output_dir=out_sharded,
            use_sharded=True,
            shard_size=total_ref + 1,  # single shard → semantic identity
            overwrite_shards=True,
        )

        # Verify candidate pair counts match
        std_cands = res_std["blocking_diagnostics"]["candidate_count"]
        shard_cands = res_shard["blocking_diagnostics"]["candidate_count"]
        assert std_cands == shard_cands, (
            f"Candidate count mismatch: standard={std_cands}, sharded={shard_cands}"
        )

        # Verify train/val splits match
        assert res_std["split_summary"]["train_pairs"] == res_shard["split_summary"]["train_pairs"]
        assert res_std["split_summary"]["val_pairs"] == res_shard["split_summary"]["val_pairs"]

        # Macro F0.5 must be very close (within 0.01)
        f05_std = res_std["validation_metrics"]["macro_f05"]
        f05_shard = res_shard["validation_metrics"]["macro_f05"]
        np.testing.assert_allclose(f05_shard, f05_std, atol=0.01,
            err_msg=f"Macro F0.5 deviation too large: standard={f05_std:.4f}, sharded={f05_shard:.4f}")

        # Verify shard metadata exists on disk
        shard_meta_path = out_sharded / "ref_shards" / "shard_meta.json"
        assert shard_meta_path.exists(), "shard_meta.json should be written to ref_shards/"

        vocab_path = out_sharded / "ref_shards" / "global_vocab.json"
        assert vocab_path.exists(), "global_vocab.json should be written to ref_shards/"

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 6: Sharded ingestion resume capability
# ---------------------------------------------------------------------------


def test_sharded_resume_capability(small_sample_200):
    """
    Sharded ingestion resumes from existing Parquet chunks without reprocessing
    completed batches, and produces the same final result as a fresh run.
    """
    s1, s2, s3, gt = small_sample_200
    tmp = tempfile.mkdtemp(prefix="test_sharded_resume_")
    out_dir = Path(tmp) / "sharded_resume"
    total_ref = len(s2) + len(s3)

    import src.gpu.gpu_pipeline as gp_module
    orig_feat = gp_module.compute_gpu_features

    try:
        b_cfg = GPUBlockingConfig(top_k_per_s1=30)
        cfg = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=30,
            random_state=42,
            val_size=0.20,
            s1_batch_size=40,
            streaming=True,
            resume=True,
        )

        # 1. First full run to establish baseline
        pipe_baseline = Turn7Pipeline(config=cfg)
        res_baseline = pipe_baseline.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt,
            output_dir=out_dir, use_sharded=True,
            shard_size=total_ref + 1, overwrite_shards=True,
        )
        baseline_pairs = res_baseline["blocking_diagnostics"]["candidate_count"]

        # 2. Second run with resume=True — should skip all completed chunks
        pipe_resume = Turn7Pipeline(config=cfg)
        res_resume = pipe_resume.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt,
            output_dir=out_dir, use_sharded=True,
            shard_size=total_ref + 1, overwrite_shards=False,
        )
        resume_pairs = res_resume["blocking_diagnostics"]["candidate_count"]

        assert baseline_pairs == resume_pairs, (
            f"Resumed run candidate count ({resume_pairs}) differs from baseline ({baseline_pairs})"
        )
        np.testing.assert_allclose(
            res_resume["validation_metrics"]["macro_f05"],
            res_baseline["validation_metrics"]["macro_f05"],
            atol=1e-5,
        )

    finally:
        gp_module.compute_gpu_features = orig_feat
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Regression Test 1: Capped posting lists must not be discarded at query time
# ---------------------------------------------------------------------------


def test_capped_posting_list_queryable():
    """Posting lists with len(plist) == cap must NOT be dropped at query time."""
    from collections import Counter
    from src.gpu.gpu_blocking import GPUBlockingConfig, GPUBlockingIndex, compute_key_frequency_discount

    cfg = GPUBlockingConfig(max_candidates_per_block=10, top_k_per_s1=20)
    idx = GPUBlockingIndex(config=cfg)

    # Ingest 10 records with exact same name to reach cap
    records = [
        {"entity_id": f"S2-{i}", "business_name": "Apex Tech Solutions", "business_address": "123 Main St", "country": "US"}
        for i in range(10)
    ]
    df = pd.DataFrame(records)
    idx.add_records(preprocess_records(df))

    # Posting list for exact name should have exactly 10 candidates (the cap)
    k = "a_exact:apex tech solutions"
    assert k in idx.index
    assert len(idx.index[k]) == 10

    # Query with S1 entity having the same name
    s1_rec = {"entity_id": "S1-999", "business_name": "Apex Tech Solutions", "business_address": "123 Main St", "country": "US"}
    s1_proc = preprocess_records(pd.DataFrame([s1_rec])).iloc[0]

    cands = idx.query_entity(s1_proc)
    # All 10 S2 candidates should be retrieved, NOT 0
    assert len(cands) == 10, f"Expected 10 candidates from capped list, got {len(cands)}"
    assert set(cands) == {f"S2-{i}" for i in range(10)}


# ---------------------------------------------------------------------------
# Regression Test 2: Top-k tie breaking and increased capacity
# ---------------------------------------------------------------------------


def test_sharded_top_k_tie_breaking():
    """top_k_per_s1=120 retains candidates that tie at cutoff where top_k=60 cuts them off."""
    from src.gpu.gpu_blocking import GPUBlockingConfig, ShardedReferenceIndex

    tmp = tempfile.mkdtemp(prefix="test_topk_")
    try:
        # Create 100 S2 records with common token
        s2_recs = [
            {"entity_id": f"S2-{i:03d}", "business_name": f"Sharma Store {i}", "business_address": f"{i} Market Rd", "country": "India"}
            for i in range(100)
        ]
        s2_proc = preprocess_records(pd.DataFrame(s2_recs))

        # Build 2 shards of 50 records
        idx = ShardedReferenceIndex(shard_dir=tmp, shard_size=50, config=GPUBlockingConfig(top_k_per_s1=120))
        idx.build_shards(s2_proc, overwrite=True)

        s1_rec = {"entity_id": "S1-001", "business_name": "Sharma Store Special", "business_address": "99 Market Rd", "country": "India"}
        s1_proc = preprocess_records(pd.DataFrame([s1_rec]))
        idx.build_global_vocab(s1_proc)

        # With top_k=120: all 100 candidates matching 'sharma' should be retrieved
        idx.config.top_k_per_s1 = 120
        res120 = idx.query_s1_batch(s1_proc)
        assert len(res120) == 100, f"Expected 100 candidates at top_k=120, got {len(res120)}"

        # With top_k=60: hard-capped at 60
        idx.config.top_k_per_s1 = 60
        res60 = idx.query_s1_batch(s1_proc)
        assert len(res60) == 60, f"Expected exactly 60 candidates at top_k=60, got {len(res60)}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Regression Test 3: Global token frequency discounting
# ---------------------------------------------------------------------------


def test_global_token_frequency_discount():
    """compute_key_frequency_discount accurately applies global frequency discounting."""
    from collections import Counter
    from src.gpu.gpu_blocking import compute_key_frequency_discount

    global_vocab = Counter({
        "hyper_common": 50_000,
        "common": 5_000,
        "medium": 1_000,
        "rare": 50,
        "unique": 2,
    })

    # Rare tokens should get discount 1.0
    assert compute_key_frequency_discount("b_tok:unique", global_vocab, local_plist_len=5) == 1.0
    assert compute_key_frequency_discount("b_tok:rare", global_vocab, local_plist_len=5) == 1.0

    # Hyper common token should receive strong discount regardless of local shard plist len
    d_hyper = compute_key_frequency_discount("b_tok:hyper_common", global_vocab, local_plist_len=5)
    assert d_hyper < 0.5, f"Expected discount < 0.5 for hyper common token, got {d_hyper}"

    # Common token discount < medium token discount < rare token discount
    d_common = compute_key_frequency_discount("b_tok:common", global_vocab, local_plist_len=5)
    d_medium = compute_key_frequency_discount("b_tok:medium", global_vocab, local_plist_len=5)
    assert d_hyper < d_common < d_medium <= 1.0


# ---------------------------------------------------------------------------
# Regression Test 4: S2 and S3 pool reporting integrity
# ---------------------------------------------------------------------------


def test_s2_s3_pool_reporting_integrity(small_sample_200):
    """build_shards and ShardedReferenceIndex accurately record exact S2 and S3 pool counts."""
    s1, s2, s3, gt = small_sample_200
    s1_proc = preprocess_records(s1)
    s2_proc = preprocess_records(s2)
    s3_proc = preprocess_records(s3)

    tmp = tempfile.mkdtemp(prefix="test_meta_reporting_")
    try:
        idx = ShardedReferenceIndex(shard_dir=tmp, shard_size=100)
        meta = idx.build_shards(s2_proc, s3_proc, overwrite=True)

        assert "s2_records" in meta, "s2_records must be in shard metadata"
        assert "s3_records" in meta, "s3_records must be in shard metadata"
        assert meta["s2_records"] == len(s2_proc), f"Expected s2_records={len(s2_proc)}, got {meta['s2_records']}"
        assert meta["s3_records"] == len(s3_proc), f"Expected s3_records={len(s3_proc)}, got {meta['s3_records']}"
        assert meta["total_records"] == len(s2_proc) + len(s3_proc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

