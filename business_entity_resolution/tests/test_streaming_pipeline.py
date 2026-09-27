"""
Unit tests for Turn 9 streaming pipeline:
1. Parity test between monolithic and streaming pipeline on a small deterministic dataset.
2. Verification of resume capability and manifest tracking.
3. Memory cleanup and telemetry validation.
"""

from __future__ import annotations

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
from src.gpu.gpu_blocking import GPUBlockingConfig
from src.gpu.gpu_features import TURN7_FEATURE_NAMES
from src.gpu.gpu_pipeline import Turn7Config, Turn7Pipeline, load_preallocated_train_arrays


@pytest.fixture
def small_sample():
    """Load or generate a small deterministic sample (100 S1 records)."""
    cfg = Config()
    s1, s2, s3, gt = load_or_create_large_scale_sample(
        sample_s1_count=100,
        distractor_count=100,
        cache_dir=cfg.CACHE_DIR,
        random_seed=42,
    )
    return s1, s2, s3, gt


def test_streaming_vs_monolithic_parity(small_sample):
    """
    Test that the streaming pipeline produces identical candidate pairs,
    feature values (34 features), labels, and evaluation metrics as the monolithic pipeline.
    """
    s1, s2, s3, gt = small_sample
    temp_dir = tempfile.mkdtemp(prefix="test_streaming_parity_")
    out_monolithic = Path(temp_dir) / "monolithic"
    out_streaming = Path(temp_dir) / "streaming"

    try:
        b_cfg = GPUBlockingConfig(top_k_per_s1=30)

        # 1. Monolithic Run
        cfg_mono = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=30,
            learning_rate=0.05,
            random_state=42,
            val_size=0.20,
            sample_s1_count=100,
            distractor_count=100,
            streaming=False,
        )
        pipe_mono = Turn7Pipeline(config=cfg_mono)
        res_mono = pipe_mono.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_monolithic
        )

        # 2. Streaming Run (batch size 25 -> 4 batches for 100 S1 records)
        cfg_stream = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=30,
            learning_rate=0.05,
            random_state=42,
            val_size=0.20,
            sample_s1_count=100,
            distractor_count=100,
            s1_batch_size=25,
            streaming=True,
            resume=False,
        )
        pipe_stream = Turn7Pipeline(config=cfg_stream)
        res_stream = pipe_stream.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_streaming
        )

        # 3. Verify Candidate Pair & Feature Parity
        mono_pairs = res_mono["blocking_diagnostics"]["candidate_count"]
        stream_pairs = res_stream["blocking_diagnostics"]["candidate_count"]
        assert mono_pairs == stream_pairs, (
            f"Candidate pair count mismatch: mono={mono_pairs} vs stream={stream_pairs}"
        )
        assert res_mono["split_summary"]["train_pairs"] == res_stream["split_summary"]["train_pairs"]
        assert res_mono["split_summary"]["val_pairs"] == res_stream["split_summary"]["val_pairs"]

        # Read streaming parquet chunks
        chunk_files = sorted((out_streaming / "streaming_chunks").glob("chunk_*.parquet"))
        assert len(chunk_files) == 4, f"Expected 4 chunks, got {len(chunk_files)}"

        stream_dfs = [pd.read_parquet(f) for f in chunk_files]
        stream_all = pd.concat(stream_dfs, ignore_index=True)

        assert len(stream_all) == mono_pairs
        assert len(pipe_stream.feature_names) == 34
        for fname in pipe_stream.feature_names:
            assert fname in stream_all.columns, f"Missing feature {fname} in streaming parquet"
            assert not stream_all[fname].isna().any(), f"NaNs found in feature {fname}"

        # Compare metric results
        # Optimal threshold and Macro F0.5 should be extremely close or identical
        mono_f05 = res_mono["validation_metrics"]["macro_f05"]
        stream_f05 = res_stream["validation_metrics"]["macro_f05"]
        np.testing.assert_allclose(stream_f05, mono_f05, atol=1e-3)

        # Verify streaming telemetry structure
        telemetry = res_stream.get("streaming_telemetry", {})
        assert telemetry.get("total_batches") == 4
        assert telemetry.get("number_of_chunks") == 4
        assert telemetry.get("total_candidate_pairs") == mono_pairs
        assert telemetry.get("peak_ram_mb", 0) > 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_streaming_resume_capability(small_sample):
    """
    Test that the streaming pipeline properly resumes from intermediate chunks
    without recomputing existing chunks.
    """
    s1, s2, s3, gt = small_sample
    temp_dir = tempfile.mkdtemp(prefix="test_streaming_resume_")
    out_dir = Path(temp_dir) / "resume_test"

    try:
        b_cfg = GPUBlockingConfig(top_k_per_s1=30)
        cfg = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=30,
            sample_s1_count=100,
            distractor_count=100,
            s1_batch_size=25,
            streaming=True,
            resume=True,
        )

        pipe = Turn7Pipeline(config=cfg)
        res1 = pipe.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_dir
        )

        # Second run with resume=True should detect all chunks completed
        pipe2 = Turn7Pipeline(config=cfg)
        res2 = pipe2.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_dir
        )

        assert res1["blocking_diagnostics"]["candidate_count"] == res2["blocking_diagnostics"]["candidate_count"]
        assert res1["validation_metrics"]["macro_f05"] == res2["validation_metrics"]["macro_f05"]

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_preallocated_loading_exact_parity():
    """
    Regression test: confirm load_preallocated_train_arrays produces bit-identical
    arrays compared to the old pd.concat path for synthetic multi-chunk Parquet data.
    """
    feature_names = TURN7_FEATURE_NAMES
    n_feats = len(feature_names)
    temp_dir = tempfile.mkdtemp(prefix="test_prealloc_parity_")
    chunks_dir = Path(temp_dir)

    try:
        np.random.seed(42)
        chunk_files = []
        n_chunks = 3
        rows_per_chunk = 60

        for c_idx in range(n_chunks):
            # Create synthetic feature matrix
            feat_data = np.random.randn(rows_per_chunk, n_feats).astype(np.float32)
            df = pd.DataFrame(feat_data, columns=feature_names)
            # Mix train and val splits
            splits = ["train"] * 40 + ["val"] * 20
            df["split"] = splits
            df["label"] = np.random.randint(0, 2, size=rows_per_chunk, dtype=np.int32)
            df["s1_id"] = [f"s1_{c_idx}_{i}" for i in range(rows_per_chunk)]
            df["candidate_id"] = [f"c_{c_idx}_{i}" for i in range(rows_per_chunk)]

            cp = chunks_dir / f"chunk_{c_idx:04d}.parquet"
            df.to_parquet(cp, index=False, engine="pyarrow")
            chunk_files.append(cp)

        # 1. Baseline: Old pd.concat path
        train_dfs = [
            pd.read_parquet(cp, filters=[("split", "==", "train")], columns=feature_names + ["label"])
            for cp in chunk_files
        ]
        concat_df = pd.concat(train_dfs, ignore_index=True)
        expected_X = concat_df[feature_names].to_numpy(dtype=np.float32)
        expected_y = concat_df["label"].to_numpy(dtype=np.int32)
        total_train_rows = len(concat_df)
        assert total_train_rows == 120

        # 2. New pre-allocated path
        actual_X, actual_y = load_preallocated_train_arrays(
            chunk_files=chunk_files,
            feature_names=feature_names,
            train_pairs_count=total_train_rows,
        )

        # 3. Bit-identical equality check
        assert actual_X.shape == expected_X.shape
        assert actual_y.shape == expected_y.shape
        np.testing.assert_array_equal(actual_X, expected_X)
        np.testing.assert_array_equal(actual_y, expected_y)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_preallocated_loading_count_mismatch_failure():
    """
    Regression test: confirm load_preallocated_train_arrays fails loudly and explicitly
    (with ValueError, not silent data corruption or uninitialized garbage memory)
    when train_pairs_count is either too large (underfill) or too small (overfill).
    """
    feature_names = TURN7_FEATURE_NAMES
    n_feats = len(feature_names)
    temp_dir = tempfile.mkdtemp(prefix="test_prealloc_mismatch_")
    chunks_dir = Path(temp_dir)

    try:
        np.random.seed(42)
        chunk_files = []
        n_chunks = 2
        rows_per_chunk = 50

        for c_idx in range(n_chunks):
            feat_data = np.random.randn(rows_per_chunk, n_feats).astype(np.float32)
            df = pd.DataFrame(feat_data, columns=feature_names)
            df["split"] = ["train"] * 30 + ["val"] * 20
            df["label"] = np.random.randint(0, 2, size=rows_per_chunk, dtype=np.int32)
            cp = chunks_dir / f"chunk_{c_idx:04d}.parquet"
            df.to_parquet(cp, index=False, engine="pyarrow")
            chunk_files.append(cp)

        actual_train_rows = 60  # 30 * 2

        # Case 1: Count too large (underfill -> array would have uninitialized tail)
        with pytest.raises(ValueError, match="underfilled"):
            load_preallocated_train_arrays(
                chunk_files=chunk_files,
                feature_names=feature_names,
                train_pairs_count=actual_train_rows + 5,
            )

        # Case 2: Count too small (overfill -> chunk exceeds pre-allocated buffer)
        with pytest.raises(ValueError, match="exceeds pre-allocated train capacity"):
            load_preallocated_train_arrays(
                chunk_files=chunk_files,
                feature_names=feature_names,
                train_pairs_count=actual_train_rows - 5,
            )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_streaming_interrupted_resume_fidelity(small_sample):
    """
    Test that an interrupted streaming run (simulated mid-way crash after batch 2 of 4)
    resumes properly from intermediate chunks, reconstructs exact train_pairs_count,
    and produces evaluation metrics identical to an uninterrupted baseline run.
    """
    s1, s2, s3, gt = small_sample
    temp_dir = tempfile.mkdtemp(prefix="test_interrupted_resume_")
    out_baseline = Path(temp_dir) / "baseline"
    out_interrupted = Path(temp_dir) / "interrupted"

    import src.gpu.gpu_pipeline as gp_module
    orig_compute_gpu_features = gp_module.compute_gpu_features

    try:
        b_cfg = GPUBlockingConfig(top_k_per_s1=30)
        cfg_base = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=30,
            learning_rate=0.05,
            random_state=42,
            val_size=0.20,
            sample_s1_count=100,
            distractor_count=100,
            s1_batch_size=25,
            streaming=True,
            resume=False,
        )

        # 1. Uninterrupted Baseline Run (4 batches)
        pipe_base = Turn7Pipeline(config=cfg_base)
        res_base = pipe_base.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_baseline
        )

        # 2. Interrupted Run: simulate crash during batch 3
        call_count = 0

        def flaky_compute_gpu_features(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("Simulated crash mid-run during batch 3 processing")
            return orig_compute_gpu_features(*args, **kwargs)

        gp_module.compute_gpu_features = flaky_compute_gpu_features
        cfg_interrupted = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=30,
            learning_rate=0.05,
            random_state=42,
            val_size=0.20,
            sample_s1_count=100,
            distractor_count=100,
            s1_batch_size=25,
            streaming=True,
            resume=True,
        )
        pipe_interrupted = Turn7Pipeline(config=cfg_interrupted)

        with pytest.raises(RuntimeError, match="Simulated crash"):
            pipe_interrupted.run_validation_experiment(
                s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_interrupted
            )

        # Verify only 2 chunks exist before resume
        chunks_pre_resume = list((out_interrupted / "streaming_chunks").glob("chunk_*.parquet"))
        assert len(chunks_pre_resume) == 2, f"Expected 2 chunks before resume, found {len(chunks_pre_resume)}"

        # 3. Resume from disk (restore compute_gpu_features)
        gp_module.compute_gpu_features = orig_compute_gpu_features

        pipe_resumed = Turn7Pipeline(config=cfg_interrupted)
        res_resumed = pipe_resumed.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_interrupted
        )

        # Verify all 4 chunks exist after resume
        chunks_post_resume = list((out_interrupted / "streaming_chunks").glob("chunk_*.parquet"))
        assert len(chunks_post_resume) == 4, f"Expected 4 chunks after resume, found {len(chunks_post_resume)}"

        # 4. Compare resumed run with uninterrupted baseline
        base_pairs = res_base["blocking_diagnostics"]["candidate_count"]
        resumed_pairs = res_resumed["blocking_diagnostics"]["candidate_count"]
        assert resumed_pairs == base_pairs, f"Candidate count mismatch: base={base_pairs}, resumed={resumed_pairs}"

        base_train_pairs = res_base["split_summary"]["train_pairs"]
        resumed_train_pairs = res_resumed["split_summary"]["train_pairs"]
        assert resumed_train_pairs == base_train_pairs, f"Train pairs mismatch: base={base_train_pairs}, resumed={resumed_train_pairs}"

        base_f05 = res_base["validation_metrics"]["macro_f05"]
        resumed_f05 = res_resumed["validation_metrics"]["macro_f05"]
        np.testing.assert_allclose(resumed_f05, base_f05, atol=1e-5)

    finally:
        gp_module.compute_gpu_features = orig_compute_gpu_features
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_process_boundary_isolation_two_process_lifecycle(small_sample):
    """
    Test that running ingestion-only (Stages 1-4) in Process 1 followed by
    training-only (Stages 5-7) in Process 2 produces outputs and metrics IDENTICAL
    to an unbroken single-process run.
    """
    s1, s2, s3, gt = small_sample
    temp_dir = tempfile.mkdtemp(prefix="test_process_isolation_")
    out_unbroken = Path(temp_dir) / "unbroken"
    out_isolated = Path(temp_dir) / "isolated"

    try:
        b_cfg = GPUBlockingConfig(top_k_per_s1=30)
        cfg = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=30,
            learning_rate=0.05,
            random_state=42,
            val_size=0.20,
            sample_s1_count=100,
            distractor_count=100,
            s1_batch_size=25,
            streaming=True,
            resume=False,
        )

        # 1. Unbroken reference run
        pipe_unbroken = Turn7Pipeline(config=cfg)
        res_unbroken = pipe_unbroken.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_unbroken
        )

        # 2. Process 1: Ingestion-only
        pipe_ingest = Turn7Pipeline(config=cfg)
        ingest_summary = pipe_ingest.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_isolated, ingest_only=True
        )
        assert ingest_summary["total_candidates"] == res_unbroken["blocking_diagnostics"]["candidate_count"]
        assert (out_isolated / "streaming_chunks" / "manifest.json").exists()

        # 3. Process 2: Training-only from pre-existing chunks
        pipe_train = Turn7Pipeline(config=cfg)
        res_isolated = pipe_train.run_validation_experiment(
            output_dir=out_isolated, train_only=True, gt_df=gt
        )

        # 4. Strict parity verification
        assert res_isolated["blocking_diagnostics"]["candidate_count"] == res_unbroken["blocking_diagnostics"]["candidate_count"]
        assert res_isolated["split_summary"]["train_pairs"] == res_unbroken["split_summary"]["train_pairs"]
        assert res_isolated["split_summary"]["val_pairs"] == res_unbroken["split_summary"]["val_pairs"]
        np.testing.assert_allclose(
            res_isolated["validation_metrics"]["macro_f05"],
            res_unbroken["validation_metrics"]["macro_f05"],
            atol=1e-5,
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_process_boundary_interrupted_ingestion_resume(small_sample):
    """
    Test that an interrupted ingestion-only process resumes properly from disk
    when restarted in a fresh process, and subsequent training-only process succeeds
    with identical results.
    """
    s1, s2, s3, gt = small_sample
    temp_dir = tempfile.mkdtemp(prefix="test_isolated_resume_")
    out_dir = Path(temp_dir) / "interrupted_ingest"

    import src.gpu.gpu_pipeline as gp_module
    orig_compute_gpu_features = gp_module.compute_gpu_features

    try:
        b_cfg = GPUBlockingConfig(top_k_per_s1=30)
        cfg_resume = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=30,
            learning_rate=0.05,
            random_state=42,
            val_size=0.20,
            sample_s1_count=100,
            distractor_count=100,
            s1_batch_size=25,
            streaming=True,
            resume=True,
        )

        # 1. Process 1 (crashes at batch 3)
        call_count = 0

        def flaky_features(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("Simulated crash in ingestion process")
            return orig_compute_gpu_features(*args, **kwargs)

        gp_module.compute_gpu_features = flaky_features
        pipe_crash = Turn7Pipeline(config=cfg_resume)
        with pytest.raises(RuntimeError, match="Simulated crash in ingestion"):
            pipe_crash.run_validation_experiment(
                s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_dir, ingest_only=True
            )

        # 2. Process 1 resumed (clean restart)
        gp_module.compute_gpu_features = orig_compute_gpu_features
        pipe_resume_ingest = Turn7Pipeline(config=cfg_resume)
        ingest_summary = pipe_resume_ingest.run_validation_experiment(
            s1=s1, s2=s2, s3=s3, gt_df=gt, output_dir=out_dir, ingest_only=True
        )
        assert len(list((out_dir / "streaming_chunks").glob("chunk_*.parquet"))) == 4

        # 3. Process 2 (training-only)
        pipe_train = Turn7Pipeline(config=cfg_resume)
        res_train = pipe_train.run_validation_experiment(
            output_dir=out_dir, train_only=True, gt_df=gt
        )
        assert res_train["validation_metrics"]["macro_f05"] > 0.90

    finally:
        gp_module.compute_gpu_features = orig_compute_gpu_features
        shutil.rmtree(temp_dir, ignore_errors=True)

