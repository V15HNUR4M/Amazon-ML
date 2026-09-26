"""
test_final_training.py — Unit and integration tests for Turn 6 final training pipeline.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_final_model import run_final_training
from src.blocking import BlockingConfig, build_candidates
from src.config import Config
from src.features import DEFAULT_FEATURE_NAMES, compute_features
from src.model import LightGBMMatcher
from src.pair_builder import build_train_pairs
from src.preprocessing import preprocess_records


@pytest.fixture
def synthetic_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a minimal synthetic dataset with 6 S1 entities, matches, and distractors."""
    s1 = pd.DataFrame(
        [
            {"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"},
            {"entity_id": "S1-002", "business_name": "Global Tech LLC", "business_address": "456 Market Ave", "country": "US"},
            {"entity_id": "S1-003", "business_name": "Zenith Solutions", "business_address": "789 Broadway", "country": "GB"},
            {"entity_id": "S1-004", "business_name": "Nexus Industries", "business_address": "101 Cyber Way", "country": "CA"},
            {"entity_id": "S1-005", "business_name": "Solo Venture", "business_address": "55 Independent Blvd", "country": "US"},  # singleton
            {"entity_id": "S1-006", "business_name": "Alpha Omega Systems", "business_address": "88 Frontier Rd", "country": "AU"},
        ]
    )

    s2 = pd.DataFrame(
        [
            {"entity_id": "S2-001", "business_name": "Acme Corporation", "business_address": "123 Main Street", "country": "US"},
            {"entity_id": "S2-002", "business_name": "Global Tech Limited", "business_address": "456 Market Ave", "country": "US"},
            {"entity_id": "S2-003", "business_name": "Zenith Solutions UK", "business_address": "789 Broadway Rd", "country": "GB"},
            {"entity_id": "S2-999", "business_name": "Acme Hardware Store", "business_address": "999 Far Away", "country": "US"},  # hard negative for S1-001
        ]
    )

    s3 = pd.DataFrame(
        [
            {"entity_id": "S3-001", "business_name": "Acme Inc", "business_address": "123 Main St", "country": "US"},
            {"entity_id": "S3-004", "business_name": "Nexus Ind", "business_address": "101 Cyber Way", "country": "CA"},
            {"entity_id": "S3-006", "business_name": "Alpha Omega Sys", "business_address": "88 Frontier Road", "country": "AU"},
            {"entity_id": "S3-888", "business_name": "Nexus Logistics LLC", "business_address": "123 Nowhere", "country": "CA"},  # hard negative for S1-004
        ]
    )

    gt = pd.DataFrame(
        [
            {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-001,S3-001"},
            {"source1_entity_id": "S1-002", "matched_entity_ids": "S2-002"},
            {"source1_entity_id": "S1-003", "matched_entity_ids": "S2-003"},
            {"source1_entity_id": "S1-004", "matched_entity_ids": "S3-004"},
            {"source1_entity_id": "S1-005", "matched_entity_ids": ""},  # singleton (0 matches)
            {"source1_entity_id": "S1-006", "matched_entity_ids": "S3-006"},
        ]
    )

    return s1, s2, s3, gt


def test_final_training_on_synthetic_data(synthetic_training_data, tmp_path: Path):
    """Verify that the final training pipeline fits on 100% of labeled pairs and produces valid artifacts."""
    s1, s2, s3, gt = synthetic_training_data

    # Preprocess
    s1_proc = preprocess_records(s1)
    s2_proc = preprocess_records(s2)
    s3_proc = preprocess_records(s3)

    # Candidates via blocking
    b_cfg = BlockingConfig()
    candidates = build_candidates(s1_proc, s2_proc, s3_proc, config=b_cfg)
    assert len(candidates) > 0

    # Build pairs
    pairs = build_train_pairs(s1_proc, s2_proc, s3_proc, candidates, gt)
    assert len(pairs) > 0
    assert (pairs["label"] == 1).sum() > 0
    assert (pairs["label"] == 0).sum() > 0

    # Compute features
    feats = compute_features(pairs)
    assert feats.shape[1] == 21
    for col in DEFAULT_FEATURE_NAMES:
        pairs[col] = feats[col].values

    # Train LightGBM matcher with Turn 5.5 validated configuration
    matcher = LightGBMMatcher(n_estimators=100, learning_rate=0.05, random_state=42)
    matcher.fit(pairs, pairs["label"])
    assert matcher.is_fitted

    # Save to custom paths in tmp_path
    model_path = tmp_path / "turn6_final_matcher.joblib"
    config_path = tmp_path / "turn6_final_model_config.json"
    matcher.save(model_path)

    metadata = {
        "model_path": str(model_path.resolve()),
        "model_type": "LightGBM",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "random_state": 42,
        "threshold": 0.90,
        "number_of_s1_training_entities": len(s1),
        "number_of_labelled_pairs": len(pairs),
        "positive_pair_count": int((pairs["label"] == 1).sum()),
        "negative_pair_count": int((pairs["label"] == 0).sum()),
        "feature_count": 21,
        "feature_names": list(DEFAULT_FEATURE_NAMES),
        "training_runtime_s": 0.123,
        "peak_rss_mb": 150.0,
        "is_full_training": True,
        "validation_split": "none (100% of labeled training data used)",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Assertions on files
    assert model_path.exists()
    assert config_path.exists()

    # Verify model artifact loading and inference
    loaded = LightGBMMatcher.load(model_path)
    assert isinstance(loaded, LightGBMMatcher)
    assert loaded.is_fitted

    proba = loaded.predict_proba(pairs)
    assert len(proba) == len(pairs)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()

    # Verify config metadata content
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_loaded = json.load(f)

    assert cfg_loaded["model_type"] == "LightGBM"
    assert cfg_loaded["n_estimators"] == 100
    assert cfg_loaded["learning_rate"] == 0.05
    assert cfg_loaded["random_state"] == 42
    assert cfg_loaded["threshold"] == 0.90
    assert cfg_loaded["feature_count"] == 21
    assert len(cfg_loaded["feature_names"]) == 21
    assert cfg_loaded["number_of_s1_training_entities"] == 6
    assert cfg_loaded["validation_split"] == "none (100% of labeled training data used)"


def test_final_training_safety_prevents_overwriting_turn5_5(tmp_path: Path):
    """Verify that attempting to overwrite Turn 5.5 artifacts raises a ValueError."""
    fake_cache = tmp_path / "cache"
    fake_cache.mkdir(parents=True)

    turn5_5_model = fake_cache / "turn5_5_best_matcher.joblib"
    turn5_5_thresh = fake_cache / "turn5_5_optimal_threshold.json"

    turn5_5_model.write_text("turn5.5 model data", encoding="utf-8")
    turn5_5_thresh.write_text('{"threshold": 0.90}', encoding="utf-8")

    # Trying to overwrite Turn 5.5 model must raise
    with pytest.raises(ValueError, match="Safety error: output_model_path points to Turn 5.5 model"):
        run_final_training(
            cache_dir=fake_cache,
            output_model_path=turn5_5_model,
        )

    # Trying to overwrite Turn 5.5 threshold config must raise
    with pytest.raises(ValueError, match="Safety error: output_config_path points to Turn 5.5"):
        run_final_training(
            cache_dir=fake_cache,
            output_config_path=turn5_5_thresh,
        )

    # Verify Turn 5.5 files remain unmodified
    assert turn5_5_model.read_text(encoding="utf-8") == "turn5.5 model data"
    assert turn5_5_thresh.read_text(encoding="utf-8") == '{"threshold": 0.90}'


def test_final_training_cli_help():
    """Verify scripts/train_final_model.py CLI prints help without error."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_final_model.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--max-s1" in result.stdout
    assert "--output-model" in result.stdout
    assert "--output-config" in result.stdout
    assert "--chunked" in result.stdout
    assert "--no-chunked" in result.stdout


def test_chunked_streaming_pipeline_synthetic(synthetic_training_data, tmp_path: Path):
    """Regression test proving the chunked/disk-backed path works end-to-end without loading all data simultaneously."""
    s1, s2, s3, gt = synthetic_training_data
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    out_model = cache_dir / "test_chunked_matcher.joblib"
    out_cfg = cache_dir / "test_chunked_config.json"

    # Run chunked training with micro-batch size of 2 S1 entities per batch
    metadata = run_final_training(
        chunked=True,
        batch_size=2,
        cache_dir=cache_dir,
        output_model_path=out_model,
        output_config_path=out_cfg,
        s1_data=s1,
        s2_data=s2,
        s3_data=s3,
        gt_data=gt,
        force_rebuild=True,
    )

    # 1. Model artifact exists, is valid, and can predict probabilities
    assert out_model.exists()
    matcher = LightGBMMatcher.load(out_model)
    assert isinstance(matcher, LightGBMMatcher)
    assert matcher.is_fitted

    # 2. Config artifact exists and verifies requirement 13 specifications
    assert out_cfg.exists()
    with open(out_cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["model_type"] == "LightGBM"
    assert cfg["is_full_training"] is True
    assert cfg["number_of_s1_training_entities"] == 6
    assert cfg["number_of_labelled_pairs"] > 0
    assert cfg["labelled_pair_count"] == cfg["number_of_labelled_pairs"]
    assert cfg["feature_count"] == 21
    assert len(cfg["feature_names"]) == 21
    assert cfg["threshold"] == 0.90
    assert cfg["chunked_disk_backed_training"] is True
    assert cfg["whether_chunked_or_disk_backed_training_was_used"] is True
    assert "training_hyperparameters" in cfg
    assert cfg["training_hyperparameters"]["n_estimators"] == 100
    assert cfg["training_hyperparameters"]["learning_rate"] == 0.05
    assert cfg["training_hyperparameters"]["class_weight"] == "balanced"
    assert "runtime" in cfg
    assert "peak_rss_mb" in cfg
    assert "peak_rss" in cfg
    assert cfg["peak_rss_mb"] > 0

    # 3. Disk-backed binary files were created and stored
    bin_dir = cache_dir / "turn6_train_binary"
    assert bin_dir.exists()
    x_bins = list(bin_dir.glob("features_*.bin"))
    y_bins = list(bin_dir.glob("labels_*.bin"))
    assert len(x_bins) >= 1
    assert len(y_bins) >= 1
    assert x_bins[0].stat().st_size > 0
    assert y_bins[0].stat().st_size > 0

    # 4. SQLite blocking DB exists and has records
    db_file = cache_dir / "turn6_train_blocking.db"
    assert db_file.exists()
    assert db_file.stat().st_size > 0


def test_chunked_streaming_with_max_s1_cap(synthetic_training_data, tmp_path: Path):
    """Test chunked streaming pipeline with max_s1 capping (e.g. 3 entities)."""
    s1, s2, s3, gt = synthetic_training_data
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    metadata = run_final_training(
        max_s1=3,
        chunked=True,
        batch_size=2,
        cache_dir=cache_dir,
        output_model_path=cache_dir / "matcher_cap.joblib",
        output_config_path=cache_dir / "config_cap.json",
        s1_data=s1,
        s2_data=s2,
        s3_data=s3,
        gt_data=gt,
        force_rebuild=True,
    )

    assert metadata["number_of_s1_training_entities"] == 3
    assert metadata["is_full_training"] is False
    assert metadata["chunked_disk_backed_training"] is True


def test_chunked_streaming_with_sample_neg_ratio(synthetic_training_data, tmp_path: Path):
    """Test chunked streaming pipeline with negative-to-positive subsampling."""
    s1, s2, s3, gt = synthetic_training_data
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    metadata = run_final_training(
        chunked=True,
        batch_size=2,
        sample_neg_ratio=1.0,
        cache_dir=cache_dir,
        output_model_path=cache_dir / "matcher_sampled.joblib",
        output_config_path=cache_dir / "config_sampled.json",
        s1_data=s1,
        s2_data=s2,
        s3_data=s3,
        gt_data=gt,
        force_rebuild=True,
    )

    assert metadata["number_of_s1_training_entities"] == 6
    assert metadata["positive_pair_count"] > 0
    # With ratio 1.0, negatives are capped proportional to positives
    assert metadata["negative_pair_count"] <= metadata["positive_pair_count"] + 1


def test_chunked_disk_backed_reloads_existing_index(synthetic_training_data, tmp_path: Path):
    """Test that existing finalized SQLite candidate index is reused on subsequent runs."""
    s1, s2, s3, gt = synthetic_training_data
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    # First run builds the index
    meta1 = run_final_training(
        chunked=True,
        batch_size=3,
        cache_dir=cache_dir,
        output_model_path=cache_dir / "matcher1.joblib",
        output_config_path=cache_dir / "config1.json",
        s1_data=s1,
        s2_data=s2,
        s3_data=s3,
        gt_data=gt,
        force_rebuild=True,
    )

    # Second run without force_rebuild should load the existing index
    meta2 = run_final_training(
        chunked=True,
        batch_size=3,
        cache_dir=cache_dir,
        output_model_path=cache_dir / "matcher2.joblib",
        output_config_path=cache_dir / "config2.json",
        s1_data=s1,
        s2_data=s2,
        s3_data=s3,
        gt_data=gt,
        force_rebuild=False,
    )

    assert meta1["number_of_labelled_pairs"] == meta2["number_of_labelled_pairs"]
    assert meta1["positive_pair_count"] == meta2["positive_pair_count"]
    assert meta1["negative_pair_count"] == meta2["negative_pair_count"]


def test_disk_index_add_ground_truth_dict(tmp_path: Path):
    """Test that DiskBlockingIndex.add_ground_truth supports dict of sets without crashing."""
    from src.blocking import DiskBlockingIndex

    db_path = tmp_path / "test_gt.db"
    idx = DiskBlockingIndex(db_path=db_path)
    gt_dict = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
        "S1-3": set(),
    }
    idx.add_ground_truth(gt_dict)

    fetched = idx.get_ground_truth_for_s1(["S1-1", "S1-2", "S1-3", "S1-999"])
    assert fetched["S1-1"] == {"S2-10", "S3-20"}
    assert fetched["S1-2"] == {"S2-30"}
    assert fetched["S1-3"] == set()
    assert fetched["S1-999"] == set()
    idx.close()


def test_chunked_mode_defaults_to_true_when_chunked_is_none(synthetic_training_data, tmp_path: Path):
    """Test that chunked disk-backed pipeline is active by default even when chunked=None."""
    s1, s2, s3, gt = synthetic_training_data
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    meta = run_final_training(
        chunked=None,  # Not explicitly passed
        batch_size=2,
        cache_dir=cache_dir,
        output_model_path=cache_dir / "default_chunked.joblib",
        output_config_path=cache_dir / "default_chunked.json",
        s1_data=s1,
        s2_data=s2,
        s3_data=s3,
        gt_data=gt,
        force_rebuild=True,
    )

    assert meta["chunked_disk_backed_training"] is True
    assert meta["whether_chunked_or_disk_backed_training_was_used"] is True
    assert meta["whether chunked/disk-backed training was used"] is True
    assert meta["labelled pair count"] == meta["labelled_pair_count"]
    assert meta["peak RSS"] == meta["peak_rss"]


def test_sample_path_without_preloaded_data_uses_sample_db(synthetic_training_data, tmp_path: Path):
    """Test that calling run_final_training with max_s1 and without preloaded data uses dedicated sample DB."""
    import pickle
    s1, s2, s3, gt = synthetic_training_data
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    # Pre-populate sample cache in tmp_path cache so test runs instantly without scanning 10M-row raw TSVs
    sample_file = cache_dir / "benchmark_sample_s1_6_dist_6.pkl"
    with open(sample_file, "wb") as f:
        pickle.dump({"s1": s1, "s2": s2, "s3": s3, "gt": gt}, f)

    meta = run_final_training(
        max_s1=6,
        distractors=6,
        chunked=True,
        batch_size=3,
        cache_dir=cache_dir,
        output_model_path=cache_dir / "sample_model.joblib",
        output_config_path=cache_dir / "sample_config.json",
        force_rebuild=False,
    )

    assert meta["number_of_s1_training_entities"] == 6
    assert meta["chunked_disk_backed_training"] is True
    assert meta["is_full_training"] is False

    # Sample DB should exist and NOT be named turn6_train_blocking.db
    sample_db = cache_dir / "turn6_train_blocking_s1_6.db"
    assert sample_db.exists()
    assert not (cache_dir / "turn6_train_blocking.db").exists()



