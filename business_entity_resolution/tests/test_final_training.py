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
    result = subprocess.run(
        [sys.executable, "scripts/train_final_model.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--max-s1" in result.stdout
    assert "--output-model" in result.stdout
    assert "--output-config" in result.stdout
