"""
Unit tests for Turn 7 GPU Pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.gpu.gpu_blocking import GPUBlockingConfig
from src.gpu.gpu_pipeline import Turn7Config, Turn7Pipeline
from src.gpu.gpu_utils import get_device_info


def test_device_info() -> None:
    info = get_device_info()
    assert "cuda_available" in info
    assert "device_name" in info
    assert "process_rss_mb" in info


def test_turn7_pipeline_synthetic_execution() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create enough synthetic data with both positive and negative pairs
        s1_rows = []
        s2_rows = []
        s3_rows = []
        gt_rows = []

        for i in range(1, 15):
            s1_rows.append({
                "entity_id": f"S1-{i}",
                "business_name": f"Enterprise Company {i} Corp",
                "business_address": f"{100 + i} Main Street",
                "country": "US",
            })
            s2_rows.append({
                "entity_id": f"S2-{i}",
                "business_name": f"Enterprise Company {i} Inc",
                "business_address": f"{100 + i} Main St",
                "country": "US",
            })
            s3_rows.append({
                "entity_id": f"S3-{i}",
                "business_name": f"enterprise{i}.com",
                "business_address": f"{100 + i} Main",
                "country": "US",
            })
            gt_rows.append({
                "source1_entity_id": f"S1-{i}",
                "matched_entity_ids": f"S2-{i},S3-{i}" if i % 4 != 0 else "",
            })

        # Add some distractors
        for i in range(20, 25):
            s2_rows.append({
                "entity_id": f"S2-{i}",
                "business_name": f"Enterprise Different {i} LLC",
                "business_address": f"{500 + i} Elm St",
                "country": "US",
            })

        s1 = pd.DataFrame(s1_rows)
        s2 = pd.DataFrame(s2_rows)
        s3 = pd.DataFrame(s3_rows)
        gt = pd.DataFrame(gt_rows)

        cfg = Turn7Config(
            blocking_config=GPUBlockingConfig(top_k_per_s1=10),
            n_estimators=10,
            val_size=0.25,
            thresholds_to_evaluate=[0.5, 0.7, 0.9],
        )

        pipeline = Turn7Pipeline(config=cfg)
        report = pipeline.run_validation_experiment(
            s1=s1,
            s2=s2,
            s3=s3,
            gt_df=gt,
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "cache",
        )

        assert "blocking_diagnostics" in report
        assert "validation_metrics" in report
        assert "model_config" in report
        assert (tmp_path / "output" / "turn7_validation_report.json").exists()
        assert (tmp_path / "cache" / "turn7_best_matcher.joblib").exists()
