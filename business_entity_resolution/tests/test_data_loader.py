"""
test_data_loader.py — Scaffold import and interface tests for data_loader.py.

These tests verify that:
  1. The module imports cleanly.
  2. Expected functions exist with the correct names.
  3. FileNotFoundError is raised on a missing path.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_data_loader_imports():
    """data_loader module must import without error."""
    import src.data_loader  # noqa: F401


def test_load_source1_exists():
    from src.data_loader import load_source1
    assert callable(load_source1)


def test_load_source2_exists():
    from src.data_loader import load_source2
    assert callable(load_source2)


def test_load_source3_exists():
    from src.data_loader import load_source3
    assert callable(load_source3)


def test_load_ground_truth_exists():
    from src.data_loader import load_ground_truth
    assert callable(load_ground_truth)


def test_load_all_train_exists():
    from src.data_loader import load_all_train
    assert callable(load_all_train)


def test_load_all_test_exists():
    from src.data_loader import load_all_test
    assert callable(load_all_test)


def test_missing_file_raises(tmp_path: Path):
    """Loading a non-existent file must raise FileNotFoundError."""
    from src.data_loader import load_source1
    with pytest.raises(FileNotFoundError):
        load_source1(tmp_path / "nonexistent.tsv")


def test_source_columns_constant():
    from src.data_loader import SOURCE_COLUMNS
    assert "entity_id" in SOURCE_COLUMNS
    assert "business_name" in SOURCE_COLUMNS
    assert "business_address" in SOURCE_COLUMNS
    assert "country" in SOURCE_COLUMNS


def test_ground_truth_columns_constant():
    from src.data_loader import GROUND_TRUTH_COLUMNS
    assert "source1_entity_id" in GROUND_TRUTH_COLUMNS
    assert "matched_entity_ids" in GROUND_TRUTH_COLUMNS


def test_expand_ground_truth():
    import pandas as pd
    from src.data_loader import expand_ground_truth

    gt_df = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2"],
            "matched_entity_ids": ["S2-10, S3-20", ""],
        }
    )
    expanded = expand_ground_truth(gt_df)
    assert len(expanded) == 3
    assert set(expanded["source1_entity_id"]) == {"S1-1", "S1-2"}
    assert set(expanded[expanded["source1_entity_id"] == "S1-1"]["matched_id"]) == {"S2-10", "S3-20"}
    assert pd.isna(expanded[expanded["source1_entity_id"] == "S1-2"]["matched_id"].iloc[0])


def test_read_tsv_chunks_exists():
    from src.data_loader import read_tsv_chunks
    assert callable(read_tsv_chunks)
