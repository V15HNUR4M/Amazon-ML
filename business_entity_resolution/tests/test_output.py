"""
test_output.py — Scaffold interface tests for output.py.
"""

from __future__ import annotations

import pytest


def test_output_imports():
    import src.output  # noqa: F401


def test_write_matching_results_exists():
    from src.output import write_matching_results
    assert callable(write_matching_results)


def test_write_candidate_pairs_exists():
    from src.output import write_candidate_pairs
    assert callable(write_candidate_pairs)


def test_write_matching_results(tmp_path):
    from src.output import write_matching_results
    import pandas as pd

    preds = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),
    }
    out_file = tmp_path / "matching_results.tsv"
    write_matching_results(preds, out_file)
    assert out_file.exists()

    df = pd.read_csv(out_file, sep="\t", keep_default_na=False)
    assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert len(df) == 2
    row1 = df[df["source1_entity_id"] == "S1-1"].iloc[0]
    row2 = df[df["source1_entity_id"] == "S1-2"].iloc[0]
    assert row1["matched_entity_ids"] == "S2-10,S3-20"
    assert row2["matched_entity_ids"] == ""


def test_write_candidate_pairs(tmp_path):
    from src.output import write_candidate_pairs
    import pandas as pd

    cands = {
        "S1-1": ["S2-10", "S3-20", "S2-10"],  # dups should be deduplicated & sorted
        "S1-2": [],
    }
    out_file = tmp_path / "candidate_pairs.tsv"
    write_candidate_pairs(cands, out_file)
    assert out_file.exists()

    df = pd.read_csv(out_file, sep="\t", keep_default_na=False)
    assert list(df.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert len(df) == 2
    row1 = df[df["source1_entity_id"] == "S1-1"].iloc[0]
    row2 = df[df["source1_entity_id"] == "S1-2"].iloc[0]
    assert row1["candidate_entity_ids"] == "S2-10,S3-20"
    assert row2["candidate_entity_ids"] == ""
