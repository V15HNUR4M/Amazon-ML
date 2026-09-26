"""
data_loader.py — Load hackathon TSV source files into DataFrames.

All files are tab-separated (TSV). The canonical read call is:
    pd.read_csv(path, sep="\\t")

Expected columns per file
-------------------------
Source 1 / 2 / 3 (train and test):
    entity_id, business_name, business_address, country

Ground truth (train only):
    source1_entity_id, matched_entity_ids

Usage
-----
from src.data_loader import load_source1, load_source2, load_source3, load_ground_truth
from src.config import Config

cfg = Config()
s1 = load_source1(cfg.TRAIN_SOURCE1)
gt = load_ground_truth(cfg.TRAIN_GROUND_TRUTH)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected schema constants (used for validation / documentation)
# ---------------------------------------------------------------------------

SOURCE_COLUMNS: list[str] = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS: list[str] = ["source1_entity_id", "matched_entity_ids"]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _read_tsv(path: Path | str, expected_columns: list[str] | None = None) -> pd.DataFrame:
    """Read a TSV file and optionally warn about unexpected columns.

    Parameters
    ----------
    path:
        Absolute or relative path to the TSV file.
    expected_columns:
        If provided, a warning is emitted when the file's columns differ.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame, no transformation applied.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    logger.info("Loaded %s — %d rows, %d columns", path.name, len(df), len(df.columns))

    if expected_columns is not None:
        missing = set(expected_columns) - set(df.columns)
        extra = set(df.columns) - set(expected_columns)
        if missing:
            logger.warning("Columns missing from %s: %s", path.name, missing)
        if extra:
            logger.warning("Unexpected extra columns in %s: %s", path.name, extra)

    return df


def read_tsv_chunks(
    path: Path | str,
    chunksize: int = 50000,
    expected_columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Read a TSV file in chunks as an iterator to conserve RAM.

    Parameters
    ----------
    path:
        Path to the TSV file.
    chunksize:
        Number of rows per chunk (default: 50,000).
    expected_columns:
        Optional list of expected column names for validation.

    Yields
    ------
    pd.DataFrame
        DataFrame chunk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False,
    )
    first = True
    for chunk in reader:
        if first and expected_columns is not None:
            missing = set(expected_columns) - set(chunk.columns)
            extra = set(chunk.columns) - set(expected_columns)
            if missing:
                logger.warning("Columns missing from %s: %s", path.name, missing)
            if extra:
                logger.warning("Unexpected extra columns in %s: %s", path.name, extra)
            first = False
        yield chunk


# ---------------------------------------------------------------------------
# Public loading functions
# ---------------------------------------------------------------------------

def load_source1(path: Path | str) -> pd.DataFrame:
    """Load Source 1 records (train or test).

    Parameters
    ----------
    path:
        Path to train_source1.tsv or test_source1.tsv.

    Returns
    -------
    pd.DataFrame with columns: entity_id, business_name, business_address, country
    """
    return _read_tsv(path, expected_columns=SOURCE_COLUMNS)


def load_source2(path: Path | str) -> pd.DataFrame:
    """Load Source 2 records (train or test).

    Parameters
    ----------
    path:
        Path to train_source2.tsv or test_source2.tsv.

    Returns
    -------
    pd.DataFrame with columns: entity_id, business_name, business_address, country
    """
    return _read_tsv(path, expected_columns=SOURCE_COLUMNS)


def load_source3(path: Path | str) -> pd.DataFrame:
    """Load Source 3 records (train or test).

    Parameters
    ----------
    path:
        Path to train_source3.tsv or test_source3.tsv.

    Returns
    -------
    pd.DataFrame with columns: entity_id, business_name, business_address, country
    """
    return _read_tsv(path, expected_columns=SOURCE_COLUMNS)


def load_ground_truth(path: Path | str) -> pd.DataFrame:
    """Load the ground-truth match file (train only).

    The ``matched_entity_ids`` column contains comma-separated IDs from
    Source 2 and/or Source 3.  An empty string means the Source 1 entity
    is a singleton (no match).

    Parameters
    ----------
    path:
        Path to train_ground_truth.tsv.

    Returns
    -------
    pd.DataFrame with columns: source1_entity_id, matched_entity_ids
    """
    return _read_tsv(path, expected_columns=GROUND_TRUTH_COLUMNS)


def expand_ground_truth(gt: pd.DataFrame) -> pd.DataFrame:
    """Expand ground-truth into one row per (source1_entity_id, matched_id) pair.

    Singleton rows (empty matched_entity_ids) produce a row with
    matched_id = NaN.

    Parameters
    ----------
    gt:
        Raw ground-truth DataFrame from :func:`load_ground_truth`.

    Returns
    -------
    pd.DataFrame with columns: source1_entity_id, matched_id
        matched_id is NaN for singletons.

    Notes
    -----
    Not called during scaffolding.  Will be used when building training pairs.
    """
    if gt is None:
        raise ValueError("Ground-truth DataFrame cannot be None")

    rows: list[dict[str, str | float]] = []
    for _, row in gt.iterrows():
        s1_id = str(row["source1_entity_id"])
        matched_raw = str(row.get("matched_entity_ids", "") or "")
        matches = [
            m.strip()
            for m in matched_raw.split(",")
            if m.strip() and m.strip().lower() not in ("nan", "none", "null")
        ]
        if not matches:
            rows.append({"source1_entity_id": s1_id, "matched_id": float("nan")})
        else:
            for m in matches:
                rows.append({"source1_entity_id": s1_id, "matched_id": m})

    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_id"])


def load_all_train(config) -> dict[str, pd.DataFrame]:
    """Convenience loader: return all train DataFrames in a dict.

    Parameters
    ----------
    config:
        A :class:`src.config.Config` instance.

    Returns
    -------
    dict with keys: 'source1', 'source2', 'source3', 'ground_truth'
    """
    return {
        "source1": load_source1(config.TRAIN_SOURCE1),
        "source2": load_source2(config.TRAIN_SOURCE2),
        "source3": load_source3(config.TRAIN_SOURCE3),
        "ground_truth": load_ground_truth(config.TRAIN_GROUND_TRUTH),
    }


def load_all_test(config) -> dict[str, pd.DataFrame]:
    """Convenience loader: return all test DataFrames in a dict.

    Parameters
    ----------
    config:
        A :class:`src.config.Config` instance.

    Returns
    -------
    dict with keys: 'source1', 'source2', 'source3'
    """
    return {
        "source1": load_source1(config.TEST_SOURCE1),
        "source2": load_source2(config.TEST_SOURCE2),
        "source3": load_source3(config.TEST_SOURCE3),
    }
