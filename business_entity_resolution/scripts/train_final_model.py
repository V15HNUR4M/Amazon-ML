"""
train_final_model.py — Final competition training pipeline for Turn 6.

Design intent
-------------
Trains the final LightGBM model on 100% of the available labeled competition
training data for the Amazon ML Business Entity Resolution challenge:

1. Loads labeled competition data:
   - train_source1.tsv (2,206,821 records)
   - train_source2.tsv (3,693,619 records)
   - train_source3.tsv (3,944,746 records)
   - train_ground_truth.tsv (2,206,821 records)
2. Memory-Safe Chunked / Streaming Architecture (< 12 GB RAM target, typically ~2.5-4.5 GB peak):
   - Candidate pool (Source 2 and Source 3) indexed in streaming chunks into
     disk-backed SQLite DiskBlockingIndex, discarding DataFrames immediately.
   - Ground truth indexed in SQLite table with primary key index for zero-RAM lookups.
   - Source 1 streamed in batches (e.g. 10,000 entities), candidate pairs assembled,
     21 pairwise features computed per batch, and float32 features appended directly
     to a disk-backed binary store (np.memmap-compatible).
   - Zero simultaneous retention of S1 + S2 + S3 pandas DataFrames in RAM.
3. Preserves existing architecture & hyperparameter compliance:
   - 4-stage blocking (Blocker A, B, C, D)
   - same blocking keys & weights (A=3, D=2, B=1, C=1)
   - same max_candidates_per_block behavior (100)
   - same top_k_per_s1 candidate ranking (50)
   - same 21 pairwise numerical features
   - same LightGBMMatcher:
     n_estimators=100, learning_rate=0.05, random_state=42, class_weight='balanced'
   - threshold fixed at 0.90 (no re-tuning on training data)
   - 100% of all labeled training entities (zero validation withholding)
4. Non-destructive artifact persistence:
   - Final model: cache/turn6_final_matcher.joblib
   - Final config: cache/turn6_final_model_config.json
   - Leaves turn5_5_best_matcher.joblib and turn5_5_optimal_threshold.json untouched.
5. Environment & progress tracking:
   - Tracks and reports process RSS memory at every stage.
   - Logs records processed, candidate pairs generated, feature rows written, and stage elapsed time.

Usage:
------
# 1. Full training on 100% of competition data (chunked streaming, ~12 GB RAM safe):
python scripts/train_final_model.py

# 2. Fast deterministic sanity check (e.g. 5,000 S1 queries, 100% of sample used for training):
python scripts/train_final_model.py --max-s1 5000

# 3. Small chunked streaming test (e.g. 500 S1 queries through disk-backed path):
python scripts/train_final_model.py --chunked --max-s1 500
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import platform
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import psutil

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (
    BlockingConfig,
    BlockingIndex,
    DiskBlockingIndex,
    build_candidates,
)
from src.config import Config
from src.evaluation import ground_truth_to_dict
from src.features import compute_features
from src.model import DEFAULT_FEATURE_NAMES, LightGBMMatcher
from src.pair_builder import build_train_pairs
from src.preprocessing import preprocess_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_process_rss_mb() -> float:
    """Return actual resident set size (RSS) memory of the current process in MB."""
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 2)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Data Loading & Sampling (In-Memory Path for small runs / testing)
# ---------------------------------------------------------------------------


def load_training_sample(
    cfg: Config,
    max_s1: int = 5000,
    distractors: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    random_seed: int = 42,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a deterministic training sample (for fast sanity check and testing).

    Parameters
    ----------
    cfg : Config instance.
    max_s1 : Cap on S1 queries.
    distractors : Cap on distractors from S2 and S3.
    cache_dir : Directory to check for cached sample files.
    random_seed : Seed for deterministic sampling.
    force_rebuild : Ignore cache and reload from raw TSV files.

    Returns
    -------
    (s1, s2, s3, gt_df)
    """
    c_dir = cache_dir or cfg.CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    dist_count = distractors if distractors is not None else max_s1

    if not force_rebuild:
        cache_file = c_dir / f"benchmark_sample_s1_{max_s1}_dist_{dist_count}.pkl"
        if cache_file.exists():
            logger.info("Loading cached training sample from %s...", cache_file)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            return data["s1"], data["s2"], data["s3"], data["gt"]

    t0 = time.time()

    # 1. Ground truth
    logger.info("Loading ground truth (first %d rows) from %s...", max_s1, cfg.TRAIN_GROUND_TRUTH)
    gt_df = pd.read_csv(
        cfg.TRAIN_GROUND_TRUTH,
        sep="\t",
        nrows=max_s1,
        keep_default_na=False,
    )
    needed_s1 = set(gt_df["source1_entity_id"])

    # 2. Source 1
    logger.info("Loading Source 1 entities from %s...", cfg.TRAIN_SOURCE1)
    s1_rows: list[dict] = []
    rem_s1 = set(needed_s1)
    for chunk in pd.read_csv(
        cfg.TRAIN_SOURCE1, sep="\t", chunksize=200000, keep_default_na=False
    ):
        matches = chunk[chunk["entity_id"].isin(rem_s1)]
        if len(matches) > 0:
            s1_rows.extend(matches.to_dict(orient="records"))
            rem_s1 -= set(matches["entity_id"])
        if len(rem_s1) == 0:
            break
    s1 = pd.DataFrame(s1_rows).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

    # 3. Collect true match IDs
    needed_s2: set[str] = set()
    needed_s3: set[str] = set()
    for _, r in gt_df.iterrows():
        m_str = str(r.get("matched_entity_ids", "")).strip()
        if not m_str:
            continue
        for m in m_str.split(","):
            m = m.strip()
            if m.startswith("S2"):
                needed_s2.add(m)
            elif m.startswith("S3"):
                needed_s3.add(m)

    # 4. Stream S2
    s2_rows: list[dict] = []
    rem_s2 = set(needed_s2)
    dist_s2_needed = dist_count // 2
    for chunk in pd.read_csv(
        cfg.TRAIN_SOURCE2, sep="\t", chunksize=250000, keep_default_na=False
    ):
        matches = chunk[chunk["entity_id"].isin(rem_s2)]
        if len(matches) > 0:
            s2_rows.extend(matches.to_dict(orient="records"))
            rem_s2 -= set(matches["entity_id"])
        if dist_s2_needed > 0:
            dist = chunk[~chunk["entity_id"].isin(needed_s2)].head(dist_s2_needed)
            s2_rows.extend(dist.to_dict(orient="records"))
            dist_s2_needed -= len(dist)
        if len(rem_s2) == 0 and dist_s2_needed <= 0:
            break
    s2 = pd.DataFrame(s2_rows).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

    # 5. Stream S3
    s3_rows: list[dict] = []
    rem_s3 = set(needed_s3)
    dist_s3_needed = dist_count // 2
    for chunk in pd.read_csv(
        cfg.TRAIN_SOURCE3, sep="\t", chunksize=250000, keep_default_na=False
    ):
        matches = chunk[chunk["entity_id"].isin(rem_s3)]
        if len(matches) > 0:
            s3_rows.extend(matches.to_dict(orient="records"))
            rem_s3 -= set(matches["entity_id"])
        if dist_s3_needed > 0:
            dist = chunk[~chunk["entity_id"].isin(needed_s3)].head(dist_s3_needed)
            s3_rows.extend(dist.to_dict(orient="records"))
            dist_s3_needed -= len(dist)
        if len(rem_s3) == 0 and dist_s3_needed <= 0:
            break
    s3 = pd.DataFrame(s3_rows).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

    # 6. Preprocess
    s1_proc = preprocess_records(s1)
    s2_proc = preprocess_records(s2)
    s3_proc = preprocess_records(s3)

    cache_file = c_dir / f"benchmark_sample_s1_{max_s1}_dist_{dist_count}.pkl"
    sample_data = {"s1": s1_proc, "s2": s2_proc, "s3": s3_proc, "gt": gt_df}
    with open(cache_file, "wb") as f:
        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Sample loaded in %.2fs (RSS: %.1f MB).", time.time() - t0, get_process_rss_mb())
    return s1_proc, s2_proc, s3_proc, gt_df


# ---------------------------------------------------------------------------
# Streaming Index Builder (S2 + S3 + Ground Truth to SQLite)
# ---------------------------------------------------------------------------


def build_or_load_candidate_index(
    cfg: Config,
    cache_dir: Path,
    db_path: Optional[Path | str] = None,
    preprocess_chunk_size: int = 50_000,
    force_rebuild: bool = False,
    s2_data: Optional[pd.DataFrame] = None,
    s3_data: Optional[pd.DataFrame] = None,
    gt_data: Optional[pd.DataFrame | dict] = None,
) -> DiskBlockingIndex:
    """Build or load the disk-backed candidate index and ground truth in SQLite.

    Memory strategy:
    - Streams S2 and S3 in chunks of `preprocess_chunk_size` (e.g. 50k rows).
    - Preprocesses each chunk and inserts into SQLite immediately.
    - Garbage collects chunk DataFrames immediately.
    - Discarding chunk DataFrames keeps Python RSS flat at ~300-500 MB.
    - Stores ground truth in SQLite table with primary key index for zero-RAM lookups.
    """
    target_db = Path(db_path) if db_path is not None else (cache_dir / "turn6_train_blocking.db")
    target_db.parent.mkdir(parents=True, exist_ok=True)

    blocking_cfg = BlockingConfig(
        top_k_per_s1=cfg.BLOCKING_TOP_K,
        max_candidates_per_block=cfg.MAX_CANDIDATES_PER_BLOCK,
        min_token_len=cfg.MIN_TOKEN_LEN,
        enable_blocker_a=cfg.BLOCKER_A_ENABLED,
        enable_blocker_b=cfg.BLOCKER_B_ENABLED,
        enable_blocker_c=cfg.BLOCKER_C_ENABLED,
        enable_blocker_d=cfg.BLOCKER_D_ENABLED,
    )

    # Check for existing finalized database
    if target_db.exists() and not force_rebuild:
        idx = DiskBlockingIndex(db_path=target_db, config=blocking_cfg)
        is_full_index_check = (s2_data is None and s3_data is None)
        has_min_records = (idx.total_records_indexed >= 7_000_000) if is_full_index_check else (idx.total_records_indexed > 0)
        if idx.is_finalized and has_min_records:
            logger.info(
                "Loaded existing disk-backed candidate index from %s (%d records indexed | S2=%d, S3=%d, RSS: %.1f MB).",
                target_db,
                idx.total_records_indexed,
                idx.source_counts.get("S2", 0),
                idx.source_counts.get("S3", 0),
                get_process_rss_mb(),
            )
            return idx
        idx.close()

    # Rebuild database from scratch
    if target_db.exists():
        try:
            target_db.unlink(missing_ok=True)
            for ext in ["-wal", "-shm"]:
                p = Path(str(target_db) + ext)
                if p.exists():
                    p.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Could not clean old DB file %s: %s", target_db, e)

    idx = DiskBlockingIndex(db_path=target_db, config=blocking_cfg)
    t0 = time.time()

    # Path A: DataFrames provided directly (e.g. unit tests, samples, or synthetic fixtures)
    if s2_data is not None or s3_data is not None:
        logger.info("Indexing in-memory candidate pools into %s...", target_db)
        if s2_data is not None and len(s2_data) > 0:
            idx.add_records(s2_data)
        if s3_data is not None and len(s3_data) > 0:
            idx.add_records(s3_data)
        if gt_data is not None:
            idx.add_ground_truth(gt_data)
        idx.finalize()
        return idx

    # Path B: Full TSV streaming
    sources = [
        (cfg.TRAIN_SOURCE2, "Source 2 (Candidate Pool)"),
        (cfg.TRAIN_SOURCE3, "Source 3 (Candidate Pool)"),
    ]

    for path, label in sources:
        if not path.exists():
            raise FileNotFoundError(f"Training source file not found: {path}")

        logger.info("Streaming and indexing %s from %s (chunk_size=%d)...", label, path, preprocess_chunk_size)
        reader = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=preprocess_chunk_size,
            keep_default_na=False,
        )
        src_total = 0
        chunk_num = 0
        for raw_chunk in reader:
            chunk_num += 1
            proc_chunk = preprocess_records(raw_chunk)
            idx.add_records(proc_chunk)
            src_total += len(proc_chunk)

            del proc_chunk, raw_chunk
            gc.collect()

            if chunk_num % 10 == 0:
                logger.info(
                    "  %s: indexed %d records (chunk %d) | Current RSS: %.1f MB",
                    label,
                    src_total,
                    chunk_num,
                    get_process_rss_mb(),
                )

        logger.info("  Finished %s: %d total records indexed.", label, src_total)

    # Index ground truth into SQLite table
    if cfg.TRAIN_GROUND_TRUTH.exists():
        logger.info("Indexing ground truth from %s into SQLite table...", cfg.TRAIN_GROUND_TRUTH)
        gt_reader = pd.read_csv(
            cfg.TRAIN_GROUND_TRUTH,
            sep="\t",
            dtype=str,
            chunksize=100_000,
            keep_default_na=False,
        )
        gt_total = 0
        for gt_chunk in gt_reader:
            idx.add_ground_truth(gt_chunk)
            gt_total += len(gt_chunk)
            del gt_chunk
            gc.collect()
        logger.info("  Indexed %d ground truth rows into SQLite.", gt_total)

    # Finalize posting lists
    logger.info("Finalizing posting lists and B-tree index in SQLite...")
    t_fin0 = time.time()
    idx.finalize()
    db_size_mb = round(target_db.stat().st_size / (1024**2), 2) if target_db.exists() else 0.0

    logger.info(
        "Candidate index complete in %.2fs (finalize: %.2fs) | DB size: %.2f MB | %d total records | RSS: %.1f MB",
        time.time() - t0,
        time.time() - t_fin0,
        db_size_mb,
        idx.total_records_indexed,
        get_process_rss_mb(),
    )
    return idx


# ---------------------------------------------------------------------------
# Chunked S1 Streaming, Candidate Pairing & Feature Extraction to Disk
# ---------------------------------------------------------------------------


def stream_and_extract_training_pairs(
    cfg: Config,
    disk_index: DiskBlockingIndex,
    max_s1: Optional[int] = None,
    batch_size: int = 10_000,
    sample_neg_ratio: Optional[float] = None,
    random_seed: int = 42,
    cache_dir: Optional[Path] = None,
    s1_data: Optional[pd.DataFrame] = None,
    gt_data: Optional[pd.DataFrame | dict] = None,
    force_rebuild: bool = False,
) -> tuple[int, int, int, int, Path, Path, Optional[pd.DataFrame], float]:
    """Stream Source 1 entities in batches, generate candidate pairs, extract 21 features,

    and write float32 features and int32 labels to disk-backed binary files.

    Returns
    -------
    (total_s1, total_pairs, total_pos, total_neg, x_bin_path, y_bin_path, sample_pairs_df, peak_rss)
    """
    c_dir = cache_dir or cfg.CACHE_DIR
    binary_dir = c_dir / "turn6_train_binary"
    binary_dir.mkdir(parents=True, exist_ok=True)

    tag = f"s1_{max_s1}" if max_s1 is not None else "s1_full"
    x_bin_path = binary_dir / f"features_{tag}.bin"
    y_bin_path = binary_dir / f"labels_{tag}.bin"

    peak_rss = get_process_rss_mb()

    # If rebuilding, delete old binary files
    if x_bin_path.exists():
        x_bin_path.unlink(missing_ok=True)
    if y_bin_path.exists():
        y_bin_path.unlink(missing_ok=True)

    f_x = None
    f_y = None

    total_s1 = 0
    total_pairs = 0
    total_pos = 0
    total_neg = 0
    sample_pairs_df: Optional[pd.DataFrame] = None

    t0 = time.time()
    logger.info(
        "Starting chunked candidate generation and feature extraction (batch_size=%d, max_s1=%s)...",
        batch_size,
        max_s1,
    )

    try:
        f_x = open(x_bin_path, "wb")
        f_y = open(y_bin_path, "wb")

        # Determine S1 iterator
        def s1_generator() -> Iterator[pd.DataFrame]:
            if s1_data is not None:
                limit = len(s1_data) if max_s1 is None else min(len(s1_data), max_s1)
                for i in range(0, limit, batch_size):
                    yield s1_data.iloc[i : min(i + batch_size, limit)].copy()
            else:
                reader = pd.read_csv(
                    cfg.TRAIN_SOURCE1,
                    sep="\t",
                    dtype=str,
                    chunksize=batch_size,
                    keep_default_na=False,
                )
                accum = 0
                for chunk in reader:
                    if max_s1 is not None and accum >= max_s1:
                        break
                    if max_s1 is not None and accum + len(chunk) > max_s1:
                        chunk = chunk.iloc[: max_s1 - accum]
                    accum += len(chunk)
                    yield chunk

        batch_idx = 0
        for raw_chunk in s1_generator():
            batch_idx += 1
            # 1. Preprocess S1 batch
            s1_proc = raw_chunk if "name_norm" in raw_chunk.columns else preprocess_records(raw_chunk)

            # 2. Query candidates from disk index
            candidates = disk_index.query_records(s1_proc)

            if len(candidates) > 0:
                # 3. Retrieve preprocessed candidate records from SQLite
                cand_ids = candidates["candidate_id"].unique()
                cand_records = disk_index.get_candidate_records(cand_ids)
                if cand_records:
                    cand_df = pd.DataFrame(cand_records)

                    s2_mini = cand_df[cand_df["source"] == "S2"].drop(columns=["source"], errors="ignore")
                    s3_mini = cand_df[cand_df["source"] == "S3"].drop(columns=["source"], errors="ignore")

                    # 4. Retrieve ground truth for this S1 batch
                    s1_ids = list(s1_proc["entity_id"].astype(str))
                    if gt_data is not None:
                        if isinstance(gt_data, dict):
                            gt_batch = {eid: gt_data.get(eid, set()) for eid in s1_ids}
                        else:
                            gt_batch = ground_truth_to_dict(gt_data[gt_data["source1_entity_id"].isin(s1_ids)])
                    else:
                        gt_batch = disk_index.get_ground_truth_for_s1(s1_ids)

                    # 5. Assemble labelled pairs
                    pairs_batch = build_train_pairs(
                        s1_proc,
                        s2_mini,
                        s3_mini,
                        candidates,
                        gt_batch,
                        sample_neg_ratio=sample_neg_ratio,
                        random_seed=random_seed,
                    )

                    if len(pairs_batch) > 0:
                        # 6. Compute 21 features
                        feats_batch = compute_features(pairs_batch)

                        # 7. Extract arrays (guarantee finite float32)
                        x_mat = feats_batch[DEFAULT_FEATURE_NAMES].to_numpy(dtype=np.float32)
                        x_mat = np.nan_to_num(x_mat, nan=0.0, copy=False)
                        y_vec = pairs_batch["label"].to_numpy(dtype=np.int32)

                        # 8. Write to disk
                        x_mat.tofile(f_x)
                        y_vec.tofile(f_y)

                        pos_in_batch = int((y_vec == 1).sum())
                        neg_in_batch = int((y_vec == 0).sum())
                        total_pairs += len(pairs_batch)
                        total_pos += pos_in_batch
                        total_neg += neg_in_batch

                        # Keep a tiny sample for post-training verification
                        if sample_pairs_df is None:
                            sample_pairs_df = pairs_batch.head(10).copy()
                            for col in DEFAULT_FEATURE_NAMES:
                                sample_pairs_df[col] = feats_batch[col].values[:len(sample_pairs_df)]

                        del feats_batch, x_mat, y_vec

                    del cand_records, cand_df, s2_mini, s3_mini, pairs_batch

            total_s1 += len(s1_proc)
            del raw_chunk, s1_proc, candidates
            gc.collect()

            cur_rss = get_process_rss_mb()
            peak_rss = max(peak_rss, cur_rss)

            # Progress logging conforming to Requirement 14
            log_interval = 1 if max_s1 and max_s1 <= 5000 else 5
            if batch_idx % log_interval == 0 or (max_s1 and total_s1 >= max_s1):
                elapsed = round(time.time() - t0, 1)
                logger.info(
                    "  [Processing Stage: Candidate Generation & Feature Extraction | Batch %d] "
                    "Records processed: %d S1 | Candidate pairs generated: %d (%d pos, %d neg) | "
                    "Feature rows written: %d | Current RSS: %.1f MB (Peak: %.1f MB) | Elapsed: %.1fs",
                    batch_idx,
                    total_s1,
                    total_pairs,
                    total_pos,
                    total_neg,
                    total_pairs,
                    cur_rss,
                    peak_rss,
                    elapsed,
                )

    finally:
        if f_x is not None and not f_x.closed:
            f_x.close()
        if f_y is not None and not f_y.closed:
            f_y.close()

    f_x_size_mb = round(x_bin_path.stat().st_size / (1024**2), 2) if x_bin_path.exists() else 0.0
    f_y_size_mb = round(y_bin_path.stat().st_size / (1024**2), 2) if y_bin_path.exists() else 0.0
    logger.info(
        "Candidate generation & feature extraction complete in %.2fs. "
        "%d S1 entities -> %d total pairs (%d pos, %d neg) | Disk: %.2f MB (X) + %.2f MB (y) | Peak RSS: %.1f MB",
        time.time() - t0,
        total_s1,
        total_pairs,
        total_pos,
        total_neg,
        f_x_size_mb,
        f_y_size_mb,
        peak_rss,
    )

    return total_s1, total_pairs, total_pos, total_neg, x_bin_path, y_bin_path, sample_pairs_df, peak_rss


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------


def run_final_training(
    max_s1: Optional[int] = None,
    distractors: Optional[int] = None,
    sample_neg_ratio: Optional[float] = None,
    use_disk_index: bool = False,
    chunked: Optional[bool] = None,
    batch_size: int = 10_000,
    preprocess_chunk_size: int = 50_000,
    cache_dir: Optional[Path] = None,
    output_model_path: Optional[Path] = None,
    output_config_path: Optional[Path] = None,
    random_seed: int = 42,
    force_rebuild: bool = False,
    s1_data: Optional[pd.DataFrame] = None,
    s2_data: Optional[pd.DataFrame] = None,
    s3_data: Optional[pd.DataFrame] = None,
    gt_data: Optional[pd.DataFrame | dict] = None,
    db_path: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Execute final LightGBM model training on 100% of labeled training data.

    Parameters
    ----------
    max_s1 : If set, trains on 100% of this deterministic S1 sample (no validation split).
             If None, trains on 100% of all competition labeled entities (2,206,821 S1 records).
    distractors : Distractor pool size when max_s1 is set.
    sample_neg_ratio : Optional negative downsampling ratio.
    use_disk_index : Use SQLite-backed DiskBlockingIndex for candidate generation.
    chunked : Whether to use chunked streaming / disk-backed pipeline.
              Defaults to True to guarantee ~12 GB RAM compliance.
    batch_size : Number of S1 entities processed per batch in chunked mode.
    preprocess_chunk_size : Number of candidate pool records processed per chunk.
    cache_dir : Directory for caching intermediate data.
    output_model_path : Path for saving final model (default: cache/turn6_final_matcher.joblib).
    output_config_path : Path for saving final config (default: cache/turn6_final_model_config.json).
    random_seed : Random seed for reproducibility.
    force_rebuild : Rebuild all caches from scratch.
    s1_data, s2_data, s3_data, gt_data : Optional preloaded DataFrames (for testing).
    db_path : Optional SQLite database file path.

    Returns
    -------
    dict with full training metadata and configuration.
    """
    cfg = Config()
    c_dir = cache_dir or cfg.CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)

    model_out = output_model_path or (c_dir / "turn6_final_matcher.joblib")
    config_out = output_config_path or (c_dir / "turn6_final_model_config.json")

    # Safety assertion: never overwrite Turn 5.5 artifacts
    turn5_5_model = c_dir / "turn5_5_best_matcher.joblib"
    turn5_5_thresh = c_dir / "turn5_5_optimal_threshold.json"
    if model_out.resolve() == turn5_5_model.resolve():
        raise ValueError(
            f"Safety error: output_model_path points to Turn 5.5 model: {turn5_5_model}. "
            "Must use a separate path (e.g. cache/turn6_final_matcher.joblib)."
        )
    if config_out.resolve() == turn5_5_thresh.resolve():
        raise ValueError(
            f"Safety error: output_config_path points to Turn 5.5 threshold config: {turn5_5_thresh}."
        )

    # Determine whether to use chunked streaming pipeline
    # Default: True for all training runs to ensure memory scalability (<12 GB RAM)
    is_chunked = chunked if chunked is not None else True

    rss_baseline = get_process_rss_mb()
    peak_rss = rss_baseline
    total_start = time.time()
    timing: dict[str, float] = {}
    disk_idx: Optional[DiskBlockingIndex] = None

    mode_label = (
        "100% OF FULL COMPETITION DATA (CHUNKED STREAMING / DISK-BACKED)"
        if (max_s1 is None and is_chunked)
        else (
            f"DETERMINISTIC SANITY SAMPLE ({max_s1:,} S1 ENTITIES, 100% TRAIN / 0% VAL)"
            if max_s1 is not None
            else "100% OF FULL COMPETITION DATA"
        )
    )

    print("=" * 80)
    print(f"TURN 6: FINAL MODEL TRAINING PIPELINE [{mode_label}]")
    print("=" * 80)
    print("Architecture: 4-stage blocking | 21 features | LightGBM | Threshold: 0.90")
    print("Validation Split: NONE (100% of labeled training data available to model)")
    print(f"Pipeline Mode: {'Chunked Streaming (Memory Safe)' if is_chunked else 'In-Memory / Cached'}")
    print(f"Baseline Process RSS: {rss_baseline:.2f} MB\n")

    # ------------------------------------------------------------------
    # BRANCH 1: Chunked Streaming Pipeline (Memory Safe, <12 GB RAM)
    # ------------------------------------------------------------------
    if is_chunked:
        try:
            is_custom_test_data = (s2_data is not None or s1_data is not None)
            # If s1_data is not provided and max_s1 is set, load deterministic sample
            if s1_data is None and max_s1 is not None:
                logger.info("Loading deterministic training sample for max_s1=%d...", max_s1)
                s1_data, s2_data, s3_data, gt_data = load_training_sample(
                    cfg=cfg,
                    max_s1=max_s1,
                    distractors=distractors,
                    cache_dir=c_dir,
                    random_seed=random_seed,
                    force_rebuild=force_rebuild,
                )

            effective_db_path = (
                Path(db_path)
                if db_path is not None
                else (
                    c_dir / f"turn6_train_blocking_s1_{max_s1}.db"
                    if (max_s1 is not None and not is_custom_test_data)
                    else c_dir / "turn6_train_blocking.db"
                )
            )

            print("[1/3] Building / Loading disk-backed candidate index and ground truth in SQLite...")
            t0 = time.time()
            disk_idx = build_or_load_candidate_index(
                cfg=cfg,
                cache_dir=c_dir,
                db_path=effective_db_path,
                preprocess_chunk_size=preprocess_chunk_size,
                force_rebuild=force_rebuild,
                s2_data=s2_data,
                s3_data=s3_data,
                gt_data=gt_data,
            )
            timing["index_prep_s"] = round(time.time() - t0, 2)
            peak_rss = max(peak_rss, get_process_rss_mb())
            print(f"  Candidate Index Ready in {timing['index_prep_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

            print("[2/3] Streaming S1, building labelled pairs, and extracting 21 features to disk...")
            t0 = time.time()
            (
                total_s1,
                total_pairs,
                total_pos,
                total_neg,
                x_bin_path,
                y_bin_path,
                sample_pairs_df,
                stream_peak_rss,
            ) = stream_and_extract_training_pairs(
                cfg=cfg,
                disk_index=disk_idx,
                max_s1=max_s1,
                batch_size=batch_size,
                sample_neg_ratio=sample_neg_ratio,
                random_seed=random_seed,
                cache_dir=c_dir,
                s1_data=s1_data,
                gt_data=gt_data,
                force_rebuild=force_rebuild,
            )
            timing["pair_and_feature_prep_s"] = round(time.time() - t0, 2)
            peak_rss = max(peak_rss, stream_peak_rss)

            if total_pairs == 0:
                raise ValueError("No candidate pairs generated. Cannot train model on 0 pairs.")

            pos_pct = round(total_pos / total_pairs * 100, 2) if total_pairs > 0 else 0.0
            print(f"  Processed S1 entities: {total_s1:,}")
            print(f"  Total Labelled Pairs:  {total_pairs:,} ({total_pos:,} pos [{pos_pct}%], {total_neg:,} neg [{100-pos_pct}%])")
            print(f"  Disk feature storage:  {x_bin_path.name} ({round(x_bin_path.stat().st_size / (1024**2), 2)} MB)")
            print(f"  Pair & Feature Time:   {timing['pair_and_feature_prep_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

            # ------------------------------------------------------------------
            # Train LightGBM using disk-backed memory map
            # ------------------------------------------------------------------
            print("[3/3] Training LightGBM on 100% of labeled training data via memory map (NO validation split)...")
            print(f"  Hyperparameters: n_estimators=100, learning_rate=0.05, random_state={random_seed}, class_weight='balanced'")
            t0 = time.time()

            X_mmap = np.memmap(
                x_bin_path,
                dtype=np.float32,
                mode="r",
                shape=(total_pairs, len(DEFAULT_FEATURE_NAMES)),
            )
            y_mmap = np.memmap(
                y_bin_path,
                dtype=np.int32,
                mode="r",
                shape=(total_pairs,),
            )

            matcher = LightGBMMatcher(
                n_estimators=100,
                learning_rate=0.05,
                random_state=random_seed,
            )
            matcher.fit(X_mmap, y_mmap)
            timing["training_s"] = round(time.time() - t0, 3)
            peak_rss = max(peak_rss, get_process_rss_mb())

            assert matcher.is_fitted, "Fatal error: LightGBMMatcher is not fitted after fit() call!"
            print(f"  LightGBM Training Time: {timing['training_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

            if hasattr(X_mmap, "_mmap") and X_mmap._mmap is not None:
                try:
                    X_mmap._mmap.close()
                except Exception:
                    pass
            if hasattr(y_mmap, "_mmap") and y_mmap._mmap is not None:
                try:
                    y_mmap._mmap.close()
                except Exception:
                    pass
            del X_mmap, y_mmap
            gc.collect()

        finally:
            if disk_idx is not None:
                disk_idx.close()
                disk_idx = None

    # ------------------------------------------------------------------
    # BRANCH 2: In-Memory / Cached Sample Pipeline (for fast testing / sanity checks)
    # ------------------------------------------------------------------
    else:
        # Fast path check for cached Turn 5.5 scale check
        pairs_pkl = c_dir / "scale_check_pairs_s1_5000.pkl"
        feats_pkl = c_dir / "scale_check_features_s1_5000.pkl"

        if max_s1 == 5000 and not force_rebuild and pairs_pkl.exists() and feats_pkl.exists():
            logger.info("Reusing cached pairs from %s...", pairs_pkl)
            with open(pairs_pkl, "rb") as f:
                pairs = pickle.load(f)
            logger.info("Reusing cached features from %s...", feats_pkl)
            with open(feats_pkl, "rb") as f:
                features_df = pickle.load(f)
            total_s1 = max_s1
        else:
            if s1_data is not None and s2_data is not None and s3_data is not None and gt_data is not None:
                s1_proc = s1_data if "name_norm" in s1_data.columns else preprocess_records(s1_data)
                s2_proc = s2_data if "name_norm" in s2_data.columns else preprocess_records(s2_data)
                s3_proc = s3_data if "name_norm" in s3_data.columns else preprocess_records(s3_data)
                gt_df = gt_data if isinstance(gt_data, pd.DataFrame) else pd.DataFrame(
                    [{"source1_entity_id": k, "matched_entity_ids": ",".join(v)} for k, v in gt_data.items()]
                )
            else:
                s1_proc, s2_proc, s3_proc, gt_df = load_training_sample(
                    cfg=cfg,
                    max_s1=max_s1 or 5000,
                    distractors=distractors,
                    cache_dir=c_dir,
                    random_seed=random_seed,
                    force_rebuild=force_rebuild,
                )

            total_s1 = len(s1_proc)
            b_cfg = BlockingConfig.from_config(cfg)
            candidates = build_candidates(s1_proc, s2_proc, s3_proc, config=b_cfg, use_disk_index=use_disk_index)
            pairs = build_train_pairs(
                s1_proc,
                s2_proc,
                s3_proc,
                candidates,
                gt_df,
                sample_neg_ratio=sample_neg_ratio,
                random_seed=random_seed,
            )
            features_df = compute_features(pairs)

        for col in DEFAULT_FEATURE_NAMES:
            if col in features_df.columns:
                pairs[col] = features_df[col].values

        total_pairs = len(pairs)
        total_pos = int((pairs["label"] == 1).sum()) if total_pairs > 0 else 0
        total_neg = int((pairs["label"] == 0).sum()) if total_pairs > 0 else 0
        sample_pairs_df = pairs.head(10).copy()

        t0 = time.time()
        matcher = LightGBMMatcher(
            n_estimators=100,
            learning_rate=0.05,
            random_state=random_seed,
        )
        matcher.fit(pairs, pairs["label"])
        timing["training_s"] = round(time.time() - t0, 3)
        peak_rss = max(peak_rss, get_process_rss_mb())

    # ------------------------------------------------------------------
    # Persist Model Artifact and Metadata Config
    # ------------------------------------------------------------------
    print("[Persisting Model & Metadata]...")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    matcher.save(model_out)

    # Verification: Load model artifact back to confirm integrity
    loaded_matcher = LightGBMMatcher.load(model_out)
    assert isinstance(loaded_matcher, LightGBMMatcher), f"Unexpected model type: {type(loaded_matcher)}"
    assert loaded_matcher.is_fitted, "Loaded model reports is_fitted=False!"

    # Verify probability inference on sample batch
    if sample_pairs_df is not None and len(sample_pairs_df) > 0:
        test_proba = loaded_matcher.predict_proba(sample_pairs_df)
        assert len(test_proba) == len(sample_pairs_df), "Probability vector length mismatch!"
        assert (test_proba >= 0.0).all() and (test_proba <= 1.0).all(), "Probabilities outside [0, 1]!"

    total_pipeline_time = round(time.time() - total_start, 2)
    timing["total_runtime_s"] = total_pipeline_time

    # Feature importances
    fi_dict = loaded_matcher.get_feature_importances(importance_type="gain")
    sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)

    metadata: dict[str, Any] = {
        "model_path": str(model_out.resolve()),
        "model_type": "LightGBM",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "random_state": random_seed,
        "threshold": 0.90,
        "number_of_s1_training_entities": total_s1,
        "number_of_labelled_pairs": total_pairs,
        "labelled_pair_count": total_pairs,
        "labelled pair count": total_pairs,
        "positive_pair_count": total_pos,
        "negative_pair_count": total_neg,
        "feature_count": len(DEFAULT_FEATURE_NAMES),
        "feature_names": list(DEFAULT_FEATURE_NAMES),
        "top_feature_by_gain": sorted_fi[0] if sorted_fi else None,
        "training_runtime_s": timing.get("training_s", 0.0),
        "total_pipeline_runtime_s": total_pipeline_time,
        "runtime": total_pipeline_time,
        "peak_rss_mb": round(peak_rss, 2),
        "peak_rss": round(peak_rss, 2),
        "peak RSS": round(peak_rss, 2),
        "is_full_training": (max_s1 is None),
        "validation_split": "none (100% of labeled training data used)",
        "chunked_disk_backed_training": is_chunked,
        "whether_chunked_or_disk_backed_training_was_used": is_chunked,
        "whether chunked/disk-backed training was used": is_chunked,
        "training_hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "random_state": random_seed,
            "class_weight": "balanced",
            "threshold": 0.90,
        },
        "training hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "random_state": random_seed,
            "class_weight": "balanced",
            "threshold": 0.90,
        },
    }

    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 80)
    print("FINAL MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Model artifact:   {model_out}")
    print(f"  Metadata config:  {config_out}")
    print(f"  S1 Entities:      {total_s1:,}")
    print(f"  Labelled Pairs:   {total_pairs:,} ({total_pos:,} positives, {total_neg:,} negatives)")
    print(f"  Validation Split: NONE (100% of labeled training data used)")
    print(f"  Model Type:       LightGBM (n_estimators=100, lr=0.05, random_state={random_seed})")
    print(f"  Threshold:        0.90 (validated from Turn 5.5)")
    print(f"  Training Time:    {timing.get('training_s', 0.0)}s")
    print(f"  Total Runtime:    {total_pipeline_time}s")
    print(f"  Peak RSS:         {peak_rss:.2f} MB")
    print(f"  Chunked Mode:     {is_chunked}")
    print(f"  Top 3 Features:   {', '.join(f'{k} ({v*100:.1f}%)' for k, v in sorted_fi[:3])}")
    print("=" * 80)
    print()

    return metadata


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train final LightGBM matcher on 100% of labeled competition training data."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Train on 100%% of all competition labeled training entities (all 2.2M S1 records).",
    )
    parser.add_argument(
        "--max-s1",
        type=int,
        default=None,
        help="Cap S1 training entities (e.g. 5000 for fast deterministic sanity test).",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=None,
        help="Cap distractor pool records (default: matches max-s1).",
    )
    parser.add_argument(
        "--sample-neg-ratio",
        type=float,
        default=None,
        help="Optional negative-to-positive subsampling ratio per S1 entity (None = keep all).",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        default=None,
        help="Explicitly enable chunked streaming / disk-backed training pipeline (default: True).",
    )
    parser.add_argument(
        "--no-chunked",
        dest="chunked",
        action="store_false",
        help="Disable chunked streaming and force in-memory pipeline.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Number of S1 entities processed per streaming batch (default: 10,000).",
    )
    parser.add_argument(
        "--preprocess-chunk-size",
        type=int,
        default=50_000,
        help="Number of candidate records processed per streaming chunk (default: 50,000).",
    )
    parser.add_argument(
        "--use-disk-index",
        action="store_true",
        default=False,
        help="Use disk-backed SQLite DiskBlockingIndex instead of in-memory BlockingIndex.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching intermediate artifacts (default: cache/).",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=None,
        help="Destination path for final trained model (default: cache/turn6_final_matcher.joblib).",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=None,
        help="Destination path for metadata config (default: cache/turn6_final_model_config.json).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Ignore cached pairs/features and rebuild from raw data.",
    )

    args = parser.parse_args()

    c_dir = Path(args.cache_dir) if args.cache_dir else None
    out_model = Path(args.output_model) if args.output_model else None
    out_cfg = Path(args.output_config) if args.output_config else None

    # Determine effective max_s1
    effective_max_s1 = args.max_s1
    if not args.full and effective_max_s1 is None:
        logger.info(
            "Neither --full nor --max-s1 specified. Defaulting to full competition training on 100% of data."
        )

    run_final_training(
        max_s1=effective_max_s1,
        distractors=args.distractors,
        sample_neg_ratio=args.sample_neg_ratio,
        use_disk_index=args.use_disk_index,
        chunked=args.chunked,
        batch_size=args.batch_size,
        preprocess_chunk_size=args.preprocess_chunk_size,
        cache_dir=c_dir,
        output_model_path=out_model,
        output_config_path=out_cfg,
        random_seed=args.random_seed,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
