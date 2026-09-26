"""
generate_submission.py - Turn 6: Full TEST inference + submission generation.

Implements a 7-step production pipeline reusing the validated Turn 5.5 architecture:
    TEST Source 1 (all ~1.73M rows)
         | [chunked streaming + preprocessing]
    TEST Source 2 + TEST Source 3
         | [4-stage blocking -> BlockingIndex]
    Candidate pairs
         | [21 canonical numerical pairwise features]
    LightGBM matcher (Turn 5.5, threshold = 0.90)
         | [predict_matches]
    matching_results.tsv  +  candidate_pairs.tsv
         | [zipfile]
    submission.zip

Usage
-----
Smoke test (first 2,000 S1 entities only):
    python scripts/generate_submission.py --smoke-test --batch-size 500

Full run:
    python scripts/generate_submission.py

Google Colab (set BER_DATASET_ROOT env var before running):
    import os
    os.environ["BER_DATASET_ROOT"] = "/content/drive/MyDrive/dataset"
    !python scripts/generate_submission.py --batch-size 5000

Outputs
-------
output/matching_results.tsv
output/candidate_pairs.tsv
output/submission.zip
output/turn6_inference_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path (Colab-compatible)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import BlockingConfig, BlockingIndex, DiskBlockingIndex
from src.config import Config
from src.features import compute_features
from src.model import DEFAULT_FEATURE_NAMES, LightGBMMatcher
from src.output import write_candidate_pairs, write_matching_results
from src.pair_builder import build_test_pairs
from src.prediction import predict_matches
from src.preprocessing import preprocess_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Validated threshold from Turn 5.5 (LightGBM, Macro F0.5 = 0.958 on 1000 val entities)
VALIDATED_THRESHOLD: float = 0.90
VALIDATED_MODEL: str = "LightGBM"

# Default batch size for processing S1 entities in chunks
DEFAULT_BATCH_S1: int = 10_000

# Chunk sizes for streaming large TSV files
CHUNK_SIZE_S1: int = 200_000
CHUNK_SIZE_S23: int = 250_000


# ---------------------------------------------------------------------------
# Memory tracker (optional: psutil)
# ---------------------------------------------------------------------------

try:
    import psutil

    def _rss_mb() -> float:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)

except ImportError:
    def _rss_mb() -> float:
        return -1.0


# ---------------------------------------------------------------------------
# Step 1: Load validated Turn 5.5 model artifact
# ---------------------------------------------------------------------------


def load_validated_model(cache_dir: Path) -> LightGBMMatcher:
    """Load the Turn 5.5 LightGBM model from joblib file.

    LightGBMMatcher.load() is a @classmethod that returns a fully-fitted instance.
    We must capture and return its return value (not call it on a blank instance).
    """
    model_path = cache_dir / "turn5_5_best_matcher.joblib"
    if not model_path.exists():
        model_path = cache_dir / "best_matcher.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found in {cache_dir}. "
            "Run scripts/run_validation_scale_check.py first."
        )
    logger.info("Loading trained model from %s ...", model_path)
    # load() is a @classmethod — must use the returned fitted instance
    matcher = LightGBMMatcher.load(model_path)
    assert isinstance(matcher, LightGBMMatcher), f"Unexpected type: {type(matcher)}"
    assert matcher.is_fitted, "Loaded model reports is_fitted=False — model file may be corrupt."
    return matcher


# Fields needed for feature computation (subset of all preprocessed columns)
_LOOKUP_FIELDS = (
    "name_norm", "name_compact", "name_no_suffix",
    "address_norm", "address_is_missing",
    "country_norm", "name_script", "is_domain", "domain_stem", "legal_suffix",
)


def build_index_and_lookup(
    paths: list[tuple[Path, str]],
    cfg: Config,
    preprocess_chunk_size: int = 50_000,
    max_rows_per_source: Optional[int] = None,
    db_path: Optional[Path | str] = None,
    cache_index_path: Optional[Path] = None,
    cache_lookup_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> tuple[DiskBlockingIndex, Any]:
    """Stream-build the disk-backed blocking index in one memory-safe pass.

    Memory strategy (solves RAM explosion on Colab with ~10M candidate records):
    - Reads S2/S3 in chunks of preprocess_chunk_size (default 50k rows).
    - Preprocesses each chunk for blocking keys and feature fields.
    - Immediately commits preprocessed records and blocking keys to SQLite.
    - Discards chunk DataFrame immediately (gc.collect()) — RAM stays O(chunk), not O(total).
    - Finalizes SQLite posting lists with max_candidates_per_block=100 cap and B-tree indexing.
    - Python heap stays flat: ~250-450 MB vs >8.78 GB OOM before.

    Parameters
    ----------
    paths : List of (path, label) tuples for each source TSV to index.
    cfg : Config instance.
    preprocess_chunk_size : Rows per preprocessing chunk (default 50k).
    max_rows_per_source : Cap on rows per source (for smoke testing).
    db_path : Path to disk-backed SQLite database.
    force_rebuild : If True, ignore existing database and rebuild from scratch.

    Returns
    -------
    (DiskBlockingIndex, DiskBlockingIndex)
    """
    import gc

    target_db = Path(db_path) if db_path is not None else (cfg.CACHE_DIR / "turn6_blocking.db")
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

    # Load from existing finalized database if available and not force_rebuild
    if target_db.exists() and not force_rebuild:
        idx = DiskBlockingIndex(db_path=target_db, config=blocking_cfg)
        if idx.is_finalized and idx.total_records_indexed > 0:
            logger.info(
                "  Loaded existing disk-backed index from %s (%d records indexed | S2=%d, S3=%d).",
                target_db,
                idx.total_records_indexed,
                idx.source_counts.get("S2", 0),
                idx.source_counts.get("S3", 0),
            )
            return idx, idx
        idx.close()

    # Rebuilding from scratch
    if target_db.exists():
        try:
            target_db.unlink()
            for ext in ["-wal", "-shm"]:
                p = Path(str(target_db) + ext)
                if p.exists():
                    p.unlink()
        except Exception:
            pass

    idx = DiskBlockingIndex(db_path=target_db, config=blocking_cfg)
    total_indexed = 0
    t0 = time.time()

    logger.info("  Index/database location: %s", target_db)
    logger.info("  Chunk size: %d rows", preprocess_chunk_size)

    for path, label in paths:
        source_total = 0
        chunk_num = 0
        logger.info(
            "  Streaming + indexing %s from %s (max_rows=%s, chunk_size=%d) ...",
            label, path, max_rows_per_source, preprocess_chunk_size,
        )
        reader = pd.read_csv(
            path, sep="\t", dtype=str,
            chunksize=preprocess_chunk_size, keep_default_na=False,
        )
        for raw_chunk in reader:
            if max_rows_per_source is not None and source_total >= max_rows_per_source:
                break
            if max_rows_per_source is not None:
                raw_chunk = raw_chunk.iloc[: max_rows_per_source - source_total]

            chunk_num += 1
            proc_chunk = preprocess_records(raw_chunk)

            # Add to disk-backed index
            idx.add_records(proc_chunk)

            source_total += len(proc_chunk)
            total_indexed += len(proc_chunk)

            del proc_chunk, raw_chunk
            gc.collect()

            if chunk_num % 5 == 0:
                elapsed = round(time.time() - t0, 1)
                logger.info(
                    "  %s: indexed %d records (chunk %d) in %.1fs | RSS=%.1f MB",
                    label, source_total, chunk_num, elapsed, _rss_mb(),
                )

        elapsed_src = round(time.time() - t0, 1)
        logger.info(
            "  %s done: %d records indexed in %.1fs | RSS=%.1f MB",
            label, source_total, elapsed_src, _rss_mb(),
        )

    # Finalize posting lists and indexes
    t_fin0 = time.time()
    idx.finalize()
    t_fin = round(time.time() - t_fin0, 2)
    elapsed = round(time.time() - t0, 2)
    db_size_mb = round(target_db.stat().st_size / (1024 ** 2), 2) if target_db.exists() else 0.0

    logger.info(
        "  Disk-backed index finalized at %s (%.2f MB on disk, %d total records) in %.2fs (finalize: %.2fs) | RSS=%.1f MB",
        target_db, db_size_mb, idx.total_records_indexed, elapsed, t_fin, _rss_mb(),
    )

    return idx, idx



# ---------------------------------------------------------------------------
# Step 4: Process S1 entities in one batch
# ---------------------------------------------------------------------------


def process_s1_batch(
    s1_batch: pd.DataFrame,
    candidate_lookup: dict[str, dict],
    blocking_idx: BlockingIndex,
    matcher: LightGBMMatcher,
    threshold: float,
    cfg: Config,
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """Run blocking -> features -> scoring for a batch of S1 entities.

    Parameters
    ----------
    s1_batch : Preprocessed S1 entities for this batch.
    candidate_lookup : Compact dict {entity_id -> feature fields} for S2+S3 candidates.
                       Built by build_index_and_lookup(); avoids keeping full DataFrames.

    Returns
    -------
    predictions : dict s1_id -> set of matched IDs
    candidates_map : dict s1_id -> list of candidate IDs
    """
    if len(s1_batch) == 0:
        return {}, {}

    # A: Generate candidates via the pre-built index
    candidates = blocking_idx.query_records(s1_batch)

    # Build candidates_map (for candidate_pairs.tsv)
    candidates_map: dict[str, list[str]] = {
        str(eid): [] for eid in s1_batch["entity_id"]
    }
    if candidates is not None and len(candidates) > 0:
        for s1_id, grp in candidates.groupby("s1_id"):
            candidates_map[str(s1_id)] = list(grp["candidate_id"].astype(str))

    # B: Handle no-candidate case
    if candidates is None or len(candidates) == 0:
        predictions = {str(eid): set() for eid in s1_batch["entity_id"]}
        return predictions, candidates_map

    # C: Reconstruct mini candidate DataFrames (disk-backed or lookup dict)
    cand_ids = candidates["candidate_id"].astype(str).unique()
    if hasattr(blocking_idx, "get_candidate_records"):
        cand_records = blocking_idx.get_candidate_records(cand_ids)
    elif candidate_lookup is not None and hasattr(candidate_lookup, "get_candidate_records"):
        cand_records = candidate_lookup.get_candidate_records(cand_ids)
    elif candidate_lookup is not None:
        cand_records = []
        for cid in cand_ids:
            rec = candidate_lookup.get(cid)
            if rec is not None:
                row = {"entity_id": cid, **rec}
                row["source"] = "S2" if cid.startswith("S2") else "S3"
                cand_records.append(row)
    else:
        cand_records = []

    if not cand_records:
        predictions = {str(eid): set() for eid in s1_batch["entity_id"]}
        return predictions, candidates_map

    cand_df = pd.DataFrame(cand_records)
    # Split into S2/S3 mini-DataFrames for build_test_pairs
    s2_mini = cand_df[cand_df["source"] == "S2"].drop(columns=["source"])
    s3_mini = cand_df[cand_df["source"] == "S3"].drop(columns=["source"])

    # D: Build unlabelled test pairs (join preprocessed fields)
    test_pairs = build_test_pairs(s1_batch, s2_mini, s3_mini, candidates)
    if len(test_pairs) == 0:
        predictions = {str(eid): set() for eid in s1_batch["entity_id"]}
        return predictions, candidates_map

    # E: Compute 21 pairwise features
    features_df = compute_features(test_pairs)
    for col in DEFAULT_FEATURE_NAMES:
        if col in features_df.columns:
            test_pairs[col] = features_df[col].values

    # F: Score with LightGBM
    proba = matcher.predict_proba(test_pairs)

    # G: Threshold -> entity-level predictions
    s1_ids = list(s1_batch["entity_id"].astype(str))
    predictions = predict_matches(
        test_pairs, proba, threshold=threshold, s1_entities=s1_ids
    )
    return predictions, candidates_map


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------


def _summary_str(all_predictions: dict[str, set[str]]) -> str:
    total = sum(len(v) for v in all_predictions.values())
    matched = sum(1 for v in all_predictions.values() if len(v) > 0)
    return f"{total:,} across {matched:,} matched entities"


def run_inference(
    smoke_test: bool = False,
    smoke_test_n: int = 2000,
    pool_rows: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_S1,
    threshold: float = VALIDATED_THRESHOLD,
    cache_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    db_path: Optional[Path | str] = None,
    cleanup_db: bool = False,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Run the full Turn 6 inference pipeline.

    Parameters
    ----------
    smoke_test : If True, only process the first `smoke_test_n` S1 entities.
    smoke_test_n : Number of S1 entities for smoke test.
    pool_rows : If set, limits S2+S3 pool to first N rows each (for fast smoke test).
               Default None = full pool. Recommended: 200_000 for smoke tests.
    batch_size : Number of S1 entities per processing batch.
    threshold : Match probability threshold (default: 0.90).
    cache_dir : Directory for caching blocking index and preprocessed data.
    output_dir : Directory for writing submission files.
    force_rebuild : If True, ignore existing caches and rebuild from scratch.
    """
    cfg = Config()
    c_dir = cache_dir or cfg.CACHE_DIR
    o_dir = output_dir or cfg.OUTPUT_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    o_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    rss_start = _rss_mb()
    timing: dict[str, float] = {}

    mode_str = f"SMOKE TEST ({smoke_test_n} S1 entities)" if smoke_test else "FULL RUN"
    print("=" * 80)
    print(f"TURN 6: TEST INFERENCE + SUBMISSION GENERATION [{mode_str}]")
    print(f"Threshold: {threshold} | Model: {VALIDATED_MODEL}")
    print("=" * 80)
    print(f"Baseline Process RSS: {rss_start:.1f} MB\n")

    # ------------------------------------------------------------------
    # [1/7] Load validated model
    # ------------------------------------------------------------------
    print("[1/7] Loading validated Turn 5.5 model...")
    t0 = time.time()
    matcher = load_validated_model(c_dir)
    timing["load_model_s"] = round(time.time() - t0, 3)
    print(f"  Model loaded in {timing['load_model_s']}s. RSS: {_rss_mb():.1f} MB\n")

    # ------------------------------------------------------------------
    # [2/7 + 3/7] Stream-build disk-backed blocking index (SQLite)
    # ------------------------------------------------------------------
    print("[2/7] Streaming S2+S3 -> disk-backed blocking index (SQLite)...")
    t0 = time.time()

    pool_paths = [
        (cfg.TEST_SOURCE2, "TEST-S2"),
        (cfg.TEST_SOURCE3, "TEST-S3"),
    ]

    prefix = "smoke_" if smoke_test else ""
    if db_path is not None:
        actual_db_path = Path(db_path)
    elif os.environ.get("BER_DB_PATH"):
        actual_db_path = Path(os.environ["BER_DB_PATH"])
    else:
        actual_db_path = c_dir / f"{prefix}turn6_blocking.db"

    chunk_size_s23 = CHUNK_SIZE_S23 if pool_rows is None else 50_000

    blocking_idx, candidate_lookup = build_index_and_lookup(
        paths=pool_paths,
        cfg=cfg,
        preprocess_chunk_size=chunk_size_s23,
        max_rows_per_source=pool_rows,
        db_path=actual_db_path,
        force_rebuild=force_rebuild,
    )
    total_pool = blocking_idx.total_records_indexed
    timing["build_index_s"] = round(time.time() - t0, 2)
    db_size_mb = round(actual_db_path.stat().st_size / (1024 ** 2), 2) if actual_db_path.exists() else 0.0
    print(f"  Pool: {total_pool:,} records indexed | DB size: {db_size_mb} MB -> {actual_db_path}")
    print(f"  Runtime: {timing['build_index_s']}s | RSS: {_rss_mb():.1f} MB\n")

    # ------------------------------------------------------------------
    # [3/7 - merged above; now: preprocess S1]
    # ------------------------------------------------------------------
    print("[3/7] Loading and preprocessing TEST S1...")
    t0 = time.time()

    s1_max_rows = smoke_test_n if smoke_test else None
    s1_cache = c_dir / "turn6_test_s1_proc.pkl"

    if (not smoke_test) and (not force_rebuild) and s1_cache.exists():
        logger.info("Loading cached TEST S1 from %s...", s1_cache)
        s1_proc = pd.read_pickle(s1_cache)
        logger.info("  Loaded %d TEST S1 records.", len(s1_proc))
    else:
        # Stream-preprocess S1 in chunks (reuse the same approach)
        s1_chunks: list[pd.DataFrame] = []
        s1_total = 0
        for raw_chunk in pd.read_csv(
            cfg.TEST_SOURCE1, sep="\t", dtype=str,
            chunksize=50_000, keep_default_na=False,
        ):
            if s1_max_rows is not None and s1_total >= s1_max_rows:
                break
            if s1_max_rows is not None:
                raw_chunk = raw_chunk.iloc[: s1_max_rows - s1_total]
            proc = preprocess_records(raw_chunk)
            s1_chunks.append(proc)
            s1_total += len(proc)
        s1_proc = pd.concat(s1_chunks, ignore_index=True) if s1_chunks else pd.DataFrame()
        timing["load_s1_s"] = round(time.time() - t0, 2)
        if not smoke_test:
            s1_proc.to_pickle(s1_cache)
            logger.info("  Cached TEST S1 to %s.", s1_cache)

    if smoke_test and len(s1_proc) > smoke_test_n:
        s1_proc = s1_proc.head(smoke_test_n).reset_index(drop=True)
    logger.info("Using %d S1 entities for inference.", len(s1_proc))

    total_s1 = len(s1_proc)
    total_s2 = blocking_idx.source_counts.get("S2", 0)
    total_s3 = blocking_idx.source_counts.get("S3", 0)

    print(f"  TEST Records: S1={total_s1:,} | S2={total_s2:,} | S3={total_s3:,}")
    print(f"  S1 preprocessing: {timing.get('load_s1_s', 0.0)}s | RSS: {_rss_mb():.1f} MB\n")

    # ------------------------------------------------------------------
    # [4/7] Batch inference over all S1 entities
    # ------------------------------------------------------------------
    total_batches = (total_s1 + batch_size - 1) // batch_size
    print(
        f"[4/7] Running batch inference "
        f"({total_s1:,} S1 entities, {total_batches} batches of {batch_size:,})..."
    )

    all_predictions: dict[str, set[str]] = {}
    all_candidates_map: dict[str, list[str]] = {}
    t_inference_start = time.time()
    total_candidates_generated = 0

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_s1)
        s1_batch = s1_proc.iloc[start:end].reset_index(drop=True)

        batch_t0 = time.time()
        batch_preds, batch_cands = process_s1_batch(
            s1_batch=s1_batch,
            candidate_lookup=candidate_lookup,
            blocking_idx=blocking_idx,
            matcher=matcher,
            threshold=threshold,
            cfg=cfg,
        )
        batch_elapsed = round(time.time() - batch_t0, 2)

        all_predictions.update(batch_preds)
        all_candidates_map.update(batch_cands)

        batch_cand_count = sum(len(v) for v in batch_cands.values())
        batch_match_count = sum(len(v) for v in batch_preds.values())
        total_candidates_generated += batch_cand_count

        pct_done = round((end / total_s1) * 100, 1)
        logger.info(
            "Batch %d/%d [%d-%d] | S1=%d | Cands=%d | Matches=%d | %.2fs | RSS=%.1f MB | %.1f%%",
            batch_idx + 1, total_batches, start, end - 1,
            len(s1_batch), batch_cand_count, batch_match_count,
            batch_elapsed, _rss_mb(), pct_done,
        )

    timing["inference_s"] = round(time.time() - t_inference_start, 2)
    print(f"\n  Inference complete. {total_s1:,} S1 entities processed.")
    print(f"  Total candidates generated: {total_candidates_generated:,}")
    print(f"  Total matches predicted:    {_summary_str(all_predictions)}")
    print(f"  Runtime: {timing['inference_s']}s | RSS: {_rss_mb():.1f} MB\n")

    # ------------------------------------------------------------------
    # [5/7] Validate coverage
    # ------------------------------------------------------------------
    print("[5/7] Validating prediction coverage...")
    all_s1_ids = set(s1_proc["entity_id"].astype(str))
    predicted_s1_ids = set(all_predictions.keys())
    missing = all_s1_ids - predicted_s1_ids

    if missing:
        logger.warning(
            "  %d S1 entities missing from predictions - adding as singletons.",
            len(missing),
        )
        for eid in missing:
            all_predictions[eid] = set()
            all_candidates_map[eid] = []

    assert len(all_predictions) == total_s1, (
        f"Coverage failure: predictions={len(all_predictions)}, expected={total_s1}"
    )

    singleton_count = sum(1 for v in all_predictions.values() if len(v) == 0)
    matched_count = total_s1 - singleton_count
    total_predicted_matches = sum(len(v) for v in all_predictions.values())

    print(f"  Coverage: {len(all_predictions):,} / {total_s1:,} S1 entities - PASSED")
    print(f"  Singletons (no match):  {singleton_count:,} ({singleton_count / total_s1 * 100:.1f}%)")
    print(f"  Matched entities:       {matched_count:,} ({matched_count / total_s1 * 100:.1f}%)")
    print(f"  Total predicted pairs:  {total_predicted_matches:,}\n")

    # ------------------------------------------------------------------
    # [6/7] Write submission files
    # ------------------------------------------------------------------
    print("[6/7] Writing submission files...")
    t0 = time.time()

    prefix = "smoke_" if smoke_test else ""
    matching_path = o_dir / f"{prefix}matching_results.tsv"
    candidates_path = o_dir / f"{prefix}candidate_pairs.tsv"
    zip_path = o_dir / f"{prefix}submission.zip"

    write_matching_results(all_predictions, matching_path)
    write_candidate_pairs(all_candidates_map, candidates_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_path, arcname="matching_results.tsv")
        zf.write(candidates_path, arcname="candidate_pairs.tsv")

    timing["write_s"] = round(time.time() - t0, 2)

    matching_size_mb = round(matching_path.stat().st_size / (1024 ** 2), 2)
    candidates_size_mb = round(candidates_path.stat().st_size / (1024 ** 2), 2)
    zip_size_mb = round(zip_path.stat().st_size / (1024 ** 2), 2)

    print(f"  matching_results.tsv:  {matching_size_mb} MB  -> {matching_path}")
    print(f"  candidate_pairs.tsv:   {candidates_size_mb} MB  -> {candidates_path}")
    print(f"  submission.zip:        {zip_size_mb} MB  -> {zip_path}")
    print(f"  Runtime: {timing['write_s']}s\n")

    # ------------------------------------------------------------------
    # [7/7] Validate output file format
    # ------------------------------------------------------------------
    print("[7/7] Validating output file format...")
    t0 = time.time()

    mr_check = pd.read_csv(matching_path, sep="\t", keep_default_na=False)
    assert list(mr_check.columns) == ["source1_entity_id", "matched_entity_ids"], (
        f"matching_results.tsv has wrong columns: {mr_check.columns.tolist()}"
    )
    assert len(mr_check) == total_s1, (
        f"matching_results.tsv row count={len(mr_check)}, expected {total_s1}"
    )
    assert mr_check["source1_entity_id"].is_unique, (
        "Duplicate source1_entity_id in matching_results.tsv!"
    )

    cp_check = pd.read_csv(candidates_path, sep="\t", keep_default_na=False)
    assert list(cp_check.columns) == ["source1_entity_id", "candidate_entity_ids"], (
        f"candidate_pairs.tsv has wrong columns: {cp_check.columns.tolist()}"
    )
    assert len(cp_check) == total_s1, (
        f"candidate_pairs.tsv row count={len(cp_check)}, expected {total_s1}"
    )
    assert cp_check["source1_entity_id"].is_unique, (
        "Duplicate source1_entity_id in candidate_pairs.tsv!"
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_contents = sorted(zf.namelist())
    assert "matching_results.tsv" in zip_contents, "matching_results.tsv missing from ZIP!"
    assert "candidate_pairs.tsv" in zip_contents, "candidate_pairs.tsv missing from ZIP!"

    timing["validation_s"] = round(time.time() - t0, 3)
    print(f"  PASS: matching_results.tsv: {len(mr_check):,} rows | correct columns | unique IDs")
    print(f"  PASS: candidate_pairs.tsv:  {len(cp_check):,} rows | correct columns | unique IDs")
    print(f"  PASS: submission.zip contains: {zip_contents}")
    print(f"  Runtime: {timing['validation_s']}s\n")

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    total_elapsed = round(time.time() - total_start, 2)
    timing["total_s"] = total_elapsed
    rss_peak = _rss_mb()

    cleanup_status = "retained for reuse"
    if cleanup_db:
        blocking_idx.cleanup()
        cleanup_status = "cleaned up database"
    else:
        blocking_idx.close()
        cleanup_status = f"preserved at {actual_db_path}"

    report: dict[str, Any] = {
        "mode": mode_str,
        "model": VALIDATED_MODEL,
        "threshold": threshold,
        "input_records": {
            "test_s1": total_s1,
            "test_s2": total_s2,
            "test_s3": total_s3,
        },
        "database": {
            "location": str(actual_db_path),
            "size_mb": db_size_mb,
            "cleanup_status": cleanup_status,
        },
        "blocking": {
            "total_candidates": total_candidates_generated,
            "avg_candidates_per_s1": round(total_candidates_generated / total_s1, 2),
        },
        "predictions": {
            "total_s1_entities": total_s1,
            "singleton_count": singleton_count,
            "matched_count": matched_count,
            "total_predicted_matches": total_predicted_matches,
            "singleton_rate_pct": round(singleton_count / total_s1 * 100, 2),
        },
        "output_files": {
            "matching_results_tsv": str(matching_path),
            "candidate_pairs_tsv": str(candidates_path),
            "submission_zip": str(zip_path),
            "matching_size_mb": matching_size_mb,
            "candidates_size_mb": candidates_size_mb,
            "zip_size_mb": zip_size_mb,
        },
        "memory": {
            "baseline_rss_mb": rss_start,
            "peak_rss_mb": rss_peak,
            "delta_rss_mb": round(rss_peak - rss_start, 1),
        },
        "timing": timing,
    }

    report_path = o_dir / f"{prefix}turn6_inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("TURN 6 COMPLETE")
    print("=" * 80)
    for label, val in [
        ("Mode", mode_str),
        ("Model", f"{VALIDATED_MODEL} (threshold={threshold})"),
        ("TEST S1 Entities", f"{total_s1:,}"),
        ("TEST S2 Indexed", f"{total_s2:,}"),
        ("TEST S3 Indexed", f"{total_s3:,}"),
        ("Index Location", f"{actual_db_path}"),
        ("Chunk Size (S2/S3)", f"{chunk_size_s23:,}"),
        ("Query Batch Size (S1)", f"{batch_size:,}"),
        ("Total Candidates (blocking)", f"{total_candidates_generated:,}"),
        ("Avg Candidates per S1", f"{report['blocking']['avg_candidates_per_s1']}"),
        ("Total Predicted Matches", f"{total_predicted_matches:,}"),
        ("Singletons", f"{singleton_count:,} ({report['predictions']['singleton_rate_pct']}%)"),
        ("matching_results.tsv", f"{matching_size_mb} MB"),
        ("candidate_pairs.tsv", f"{candidates_size_mb} MB"),
        ("submission.zip", f"{zip_size_mb} MB"),
        ("Peak RSS Memory", f"{rss_peak:.1f} MB"),
        ("Indexing Time", f"{timing.get('build_index_s', 0.0)}s"),
        ("Lookup/Inference Time", f"{timing.get('inference_s', 0.0)}s"),
        ("Total Pipeline Runtime", f"{total_elapsed}s"),
        ("Cleanup Status", f"{cleanup_status}"),
    ]:
        print(f"  {label:<35} {val}")
    print("=" * 80)
    print(f"\nInference report -> {report_path}")
    print(f"Submission ready -> {zip_path}\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turn 6: Full TEST inference and submission generation."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Run on first --smoke-n S1 entities only.",
    )
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=2000,
        help="S1 entity count for smoke test (default: 2000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_S1,
        help=f"S1 entities per processing batch (default: {DEFAULT_BATCH_S1}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=VALIDATED_THRESHOLD,
        help=f"Match probability threshold (default: {VALIDATED_THRESHOLD}).",
    )
    parser.add_argument(
        "--pool-rows",
        type=int,
        default=None,
        help=(
            "Limit S2+S3 pool to first N rows each (for fast smoke testing). "
            "Default None = full pool. Recommended: 200000 for smoke tests (~65s preprocessing)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for submission files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for preprocessed data and blocking index.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Ignore existing caches and rebuild from scratch.",
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to disk-backed SQLite database (e.g. /content/turn6_blocking.db on Colab).",
    )
    parser.add_argument(
        "--cleanup-db",
        action="store_true",
        default=False,
        help="Delete the SQLite database file after inference completes.",
    )

    args = parser.parse_args()
    o_dir = Path(args.output_dir) if args.output_dir else None
    c_dir = Path(args.cache_dir) if args.cache_dir else None

    run_inference(
        smoke_test=args.smoke_test,
        smoke_test_n=args.smoke_n,
        pool_rows=args.pool_rows,
        batch_size=args.batch_size,
        threshold=args.threshold,
        cache_dir=c_dir,
        output_dir=o_dir,
        db_path=args.db_path,
        cleanup_db=args.cleanup_db,
        force_rebuild=args.force_rebuild,
    )
