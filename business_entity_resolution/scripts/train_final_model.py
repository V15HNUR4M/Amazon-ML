"""
train_final_model.py — Final competition training pipeline for Turn 6.

Design intent
-------------
Trains the final LightGBM model on 100% of the available labeled competition
training data for the Amazon ML Business Entity Resolution challenge:

1. Loads labeled competition data:
   - train_source1.tsv
   - train_source2.tsv
   - train_source3.tsv
   - train_ground_truth.tsv
2. Reuses existing validated modules:
   - src.preprocessing (preprocess_records)
   - src.blocking (BlockingConfig, BlockingIndex, DiskBlockingIndex, build_candidates)
   - src.pair_builder (build_train_pairs)
   - src.features (compute_features, DEFAULT_FEATURE_NAMES)
   - src.model (LightGBMMatcher)
3. Zero validation withholding:
   - Trains on 100% of available labeled training entities (NO 80/20 train/val split).
4. Strict hyperparameter compliance:
   - LightGBMMatcher(n_estimators=100, learning_rate=0.05, random_state=42)
5. Validated threshold preservation:
   - Threshold fixed at 0.90 (no re-tuning on training data).
6. Non-destructive artifact persistence:
   - Final model: cache/turn6_final_matcher.joblib
   - Final config: cache/turn6_final_model_config.json
   - Leaves turn5_5_best_matcher.joblib and turn5_5_optimal_threshold.json untouched.
7. Environment & memory portability:
   - Uses BER_DATASET_ROOT for local and Colab execution.
   - Tracks and reports process RSS memory at every stage.

Usage:
------
# 1. Full training on 100% of competition data:
python scripts/train_final_model.py

# 2. Fast deterministic sanity check (e.g. 5,000 S1 queries, 100% of sample used for training):
python scripts/train_final_model.py --max-s1 5000
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
import sys
import time
from pathlib import Path
from typing import Any, Optional

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
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 2)


# ---------------------------------------------------------------------------
# Data Loading & Sampling
# ---------------------------------------------------------------------------


def load_training_data(
    cfg: Config,
    max_s1: Optional[int] = None,
    distractors: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    random_seed: int = 42,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load labeled training data (either full dataset or deterministic sample).

    Parameters
    ----------
    cfg : Config instance with paths to dataset files.
    max_s1 : Optional cap on S1 queries. None loads 100% of labeled S1 entities.
    distractors : Optional cap on distractors from S2 and S3.
    cache_dir : Directory to check for cached sample files.
    random_seed : Seed for deterministic sampling.
    force_rebuild : Ignore cache and reload from raw TSV files.

    Returns
    -------
    (s1, s2, s3, gt_df)
    """
    c_dir = cache_dir or cfg.CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)

    dist_count = distractors if distractors is not None else (max_s1 or 5000)

    # Check for existing sample cache if max_s1 is specified
    if max_s1 is not None and not force_rebuild:
        cache_file = c_dir / f"benchmark_sample_s1_{max_s1}_dist_{dist_count}.pkl"
        if cache_file.exists():
            logger.info("Loading cached training sample from %s...", cache_file)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            return data["s1"], data["s2"], data["s3"], data["gt"]

    t0 = time.time()

    # 1. Ground truth
    logger.info("Loading ground truth from %s...", cfg.TRAIN_GROUND_TRUTH)
    if max_s1 is not None:
        gt_df = pd.read_csv(
            cfg.TRAIN_GROUND_TRUTH,
            sep="\t",
            nrows=max_s1,
            keep_default_na=False,
        )
    else:
        gt_df = pd.read_csv(
            cfg.TRAIN_GROUND_TRUTH,
            sep="\t",
            keep_default_na=False,
        )
    needed_s1 = set(gt_df["source1_entity_id"])
    logger.info("  Loaded %d ground truth rows (RSS: %.1f MB).", len(gt_df), get_process_rss_mb())

    # 2. Source 1
    logger.info("Loading Source 1 entities from %s...", cfg.TRAIN_SOURCE1)
    if max_s1 is not None:
        # Stream S1 chunks to collect matching records
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
    else:
        s1 = pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", keep_default_na=False)

    logger.info("  Loaded %d Source 1 records (RSS: %.1f MB).", len(s1), get_process_rss_mb())

    # 3. Collect true match IDs from S2 and S3
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

    logger.info("  Ground truth requires %d S2 and %d S3 true match records.", len(needed_s2), len(needed_s3))

    # 4. Stream S2
    logger.info("Loading Source 2 records from %s...", cfg.TRAIN_SOURCE2)
    s2_rows: list[dict] = []
    rem_s2 = set(needed_s2)
    dist_s2_needed = dist_count // 2 if max_s1 is not None else 0

    for chunk in pd.read_csv(
        cfg.TRAIN_SOURCE2, sep="\t", chunksize=250000, keep_default_na=False
    ):
        if max_s1 is not None:
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
        else:
            # Full training: keep true matches + optional distractors
            matches = chunk[chunk["entity_id"].isin(needed_s2)]
            if len(matches) > 0:
                s2_rows.extend(matches.to_dict(orient="records"))
                rem_s2 -= set(matches["entity_id"])
            if len(rem_s2) == 0:
                break

    s2 = pd.DataFrame(s2_rows).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    logger.info("  Loaded %d Source 2 records (RSS: %.1f MB).", len(s2), get_process_rss_mb())

    # 5. Stream S3
    logger.info("Loading Source 3 records from %s...", cfg.TRAIN_SOURCE3)
    s3_rows: list[dict] = []
    rem_s3 = set(needed_s3)
    dist_s3_needed = dist_count // 2 if max_s1 is not None else 0

    for chunk in pd.read_csv(
        cfg.TRAIN_SOURCE3, sep="\t", chunksize=250000, keep_default_na=False
    ):
        if max_s1 is not None:
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
        else:
            # Full training: keep true matches + optional distractors
            matches = chunk[chunk["entity_id"].isin(needed_s3)]
            if len(matches) > 0:
                s3_rows.extend(matches.to_dict(orient="records"))
                rem_s3 -= set(matches["entity_id"])
            if len(rem_s3) == 0:
                break

    s3 = pd.DataFrame(s3_rows).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    logger.info("  Loaded %d Source 3 records (RSS: %.1f MB).", len(s3), get_process_rss_mb())

    # 6. Preprocess records
    logger.info("Preprocessing records across all sources...")
    s1_proc = preprocess_records(s1)
    s2_proc = preprocess_records(s2)
    s3_proc = preprocess_records(s3)

    # Cache sample if requested
    if max_s1 is not None:
        cache_file = c_dir / f"benchmark_sample_s1_{max_s1}_dist_{dist_count}.pkl"
        sample_data = {"s1": s1_proc, "s2": s2_proc, "s3": s3_proc, "gt": gt_df}
        with open(cache_file, "wb") as f:
            pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved training sample to %s", cache_file)

    logger.info("Data loading complete in %.2fs (RSS: %.1f MB).", time.time() - t0, get_process_rss_mb())
    return s1_proc, s2_proc, s3_proc, gt_df


# ---------------------------------------------------------------------------
# Candidate Generation & Pair Construction
# ---------------------------------------------------------------------------


def build_or_load_training_pairs(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt_df: pd.DataFrame,
    cfg: Config,
    max_s1: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    sample_neg_ratio: Optional[float] = None,
    random_seed: int = 42,
    use_disk_index: bool = False,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate candidate pairs, label them via ground truth, and compute 21 features.

    Parameters
    ----------
    s1, s2, s3 : Preprocessed source DataFrames.
    gt_df : Ground truth DataFrame.
    cfg : Config instance.
    max_s1 : Sample size cap (for cache naming).
    cache_dir : Directory to check for cached features/pairs.
    sample_neg_ratio : Ratio of negative to positive pairs.
    random_seed : Seed for negative sampling.
    use_disk_index : Use DiskBlockingIndex (SQLite) for candidate generation.
    force_rebuild : Ignore caches and recompute.

    Returns
    -------
    (pairs, features_df)
    """
    c_dir = cache_dir or cfg.CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing precomputed pairs and features (e.g. Turn 5.5 scale check cache)
    if max_s1 == 5000 and not force_rebuild:
        pairs_pkl = c_dir / "scale_check_pairs_s1_5000.pkl"
        feats_pkl = c_dir / "scale_check_features_s1_5000.pkl"
        if pairs_pkl.exists() and feats_pkl.exists():
            logger.info("Reusing cached pairs from %s...", pairs_pkl)
            with open(pairs_pkl, "rb") as f:
                pairs = pickle.load(f)
            logger.info("Reusing cached features from %s...", feats_pkl)
            with open(feats_pkl, "rb") as f:
                features_df = pickle.load(f)
            return pairs, features_df

    # 1. Blocking: Generate candidate pairs
    logger.info("Generating candidate pairs via 4-stage blocking...")
    t0 = time.time()
    candidates = build_candidates(s1, s2, s3, config=cfg, use_disk_index=use_disk_index)
    logger.info(
        "Generated %d candidate pairs in %.2fs (RSS: %.1f MB).",
        len(candidates),
        time.time() - t0,
        get_process_rss_mb(),
    )

    # 2. Labelled Pair Construction
    logger.info("Assembling labelled training pairs...")
    t0 = time.time()
    pairs = build_train_pairs(
        s1,
        s2,
        s3,
        candidates,
        gt_df,
        sample_neg_ratio=sample_neg_ratio,
        random_seed=random_seed,
    )
    logger.info(
        "Built %d labelled pairs (%d pos, %d neg) in %.2fs (RSS: %.1f MB).",
        len(pairs),
        (pairs["label"] == 1).sum() if len(pairs) > 0 else 0,
        (pairs["label"] == 0).sum() if len(pairs) > 0 else 0,
        time.time() - t0,
        get_process_rss_mb(),
    )

    # 3. Feature Extraction (21 Features)
    logger.info("Computing 21 pairwise numerical features...")
    t0 = time.time()
    features_df = compute_features(pairs)
    logger.info(
        "Computed %d features for %d pairs in %.2fs (RSS: %.1f MB).",
        features_df.shape[1],
        len(features_df),
        time.time() - t0,
        get_process_rss_mb(),
    )

    return pairs, features_df


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------


def run_final_training(
    max_s1: Optional[int] = None,
    distractors: Optional[int] = None,
    sample_neg_ratio: Optional[float] = None,
    use_disk_index: bool = False,
    cache_dir: Optional[Path] = None,
    output_model_path: Optional[Path] = None,
    output_config_path: Optional[Path] = None,
    random_seed: int = 42,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Execute final LightGBM model training on 100% of labeled training data.

    Parameters
    ----------
    max_s1 : If set, trains on 100% of this deterministic S1 sample (no validation split).
             If None, trains on 100% of all competition labeled entities.
    distractors : Distractor pool size when max_s1 is set.
    sample_neg_ratio : Optional negative downsampling ratio.
    use_disk_index : Use SQLite-backed DiskBlockingIndex for candidate generation.
    cache_dir : Directory for caching intermediate data.
    output_model_path : Path for saving final model (default: cache/turn6_final_matcher.joblib).
    output_config_path : Path for saving final config (default: cache/turn6_final_model_config.json).
    random_seed : Random seed for reproducibility.
    force_rebuild : Rebuild all caches from scratch.

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

    rss_baseline = get_process_rss_mb()
    peak_rss = rss_baseline
    total_start = time.time()
    timing: dict[str, float] = {}

    mode_label = (
        f"100% OF FULL COMPETITION DATA"
        if max_s1 is None
        else f"DETERMINISTIC SANITY SAMPLE ({max_s1:,} S1 ENTITIES, 100% TRAIN / 0% VAL)"
    )

    print("=" * 80)
    print(f"TURN 6: FINAL MODEL TRAINING PIPELINE [{mode_label}]")
    print("=" * 80)
    print(f"Architecture: 4-stage blocking | 21 features | LightGBM | Threshold: 0.90")
    print(f"Validation Split: NONE (100% of labeled training data available to model)")
    print(f"Baseline Process RSS: {rss_baseline:.2f} MB\n")

    # ------------------------------------------------------------------
    # Step 1: Load training data
    # ------------------------------------------------------------------
    print("[1/4] Loading labeled training data...")
    t0 = time.time()
    s1, s2, s3, gt_df = load_training_data(
        cfg=cfg,
        max_s1=max_s1,
        distractors=distractors,
        cache_dir=c_dir,
        random_seed=random_seed,
        force_rebuild=force_rebuild,
    )
    timing["data_loading_s"] = round(time.time() - t0, 2)
    peak_rss = max(peak_rss, get_process_rss_mb())

    print(f"  Loaded Records: {len(s1):,} S1 queries, {len(s2):,} S2 pool, {len(s3):,} S3 pool")
    print(f"  Ground Truth:   {len(gt_df):,} S1 entities labeled")
    print(f"  Runtime: {timing['data_loading_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

    # ------------------------------------------------------------------
    # Step 2: Build candidates, assemble pairs, and compute 21 features
    # ------------------------------------------------------------------
    print("[2/4] Generating candidates, assembling pairs, and computing 21 features...")
    t0 = time.time()
    pairs, features_df = build_or_load_training_pairs(
        s1=s1,
        s2=s2,
        s3=s3,
        gt_df=gt_df,
        cfg=cfg,
        max_s1=max_s1,
        cache_dir=c_dir,
        sample_neg_ratio=sample_neg_ratio,
        random_seed=random_seed,
        use_disk_index=use_disk_index,
        force_rebuild=force_rebuild,
    )
    timing["pair_and_feature_prep_s"] = round(time.time() - t0, 2)
    peak_rss = max(peak_rss, get_process_rss_mb())

    # Merge features into pairs DataFrame
    for col in DEFAULT_FEATURE_NAMES:
        if col in features_df.columns:
            pairs[col] = features_df[col].values

    total_pairs = len(pairs)
    total_pos = int((pairs["label"] == 1).sum()) if total_pairs > 0 else 0
    total_neg = int((pairs["label"] == 0).sum()) if total_pairs > 0 else 0
    pos_pct = round(total_pos / total_pairs * 100, 2) if total_pairs > 0 else 0.0

    print(f"  Total Labelled Pairs: {total_pairs:,} ({total_pos:,} pos [{pos_pct}%], {total_neg:,} neg [{100-pos_pct}%])")
    print(f"  Feature Matrix:       {features_df.shape[1]} features across {len(features_df):,} rows")
    print(f"  Runtime: {timing['pair_and_feature_prep_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

    # ------------------------------------------------------------------
    # Step 3: Train LightGBM on 100% of labeled training data (NO SPLIT)
    # ------------------------------------------------------------------
    print("[3/4] Training LightGBM on 100% of labeled training data (NO validation split)...")
    print(f"  Hyperparameters: n_estimators=100, learning_rate=0.05, random_state={random_seed}")
    t0 = time.time()

    matcher = LightGBMMatcher(
        n_estimators=100,
        learning_rate=0.05,
        random_state=random_seed,
    )
    matcher.fit(pairs, pairs["label"])
    timing["training_s"] = round(time.time() - t0, 3)
    peak_rss = max(peak_rss, get_process_rss_mb())

    assert matcher.is_fitted, "Fatal error: LightGBMMatcher is not fitted after fit() call!"
    print(f"  LightGBM Training Time: {timing['training_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

    # ------------------------------------------------------------------
    # Step 4: Persist Model Artifact and Metadata Config
    # ------------------------------------------------------------------
    print("[4/4] Persisting final model artifact and metadata...")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    matcher.save(model_out)

    # Verification: Load model artifact back to confirm integrity
    loaded_matcher = LightGBMMatcher.load(model_out)
    assert isinstance(loaded_matcher, LightGBMMatcher), f"Unexpected model type: {type(loaded_matcher)}"
    assert loaded_matcher.is_fitted, "Loaded model reports is_fitted=False!"

    # Verify probability inference on sample batch
    sample_batch = pairs.head(min(10, len(pairs)))
    test_proba = loaded_matcher.predict_proba(sample_batch)
    assert len(test_proba) == len(sample_batch), "Probability vector length mismatch!"
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
        "number_of_s1_training_entities": len(s1),
        "number_of_labelled_pairs": total_pairs,
        "positive_pair_count": total_pos,
        "negative_pair_count": total_neg,
        "feature_count": len(DEFAULT_FEATURE_NAMES),
        "feature_names": list(DEFAULT_FEATURE_NAMES),
        "top_feature_by_gain": sorted_fi[0] if sorted_fi else None,
        "training_runtime_s": timing["training_s"],
        "total_pipeline_runtime_s": total_pipeline_time,
        "peak_rss_mb": round(peak_rss, 2),
        "is_full_training": (max_s1 is None),
        "validation_split": "none (100% of labeled training data used)",
    }

    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 80)
    print("FINAL MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Model artifact:   {model_out}")
    print(f"  Metadata config:  {config_out}")
    print(f"  S1 Entities:      {len(s1):,}")
    print(f"  Labelled Pairs:   {total_pairs:,} ({total_pos:,} positives, {total_neg:,} negatives)")
    print(f"  Validation Split: NONE (100% of labeled training data used)")
    print(f"  Model Type:       LightGBM (n_estimators=100, lr=0.05, random_state={random_seed})")
    print(f"  Threshold:        0.90 (validated from Turn 5.5)")
    print(f"  Training Time:    {timing['training_s']}s")
    print(f"  Total Runtime:    {total_pipeline_time}s")
    print(f"  Peak RSS:         {peak_rss:.2f} MB")
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
        # If neither --full nor --max-s1 is provided, default to full training on competition data
        logger.info(
            "Neither --full nor --max-s1 specified. Defaulting to full competition training on 100% of data."
        )

    run_final_training(
        max_s1=effective_max_s1,
        distractors=args.distractors,
        sample_neg_ratio=args.sample_neg_ratio,
        use_disk_index=args.use_disk_index,
        cache_dir=c_dir,
        output_model_path=out_model,
        output_config_path=out_cfg,
        random_seed=args.random_seed,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
