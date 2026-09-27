"""
train_gpu_model.py — Final Turn 7 Full-Scale Training Pipeline.

Safety Notice:
Per Competition Directive, this script requires explicit confirmation flag
`--confirm-full-training` when running without `--max-s1`.

Design:
- Uses SQLite DiskGPUBlockingIndex to stream candidate records (S2, S3, Ground Truth)
  with bounded RAM (< 4 GB peak).
- Computes 34 Turn 7 pairwise features in streaming batches directly to disk-backed binary files.
- Trains final LightGBM model with GPU acceleration if CUDA is available, or multi-threaded CPU.
- Persists final model to cache/turn7_final_matcher.joblib and config to cache/turn7_final_model_config.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import psutil

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.evaluation import ground_truth_to_dict
from src.gpu.gpu_blocking import (
    DiskGPUBlockingIndex,
    GPUBlockingConfig,
    build_gpu_candidates,
)
from src.gpu.gpu_features import (
    BASELINE_FEATURE_NAMES,
    TURN7_FEATURE_NAMES,
    TokenIDFStore,
    compute_gpu_features,
)
from src.gpu.gpu_utils import (
    DeviceContext,
    clear_gpu_cache,
    get_device_info,
    get_gpu_memory_mb,
    get_process_rss_mb,
    is_cuda_available,
)
from src.model import LightGBMMatcher
from src.pair_builder import build_train_pairs
from src.preprocessing import preprocess_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_streaming_gpu_index(
    cfg: Config,
    db_path: Path,
    preprocess_chunk_size: int = 50_000,
    force_rebuild: bool = False,
) -> DiskGPUBlockingIndex:
    """Stream and index candidate pool (S2 + S3) and ground truth into SQLite."""
    b_cfg = GPUBlockingConfig(top_k_per_s1=60, max_candidates_per_block=150)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not force_rebuild:
        idx = DiskGPUBlockingIndex(db_path=db_path, config=b_cfg)
        if idx.is_finalized and idx.total_records_indexed > 0:
            logger.info("Loaded existing finalized DiskGPUBlockingIndex (%d records)", idx.total_records_indexed)
            return idx
        idx.close()

    if db_path.exists():
        db_path.unlink(missing_ok=True)

    idx = DiskGPUBlockingIndex(db_path=db_path, config=b_cfg)
    sources = [
        (cfg.TRAIN_SOURCE2, "Source 2"),
        (cfg.TRAIN_SOURCE3, "Source 3"),
    ]

    for path, label in sources:
        if not path.exists():
            continue
        logger.info("Indexing %s from %s...", label, path)
        reader = pd.read_csv(path, sep="\t", dtype=str, chunksize=preprocess_chunk_size, keep_default_na=False)
        for chunk in reader:
            proc = preprocess_records(chunk)
            idx.add_records(proc)
            del proc, chunk
            gc.collect()

    if cfg.TRAIN_GROUND_TRUTH.exists():
        logger.info("Indexing ground truth from %s...", cfg.TRAIN_GROUND_TRUTH)
        gt_reader = pd.read_csv(cfg.TRAIN_GROUND_TRUTH, sep="\t", dtype=str, chunksize=100_000, keep_default_na=False)
        for chunk in gt_reader:
            idx.add_ground_truth(chunk)
            del chunk
            gc.collect()

    idx.finalize()
    logger.info("DiskGPUBlockingIndex finalized: %d records indexed.", idx.total_records_indexed)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn 7 Full Training Pipeline.")
    parser.add_argument("--max-s1", type=int, default=None, help="Cap on S1 entities for testing.")
    parser.add_argument("--batch-size", type=int, default=10000, help="S1 batch size for chunked streaming.")
    parser.add_argument("--confirm-full-training", action="store_true", help="Explicit confirmation for 2.2M run.")
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()

    if args.max_s1 is None and not args.confirm_full_training:
        print("\n" + "!" * 80)
        print("SAFETY GUARD TRIGGERED:")
        print("You are attempting to launch full 2.2M training without '--confirm-full-training'.")
        print("Per directive: Complete the 5k validation benchmark and 50k GPU benchmark first.")
        print("To proceed with a bounded test, use e.g. '--max-s1 5000'.")
        print("!" * 80 + "\n")
        sys.exit(1)

    cfg = Config()
    dev_info = get_device_info()
    print("=" * 80)
    print("TURN 7 MODEL TRAINING")
    print("=" * 80)
    print(f"Device: {dev_info['device_name']} | Max S1: {args.max_s1 or 'FULL (2.2M)'}")
    print(f"Features: 34 (21 baseline + 13 Turn 7) | Batch Size: {args.batch_size}")
    print(f"Current Process RSS: {get_process_rss_mb():.2f} MB\n")

    db_path = cfg.CACHE_DIR / "turn7_streaming_blocking.db"
    disk_idx = build_streaming_gpu_index(cfg, db_path=db_path)

    # Output paths
    model_path = cfg.CACHE_DIR / "turn7_final_matcher.joblib"
    config_path = cfg.CACHE_DIR / "turn7_final_model_config.json"
    binary_dir = cfg.CACHE_DIR / "turn7_train_binary"
    binary_dir.mkdir(parents=True, exist_ok=True)

    tag = f"s1_{args.max_s1}" if args.max_s1 is not None else "s1_full"
    x_bin_path = binary_dir / f"features_{tag}.bin"
    y_bin_path = binary_dir / f"labels_{tag}.bin"

    if x_bin_path.exists():
        x_bin_path.unlink(missing_ok=True)
    if y_bin_path.exists():
        y_bin_path.unlink(missing_ok=True)

    logger.info("Initializing TokenIDFStore...")
    idf_store = TokenIDFStore()

    total_s1 = 0
    total_pairs = 0
    total_pos = 0
    total_neg = 0
    sample_pairs_df: Optional[pd.DataFrame] = None
    t0 = time.time()

    logger.info(
        "Starting chunked candidate generation and feature extraction (batch_size=%d, max_s1=%s)...",
        args.batch_size,
        args.max_s1,
    )

    try:
        f_x = open(x_bin_path, "wb")
        f_y = open(y_bin_path, "wb")

        s1_reader = pd.read_csv(
            cfg.TRAIN_SOURCE1,
            sep="\t",
            dtype=str,
            chunksize=args.batch_size,
            keep_default_na=False,
        )

        accum = 0
        batch_idx = 0
        for chunk in s1_reader:
            if args.max_s1 is not None and accum >= args.max_s1:
                break
            if args.max_s1 is not None and accum + len(chunk) > args.max_s1:
                chunk = chunk.iloc[: args.max_s1 - accum]
            accum += len(chunk)
            batch_idx += 1

            s1_proc = preprocess_records(chunk)
            candidates = disk_idx.query_records(s1_proc)

            if len(candidates) > 0:
                cand_ids = candidates["candidate_id"].unique()
                cand_records = disk_idx.get_candidate_records(cand_ids)
                if cand_records:
                    cand_df = pd.DataFrame(cand_records)
                    s2_mini = cand_df[cand_df["source"] == "S2"].drop(columns=["source"], errors="ignore")
                    s3_mini = cand_df[cand_df["source"] == "S3"].drop(columns=["source"], errors="ignore")

                    s1_ids = list(s1_proc["entity_id"].astype(str))
                    gt_batch = disk_idx.get_ground_truth_for_s1(s1_ids)

                    pairs_batch = build_train_pairs(
                        s1_proc,
                        s2_mini,
                        s3_mini,
                        candidates,
                        gt_batch,
                        random_seed=args.random_seed,
                    )

                    if len(pairs_batch) > 0:
                        feats_batch = compute_gpu_features(
                            pairs_batch,
                            idf_store=idf_store,
                            include_turn7_features=True,
                        )

                        x_bytes = feats_batch.values.astype(np.float32).tobytes()
                        y_bytes = pairs_batch["label"].values.astype(np.int32).tobytes()
                        f_x.write(x_bytes)
                        f_y.write(y_bytes)

                        b_pos = int((pairs_batch["label"] == 1).sum())
                        b_neg = int((pairs_batch["label"] == 0).sum())
                        total_pos += b_pos
                        total_neg += b_neg
                        total_pairs += len(pairs_batch)

                        if sample_pairs_df is None:
                            sample_pairs_df = pairs_batch.head(10).copy()

            total_s1 += len(chunk)
            if batch_idx % 5 == 0 or args.max_s1 is not None:
                logger.info(
                    "Batch %d: S1=%d, Pairs=%d (pos=%d, neg=%d) | RSS=%.1f MB",
                    batch_idx,
                    total_s1,
                    total_pairs,
                    total_pos,
                    total_neg,
                    get_process_rss_mb(),
                )

        f_x.close()
        f_y.close()
    finally:
        disk_idx.close()

    elapsed = time.time() - t0
    logger.info(
        "Candidate generation and feature extraction finished in %.1fs: %d S1 entities -> %d pairs (%d pos, %d neg)",
        elapsed,
        total_s1,
        total_pairs,
        total_pos,
        total_neg,
    )

    if total_pairs == 0:
        logger.warning("No candidate pairs generated. Aborting model training.")
        return

    # Train LightGBM model on disk-backed feature store
    logger.info("Training LightGBM model on %d pairwise features...", total_pairs)
    n_features = len(TURN7_FEATURE_NAMES)
    x_memmap = np.memmap(x_bin_path, dtype=np.float32, mode="r", shape=(total_pairs, n_features))
    y_memmap = np.memmap(y_bin_path, dtype=np.int32, mode="r", shape=(total_pairs,))

    x_df = pd.DataFrame(x_memmap, columns=TURN7_FEATURE_NAMES)
    y_series = pd.Series(y_memmap)

    matcher = LightGBMMatcher(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        random_state=args.random_seed,
        class_weight="balanced",
        feature_names=TURN7_FEATURE_NAMES,
    )
    t_train = time.time()
    matcher.fit(x_df, y_series)
    logger.info("Model fitted in %.2fs. Saving model to %s...", time.time() - t_train, model_path)
    matcher.save(model_path)

    # Save model config metadata
    model_config = {
        "turn": 7,
        "features": TURN7_FEATURE_NAMES,
        "n_features": len(TURN7_FEATURE_NAMES),
        "total_s1_trained": total_s1,
        "total_pairs_trained": total_pairs,
        "total_positives": total_pos,
        "total_negatives": total_neg,
        "optimal_threshold": 0.90,
        "training_time_s": round(time.time() - t_train, 2),
        "total_time_s": round(time.time() - t0, 2),
        "peak_rss_mb": get_process_rss_mb(),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    print(f"\nModel successfully saved to: {model_path}")
    print(f"Config successfully saved to: {config_path}")
    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
