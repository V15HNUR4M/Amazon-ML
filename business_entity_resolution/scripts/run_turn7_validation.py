"""
run_turn7_validation.py — Turn 7 GPU-Accelerated & Accuracy-Optimized Validation Experiment.

Usage:
------
# Artificial-pool benchmark (existing behaviour, unchanged):
python scripts/run_turn7_validation.py --sample-source1 5000 --distractors 5000

# Full-reference corpus (NEW — Colab memory-safe sharded mode):
python scripts/run_turn7_validation.py --sharded --sample-source1 5000
python scripts/run_turn7_validation.py --sharded --sample-source1 10000

# Process-boundary isolation (unchanged):
python scripts/run_turn7_validation.py --ingest-only --output-dir output/run1
python scripts/run_turn7_validation.py --train-only  --output-dir output/run1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.gpu.gpu_blocking import GPUBlockingConfig
from src.gpu.gpu_pipeline import Turn7Config, Turn7Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_real_full_corpus(cfg: Config, sample_s1_count: int, random_seed: int):
    """Load S1 sample + REAL FULL S2/S3 reference universe for sharded runs.

    Does NOT use the artificial benchmark pool generator.
    S2 and S3 are loaded in their entirety from disk.
    S1 is sampled deterministically.
    """
    import numpy as np
    import pandas as pd

    logger.info("Loading REAL full S2 corpus from %s ...", cfg.TRAIN_SOURCE2)
    s2 = pd.read_csv(cfg.TRAIN_SOURCE2, sep="\t", keep_default_na=False, low_memory=False)
    logger.info("  S2: %d records", len(s2))

    logger.info("Loading REAL full S3 corpus from %s ...", cfg.TRAIN_SOURCE3)
    s3 = pd.read_csv(cfg.TRAIN_SOURCE3, sep="\t", keep_default_na=False, low_memory=False)
    logger.info("  S3: %d records", len(s3))

    logger.info("Loading ground truth from %s ...", cfg.TRAIN_GROUND_TRUTH)
    gt = pd.read_csv(cfg.TRAIN_GROUND_TRUTH, sep="\t", keep_default_na=False)
    logger.info("  GT: %d rows", len(gt))

    logger.info("Loading S1 from %s ...", cfg.TRAIN_SOURCE1)
    s1_full = pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", keep_default_na=False, low_memory=False)
    logger.info("  S1 full: %d records", len(s1_full))

    # Deterministic sample of S1
    rng = np.random.RandomState(random_seed)
    if sample_s1_count >= len(s1_full):
        s1 = s1_full.copy()
    else:
        idx = rng.choice(len(s1_full), size=sample_s1_count, replace=False)
        idx.sort()
        s1 = s1_full.iloc[idx].reset_index(drop=True)

    # Restrict ground truth to the sampled S1 entities
    s1_ids = set(s1["entity_id"].astype(str))
    s1_col = "source1_entity_id" if "source1_entity_id" in gt.columns else "s1_id"
    gt_sampled = gt[gt[s1_col].astype(str).isin(s1_ids)].copy()

    logger.info(
        "Sampled S1: %d records | Reference pool: %d S2 + %d S3 = %d total | GT rows: %d",
        len(s1),
        len(s2),
        len(s3),
        len(s2) + len(s3),
        len(gt_sampled),
    )
    return s1, s2, s3, gt_sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Turn 7 GPU-accelerated validation benchmark.")
    parser.add_argument("--sample-source1", type=int, default=5000)
    parser.add_argument("--distractors", type=int, default=5000)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--s1-batch-size", type=int, default=25000, help="S1 batch size for streaming pipeline")
    parser.add_argument("--non-streaming", action="store_true", help="Disable streaming mode (use monolithic pipeline)")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume support in streaming mode")
    parser.add_argument("--disable-turn7-features", action="store_true")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Execute Stages 1-4 (streaming ingestion to Parquet chunks) and exit cleanly",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Execute Stages 5-7 (LightGBM training & validation) from pre-existing Parquet chunks",
    )
    # --- NEW: Sharded full-reference-corpus flags ---
    parser.add_argument(
        "--sharded",
        action="store_true",
        help=(
            "Use sharded reference index against the REAL FULL S2+S3 corpus "
            "(Colab memory-safe). Ignores --distractors."
        ),
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=200_000,
        help="Number of reference records per shard (default: 200000). "
             "At ~3 KB/record each shard uses ~600 MB RAM.",
    )
    parser.add_argument(
        "--overwrite-shards",
        action="store_true",
        help="Force re-build of on-disk shard Parquet files even if they already exist.",
    )

    args = parser.parse_args()
    cfg = Config()
    c_dir = Path(args.cache_dir) if args.cache_dir else cfg.CACHE_DIR
    o_dir = Path(args.output_dir) if args.output_dir else cfg.OUTPUT_DIR

    # 1. Configure Turn 7 Pipeline
    b_cfg = GPUBlockingConfig(top_k_per_s1=args.top_k)
    t7_cfg = Turn7Config(
        blocking_config=b_cfg,
        include_turn7_features=not args.disable_turn7_features,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        random_state=args.random_seed,
        val_size=args.val_size,
        sample_s1_count=args.sample_source1,
        distractor_count=args.distractors,
        s1_batch_size=args.s1_batch_size,
        streaming=not args.non_streaming,
        resume=not args.no_resume,
    )
    pipeline = Turn7Pipeline(config=t7_cfg)

    # 2. If train-only, execute Stages 5-7 directly without loading S1/S2/S3
    if args.train_only:
        logger.info("Executing training-only stage (Stages 5-7) from Parquet chunks in clean process...")
        pipeline.run_training_from_chunks(output_dir=o_dir, cache_dir=c_dir)
        return

    # 3. Load data
    if args.sharded:
        # Full-reference corpus path: load real S1 sample + full S2/S3
        logger.info(
            "SHARDED MODE: Loading real full reference corpus (S2+S3) with %d S1 records...",
            args.sample_source1,
        )
        s1, s2, s3, gt = _load_real_full_corpus(cfg, args.sample_source1, args.random_seed)
    else:
        # Legacy artificial-pool benchmark (unchanged behaviour)
        from scripts.run_validation_scale_check import load_or_create_large_scale_sample  # noqa: PLC0415
        logger.info("Loading deterministic validation sample (%d S1 queries)...", args.sample_source1)
        s1, s2, s3, gt = load_or_create_large_scale_sample(
            sample_s1_count=args.sample_source1,
            distractor_count=args.distractors,
            cache_dir=c_dir,
            random_seed=args.random_seed,
        )

    # 4. Execute Turn 7 Pipeline
    pipeline.run_validation_experiment(
        s1=s1,
        s2=s2,
        s3=s3,
        gt_df=gt,
        output_dir=o_dir,
        cache_dir=c_dir,
        ingest_only=args.ingest_only,
        use_sharded=args.sharded,
        shard_size=args.shard_size,
        overwrite_shards=args.overwrite_shards,
    )


if __name__ == "__main__":
    main()
