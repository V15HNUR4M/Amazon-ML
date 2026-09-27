"""
run_full_ref_validation.py — Test A & Test B: Full-Reference Corpus Validation.

Runs the sharded blocking pipeline against the REAL COMPLETE 10,320,219-record
reference universe (S2 + S3), with a small S1 sample to validate Colab memory safety.

Usage
-----
# Test A: 5k S1 records against full reference corpus
python scripts/run_full_ref_validation.py --test A --s1-count 5000 --output-dir output/full_ref_5k

# Test B: 10k S1 records against full reference corpus
python scripts/run_full_ref_validation.py --test B --s1-count 10000 --output-dir output/full_ref_10k

# Ingest-only (useful for estimating before running training):
python scripts/run_full_ref_validation.py --test A --s1-count 5000 --ingest-only

# Skip training; just get candidate recall & memory estimates:
python scripts/run_full_ref_validation.py --test A --s1-count 5000 --ingest-only \\
    --output-dir output/full_ref_5k

Environment variables
---------------------
BER_DATASET_ROOT: Override dataset root path (useful in Colab with Google Drive).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import psutil

from src.config import Config
from src.gpu.gpu_blocking import GPUBlockingConfig
from src.gpu.gpu_pipeline import Turn7Config, Turn7Pipeline
from src.gpu.gpu_utils import get_gpu_memory_mb, get_process_rss_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _rss_gb() -> float:
    return get_process_rss_mb() / 1024.0


def _vram_gb() -> float:
    info = get_gpu_memory_mb()
    return info.get("peak_allocated_mb", 0.0) / 1024.0


def load_s1_and_gt(cfg: Config, s1_count: int, random_seed: int) -> tuple:
    """Load ONLY S1 sample and ground truth. S2/S3 are streamed from TSV by the pipeline.

    MEMORY-SAFE: Never loads the 10.3M reference corpus into RAM.
    Peak RAM from this function: ~800 MB (S1_full + GT).
    """
    t0 = time.time()

    print(f"Loading S1 sample ({s1_count:,} from {cfg.TRAIN_SOURCE1}) ...")
    s1_full = pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", keep_default_na=False, low_memory=False)
    rng = np.random.RandomState(random_seed)
    n = min(s1_count, len(s1_full))
    idx = rng.choice(len(s1_full), size=n, replace=False)
    idx.sort()
    s1 = s1_full.iloc[idx].reset_index(drop=True)
    del s1_full
    gc.collect()
    print(f"  S1 sample:  {len(s1):,} records | {_rss_gb():.2f} GB RSS")

    print(f"Loading ground truth from {cfg.TRAIN_GROUND_TRUTH} ...")
    gt = pd.read_csv(cfg.TRAIN_GROUND_TRUTH, sep="\t", keep_default_na=False)
    print(f"  GT:         {len(gt):,} rows")

    # Filter ground truth to only the sampled S1 entities
    s1_ids = set(s1["entity_id"].astype(str))
    s1_col = "source1_entity_id" if "source1_entity_id" in gt.columns else "s1_id"
    gt_sampled = gt[gt[s1_col].astype(str).isin(s1_ids)].copy()
    del gt
    gc.collect()

    load_time = round(time.time() - t0, 2)
    print(f"\nData loading complete in {load_time}s:")
    print(f"  S1 sample:      {len(s1):,} records")
    print(f"  S2+S3 ref:      10,320,219 records (streamed from TSV — not loaded into RAM)")
    print(f"  GT rows:        {len(gt_sampled):,}")
    print(f"  Current RSS:    {_rss_gb():.2f} GB  [S2/S3 not loaded]")
    print()

    return s1, gt_sampled


# Keep the original loader name as an alias for any callers that might use it
def load_full_corpus(cfg: Config, s1_count: int, random_seed: int) -> tuple:
    """Deprecated alias for load_s1_and_gt. S2/S3 are no longer loaded into RAM."""
    s1, gt = load_s1_and_gt(cfg, s1_count, random_seed)
    return s1, None, None, gt


def estimate_full_scale(
    ingestion_summary: dict,
    total_s1_full: int = 2_206_821,
) -> dict:
    """Extrapolate measured metrics to predict full 2.2M S1 run behaviour."""
    s1_sample = ingestion_summary.get("source1_entities", 1)
    candidates = ingestion_summary.get("total_candidates", 0)
    scale_factor = total_s1_full / s1_sample

    cands_per_s1 = candidates / max(s1_sample, 1)
    est_total_cands = cands_per_s1 * total_s1_full

    # Training memory: X_train = N_pairs × 34 × 4 bytes
    train_frac = ingestion_summary.get("train_pairs", 0) / max(candidates, 1)
    est_train_pairs = est_total_cands * train_frac
    est_xtrain_gb = (est_train_pairs * 34 * 4) / (1024**3)

    # Disk: measured disk per candidate pair
    disk_bytes = ingestion_summary.get("total_disk_bytes", 0)
    bytes_per_pair = disk_bytes / max(candidates, 1)
    est_disk_gb = bytes_per_pair * est_total_cands / (1024**3)

    # Timing: blocking time per S1 entity (sharded, so scales with S1 × n_shards)
    blocking_s = ingestion_summary.get("timing_ingest", {}).get("blocking_s", 0.0)
    blocking_per_s1 = blocking_s / max(s1_sample, 1)
    est_blocking_h = (blocking_per_s1 * total_s1_full) / 3600

    return {
        "s1_sample_size": s1_sample,
        "total_s1_full": total_s1_full,
        "scale_factor": round(scale_factor, 1),
        "candidates_per_s1": round(cands_per_s1, 1),
        "estimated_total_candidate_pairs": int(est_total_cands),
        "estimated_train_pairs": int(est_train_pairs),
        "estimated_X_train_GB": round(est_xtrain_gb, 2),
        "estimated_disk_GB": round(est_disk_gb, 2),
        "estimated_blocking_hours": round(est_blocking_h, 2),
    }


def check_memory_safety(
    ingestion_summary: dict,
    colab_ram_gb: float = 12.7,
    colab_vram_gb: float = 15.0,
) -> dict:
    """Emit GO / NOT-READY verdict based on measured peak RAM and scaling projections."""
    peak_rss_gb = ingestion_summary.get("peak_rss_mb", 0) / 1024.0
    peak_vram_gb = ingestion_summary.get("peak_vram_mb", 0) / 1024.0
    est = estimate_full_scale(ingestion_summary)

    # Blocking memory is bounded by shard_size × 3.08 KB + S1 batch + baseline
    # (already measured in peak_rss_mb)
    blocking_safe = peak_rss_gb < colab_ram_gb * 0.85

    # Training memory: is estimated X_train safe?
    # Process 2 (training-only) peak = X_train + y_train + LightGBM dataset
    est_train_peak_gb = est["estimated_X_train_GB"] + 0.31 + 3.50
    training_safe = est_train_peak_gb < colab_ram_gb

    vram_safe = peak_vram_gb < colab_vram_gb * 0.80

    verdict = "GO" if (blocking_safe and vram_safe) else "NOT-READY"
    if not training_safe:
        verdict = "NOT-READY (training stage)"

    return {
        "verdict": verdict,
        "blocking_peak_rss_gb": round(peak_rss_gb, 3),
        "blocking_peak_vram_gb": round(peak_vram_gb, 3),
        "colab_ram_budget_gb": colab_ram_gb,
        "colab_vram_budget_gb": colab_vram_gb,
        "blocking_ram_safe": blocking_safe,
        "training_est_peak_gb": round(est_train_peak_gb, 2),
        "training_ram_safe": training_safe,
        "vram_safe": vram_safe,
        "scale_estimate": est,
    }


def run_test(
    test_label: str,
    s1_count: int,
    shard_size: int,
    s1_batch_size: int,
    top_k: int,
    val_size: float,
    random_seed: int,
    output_dir: Path,
    ingest_only: bool,
    overwrite_shards: bool,
    train_only: bool,
) -> dict:
    cfg = Config()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.CACHE_DIR

    print("=" * 80)
    print(f"FULL REFERENCE CORPUS VALIDATION — TEST {test_label}")
    print(f"S1 Sample: {s1_count:,} records | Full S2+S3 reference: 10,320,219 records")
    print(f"Shard size: {shard_size:,} records | S1 batch: {s1_batch_size:,}")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    if train_only:
        print("Train-only mode: loading manifest and running LightGBM training from Parquet chunks.")
        b_cfg = GPUBlockingConfig(top_k_per_s1=top_k)
        t7_cfg = Turn7Config(
            blocking_config=b_cfg,
            include_turn7_features=True,
            n_estimators=120,
            learning_rate=0.05,
            random_state=random_seed,
            val_size=val_size,
            s1_batch_size=s1_batch_size,
            streaming=True,
            resume=True,
        )
        pipeline = Turn7Pipeline(config=t7_cfg)
        result = pipeline.run_training_from_chunks(output_dir=output_dir, cache_dir=cache_dir)
        return result

    # -----------------------------------------------------------------------
    # MEMORY-SAFE LOAD: Only S1 + GT loaded into RAM.
    # S2/S3 are streamed chunk-by-chunk from TSV during shard building.
    # Peak RAM savings: ~12-15 GB versus the previous full-DF approach.
    # -----------------------------------------------------------------------
    rss_before_load = _rss_gb()
    s1, gt = load_s1_and_gt(cfg, s1_count, random_seed)
    rss_after_load = _rss_gb()
    print(f"RSS after S1+GT load: {rss_after_load:.2f} GB (delta: +{rss_after_load - rss_before_load:.2f} GB)")
    print(f"Available system RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB\n")

    # Build pipeline
    b_cfg = GPUBlockingConfig(top_k_per_s1=top_k)
    t7_cfg = Turn7Config(
        blocking_config=b_cfg,
        include_turn7_features=True,
        n_estimators=120,
        learning_rate=0.05,
        random_state=random_seed,
        val_size=val_size,
        s1_batch_size=s1_batch_size,
        streaming=True,
        resume=True,
    )
    pipeline = Turn7Pipeline(config=t7_cfg)

    # Pass S2/S3 as TSV paths — the pipeline streams them directly without loading full DFs.
    t_total = time.time()
    result = pipeline.run_validation_experiment(
        s1=s1,
        gt_df=gt,
        output_dir=output_dir,
        cache_dir=cache_dir,
        ingest_only=ingest_only,
        use_sharded=True,
        shard_size=shard_size,
        overwrite_shards=overwrite_shards,
        s2_path=cfg.TRAIN_SOURCE2,
        s3_path=cfg.TRAIN_SOURCE3,
    )
    total_runtime = round(time.time() - t_total, 2)

    # --- Report ---
    print("\n" + "=" * 80)
    print(f"TEST {test_label} RESULTS SUMMARY")
    print("=" * 80)

    if ingest_only:
        ingest = result
        print(f"Mode:                   Ingest-only (no training)")
        print(f"S1 Entities:            {ingest.get('source1_entities', 0):,}")
        print(f"Reference Pool (S2+S3): {ingest.get('candidate_pool_size', 0):,}")
        print(f"Total Candidate Pairs:  {ingest.get('total_candidates', 0):,}")
        cand_recall = ingest.get("candidate_recall", 0.0)
        print(f"Candidate Recall:       {cand_recall:.2f}%")
        zero_cand = ingest.get("zero_candidate_s1", "N/A")
        print(f"Peak RSS:               {ingest.get('peak_rss_mb', 0)/1024:.2f} GB")
        print(f"Peak VRAM:              {ingest.get('peak_vram_mb', 0)/1024:.2f} GB")
        print(f"Total Runtime:          {total_runtime:.2f}s")
        print(f"Disk Output:            {ingest.get('total_disk_mb', 0):.2f} MB")
        print(f"Shards:                 {ingest.get('n_shards', '?')} × {shard_size:,} records")
        print()

        # Memory safety check
        safety = check_memory_safety(ingest)
        scale = safety["scale_estimate"]
        print("Memory Safety Assessment:")
        print(f"  Blocking peak RSS:        {safety['blocking_peak_rss_gb']:.2f} GB (budget: {safety['colab_ram_budget_gb']} GB)")
        print(f"  Blocking RAM safe:        {'YES' if safety['blocking_ram_safe'] else 'NO'}")
        print(f"  Est. training peak (2.2M): {safety['training_est_peak_gb']:.2f} GB")
        print(f"  Training RAM safe:        {'YES' if safety['training_ram_safe'] else 'NO'}")
        print(f"  VRAM safe:                {'YES' if safety['vram_safe'] else 'NO'}")
        print()
        print(f"Full 2.2M Scale Estimates:")
        print(f"  Total candidate pairs:    {scale['estimated_total_candidate_pairs']:,}")
        print(f"  Train pairs:              {scale['estimated_train_pairs']:,}")
        print(f"  X_train memory:           {scale['estimated_X_train_GB']:.2f} GB")
        print(f"  Disk usage:               {scale['estimated_disk_GB']:.2f} GB")
        print(f"  Blocking time:            {scale['estimated_blocking_hours']:.2f} hours")
        print()
        print(f"VERDICT: {safety['verdict']}")
        print("=" * 80)

        # Save full report to JSON
        report = {
            "test": test_label,
            "s1_count": s1_count,
            "shard_size": shard_size,
            "ingestion_summary": ingest,
            "memory_safety": safety,
        }
        report_path = output_dir / f"full_ref_validation_test_{test_label.lower()}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Full report saved: {report_path}")

    else:
        # Full end-to-end result
        block_diag = result.get("blocking_diagnostics", {})
        val_metrics = result.get("validation_metrics", {})
        resource = result.get("resource_diagnostics", {})
        stream_tel = result.get("streaming_telemetry", {})

        print(f"S1 Entities:            {result.get('sample_size', {}).get('source1_entities', 0):,}")
        print(f"Reference Pool:         {result.get('sample_size', {}).get('candidate_pool_size', 0):,}")
        print(f"Candidate Pairs:        {block_diag.get('candidate_count', 0):,}")
        print(f"Candidate Recall:       {block_diag.get('candidate_recall_pct', 0):.2f}%")
        print(f"Validation Macro F0.5:  {val_metrics.get('macro_f05', 0):.4f}")
        print(f"Optimal Threshold:      {result.get('model_config', {}).get('optimal_threshold', 0):.2f}")
        print(f"Mean Entity Precision:  {val_metrics.get('mean_entity_precision', 0):.4f}")
        print(f"Mean Entity Recall:     {val_metrics.get('mean_entity_recall', 0):.4f}")
        print(f"Peak RSS:               {resource.get('peak_rss_mb', 0)/1024:.2f} GB")
        print(f"Peak VRAM:              {resource.get('peak_vram_mb', 0)/1024:.2f} GB")
        print(f"Total Runtime:          {total_runtime:.2f}s")
        print(f"Disk Chunks:            {stream_tel.get('total_chunk_size_mb', 0):.2f} MB")
        print("=" * 80)

        report_path = output_dir / f"full_ref_validation_test_{test_label.lower()}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"test": test_label, "s1_count": s1_count, "result": result}, f, indent=2)
        print(f"Full report saved: {report_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-reference-corpus validation: Test A (5k S1) and Test B (10k S1)."
    )
    parser.add_argument("--test", choices=["A", "B", "AB"], default="A", help="Which test to run.")
    parser.add_argument("--s1-count", type=int, default=None, help="Override S1 count (default: 5k for A, 10k for B).")
    parser.add_argument("--shard-size", type=int, default=200_000, help="Reference records per shard (default: 200000).")
    parser.add_argument("--s1-batch-size", type=int, default=2500, help="S1 query batch size (default: 2500 for full-ref).")
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--ingest-only", action="store_true", help="Stop after candidate generation; skip LightGBM.")
    parser.add_argument("--train-only", action="store_true", help="Skip ingestion; run LightGBM from existing chunks.")
    parser.add_argument("--overwrite-shards", action="store_true", help="Force re-build shard files.")
    args = parser.parse_args()

    cfg = Config()
    test_sizes = {"A": 5_000, "B": 10_000}

    tests_to_run = ["A", "B"] if args.test == "AB" else [args.test]

    for test_label in tests_to_run:
        s1_count = args.s1_count if args.s1_count is not None else test_sizes[test_label]
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = cfg.OUTPUT_DIR / f"full_ref_{test_label}_{s1_count}"

        run_test(
            test_label=test_label,
            s1_count=s1_count,
            shard_size=args.shard_size,
            s1_batch_size=args.s1_batch_size,
            top_k=args.top_k,
            val_size=args.val_size,
            random_seed=args.random_seed,
            output_dir=out_dir,
            ingest_only=args.ingest_only,
            overwrite_shards=args.overwrite_shards,
            train_only=args.train_only,
        )


if __name__ == "__main__":
    main()
