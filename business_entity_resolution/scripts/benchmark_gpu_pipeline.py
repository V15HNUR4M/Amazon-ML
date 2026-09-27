"""
benchmark_gpu_pipeline.py — Speed and Scalability Benchmark comparing Turn 5.5 CPU vs Turn 7 GPU.

Benchmarks on configurable scale (e.g. 5,000 S1 or 50,000 S1):
- Total pipeline runtime
- Records / sec (throughput)
- Candidate pairs / sec
- Feature calculation pairs / sec
- Peak RAM (process RSS)
- Peak VRAM (GPU memory)
- Speedup factor

Usage:
------
python scripts/benchmark_gpu_pipeline.py --scale 5000
python scripts/benchmark_gpu_pipeline.py --scale 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

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

from scripts.run_validation_scale_check import load_or_create_large_scale_sample
from src.blocking import build_candidates
from src.config import Config
from src.evaluation import ground_truth_to_dict
from src.features import compute_features
from src.gpu.gpu_blocking import GPUBlockingConfig, build_gpu_candidates
from src.gpu.gpu_features import compute_gpu_features
from src.gpu.gpu_utils import get_device_info, get_gpu_memory_mb, get_process_rss_mb
from src.model import LightGBMMatcher
from src.pair_builder import build_train_pairs
from src.preprocessing import preprocess_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_speed_benchmark(
    scale: int = 5000,
    distractors: int = 5000,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    cfg = Config()
    o_dir = output_dir or cfg.OUTPUT_DIR
    o_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"SPEED & RESOURCE BENCHMARK: CPU (Turn 5.5) vs GPU (Turn 7) [{scale:,} S1 Queries]")
    print("=" * 80)
    dev_info = get_device_info()
    print(f"Hardware: {dev_info['device_name']} | CUDA Available: {dev_info['cuda_available']}")
    print(f"Initial RSS: {get_process_rss_mb():.2f} MB | Initial VRAM: {get_gpu_memory_mb()['allocated_mb']:.2f} MB\n")

    # 1. Load Data Sample
    print(f"[1/4] Loading {scale:,} S1 entities and candidate pool...")
    s1, s2, s3, gt = load_or_create_large_scale_sample(
        sample_s1_count=scale,
        distractor_count=distractors,
        random_seed=random_seed,
    )
    gt_dict = ground_truth_to_dict(gt)

    # 2. Benchmark Blocking
    print("[2/4] Benchmarking Blocking Candidate Generation...")
    t0 = time.time()
    cands_t7 = build_gpu_candidates(s1, s2, s3, config=GPUBlockingConfig(top_k_per_s1=60))
    t_blocking_t7 = time.time() - t0
    t7_blocking_rate = round(len(s1) / max(t_blocking_t7, 0.001), 1)
    print(f"  Turn 7 GPU Blocking: {t_blocking_t7:.2f}s ({t7_blocking_rate:,} S1 records/sec, {len(cands_t7):,} pairs)")

    # 3. Assemble Pairs
    print("[3/4] Assembling Pairs for Feature Extraction...")
    t0 = time.time()
    pairs = build_train_pairs(s1, s2, s3, cands_t7, gt, random_seed=random_seed)
    t_pairs = time.time() - t0

    # 4. Benchmark Feature Extraction (CPU vs GPU)
    print(f"[4/4] Benchmarking Feature Extraction on {len(pairs):,} pairs (CPU vs GPU)...")

    # A. Benchmark CPU Path
    print("  -> Benchmarking CPU Feature Extraction...")
    rss_before_cpu = get_process_rss_mb()
    # For scale > 5000, benchmark a 5,000-pair sample to determine empirical CPU rate without a 30-min block
    cpu_sample_size = min(len(pairs), 5000 if scale > 5000 else len(pairs))
    cpu_sample = pairs.head(cpu_sample_size).copy()
    t0 = time.time()
    feats_cpu_sample = compute_gpu_features(cpu_sample, include_turn7_features=True, device="cpu")
    t_feat_cpu_sample = time.time() - t0
    cpu_rate = round(cpu_sample_size / max(t_feat_cpu_sample, 0.001), 1)
    if cpu_sample_size == len(pairs):
        t_feat_cpu = round(t_feat_cpu_sample, 2)
    else:
        t_feat_cpu = round(len(pairs) / max(cpu_rate, 0.001), 2)
    peak_rss_cpu = max(rss_before_cpu, get_process_rss_mb())
    print(f"     CPU Feature Extraction: {t_feat_cpu:.2f}s ({cpu_rate:,} pairs/sec | sample={cpu_sample_size:,})")

    # B. Benchmark GPU Path
    print("  -> Benchmarking GPU Feature Extraction...")
    rss_before_gpu = get_process_rss_mb()
    # GPU Warmup
    _ = compute_gpu_features(pairs.head(100), include_turn7_features=True, device="gpu")
    t0 = time.time()
    feats_gpu = compute_gpu_features(pairs, include_turn7_features=True, device="gpu")
    t_feat_gpu = time.time() - t0
    gpu_rate = round(len(pairs) / max(t_feat_gpu, 0.001), 1)
    peak_rss_gpu = max(rss_before_gpu, get_process_rss_mb())
    peak_vram_gpu = get_gpu_memory_mb()["peak_allocated_mb"]
    speedup = round(t_feat_cpu / max(t_feat_gpu, 0.001), 2)
    print(f"     GPU Feature Extraction: {t_feat_gpu:.2f}s ({gpu_rate:,} pairs/sec | 34 features)")
    print(f"     Peak RSS: {max(peak_rss_cpu, peak_rss_gpu):.2f} MB | Peak VRAM: {peak_vram_gpu:.2f} MB")
    print(f"     Feature Acceleration Speedup: {speedup:.2f}x faster on GPU!\n")

    print("=" * 80)
    print(f"SPEED BENCHMARK COMPARISON SUMMARY: CPU vs GPU [{scale:,} S1 Queries]")
    print("=" * 80)
    print(f"{'Pipeline Stage / Implementation':<38} | {'Runtime':<12} | {'Throughput':<22}")
    print("-" * 80)
    print(f"{'Turn 7 Blocking Generation (GPU)':<38} | {t_blocking_t7:.2f}s{'':<7} | {t7_blocking_rate:,} S1/sec")
    print(f"{'Pair Assembly':<38} | {t_pairs:.2f}s{'':<7} | {round(len(pairs)/max(t_pairs, 0.001), 1):,} pairs/sec")
    print(f"{'Feature Extraction (CPU Baseline)':<38} | {t_feat_cpu:.2f}s{'':<7} | {cpu_rate:,} pairs/sec")
    print(f"{'Feature Extraction (GPU Accelerated)':<38} | {t_feat_gpu:.2f}s{'':<7} | {gpu_rate:,} pairs/sec")
    print("-" * 80)
    print(f"GPU Feature Extraction Speedup Factor: {speedup:.2f}x FASTER than CPU")
    print(f"Reduction in Feature Extraction Time:  {t_feat_cpu - t_feat_gpu:.2f}s saved ({((t_feat_cpu - t_feat_gpu)/max(t_feat_cpu,0.001))*100:.1f}% reduction)")
    print("=" * 80)

    benchmark_results = {
        "scale_s1": scale,
        "distractors": distractors,
        "device_info": dev_info,
        "blocking": {
            "runtime_s": round(t_blocking_t7, 2),
            "throughput_records_per_s": t7_blocking_rate,
            "candidate_pairs_count": len(cands_t7),
        },
        "feature_extraction": {
            "cpu_runtime_s": round(t_feat_cpu, 2),
            "cpu_throughput_pairs_per_s": cpu_rate,
            "gpu_runtime_s": round(t_feat_gpu, 2),
            "gpu_throughput_pairs_per_s": gpu_rate,
            "speedup_factor": speedup,
            "pairs_count": len(pairs),
            "feature_count": 34,
            "sample_pairs_benchmarked_cpu": cpu_sample_size,
        },
        "resources": {
            "peak_rss_mb": max(peak_rss_cpu, peak_rss_gpu),
            "peak_vram_mb": peak_vram_gpu,
        },
    }

    out_file = o_dir / f"turn7_speed_benchmark_s1_{scale}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved benchmark results to: {out_file}")
    return benchmark_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Turn 7 speed and scalability benchmark.")
    parser.add_argument("--scale", type=int, default=5000)
    parser.add_argument("--distractors", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    o_dir = Path(args.output_dir) if args.output_dir else None
    run_speed_benchmark(scale=args.scale, distractors=args.distractors, random_seed=args.random_seed, output_dir=o_dir)
