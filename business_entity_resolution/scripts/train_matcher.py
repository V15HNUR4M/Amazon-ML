"""
train_matcher.py — End-to-end ML training, threshold tuning, and validation benchmark.

Design intent
-------------
Executes the Turn 5 ML workflow in a Google Colab / local Python environment:
1. Reports system environment (Python version, RAM, Disk, GPU, packages).
2. Loads/caches candidate pairs, labels, and 21 numerical features.
3. Performs entity-level Group split (80% train, 20% validation by Source 1 entity).
4. Trains baseline Logistic Regression with standardized features and balanced weighting.
5. Trains tree-based Gradient Boosting model (LightGBM, MIT licensed, < 8B parameters).
6. Performs validation threshold sweep (0.05 to 0.95) to maximize competition metric (entity-level macro F0.5).
7. Evaluates multi-match and singleton performance.
8. Produces side-by-side Model Comparison Table.
9. Analyzes feature importance and validation errors (separating candidate generation from classifier errors).
10. Persists best model artifact and threshold configuration for Turn 6 test inference.

Usage:
------
python scripts/train_matcher.py
python scripts/train_matcher.py --sample-source1 100 --distractors 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_blocking import load_or_create_benchmark_sample
from src.blocking import build_candidates
from src.config import Config
from src.evaluation import analyze_validation_errors, ground_truth_to_dict
from src.features import compute_features
from src.model import (
    DEFAULT_FEATURE_NAMES,
    LightGBMMatcher,
    LogisticRegressionMatcher,
    entity_train_val_split,
)
from src.pair_builder import build_train_pairs
from src.prediction import predict_matches
from src.threshold import evaluate_at_threshold, find_optimal_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def report_environment() -> dict[str, Any]:
    """Report runtime environment details for Colab / local verification."""
    print("=" * 70)
    print("RUNTIME ENVIRONMENT DIAGNOSTICS")
    print("=" * 70)

    env_info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import psutil

        vm = psutil.virtual_memory()
        env_info["ram_total_gb"] = round(vm.total / (1024**3), 2)
        env_info["ram_available_gb"] = round(vm.available / (1024**3), 2)
    except Exception:
        env_info["ram_total_gb"] = "N/A"
        env_info["ram_available_gb"] = "N/A"

    try:
        disk = shutil.disk_usage(".")
        env_info["disk_free_gb"] = round(disk.free / (1024**3), 2)
    except Exception:
        env_info["disk_free_gb"] = "N/A"

    # GPU availability check (without requiring torch)
    gpu_available = False
    gpu_name = "None (CPU Execution)"
    try:
        import subprocess

        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if nvidia_smi.returncode == 0 and nvidia_smi.stdout.strip():
            gpu_available = True
            gpu_name = nvidia_smi.stdout.strip().split("\n")[0]
    except Exception:
        pass

    env_info["gpu_available"] = gpu_available
    env_info["gpu_device"] = gpu_name

    # Package versions
    pkg_versions = {}
    for pkg in ["pandas", "numpy", "sklearn", "lightgbm", "joblib"]:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            pkg_versions[pkg] = "not installed"
    env_info["packages"] = pkg_versions

    print(f"Python:       {env_info['python_version']} ({env_info['platform']})")
    print(f"CPUs:         {env_info['cpu_count']}")
    print(f"Total RAM:    {env_info['ram_total_gb']} GB (Available: {env_info['ram_available_gb']} GB)")
    print(f"Free Disk:    {env_info['disk_free_gb']} GB")
    print(f"GPU:          {env_info['gpu_device']}")
    print(f"Packages:     {env_info['packages']}")
    print("=" * 70)
    print()
    return env_info


def run_training_pipeline(
    sample_s1_count: int = 100,
    distractor_count: int = 500,
    val_size: float = 0.20,
    random_seed: int = 42,
    cache_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute complete training and validation pipeline."""
    cfg = Config()
    c_dir = cache_dir or cfg.CACHE_DIR
    o_dir = output_dir or cfg.OUTPUT_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    o_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    total_start_time = time.time()

    # 1. Environment Diagnostics
    env_info = report_environment()

    # 2. Data Loading / Benchmark Sample
    print("[1/7] Loading / building candidate data & labels...")
    s1, s2, s3, gt_df = load_or_create_benchmark_sample(
        sample_s1_count=sample_s1_count,
        distractor_count=distractor_count,
        cache_dir=c_dir,
    )
    gt_dict = ground_truth_to_dict(gt_df)

    # 3. Candidate Generation (Blocking) & Pair Building
    pairs_pkl = c_dir / f"train_pairs_s1_{sample_s1_count}_dist_{distractor_count}.pkl"
    feats_pkl = c_dir / f"features_s1_{sample_s1_count}_dist_{distractor_count}.pkl"

    if pairs_pkl.exists() and feats_pkl.exists():
        print(f"  Loading cached candidate pairs from {pairs_pkl}...")
        pairs = pd.read_pickle(pairs_pkl)
        print(f"  Loading cached feature matrix from {feats_pkl}...")
        features_df = pd.read_pickle(feats_pkl)
    else:
        print("  Generating candidate pairs via 4-stage blocking...")
        candidates = build_candidates(s1, s2, s3, config=cfg)
        print("  Labeling pairs with ground truth (positives + hard negatives)...")
        pairs = build_train_pairs(s1, s2, s3, candidates, gt_df, random_seed=random_seed)
        print("  Computing 21 pairwise numerical features...")
        features_df = compute_features(pairs)

        # Cache artifacts (pickle is universal, parquet tried optionally)
        try:
            pairs.to_pickle(pairs_pkl)
            features_df.to_pickle(feats_pkl)
            print(f"  Cached pairs and features to {c_dir}")
        except Exception as e:
            logger.warning("Could not cache files: %s", e)

    # Merge features back into pairs DataFrame for consistent slicing
    for col in DEFAULT_FEATURE_NAMES:
        if col in features_df.columns:
            pairs[col] = features_df[col].values

    total_pairs = len(pairs)
    total_pos = int((pairs["label"] == 1).sum())
    total_neg = int((pairs["label"] == 0).sum())
    pos_pct = round(total_pos / total_pairs * 100, 2) if total_pairs > 0 else 0.0

    print(f"  Total Candidate Pairs: {total_pairs}")
    print(f"  Class Distribution:    {total_pos} Positives ({pos_pct}%), {total_neg} Hard Negatives ({100-pos_pct}%)")
    print()

    # 4. Entity-Level Train / Validation Split
    print("[2/7] Splitting by Source 1 entity (GroupSplit: 80% train / 20% validation)...")
    train_pairs, val_pairs, train_gt, val_gt = entity_train_val_split(
        pairs, gt_dict, val_size=val_size, random_state=random_seed
    )

    train_s1_set = set(train_pairs["s1_id"].unique()) | set(train_gt.keys())
    val_s1_set = set(val_pairs["s1_id"].unique()) | set(val_gt.keys())

    # Leakage verification
    leakage = train_s1_set & val_s1_set
    assert len(leakage) == 0, f"DATA LEAKAGE DETECTED! Overlapping entities: {leakage}"

    train_pos = int((train_pairs["label"] == 1).sum()) if len(train_pairs) > 0 else 0
    train_neg = int((train_pairs["label"] == 0).sum()) if len(train_pairs) > 0 else 0
    val_pos = int((val_pairs["label"] == 1).sum()) if len(val_pairs) > 0 else 0
    val_neg = int((val_pairs["label"] == 0).sum()) if len(val_pairs) > 0 else 0

    print(f"  Train: {len(train_s1_set)} S1 entities | {len(train_pairs)} pairs ({train_pos} pos, {train_neg} neg)")
    print(f"  Val:   {len(val_s1_set)} S1 entities | {len(val_pairs)} pairs ({val_pos} pos, {val_neg} neg)")
    print(f"  Safety Verification: intersection(train_s1, val_s1) == empty -> PASSED (0 overlapping entities)")
    print()

    # 5. Train Model 1: Baseline Logistic Regression
    print("[3/7] Training Baseline: Logistic Regression (Standardized + Balanced)...")
    lr_start = time.time()
    lr_matcher = LogisticRegressionMatcher(random_state=random_seed)
    lr_matcher.fit(train_pairs, train_pairs["label"])
    lr_train_time = round(time.time() - lr_start, 3)

    lr_val_proba = lr_matcher.predict_proba(val_pairs)
    lr_thresh, lr_macro_f05, lr_history = find_optimal_threshold(
        val_pairs, lr_val_proba, val_gt
    )
    lr_metrics = evaluate_at_threshold(val_pairs, lr_val_proba, lr_thresh, val_gt)
    lr_metrics["runtime_s"] = lr_train_time

    print(f"  Logistic Regression trained in {lr_train_time}s")
    print(f"  Validation Optimal Threshold: {lr_thresh:.2f} -> Macro F0.5: {lr_macro_f05:.4f}")
    print()

    # 6. Train Model 2: Gradient Boosting (LightGBM)
    print("[4/7] Training Tree Model: LightGBM (MIT license, <=8B parameters, Balanced)...")
    lgb_start = time.time()
    lgb_matcher = LightGBMMatcher(n_estimators=100, learning_rate=0.05, random_state=random_seed)
    lgb_matcher.fit(train_pairs, train_pairs["label"])
    lgb_train_time = round(time.time() - lgb_start, 3)

    lgb_val_proba = lgb_matcher.predict_proba(val_pairs)
    lgb_thresh, lgb_macro_f05, lgb_history = find_optimal_threshold(
        val_pairs, lgb_val_proba, val_gt
    )
    lgb_metrics = evaluate_at_threshold(val_pairs, lgb_val_proba, lgb_thresh, val_gt)
    lgb_metrics["runtime_s"] = lgb_train_time

    print(f"  LightGBM trained in {lgb_train_time}s")
    print(f"  Validation Optimal Threshold: {lgb_thresh:.2f} -> Macro F0.5: {lgb_macro_f05:.4f}")
    print()

    # 7. Model Comparison Table
    print("=" * 88)
    print("MODEL COMPARISON TABLE (Held-Out Entity Validation)")
    print("=" * 88)
    header = (
        f"{'Model':<22} | {'Thresh':<6} | {'Pair Prec':<9} | {'Pair Rec':<8} | "
        f"{'Pair F1':<7} | {'PR-AUC':<6} | {'Macro F0.5':<10} | {'Singl Acc':<9} | {'Runtime':<7}"
    )
    print(header)
    print("-" * 88)

    for name, m in [("Logistic Regression", lr_metrics), ("LightGBM (GBDT)", lgb_metrics)]:
        row = (
            f"{name:<22} | {m['threshold']:<6.2f} | {m['pair_precision']:<9.4f} | "
            f"{m['pair_recall']:<8.4f} | {m['pair_f1']:<7.4f} | {m['pr_auc']:<6.4f} | "
            f"{m['macro_f05']:<10.4f} | {m['singleton_accuracy']:<9.4f} | {m['runtime_s']:<6.2f}s"
        )
        print(row)
    print("=" * 88)
    print()

    # 8. Feature Importance Analysis
    print("[5/7] Feature Importance Diagnostics (LightGBM Gain vs LR Weights)...")
    lgb_fi = lgb_matcher.get_feature_importances(importance_type="gain")
    sorted_fi = sorted(lgb_fi.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<4} | {'Feature Name':<30} | {'LightGBM Gain %':<16} | {'LR Rel Weight %':<16}")
    print("-" * 72)
    lr_fi = lr_matcher.get_feature_importances()
    for rank, (feat, gain) in enumerate(sorted_fi, 1):
        lr_weight = lr_fi.get(feat, 0.0)
        print(f"{rank:<4} | {feat:<30} | {gain*100:<16.2f}% | {lr_weight*100:<16.2f}%")
    print()

    # 9. Validation Error Analysis
    print("[6/7] Validation Error Analysis (Separating Candidate Generation vs Classification)...")
    best_matcher = lgb_matcher if lgb_macro_f05 >= lr_macro_f05 else lr_matcher
    best_thresh = lgb_thresh if lgb_macro_f05 >= lr_macro_f05 else lr_thresh
    best_proba = lgb_val_proba if lgb_macro_f05 >= lr_macro_f05 else lr_val_proba
    best_model_name = "LightGBM" if lgb_macro_f05 >= lr_macro_f05 else "LogisticRegression"

    val_preds = predict_matches(val_pairs, best_proba, threshold=best_thresh, s1_entities=val_gt.keys())
    error_analysis = analyze_validation_errors(
        val_gt=val_gt,
        val_predictions=val_preds,
        val_pairs=val_pairs,
        val_proba=best_proba,
        threshold=best_thresh,
    )

    print("  Validation Ground Truth Summary:")
    print(f"    Total Validation Entities: {error_analysis['total_val_entities']}")
    print(f"    Total True Matches:        {error_analysis['total_true_matches']}")
    print(f"    Correct True Positives:    {error_analysis['true_positives']}")
    print(f"    False Positives:           {error_analysis['false_positives']}")
    print(f"    Total False Negatives:     {error_analysis['total_false_negatives']}")
    print()
    print("  Error Root-Cause Breakdown:")
    print(f"    A. Candidate Generation Misses (Blocking):  {error_analysis['blocking_misses_count']} "
          f"({error_analysis['blocking_miss_rate_pct']:.2f}% of true matches)")
    print(f"       - Missed in S2: {error_analysis['s2_blocking_misses']}")
    print(f"       - Missed in S3: {error_analysis['s3_blocking_misses']}")
    print(f"    B. Classification Misses (P < {best_thresh:.2f}):          {error_analysis['classifier_misses_count']} "
          f"({error_analysis['classifier_miss_rate_pct']:.2f}% of true matches)")
    print(f"       - S2 candidate rejected: {error_analysis['s2_classifier_misses']}")
    print(f"       - S3 candidate rejected: {error_analysis['s3_classifier_misses']}")
    print(f"    C. Singleton Performance:")
    print(f"       - Total Singletons:          {error_analysis['singleton_total']}")
    print(f"       - Correctly Rejected (Empty): {error_analysis['singleton_correct']} "
          f"({error_analysis['singleton_accuracy_pct']:.2f}%)")
    print(f"       - False Alarm Singletons:     {error_analysis['singleton_false_positives']}")
    print()

    # 10. Persist Best Model and Artifacts
    print("[7/7] Persisting model artifacts and configuration...")
    best_model_path = c_dir / "best_matcher.joblib"
    best_matcher.save(best_model_path)

    threshold_config = {
        "best_model": best_model_name,
        "optimal_threshold": float(best_thresh),
        "validation_macro_f05": float(max(lgb_macro_f05, lr_macro_f05)),
        "beta": 0.5,
        "feature_names": DEFAULT_FEATURE_NAMES,
        "sample_s1_count": sample_s1_count,
        "distractor_count": distractor_count,
        "val_size": val_size,
        "random_seed": random_seed,
    }
    threshold_path = c_dir / "optimal_threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)

    report_path = o_dir / "train_validation_report.json"
    full_report = {
        "environment": env_info,
        "data_summary": {
            "total_pairs": total_pairs,
            "total_positives": total_pos,
            "total_negatives": total_neg,
            "positive_pct": pos_pct,
            "train_entities": len(train_s1_set),
            "val_entities": len(val_s1_set),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
        },
        "logistic_regression": lr_metrics,
        "lightgbm": lgb_metrics,
        "threshold_selection": threshold_config,
        "error_analysis": error_analysis,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    total_time = round(time.time() - total_start_time, 2)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Model saved to:     {best_model_path}")
    print(f"  Threshold saved to: {threshold_path}")
    print(f"  Full report saved:  {report_path}")
    print(f"  Peak RAM overhead:  {peak_mem / (1024**2):.2f} MB")
    print(f"  Total run time:     {total_time}s")
    print("=" * 70)
    print()

    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate entity resolution matcher.")
    parser.add_argument("--sample-source1", type=int, default=100)
    parser.add_argument("--distractors", type=int, default=500)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    c_dir = Path(args.cache_dir) if args.cache_dir else None
    o_dir = Path(args.output_dir) if args.output_dir else None

    run_training_pipeline(
        sample_s1_count=args.sample_source1,
        distractor_count=args.distractors,
        val_size=args.val_size,
        random_seed=args.random_seed,
        cache_dir=c_dir,
        output_dir=o_dir,
    )
