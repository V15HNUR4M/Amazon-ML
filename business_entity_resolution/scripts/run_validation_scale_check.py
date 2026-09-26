"""
run_validation_scale_check.py — Large-scale validation experiment (Turn 5.5).

Design intent
-------------
Rigorously evaluates the Turn 5 ML entity resolution pipeline on a substantially
larger, statistically confident sample of 5,000 Source-1 entities (~17,250 true matches,
~27,000 candidate pool records, ~25,000-30,000 candidate pairs):
1. Verifies whether the blocking candidate recall (~97.38%) holds at 5k scale.
2. Verifies whether entity-level GroupKFold / GroupShuffleSplit prevents leakage at scale.
3. Compares Logistic Regression vs LightGBM on:
   - Entity-level Macro F0.5 (beta = 0.5)
   - Mean entity precision & recall
   - Pair-level precision, recall, F1, PR-AUC
   - Singleton accuracy on real data distribution (~5.5% singletons)
4. Classifies validation errors into:
   - A: Candidate-generation misses (blocking misses)
   - B: Classification misses (P < threshold)
   - C: False merges (P >= threshold on non-matches)
5. Tracks true process RSS memory and per-stage wall-clock runtime.
6. Produces a side-by-side comparison table between Turn 5 and Turn 5.5.
7. Saves all Turn 5.5 artifacts without overwriting Turn 5.

Usage:
------
python scripts/run_validation_scale_check.py --sample-source1 5000 --distractors 5000
"""

from __future__ import annotations

import argparse
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

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import BlockingConfig, BlockingIndex, build_candidates
from src.config import Config
from src.evaluation import (
    analyze_validation_errors,
    entity_f05,
    entity_precision,
    entity_recall,
    ground_truth_to_dict,
    macro_f05,
)
from src.features import compute_features
from src.model import (
    DEFAULT_FEATURE_NAMES,
    LightGBMMatcher,
    LogisticRegressionMatcher,
    entity_train_val_split,
)
from src.pair_builder import build_train_pairs
from src.prediction import predict_matches
from src.preprocessing import preprocess_records
from src.threshold import evaluate_at_threshold, find_optimal_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_process_rss_mb() -> float:
    """Return actual resident set size (RSS) memory of the current process in MB."""
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 2)


# ---------------------------------------------------------------------------
# Large-Scale Data Sampler
# ---------------------------------------------------------------------------


def load_or_create_large_scale_sample(
    sample_s1_count: int = 5000,
    distractor_count: int = 5000,
    cache_dir: Optional[Path] = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load or deterministically sample S1 queries, candidate pool, and ground truth."""
    cfg = Config()
    c_dir = cache_dir or cfg.CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    cache_file = c_dir / f"benchmark_sample_s1_{sample_s1_count}_dist_{distractor_count}.pkl"

    if cache_file.exists():
        logger.info("Loading cached scale sample from %s...", cache_file)
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["s1"], data["s2"], data["s3"], data["gt"]

    logger.info(
        "Building large-scale sample (%d S1 queries + true matches + %d distractors)...",
        sample_s1_count,
        distractor_count,
    )

    # 1. Load Ground Truth sample deterministically
    gt = pd.read_csv(
        cfg.TRAIN_GROUND_TRUTH,
        sep="\t",
        nrows=sample_s1_count,
        keep_default_na=False,
    )
    needed_s1: set[str] = set(gt["source1_entity_id"])

    # 2. Stream S1 chunks to collect matching S1 records
    logger.info("  Streaming train_source1.tsv to extract %d S1 records...", len(needed_s1))
    s1_rows: list[dict] = []
    for chunk in pd.read_csv(
        cfg.TRAIN_SOURCE1, sep="\t", chunksize=200000, keep_default_na=False
    ):
        matches = chunk[chunk["entity_id"].isin(needed_s1)]
        if len(matches) > 0:
            s1_rows.extend(matches.to_dict(orient="records"))
            needed_s1 -= set(matches["entity_id"])
        if len(needed_s1) == 0:
            break
    s1 = pd.DataFrame(s1_rows).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    logger.info("  Extracted %d S1 records.", len(s1))

    # 3. Collect all true match IDs from S2 and S3
    needed_s2: set[str] = set()
    needed_s3: set[str] = set()
    for _, r in gt.iterrows():
        for m in str(r.get("matched_entity_ids", "")).split(","):
            m = m.strip()
            if m.startswith("S2"):
                needed_s2.add(m)
            elif m.startswith("S3"):
                needed_s3.add(m)

    logger.info("  Target matches needed: %d in S2, %d in S3", len(needed_s2), len(needed_s3))

    # 4. Stream S2 chunks
    logger.info("  Streaming train_source2.tsv for true matches + distractors...")
    s2_rows: list[dict] = []
    distractors_s2_needed = distractor_count // 2
    for chunk in pd.read_csv(
        cfg.TRAIN_SOURCE2, sep="\t", chunksize=250000, keep_default_na=False
    ):
        matches = chunk[chunk["entity_id"].isin(needed_s2)]
        if len(matches) > 0:
            s2_rows.extend(matches.to_dict(orient="records"))
            needed_s2 -= set(matches["entity_id"])
        if distractors_s2_needed > 0:
            dist = chunk[~chunk["entity_id"].isin(needed_s2)].head(distractors_s2_needed)
            s2_rows.extend(dist.to_dict(orient="records"))
            distractors_s2_needed -= len(dist)
        if len(needed_s2) == 0 and distractors_s2_needed <= 0:
            break

    # 5. Stream S3 chunks
    logger.info("  Streaming train_source3.tsv for true matches + distractors...")
    s3_rows: list[dict] = []
    distractors_s3_needed = distractor_count // 2
    for chunk in pd.read_csv(
        cfg.TRAIN_SOURCE3, sep="\t", chunksize=250000, keep_default_na=False
    ):
        matches = chunk[chunk["entity_id"].isin(needed_s3)]
        if len(matches) > 0:
            s3_rows.extend(matches.to_dict(orient="records"))
            needed_s3 -= set(matches["entity_id"])
        if distractors_s3_needed > 0:
            dist = chunk[~chunk["entity_id"].isin(needed_s3)].head(distractors_s3_needed)
            s3_rows.extend(dist.to_dict(orient="records"))
            distractors_s3_needed -= len(dist)
        if len(needed_s3) == 0 and distractors_s3_needed <= 0:
            break

    s2_df = pd.DataFrame(s2_rows).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    s3_df = pd.DataFrame(s3_rows).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

    # 6. Preprocess records
    logger.info("  Preprocessing large-scale sample records...")
    s1_proc = preprocess_records(s1)
    s2_proc = preprocess_records(s2_df)
    s3_proc = preprocess_records(s3_df)

    sample_data = {"s1": s1_proc, "s2": s2_proc, "s3": s3_proc, "gt": gt}
    with open(cache_file, "wb") as f:
        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("  Saved large-scale sample to %s", cache_file)
    return s1_proc, s2_proc, s3_proc, gt


# ---------------------------------------------------------------------------
# Main Scale Check Pipeline
# ---------------------------------------------------------------------------


def run_scale_check(
    sample_s1_count: int = 5000,
    distractor_count: int = 5000,
    val_size: float = 0.20,
    random_seed: int = 42,
    cache_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute complete Turn 5.5 large-scale validation experiment."""
    cfg = Config()
    c_dir = cache_dir or cfg.CACHE_DIR
    o_dir = output_dir or cfg.OUTPUT_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    o_dir.mkdir(parents=True, exist_ok=True)

    rss_baseline = get_process_rss_mb()
    peak_rss = rss_baseline
    total_start = time.time()
    timing: dict[str, float] = {}

    print("=" * 80)
    print(f"TURN 5.5: LARGER-SCALE VALIDATION SANITY CHECK ({sample_s1_count} S1 ENTITIES)")
    print("=" * 80)
    print(f"Baseline Process RSS: {rss_baseline:.2f} MB\n")

    # -----------------------------------------------------------------------
    # Stage 1: Data Loading & Natural Distribution Inspection
    # -----------------------------------------------------------------------
    print("[1/8] Loading / sampling dataset...")
    t0 = time.time()
    s1, s2, s3, gt_df = load_or_create_large_scale_sample(
        sample_s1_count=sample_s1_count,
        distractor_count=distractor_count,
        cache_dir=c_dir,
        random_seed=random_seed,
    )
    gt_dict = ground_truth_to_dict(gt_df)
    timing["data_loading_s"] = round(time.time() - t0, 2)
    peak_rss = max(peak_rss, get_process_rss_mb())

    # Analyze real data distribution
    match_counts = [len(matches) for matches in gt_dict.values()]
    singleton_count = match_counts.count(0)
    one_match_count = match_counts.count(1)
    multi_match_count = sum(1 for c in match_counts if c >= 2)
    s2_only = sum(1 for m in gt_dict.values() if m and all(x.startswith("S2") for x in m))
    s3_only = sum(1 for m in gt_dict.values() if m and all(x.startswith("S3") for x in m))
    both_sources = sum(
        1 for m in gt_dict.values() if any(x.startswith("S2") for x in m) and any(x.startswith("S3") for x in m)
    )
    total_true_matches = sum(match_counts)
    s2_true_matches = sum(1 for m in gt_dict.values() for x in m if x.startswith("S2"))
    s3_true_matches = sum(1 for m in gt_dict.values() for x in m if x.startswith("S3"))

    country_counts = s1["country_norm"].value_counts().to_dict()
    missing_addr_count = int(s1["address_is_missing"].sum()) if "address_is_missing" in s1.columns else 0
    script_counts = s1["name_script"].value_counts().to_dict() if "name_script" in s1.columns else {}

    print(f"  Loaded Records: {len(s1)} S1 queries, {len(s2)} S2 pool, {len(s3)} S3 pool")
    print(f"  Total True Matches: {total_true_matches} ({s2_true_matches} in S2, {s3_true_matches} in S3)")
    print(f"  Match Cardinality Distribution:")
    print(f"    - Singletons (0 matches): {singleton_count} ({singleton_count/len(s1)*100:.2f}%)")
    print(f"    - 1 Match:               {one_match_count} ({one_match_count/len(s1)*100:.2f}%)")
    print(f"    - Multi-Match (2+):      {multi_match_count} ({multi_match_count/len(s1)*100:.2f}%)")
    print(f"    - S2 only: {s2_only} | S3 only: {s3_only} | Both S2+S3: {both_sources}")
    print(f"  Countries: {country_counts}")
    print(f"  Missing Addresses in S1: {missing_addr_count} ({missing_addr_count/len(s1)*100:.2f}%)")
    print(f"  Scripts: {script_counts}")
    print(f"  Runtime: {timing['data_loading_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

    # -----------------------------------------------------------------------
    # Stage 2: Candidate Generation (Blocking) Diagnostics
    # -----------------------------------------------------------------------
    print("[2/8] Generating candidate pairs via 4-stage blocking...")
    cands_pkl = c_dir / f"scale_check_candidates_s1_{sample_s1_count}.pkl"
    t0 = time.time()
    if cands_pkl.exists():
        logger.info("Loading cached candidate pairs from %s...", cands_pkl)
        candidates = pd.read_pickle(cands_pkl)
    else:
        candidates = build_candidates(s1, s2, s3, config=cfg)
        candidates.to_pickle(cands_pkl)
    timing["blocking_s"] = round(time.time() - t0, 2)
    peak_rss = max(peak_rss, get_process_rss_mb())

    # Candidate diagnostics
    total_candidates = len(candidates)
    cands_by_s1 = candidates.groupby("s1_id")["candidate_id"].apply(set).to_dict()

    retrieved_true_total = 0
    retrieved_true_s2 = 0
    retrieved_true_s3 = 0

    cands_per_s1_list = [len(cands_by_s1.get(str(eid), set())) for eid in s1["entity_id"]]

    for s1_id, true_set in gt_dict.items():
        cands_set = cands_by_s1.get(str(s1_id), set())
        for tid in true_set:
            if tid in cands_set:
                retrieved_true_total += 1
                if tid.startswith("S2"):
                    retrieved_true_s2 += 1
                elif tid.startswith("S3"):
                    retrieved_true_s3 += 1

    candidate_recall = (retrieved_true_total / total_true_matches * 100) if total_true_matches > 0 else 0.0
    s2_candidate_recall = (retrieved_true_s2 / s2_true_matches * 100) if s2_true_matches > 0 else 0.0
    s3_candidate_recall = (retrieved_true_s3 / s3_true_matches * 100) if s3_true_matches > 0 else 0.0

    total_possible_comparisons = len(s1) * (len(s2) + len(s3))
    reduction_ratio = (
        (1.0 - (total_candidates / total_possible_comparisons)) * 100
        if total_possible_comparisons > 0
        else 0.0
    )

    cands_series = pd.Series(cands_per_s1_list)
    diag_metrics: dict[str, Any] = {
        "candidate_count": total_candidates,
        "total_true_matches": total_true_matches,
        "retrieved_true_matches": retrieved_true_total,
        "candidate_recall_pct": round(candidate_recall, 2),
        "s2_candidate_recall_pct": round(s2_candidate_recall, 2),
        "s3_candidate_recall_pct": round(s3_candidate_recall, 2),
        "reduction_ratio_pct": round(reduction_ratio, 4),
        "mean_candidates_per_s1": round(float(cands_series.mean()), 2),
        "median_candidates_per_s1": round(float(cands_series.median()), 2),
        "p90_candidates_per_s1": round(float(cands_series.quantile(0.90)), 2),
        "p95_candidates_per_s1": round(float(cands_series.quantile(0.95)), 2),
        "p99_candidates_per_s1": round(float(cands_series.quantile(0.99)), 2),
        "max_candidates_per_s1": int(cands_series.max()),
        "blocking_runtime_s": timing["blocking_s"],
    }

    print(f"  Candidate Pairs Generated: {total_candidates}")
    print(f"  Candidate Recall:          {candidate_recall:.2f}% (Retrieved {retrieved_true_total}/{total_true_matches})")
    print(f"    - S2 Recall:             {s2_candidate_recall:.2f}% ({retrieved_true_s2}/{s2_true_matches})")
    print(f"    - S3 Recall:             {s3_candidate_recall:.2f}% ({retrieved_true_s3}/{s3_true_matches})")
    print(f"  Reduction Ratio:           {reduction_ratio:.4f}%")
    print(f"  Candidates per S1:         Mean={diag_metrics['mean_candidates_per_s1']}, Median={diag_metrics['median_candidates_per_s1']}, "
          f"P90={diag_metrics['p90_candidates_per_s1']}, P95={diag_metrics['p95_candidates_per_s1']}, Max={diag_metrics['max_candidates_per_s1']}")
    print(f"  Runtime: {timing['blocking_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

    # -----------------------------------------------------------------------
    # Stage 3: Labelled Pair Construction
    # -----------------------------------------------------------------------
    print("[3/8] Assembling labelled candidate pairs (positives + hard negatives)...")
    pairs_pkl = c_dir / f"scale_check_pairs_s1_{sample_s1_count}.pkl"
    t0 = time.time()
    if pairs_pkl.exists():
        logger.info("Loading cached pairs from %s...", pairs_pkl)
        pairs = pd.read_pickle(pairs_pkl)
    else:
        pairs = build_train_pairs(s1, s2, s3, candidates, gt_df, random_seed=random_seed)
        pairs.to_pickle(pairs_pkl)
    timing["pair_construction_s"] = round(time.time() - t0, 2)
    peak_rss = max(peak_rss, get_process_rss_mb())

    total_pairs = len(pairs)
    total_pos = int((pairs["label"] == 1).sum())
    total_neg = int((pairs["label"] == 0).sum())
    pos_pct = round(total_pos / total_pairs * 100, 2) if total_pairs > 0 else 0.0

    print(f"  Built Pairs: {total_pairs} rows ({total_pos} positives [{pos_pct}%], {total_neg} hard negatives [{100-pos_pct}%])")
    print(f"  Runtime: {timing['pair_construction_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

    # -----------------------------------------------------------------------
    # Stage 4: Feature Extraction (21 Features)
    # -----------------------------------------------------------------------
    print("[4/8] Computing 21 pairwise numerical features...")
    feats_pkl = c_dir / f"scale_check_features_s1_{sample_s1_count}.pkl"
    t0 = time.time()
    if feats_pkl.exists():
        logger.info("Loading cached feature matrix from %s...", feats_pkl)
        features_df = pd.read_pickle(feats_pkl)
    else:
        features_df = compute_features(pairs)
        features_df.to_pickle(feats_pkl)
        logger.info("Cached features to %s", feats_pkl)
    timing["feature_generation_s"] = round(time.time() - t0, 2)
    peak_rss = max(peak_rss, get_process_rss_mb())

    feature_throughput = round(len(pairs) / (timing["feature_generation_s"] or 0.001), 1)
    feature_mem_mb = round(features_df.memory_usage(deep=True).sum() / (1024**2), 2)
    cache_file_size_mb = round(feats_pkl.stat().st_size / (1024**2), 2) if feats_pkl.exists() else 0.0

    for col in DEFAULT_FEATURE_NAMES:
        if col in features_df.columns:
            pairs[col] = features_df[col].values

    print(f"  Feature Matrix Shape:      {features_df.shape} ({len(features_df)} pairs, {features_df.shape[1]} features)")
    print(f"  Throughput:                {feature_throughput} pairs/sec")
    print(f"  In-Memory Footprint:       {feature_mem_mb} MB")
    print(f"  Cache File Size:           {cache_file_size_mb} MB")
    print(f"  Runtime: {timing['feature_generation_s']}s | Current RSS: {get_process_rss_mb():.2f} MB\n")

    # -----------------------------------------------------------------------
    # Stage 5: Entity-Level Train / Validation Split
    # -----------------------------------------------------------------------
    print("[5/8] Performing entity-level GroupSplit (80% train / 20% validation)...")
    t0 = time.time()
    train_pairs, val_pairs, train_gt, val_gt = entity_train_val_split(
        pairs, gt_dict, val_size=val_size, random_state=random_seed
    )
    timing["train_val_split_s"] = round(time.time() - t0, 3)

    train_s1_set = set(train_pairs["s1_id"].unique()) | set(train_gt.keys())
    val_s1_set = set(val_pairs["s1_id"].unique()) | set(val_gt.keys())

    # Critical safety assertion
    leakage = train_s1_set & val_s1_set
    assert len(leakage) == 0, f"FATAL DATA LEAKAGE DETECTED! Overlapping entities: {leakage}"

    train_pos = int((train_pairs["label"] == 1).sum()) if len(train_pairs) > 0 else 0
    train_neg = int((train_pairs["label"] == 0).sum()) if len(train_pairs) > 0 else 0
    val_pos = int((val_pairs["label"] == 1).sum()) if len(val_pairs) > 0 else 0
    val_neg = int((val_pairs["label"] == 0).sum()) if len(val_pairs) > 0 else 0

    val_singletons = sum(1 for m in val_gt.values() if len(m) == 0)
    val_multi = sum(1 for m in val_gt.values() if len(m) >= 2)

    print(f"  Train Split: {len(train_s1_set)} S1 entities | {len(train_pairs)} pairs ({train_pos} pos, {train_neg} neg)")
    print(f"  Val Split:   {len(val_s1_set)} S1 entities | {len(val_pairs)} pairs ({val_pos} pos, {val_neg} neg)")
    print(f"               Validation includes: {val_singletons} singletons, {val_multi} multi-matches")
    print(f"  Safety Verification: len(train_s1 & val_s1) == 0 -> PASSED (0 overlapping entities)\n")

    # -----------------------------------------------------------------------
    # Stage 6: Train & Validate Model 1 (Logistic Regression)
    # -----------------------------------------------------------------------
    print("[6/8] Training & Validating Baseline: Logistic Regression (Standardized + Balanced)...")
    t0 = time.time()
    lr_matcher = LogisticRegressionMatcher(random_state=random_seed)
    lr_matcher.fit(train_pairs, train_pairs["label"])
    timing["lr_train_s"] = round(time.time() - t0, 3)
    peak_rss = max(peak_rss, get_process_rss_mb())

    lr_val_proba = lr_matcher.predict_proba(val_pairs)
    t0 = time.time()
    lr_best_thresh, lr_best_macro_f05, lr_history = find_optimal_threshold(
        val_pairs, lr_val_proba, val_gt
    )
    timing["lr_sweep_s"] = round(time.time() - t0, 3)
    lr_metrics = evaluate_at_threshold(val_pairs, lr_val_proba, lr_best_thresh, val_gt)
    lr_metrics["runtime_s"] = timing["lr_train_s"]

    # Calculate classification recall among generated candidates
    val_true_in_cands = sum(
        1 for _, r in val_pairs[val_pairs["label"] == 1].iterrows()
    )
    val_true_predicted_lr = int(((val_pairs["label"] == 1) & (lr_val_proba >= lr_best_thresh)).sum())
    lr_cls_recall = (val_true_predicted_lr / val_true_in_cands * 100) if val_true_in_cands > 0 else 0.0
    lr_metrics["classification_recall_pct"] = round(lr_cls_recall, 2)

    print(f"  LR Training Time: {timing['lr_train_s']}s")
    print(f"  LR Optimal Threshold: {lr_best_thresh:.2f} -> Validation Macro F0.5: {lr_best_macro_f05:.4f}")
    print(f"  LR Mean Entity Precision: {lr_metrics['mean_entity_precision']:.4f} | Recall: {lr_metrics['mean_entity_recall']:.4f}")
    print(f"  LR Singleton Accuracy:    {lr_metrics['singleton_accuracy']:.4f} ({val_singletons} singletons)")
    print(f"  LR Classification Recall: {lr_cls_recall:.2f}% ({val_true_predicted_lr}/{val_true_in_cands} generated true matches)\n")

    # -----------------------------------------------------------------------
    # Stage 7: Train & Validate Model 2 (LightGBM)
    # -----------------------------------------------------------------------
    print("[7/8] Training & Validating Tree Model: LightGBM (MIT license, <=8B params, Balanced)...")
    t0 = time.time()
    lgb_matcher = LightGBMMatcher(n_estimators=100, learning_rate=0.05, random_state=random_seed)
    lgb_matcher.fit(train_pairs, train_pairs["label"])
    timing["lgb_train_s"] = round(time.time() - t0, 3)
    peak_rss = max(peak_rss, get_process_rss_mb())

    lgb_val_proba = lgb_matcher.predict_proba(val_pairs)
    t0 = time.time()
    lgb_best_thresh, lgb_best_macro_f05, lgb_history = find_optimal_threshold(
        val_pairs, lgb_val_proba, val_gt
    )
    timing["lgb_sweep_s"] = round(time.time() - t0, 3)
    lgb_metrics = evaluate_at_threshold(val_pairs, lgb_val_proba, lgb_best_thresh, val_gt)
    lgb_metrics["runtime_s"] = timing["lgb_train_s"]

    val_true_predicted_lgb = int(((val_pairs["label"] == 1) & (lgb_val_proba >= lgb_best_thresh)).sum())
    lgb_cls_recall = (val_true_predicted_lgb / val_true_in_cands * 100) if val_true_in_cands > 0 else 0.0
    lgb_metrics["classification_recall_pct"] = round(lgb_cls_recall, 2)

    print(f"  LightGBM Training Time: {timing['lgb_train_s']}s")
    print(f"  LightGBM Optimal Threshold: {lgb_best_thresh:.2f} -> Validation Macro F0.5: {lgb_best_macro_f05:.4f}")
    print(f"  LightGBM Mean Entity Precision: {lgb_metrics['mean_entity_precision']:.4f} | Recall: {lgb_metrics['mean_entity_recall']:.4f}")
    print(f"  LightGBM Singleton Accuracy:    {lgb_metrics['singleton_accuracy']:.4f} ({val_singletons} singletons)")
    print(f"  LightGBM Classification Recall: {lgb_cls_recall:.2f}% ({val_true_predicted_lgb}/{val_true_in_cands} generated true matches)\n")

    # Model comparison table
    print("=" * 104)
    print(f"TURN 5.5 MODEL COMPARISON TABLE (Held-Out Validation on {len(val_s1_set)} S1 Entities)")
    print("=" * 104)
    hdr = (
        f"{'Model':<22} | {'Thresh':<6} | {'Pair Prec':<9} | {'Pair Rec':<8} | "
        f"{'Pair F1':<7} | {'PR-AUC':<6} | {'Macro F0.5':<10} | {'Ent Prec':<8} | {'Ent Rec':<7} | {'Singl Acc':<9} | {'Runtime':<7}"
    )
    print(hdr)
    print("-" * 104)
    for name, m in [("Logistic Regression", lr_metrics), ("LightGBM (GBDT)", lgb_metrics)]:
        row = (
            f"{name:<22} | {m['threshold']:<6.2f} | {m['pair_precision']:<9.4f} | "
            f"{m['pair_recall']:<8.4f} | {m['pair_f1']:<7.4f} | {m['pr_auc']:<6.4f} | "
            f"{m['macro_f05']:<10.4f} | {m['mean_entity_precision']:<8.4f} | {m['mean_entity_recall']:<7.4f} | "
            f"{m['singleton_accuracy']:<9.4f} | {m['runtime_s']:<6.2f}s"
        )
        print(row)
    print("=" * 104)
    print()

    # -----------------------------------------------------------------------
    # Stage 8: Error Analysis (A: Blocking Miss vs B: Classifier Miss vs C: False Merge)
    # -----------------------------------------------------------------------
    print("[8/8] Classifying validation errors and analyzing failure modes...")
    t0 = time.time()
    best_model = lgb_matcher if lgb_best_macro_f05 >= lr_best_macro_f05 else lr_matcher
    best_thresh = lgb_best_thresh if lgb_best_macro_f05 >= lr_best_macro_f05 else lr_best_thresh
    best_proba = lgb_val_proba if lgb_best_macro_f05 >= lr_best_macro_f05 else lr_val_proba
    best_model_name = "LightGBM" if best_model == lgb_matcher else "LogisticRegression"

    val_preds = predict_matches(val_pairs, best_proba, threshold=best_thresh, s1_entities=val_gt.keys())
    error_analysis = analyze_validation_errors(
        val_gt=val_gt,
        val_predictions=val_preds,
        val_pairs=val_pairs,
        val_proba=best_proba,
        threshold=best_thresh,
    )
    timing["error_analysis_s"] = round(time.time() - t0, 3)

    # Detailed Error Categorization
    val_true_matches_total = error_analysis["total_true_matches"]
    blocking_miss_count = error_analysis["blocking_misses_count"]
    cls_miss_count = error_analysis["classifier_misses_count"]
    false_merge_count = error_analysis["false_positives"]
    true_positive_count = error_analysis["true_positives"]

    print(f"  Validation Ground Truth: {len(val_s1_set)} Entities | {val_true_matches_total} True Matches")
    print(f"  Correct Matches (True Positives):   {true_positive_count} ({true_positive_count/val_true_matches_total*100:.2f}%)")
    print(f"  A. Candidate-Generation Misses:     {blocking_miss_count} ({error_analysis['blocking_miss_rate_pct']:.2f}% of true matches)")
    print(f"     - S2 Blocking Misses:            {error_analysis['s2_blocking_misses']}")
    print(f"     - S3 Blocking Misses:            {error_analysis['s3_blocking_misses']}")
    print(f"  B. Classification Misses (P < {best_thresh:.2f}): {cls_miss_count} ({error_analysis['classifier_miss_rate_pct']:.2f}% of true matches)")
    print(f"     - S2 Classification Misses:      {error_analysis['s2_classifier_misses']}")
    print(f"     - S3 Classification Misses:      {error_analysis['s3_classifier_misses']}")
    print(f"  C. False Merges (P >= {best_thresh:.2f} non-match): {false_merge_count}")
    print(f"  Singleton Performance:              {error_analysis['singleton_correct']}/{error_analysis['singleton_total']} correct "
          f"({error_analysis['singleton_accuracy_pct']:.2f}%), {error_analysis['singleton_false_positives']} false alarms")
    print()

    # Feature Importance (LightGBM Gain)
    print("Feature Importance Diagnostics (Top 8 Features by Gain):")
    lgb_fi = lgb_matcher.get_feature_importances(importance_type="gain")
    sorted_fi = sorted(lgb_fi.items(), key=lambda x: x[1], reverse=True)
    for rank, (feat, gain) in enumerate(sorted_fi[:8], 1):
        print(f"  {rank}. {feat:<28}: {gain*100:6.2f}%")
    print()

    # -----------------------------------------------------------------------
    # Comparison Against Turn 5
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("COMPARISON: TURN 5 vs TURN 5.5")
    print("=" * 80)
    cmp_fmt = f"{'Metric':<32} | {'Turn 5 (Small)':<20} | {'Turn 5.5 (5k Scale)':<20}"
    print(cmp_fmt)
    print("-" * 80)
    comparison_data = [
        ("Source-1 Query Entities", "100", f"{len(s1):,}"),
        ("Validation S1 Entities", "20", f"{len(val_s1_set):,}"),
        ("Candidate Pool Size (S2+S3)", "843", f"{len(s2)+len(s3):,}"),
        ("Candidate Pairs Generated", "561", f"{total_candidates:,}"),
        ("Overall Candidate Recall", "97.38%", f"{candidate_recall:.2f}%"),
        ("S2 Candidate Recall", "96.99%", f"{s2_candidate_recall:.2f}%"),
        ("S3 Candidate Recall", "97.74%", f"{s3_candidate_recall:.2f}%"),
        ("Candidate Reduction Ratio", "99.33%", f"{reduction_ratio:.4f}%"),
        ("Mean Candidates / S1", "5.61", f"{diag_metrics['mean_candidates_per_s1']}"),
        ("Max Candidates / S1", "19", f"{diag_metrics['max_candidates_per_s1']}"),
        ("Validation Positive Pairs", "60", f"{val_pos:,}"),
        ("Validation Hard Negatives", "45", f"{val_neg:,}"),
        ("Validation Singletons", "1", f"{val_singletons:,}"),
        ("Singleton Accuracy", "100.0%", f"{lgb_metrics['singleton_accuracy']*100:.1f}%"),
        ("Best Model", "LightGBM / LR", f"{best_model_name}"),
        ("Best Threshold", "0.60 (LGB) / 0.15 (LR)", f"{lgb_best_thresh:.2f} (LGB) / {lr_best_thresh:.2f} (LR)"),
        ("Validation Macro F0.5", "0.9976", f"{max(lgb_best_macro_f05, lr_best_macro_f05):.4f}"),
        ("Mean Entity Precision", "1.0000", f"{lgb_metrics['mean_entity_precision']:.4f}"),
        ("Mean Entity Recall", "0.9836", f"{lgb_metrics['mean_entity_recall']:.4f}"),
        ("Candidate Gen Misses", "1 (1.64%)", f"{blocking_miss_count} ({error_analysis['blocking_miss_rate_pct']:.2f}%)"),
        ("Classification Misses", "0 (0.00%)", f"{cls_miss_count} ({error_analysis['classifier_miss_rate_pct']:.2f}%)"),
        ("False Merges", "0", f"{false_merge_count}"),
        ("Peak Process RSS", "5.77 MB (tracemalloc)", f"{peak_rss:.2f} MB (RSS)"),
        ("Total Pipeline Runtime", "9.07s", f"{round(time.time() - total_start, 2)}s"),
    ]
    for m_label, t5_val, t55_val in comparison_data:
        print(f"{m_label:<32} | {t5_val:<20} | {t55_val:<20}")
    print("=" * 80)
    print()

    # -----------------------------------------------------------------------
    # Artifact Persistence
    # -----------------------------------------------------------------------
    total_time = round(time.time() - total_start, 2)
    timing["total_runtime_s"] = total_time

    # 1. Best model
    best_model_path = c_dir / "turn5_5_best_matcher.joblib"
    best_model.save(best_model_path)

    # 2. Optimal threshold
    thresh_config = {
        "best_model": best_model_name,
        "optimal_threshold": float(best_thresh),
        "validation_macro_f05": float(max(lgb_best_macro_f05, lr_best_macro_f05)),
        "beta": 0.5,
        "sample_s1_count": sample_s1_count,
        "distractor_count": distractor_count,
        "val_size": val_size,
        "random_seed": random_seed,
    }
    thresh_path = c_dir / "turn5_5_optimal_threshold.json"
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump(thresh_config, f, indent=2)

    # 3. Candidate diagnostics
    diag_path = o_dir / "turn5_5_candidate_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag_metrics, f, indent=2)

    # 4. Error analysis report
    err_path = o_dir / "turn5_5_error_analysis.json"
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)

    # 5. Full validation report
    full_report = {
        "experiment_name": "Turn 5.5 Large-Scale Validation Check",
        "sample_size": {
            "source1_entities": len(s1),
            "candidate_pool_size": len(s2) + len(s3),
            "s2_pool": len(s2),
            "s3_pool": len(s3),
            "total_true_matches": total_true_matches,
        },
        "distribution": {
            "singletons": singleton_count,
            "one_match": one_match_count,
            "multi_match": multi_match_count,
            "s2_only": s2_only,
            "s3_only": s3_only,
            "both_sources": both_sources,
            "country_counts": country_counts,
            "missing_addresses": missing_addr_count,
        },
        "blocking_diagnostics": diag_metrics,
        "split_summary": {
            "train_entities": len(train_s1_set),
            "val_entities": len(val_s1_set),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "train_pos": train_pos,
            "train_neg": train_neg,
            "val_pos": val_pos,
            "val_neg": val_neg,
            "val_singletons": val_singletons,
            "val_multi_match": val_multi,
        },
        "logistic_regression": lr_metrics,
        "lightgbm": lgb_metrics,
        "error_analysis": error_analysis,
        "memory_diagnostics": {
            "baseline_rss_mb": rss_baseline,
            "peak_rss_mb": peak_rss,
            "delta_rss_mb": round(peak_rss - rss_baseline, 2),
            "feature_cache_size_mb": cache_file_size_mb,
        },
        "timing_breakdown": timing,
    }
    report_path = o_dir / "turn5_5_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Artifacts successfully saved:")
    print(f"  - Model:         {best_model_path}")
    print(f"  - Threshold:     {thresh_path}")
    print(f"  - Diagnostics:   {diag_path}")
    print(f"  - Error Report:  {err_path}")
    print(f"  - Full Report:   {report_path}")
    print(f"Peak Process RSS:  {peak_rss:.2f} MB (Delta: +{peak_rss - rss_baseline:.2f} MB)")
    print(f"Total Experiment:  {total_time}s")
    print("=" * 80)
    print()

    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Turn 5.5 large-scale validation sanity check.")
    parser.add_argument("--sample-source1", type=int, default=5000)
    parser.add_argument("--distractors", type=int, default=5000)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    c_dir = Path(args.cache_dir) if args.cache_dir else None
    o_dir = Path(args.output_dir) if args.output_dir else None

    run_scale_check(
        sample_s1_count=args.sample_source1,
        distractor_count=args.distractors,
        val_size=args.val_size,
        random_seed=args.random_seed,
        cache_dir=c_dir,
        output_dir=o_dir,
    )
