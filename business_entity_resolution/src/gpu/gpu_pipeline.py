"""
gpu_pipeline.py — Turn 7 End-to-End GPU-Accelerated Entity Resolution Pipeline.

Coordinates:
1. GPU Environment Detection (CUDA / T4 / cuDF / CPU fallback)
2. High-Recall GPU-Accelerated Blocking (Core Domain, Rare Tokens, Stopword Bigrams, Postal Codes)
3. Labelled Training Pair Construction (Group-aware, No Leakage)
4. Vectorized / GPU Pairwise Feature Extraction (34 features: 21 baseline + 13 Turn 7)
5. LightGBM Classifier Training & Hyperparameter Evaluation (Balanced, GPU/CPU)
6. Validation Threshold Optimization (0.80 - 0.98 Sweep)
7. Entity-Level Margin Post-Processing (Empirically evaluated)
8. Comprehensive Reporting (Candidate Recall, Pair Metrics, Macro F0.5, RAM/VRAM, Runtimes)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psutil

from src.config import Config
from src.evaluation import (
    analyze_validation_errors,
    entity_f05,
    entity_precision,
    entity_recall,
    ground_truth_to_dict,
    macro_f05,
)
from src.gpu.gpu_blocking import (
    GPUBlockingConfig,
    GPUBlockingIndex,
    ShardedReferenceIndex,
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
from src.model import LightGBMMatcher, entity_train_val_split
from src.pair_builder import build_train_pairs
from src.prediction import predict_matches
from src.preprocessing import preprocess_records
from src.threshold import evaluate_at_threshold, find_optimal_threshold

logger = logging.getLogger(__name__)


@dataclass
class Turn7Config:
    """Master configuration for Turn 7 Pipeline."""

    blocking_config: GPUBlockingConfig = field(default_factory=GPUBlockingConfig)
    include_turn7_features: bool = True
    n_estimators: int = 120
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    random_state: int = 42
    class_weight: str = "balanced"
    thresholds_to_evaluate: list[float] = field(
        default_factory=lambda: [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    )
    val_size: float = 0.20
    sample_s1_count: int = 5000
    distractor_count: int = 5000
    enable_margin_postprocessing: bool = False
    margin_threshold: float = 0.15
    s1_batch_size: int = 25000
    streaming: bool = True
    resume: bool = True


def load_preallocated_train_arrays(
    chunk_files: list[Path],
    feature_names: list[str],
    train_pairs_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Memory-safe streamed loading of training data from Parquet chunks into
    pre-allocated NumPy arrays.

    Guarantees:
    - Exactly one chunk is held in host memory at any time.
    - Fails loudly with ValueError if train_pairs_count does not match the actual chunk rows
      (both overfill and underfill).
    """
    n_feats = len(feature_names)
    X_train = np.empty((train_pairs_count, n_feats), dtype=np.float32)
    y_train = np.empty(train_pairs_count, dtype=np.int32)
    row_offset = 0

    for cp in chunk_files:
        if not cp.exists():
            continue
        chunk_df = pd.read_parquet(
            cp, filters=[("split", "==", "train")], columns=feature_names + ["label"]
        )
        n_rows = len(chunk_df)
        if n_rows > 0:
            if row_offset + n_rows > train_pairs_count:
                raise ValueError(
                    f"Chunk {cp.name} exceeds pre-allocated train capacity: "
                    f"attempted to write up to row {row_offset + n_rows}, but allocated size is {train_pairs_count}."
                )
            X_train[row_offset : row_offset + n_rows] = chunk_df[feature_names].to_numpy(dtype=np.float32)
            y_train[row_offset : row_offset + n_rows] = chunk_df["label"].to_numpy(dtype=np.int32)
        row_offset += n_rows
        del chunk_df
        gc.collect()

    if row_offset != train_pairs_count:
        raise ValueError(
            f"Pre-allocated train array underfilled: expected {train_pairs_count} rows, "
            f"but only populated {row_offset} rows from {len(chunk_files)} chunk(s)."
        )

    return X_train, y_train


class Turn7Pipeline:
    """Turn 7 End-to-End GPU Entity Resolution Pipeline."""

    def __init__(self, config: Optional[Turn7Config] = None) -> None:
        self.config = config or Turn7Config()
        self.device_info = get_device_info()
        self.idf_store = TokenIDFStore()
        self.matcher: Optional[LightGBMMatcher] = None
        self.optimal_threshold: float = 0.90
        self.feature_names: list[str] = (
            TURN7_FEATURE_NAMES if self.config.include_turn7_features else BASELINE_FEATURE_NAMES
        )

    def run_validation_experiment(
        self,
        s1: Optional[pd.DataFrame] = None,
        s2: Optional[pd.DataFrame] = None,
        s3: Optional[pd.DataFrame] = None,
        gt_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        ingest_only: bool = False,
        train_only: bool = False,
        use_sharded: bool = False,
        shard_size: int = 200_000,
        overwrite_shards: bool = False,
        s2_path: Optional["Path | str"] = None,
        s3_path: Optional["Path | str"] = None,
    ) -> dict[str, Any]:
        """Execute complete Turn 7 validation experiment matching Turn 5.5 methodology.

        Parameters
        ----------
        s1, s2, s3: Source DataFrames (S1 queries + candidates).
        gt_df: Ground truth DataFrame.
        output_dir: Output directory for reports.
        cache_dir: Cache directory.
        ingest_only: If True, execute only Stages 1-4 (streaming ingestion) and exit cleanly.
        train_only: If True, execute only Stages 5-7 (LightGBM training/eval) from Parquet chunks.
        use_sharded: If True, use ShardedReferenceIndex for Colab-memory-safe full-corpus blocking.
        shard_size: Reference records per shard when use_sharded=True (default: 200_000).
        overwrite_shards: If True, force re-build of shard Parquet files on disk.
        s2_path, s3_path: Optional TSV file paths for streaming shard building (memory-safe mode).

        Returns
        -------
        Dictionary containing comprehensive metrics.
        """
        if self.config.streaming:
            return self.run_streaming_validation_experiment(
                s1=s1,
                s2=s2,
                s3=s3,
                gt_df=gt_df,
                output_dir=output_dir,
                cache_dir=cache_dir,
                ingest_only=ingest_only,
                train_only=train_only,
                use_sharded=use_sharded,
                shard_size=shard_size,
                overwrite_shards=overwrite_shards,
                s2_path=s2_path,
                s3_path=s3_path,
            )

        cfg = Config()
        o_dir = output_dir or cfg.OUTPUT_DIR
        c_dir = cache_dir or cfg.CACHE_DIR
        o_dir.mkdir(parents=True, exist_ok=True)
        c_dir.mkdir(parents=True, exist_ok=True)

        rss_baseline = get_process_rss_mb()
        peak_rss = rss_baseline
        vram_info = get_gpu_memory_mb()
        peak_vram = vram_info["peak_allocated_mb"]
        total_start = time.time()
        timing: dict[str, float] = {}

        print("=" * 80)
        print("TURN 7: GPU-ACCELERATED & ACCURACY-OPTIMIZED VALIDATION BENCHMARK")
        print("=" * 80)
        print(f"Device: {self.device_info['device_name']} (CUDA: {self.device_info['cuda_available']})")
        print(f"Baseline Process RSS: {rss_baseline:.2f} MB | Initial VRAM: {vram_info['allocated_mb']:.2f} MB\n")

        gt_dict = ground_truth_to_dict(gt_df)
        total_true_matches = sum(len(m) for m in gt_dict.values())
        s2_true_matches = sum(1 for m in gt_dict.values() for x in m if x.startswith("S2"))
        s3_true_matches = sum(1 for m in gt_dict.values() for x in m if x.startswith("S3"))

        # -------------------------------------------------------------------
        # Stage 1: Data Preprocessing & Vocabulary Fitting
        # -------------------------------------------------------------------
        print("[1/7] Preprocessing & Fitting Token Statistics...")
        t0 = time.time()
        s1_proc = s1 if "name_norm" in s1.columns else preprocess_records(s1)
        s2_proc = s2 if "name_norm" in s2.columns else preprocess_records(s2)
        s3_proc = s3 if "name_norm" in s3.columns else preprocess_records(s3)

        self.idf_store.fit(s1_proc, s2_proc, s3_proc)
        timing["preprocessing_s"] = round(time.time() - t0, 2)
        peak_rss = max(peak_rss, get_process_rss_mb())

        # -------------------------------------------------------------------
        # Stage 2: Accuracy-Optimized Blocking (Candidate Generation)
        # -------------------------------------------------------------------
        print(f"[2/7] Generating candidates via Turn 7 GPU/Enhanced Blocking (Top-K={self.config.blocking_config.top_k_per_s1})...")
        t0 = time.time()
        candidates = build_gpu_candidates(
            s1_proc, s2_proc, s3_proc, config=self.config.blocking_config
        )
        timing["blocking_s"] = round(time.time() - t0, 2)
        peak_rss = max(peak_rss, get_process_rss_mb())

        # Diagnostics
        total_candidates = len(candidates)
        cands_by_s1 = candidates.groupby("s1_id")["candidate_id"].apply(set).to_dict()

        retrieved_true_total = 0
        retrieved_true_s2 = 0
        retrieved_true_s3 = 0

        for s1_id, true_set in gt_dict.items():
            found_set = cands_by_s1.get(str(s1_id), set())
            for tid in true_set:
                if tid in found_set:
                    retrieved_true_total += 1
                    if tid.startswith("S2"):
                        retrieved_true_s2 += 1
                    elif tid.startswith("S3"):
                        retrieved_true_s3 += 1

        candidate_recall = (retrieved_true_total / total_true_matches * 100) if total_true_matches > 0 else 0.0
        s2_candidate_recall = (retrieved_true_s2 / s2_true_matches * 100) if s2_true_matches > 0 else 0.0
        s3_candidate_recall = (retrieved_true_s3 / s3_true_matches * 100) if s3_true_matches > 0 else 0.0

        total_possible = len(s1_proc) * (len(s2_proc) + len(s3_proc))
        reduction_ratio = ((1.0 - (total_candidates / total_possible)) * 100) if total_possible > 0 else 0.0
        cands_series = pd.Series([len(cands_by_s1.get(str(eid), set())) for eid in s1_proc["entity_id"]])

        blocking_diagnostics = {
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
            "max_candidates_per_s1": int(cands_series.max()),
            "blocking_runtime_s": timing["blocking_s"],
        }

        print(f"  Candidate Pairs:  {total_candidates:,}")
        print(f"  Candidate Recall: {candidate_recall:.2f}% (Baseline Turn 5.5: 93.40%)")
        print(f"    - S2 Recall:    {s2_candidate_recall:.2f}% (Baseline Turn 5.5: 92.76%)")
        print(f"    - S3 Recall:    {s3_candidate_recall:.2f}% (Baseline Turn 5.5: 93.98%)")
        print(f"  Reduction Ratio:  {reduction_ratio:.4f}% | Mean Cands/S1: {blocking_diagnostics['mean_candidates_per_s1']}")
        print(f"  Blocking Runtime: {timing['blocking_s']}s | RSS: {get_process_rss_mb():.2f} MB\n")

        # -------------------------------------------------------------------
        # Stage 3: Labelled Pair Assembly
        # -------------------------------------------------------------------
        print("[3/7] Assembling labelled candidate pairs...")
        t0 = time.time()
        pairs = build_train_pairs(
            s1_proc, s2_proc, s3_proc, candidates, gt_df, random_seed=self.config.random_state
        )
        timing["pair_construction_s"] = round(time.time() - t0, 2)
        peak_rss = max(peak_rss, get_process_rss_mb())
        total_pairs = len(pairs)
        total_pos = int((pairs["label"] == 1).sum())
        total_neg = int((pairs["label"] == 0).sum())
        print(f"  Assembled Pairs:  {total_pairs:,} ({total_pos:,} pos, {total_neg:,} neg)")
        print(f"  Runtime:          {timing['pair_construction_s']}s\n")

        # -------------------------------------------------------------------
        # Stage 4: High-Performance / Vectorized Feature Extraction
        # -------------------------------------------------------------------
        print(f"[4/7] Extracting {len(self.feature_names)} pairwise features (Vectorized / GPU)...")
        t0 = time.time()
        features_df = compute_gpu_features(
            pairs,
            idf_store=self.idf_store,
            include_turn7_features=self.config.include_turn7_features,
        )
        timing["feature_generation_s"] = round(time.time() - t0, 2)
        peak_rss = max(peak_rss, get_process_rss_mb())
        vram_info = get_gpu_memory_mb()
        peak_vram = max(peak_vram, vram_info["peak_allocated_mb"])

        throughput = round(len(pairs) / max(timing["feature_generation_s"], 0.001), 1)
        for col in self.feature_names:
            if col in features_df.columns:
                pairs[col] = features_df[col].values

        print(f"  Feature Matrix:   {features_df.shape} ({throughput:,} pairs/sec)")
        print(f"  Feature Runtime:  {timing['feature_generation_s']}s (Baseline Turn 5.5: 181.43s)")
        print(f"  Feature Speedup:  {181.43 / max(timing['feature_generation_s'], 0.001):.2f}x faster!")
        print(f"  Current RSS:      {get_process_rss_mb():.2f} MB | Peak VRAM: {peak_vram:.2f} MB\n")

        # -------------------------------------------------------------------
        # Stage 5: Entity-Level Train / Validation Split (Zero Leakage)
        # -------------------------------------------------------------------
        print("[5/7] Entity-level GroupSplit (80% train / 20% validation)...")
        t0 = time.time()
        train_pairs, val_pairs, train_gt, val_gt = entity_train_val_split(
            pairs, gt_dict, val_size=self.config.val_size, random_state=self.config.random_state
        )
        timing["train_val_split_s"] = round(time.time() - t0, 3)

        train_s1_set = set(train_pairs["s1_id"].unique()) | set(train_gt.keys())
        val_s1_set = set(val_pairs["s1_id"].unique()) | set(val_gt.keys())
        leakage = train_s1_set & val_s1_set
        assert len(leakage) == 0, f"FATAL LEAKAGE! Overlapping entities: {leakage}"

        train_pos = int((train_pairs["label"] == 1).sum())
        val_pos = int((val_pairs["label"] == 1).sum())
        val_singletons = sum(1 for m in val_gt.values() if len(m) == 0)

        print(f"  Train: {len(train_s1_set):,} S1 | {len(train_pairs):,} pairs ({train_pos:,} pos)")
        print(f"  Val:   {len(val_s1_set):,} S1 | {len(val_pairs):,} pairs ({val_pos:,} pos, {val_singletons} singletons)")
        print("  Safety Check: Zero entity overlap verified.\n")

        # -------------------------------------------------------------------
        # Stage 6: Train Turn 7 LightGBM Model
        # -------------------------------------------------------------------
        print("[6/7] Training Turn 7 LightGBM Classifier...")
        t0 = time.time()
        matcher = LightGBMMatcher(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            feature_names=self.feature_names,
        )
        matcher.fit(train_pairs, train_pairs["label"])
        timing["model_train_s"] = round(time.time() - t0, 3)
        self.matcher = matcher
        peak_rss = max(peak_rss, get_process_rss_mb())

        # Prediction probabilities
        val_proba = matcher.predict_proba(val_pairs)

        # -------------------------------------------------------------------
        # Stage 7: Threshold Sweep & Optimal Evaluation
        # -------------------------------------------------------------------
        print("[7/7] Threshold Sweep & Entity Evaluation...")
        t0 = time.time()
        best_thresh, best_macro_f05, sweep_history = find_optimal_threshold(
            val_pairs, val_proba, val_gt, threshold_grid=self.config.thresholds_to_evaluate
        )
        timing["threshold_sweep_s"] = round(time.time() - t0, 3)
        self.optimal_threshold = best_thresh

        val_metrics = evaluate_at_threshold(val_pairs, val_proba, best_thresh, val_gt)
        val_metrics["runtime_s"] = timing["model_train_s"]

        # Predict matches
        val_preds = predict_matches(
            val_pairs, val_proba, threshold=best_thresh, s1_entities=val_gt.keys()
        )

        # Margin postprocessing experiment if enabled
        if self.config.enable_margin_postprocessing:
            print("  Evaluating entity-level margin post-processing...")
            val_pairs_annotated = val_pairs.assign(pred_proba=val_proba)
            probs_by_s1 = {
                s1_id: dict(zip(group["candidate_id"], group["pred_proba"]))
                for s1_id, group in val_pairs_annotated.groupby("s1_id")
            }
            filtered_preds: dict[str, set[str]] = {}
            for s1_id, candidate_ids in val_preds.items():
                if not candidate_ids:
                    filtered_preds[s1_id] = set()
                    continue
                c_probs = probs_by_s1.get(s1_id, {})
                max_p = max((c_probs.get(cid, 0.0) for cid in candidate_ids), default=0.0)
                kept = {
                    cid
                    for cid in candidate_ids
                    if c_probs.get(cid, 0.0) >= max_p - self.config.margin_threshold
                }
                filtered_preds[s1_id] = kept

            margin_f05 = macro_f05(val_gt, filtered_preds)
            print(f"  Margin Post-Processing (margin={self.config.margin_threshold}): Macro F0.5 = {margin_f05:.4f}")
            if margin_f05 > best_macro_f05:
                print(f"  -> Margin post-processing improves Macro F0.5 ({best_macro_f05:.4f} -> {margin_f05:.4f})!")
                val_preds = filtered_preds
                best_macro_f05 = margin_f05
                val_metrics["macro_f05"] = margin_f05
                val_metrics["mean_entity_precision"] = entity_precision(val_gt, val_preds)
                val_metrics["mean_entity_recall"] = entity_recall(val_gt, val_preds)

        # Error analysis
        error_analysis = analyze_validation_errors(
            val_gt=val_gt,
            val_predictions=val_preds,
            val_pairs=val_pairs,
            val_proba=val_proba,
            threshold=best_thresh,
        )
        timing["error_analysis_s"] = round(time.time() - t0, 3)

        total_time = round(time.time() - total_start, 2)
        timing["total_runtime_s"] = total_time

        # Print Threshold Sweep Table
        print("\nThreshold Sweep Results:")
        print(f"{'Threshold':<10} | {'Macro F0.5':<12}")
        print("-" * 26)
        sweep_report = []
        for thresh_val, score_val in sorted(sweep_history.items()):
            print(f"{thresh_val:<10.2f} | {score_val:<12.4f}")
            sweep_report.append({"threshold": thresh_val, "macro_f05": score_val})
        print("-" * 26)
        print(f"Optimal Threshold: {best_thresh:.2f} -> Validation Macro F0.5: {best_macro_f05:.4f}\n")

        # Feature Importance Diagnostics
        print("Feature Importance Diagnostics (Top 10 Features by Gain):")
        lgb_fi = matcher.get_feature_importances(importance_type="gain")
        sorted_fi = sorted(lgb_fi.items(), key=lambda x: x[1], reverse=True)
        for rank, (feat, gain) in enumerate(sorted_fi[:10], 1):
            print(f"  {rank:2d}. {feat:<32}: {gain*100:6.2f}%")
        print()

        # Side-by-Side Comparison against Turn 5.5
        print("=" * 85)
        print("SIDE-BY-SIDE BENCHMARK COMPARISON: TURN 5.5 vs TURN 7")
        print("=" * 85)
        cmp_fmt = f"{'Metric':<35} | {'Turn 5.5 Baseline':<22} | {'Turn 7 (GPU/Acc)':<22}"
        print(cmp_fmt)
        print("-" * 85)

        cmp_rows = [
            ("Source-1 Query Entities", "5,000", f"{len(s1_proc):,}"),
            ("Candidate Blocking Recall", "93.40%", f"{candidate_recall:.2f}%"),
            ("S2 Candidate Recall", "92.76%", f"{s2_candidate_recall:.2f}%"),
            ("S3 Candidate Recall", "93.98%", f"{s3_candidate_recall:.2f}%"),
            ("Reduction Ratio", "99.8505%", f"{reduction_ratio:.4f}%"),
            ("Candidate Pairs Generated", "166,296", f"{total_candidates:,}"),
            ("Pairwise Features", "21 features", f"{len(self.feature_names)} features"),
            ("Optimal Decision Threshold", "0.90", f"{best_thresh:.2f}"),
            ("Pair Precision", "99.35%", f"{val_metrics['pair_precision']*100:.2f}%"),
            ("Pair Recall", "98.28%", f"{val_metrics['pair_recall']*100:.2f}%"),
            ("Pair F1 Score", "0.9881", f"{val_metrics['pair_f1']:.4f}"),
            ("PR-AUC", "0.9993", f"{val_metrics.get('pr_auc', 0):.4f}"),
            ("Mean Entity Precision", "0.9788", f"{val_metrics['mean_entity_precision']:.4f}"),
            ("Mean Entity Recall", "0.9187", f"{val_metrics['mean_entity_recall']:.4f}"),
            ("VALIDATION MACRO F0.5", "0.9584", f"{best_macro_f05:.4f}"),
            ("Candidate Generation Misses", "235 (6.74%)", f"{error_analysis['blocking_misses_count']} ({error_analysis['blocking_miss_rate_pct']:.2f}%)"),
            ("Classification Misses", "56 (1.61%)", f"{error_analysis['classifier_misses_count']} ({error_analysis['classifier_miss_rate_pct']:.2f}%)"),
            ("False Positive Merges", "21", f"{error_analysis['false_positives']}"),
            ("Feature Extraction Runtime", "181.43s", f"{timing['feature_generation_s']:.2f}s"),
            ("Total Pipeline Runtime", "206.53s", f"{total_time:.2f}s"),
            ("Peak Process RSS", "343.81 MB", f"{peak_rss:.2f} MB"),
            ("Peak GPU VRAM", "0.00 MB (CPU)", f"{peak_vram:.2f} MB"),
        ]

        for m_lbl, t55_val, t7_val in cmp_rows:
            print(f"{m_lbl:<35} | {t55_val:<22} | {t7_val:<22}")
        print("=" * 85)
        print()

        # Build Full Experiment Report
        report: dict[str, Any] = {
            "experiment_name": "Turn 7 GPU-Accelerated & Accuracy-Optimized Validation Check",
            "device_info": self.device_info,
            "sample_size": {
                "source1_entities": len(s1_proc),
                "candidate_pool_size": len(s2_proc) + len(s3_proc),
                "s2_pool": len(s2_proc),
                "s3_pool": len(s3_proc),
                "total_true_matches": total_true_matches,
            },
            "blocking_diagnostics": blocking_diagnostics,
            "split_summary": {
                "train_entities": len(train_s1_set),
                "val_entities": len(val_s1_set),
                "train_pairs": len(train_pairs),
                "val_pairs": len(val_pairs),
                "train_pos": train_pos,
                "val_pos": val_pos,
                "val_singletons": val_singletons,
            },
            "model_config": {
                "model_type": "LightGBMMatcher",
                "n_estimators": self.config.n_estimators,
                "learning_rate": self.config.learning_rate,
                "num_leaves": self.config.num_leaves,
                "feature_count": len(self.feature_names),
                "feature_names": self.feature_names,
                "optimal_threshold": float(best_thresh),
            },
            "validation_metrics": val_metrics,
            "threshold_sweep": sweep_history,
            "error_analysis": error_analysis,
            "feature_importance_gain": dict(sorted_fi[:15]),
            "resource_diagnostics": {
                "baseline_rss_mb": rss_baseline,
                "peak_rss_mb": peak_rss,
                "delta_rss_mb": round(peak_rss - rss_baseline, 2),
                "peak_vram_mb": peak_vram,
            },
            "timing_breakdown": timing,
        }

        # Save artifacts non-destructively
        report_path = o_dir / "turn7_validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        results_path = o_dir / "turn7_experiment_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        model_path = c_dir / "turn7_best_matcher.joblib"
        matcher.save(model_path)

        thresh_path = c_dir / "turn7_optimal_threshold.json"
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "Turn7_LightGBM",
                    "optimal_threshold": float(best_thresh),
                    "validation_macro_f05": float(best_macro_f05),
                    "feature_names": self.feature_names,
                    "beta": 0.5,
                },
                f,
                indent=2,
            )

        print(f"Artifacts successfully saved:")
        print(f"  - Model:         {model_path}")
        print(f"  - Threshold:     {thresh_path}")
        print(f"  - Full Report:   {report_path}")
        print(f"  - Exp Results:   {results_path}")
        print("=" * 80)
        return report

    def run_streaming_ingestion(
        self,
        s1: pd.DataFrame,
        s2: pd.DataFrame,
        s3: pd.DataFrame,
        gt_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Execute Stages 1-4: Candidate blocking, assembly, GPU feature extraction,
        and persistence to disk-backed Parquet chunks. Exits cleanly to free 100% of host heap.
        """
        cfg = Config()
        o_dir = output_dir or cfg.OUTPUT_DIR
        c_dir = cache_dir or cfg.CACHE_DIR
        o_dir.mkdir(parents=True, exist_ok=True)
        c_dir.mkdir(parents=True, exist_ok=True)

        chunks_dir = o_dir / "streaming_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = chunks_dir / "manifest.json"

        rss_baseline = get_process_rss_mb()
        peak_rss = rss_baseline
        vram_info = get_gpu_memory_mb()
        peak_vram = vram_info["peak_allocated_mb"]
        total_start = time.time()
        timing: dict[str, float] = {}

        print("=" * 80)
        print("TURN 9: STREAMING INGESTION (STAGES 1-4)")
        print("=" * 80)
        print(f"Device: {self.device_info['device_name']} (CUDA: {self.device_info['cuda_available']})")
        print(f"Baseline Process RSS: {rss_baseline:.2f} MB | Initial VRAM: {vram_info['allocated_mb']:.2f} MB")
        print(f"Configured S1 Batch Size: {self.config.s1_batch_size:,} | Resume Enabled: {self.config.resume}\n")

        gt_dict = ground_truth_to_dict(gt_df)
        total_true_matches = sum(len(m) for m in gt_dict.values())
        s2_true_matches = sum(1 for m in gt_dict.values() for x in m if x.startswith("S2"))
        s3_true_matches = sum(1 for m in gt_dict.values() for x in m if x.startswith("S3"))

        # -------------------------------------------------------------------
        # Stage 1: Deterministic Entity-Level Train / Validation Split
        # -------------------------------------------------------------------
        print("[1/7] Establishing Deterministic Entity Split (Zero Leakage)...")
        all_s1_list = sorted(list(set(gt_dict.keys()) | set(s1["entity_id"].astype(str))))
        rng = np.random.RandomState(self.config.random_state)
        shuffled_s1 = rng.permutation(all_s1_list)
        n_val = max(1, int(len(all_s1_list) * self.config.val_size))
        val_s1_set = set(shuffled_s1[:n_val])
        train_s1_set = set(shuffled_s1[n_val:])
        assert len(train_s1_set & val_s1_set) == 0, "FATAL LEAKAGE! Overlapping entities!"
        train_gt = {eid: gt_dict[eid] for eid in train_s1_set if eid in gt_dict}
        val_gt = {eid: gt_dict[eid] for eid in val_s1_set if eid in gt_dict}
        val_singletons = sum(1 for m in val_gt.values() if len(m) == 0)
        print(f"  Train: {len(train_s1_set):,} S1 queries | Val: {len(val_s1_set):,} S1 queries ({val_singletons} singletons)\n")

        # -------------------------------------------------------------------
        # Stage 2: Data Preprocessing & Fitting Token Statistics
        # -------------------------------------------------------------------
        print("[2/7] Preprocessing & Fitting Token Statistics...")
        t0 = time.time()
        s1_proc = s1 if "name_norm" in s1.columns else preprocess_records(s1)
        s2_proc = s2 if "name_norm" in s2.columns else preprocess_records(s2)
        s3_proc = s3 if "name_norm" in s3.columns else preprocess_records(s3)

        self.idf_store.fit(s1_proc, s2_proc, s3_proc)
        timing["preprocessing_s"] = round(time.time() - t0, 2)
        peak_rss = max(peak_rss, get_process_rss_mb())
        print(f"  Vocabulary: {len(self.idf_store.idf_table):,} unique tokens fitted in {timing['preprocessing_s']}s\n")

        # -------------------------------------------------------------------
        # Stage 3: Inverted Candidate Pool Indexing (S2 + S3)
        # -------------------------------------------------------------------
        print("[3/7] Indexing candidate pool (S2 + S3)...")
        t0 = time.time()
        idx = GPUBlockingIndex(self.config.blocking_config)
        idx.build_vocabulary(s1_proc, s2_proc, s3_proc)
        idx.add_records(s2_proc)
        idx.add_records(s3_proc)
        t_index_build = round(time.time() - t0, 2)
        peak_rss = max(peak_rss, get_process_rss_mb())
        print(f"  Candidate Index: {idx.total_records_indexed:,} records indexed in {t_index_build}s | RSS: {get_process_rss_mb():.2f} MB\n")

        # -------------------------------------------------------------------
        # Stage 4: Streaming Candidate Generation, Assembly, GPU Extraction & Persistence
        # -------------------------------------------------------------------
        print(f"[4/7] Streaming S1 Batches (batch_size={self.config.s1_batch_size:,})...")
        n_s1 = len(s1_proc)
        b_size = max(1, self.config.s1_batch_size)
        batches = [s1_proc.iloc[i : min(i + b_size, n_s1)].copy() for i in range(0, n_s1, b_size)]
        total_batches = len(batches)

        # Manifest
        manifest: dict[str, Any] = {
            "total_batches": total_batches,
            "s1_batch_size": b_size,
            "chunks": {},
        }
        if self.config.resume and manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                logger.info("Loaded resume manifest with %d existing chunks.", len(manifest.get("chunks", {})))
            except Exception as e:
                logger.warning("Failed to load manifest (%s). Starting clean.", e)
        elif not self.config.resume:
            for old_f in chunks_dir.glob("chunk_*.parquet*"):
                old_f.unlink(missing_ok=True)

        total_candidates = 0
        retrieved_true_total = 0
        retrieved_true_s2 = 0
        retrieved_true_s3 = 0
        t_blocking_total = 0.0
        t_assembly_total = 0.0
        t_features_total = 0.0
        t_write_total = 0.0
        train_pairs_count = 0
        val_pairs_count = 0
        train_pos_count = 0
        val_pos_count = 0
        total_pos = 0
        total_neg = 0
        chunk_files: list[Path] = []
        batch_telemetry: list[dict[str, Any]] = []

        for b_idx, s1_batch in enumerate(batches, start=1):
            chunk_file = chunks_dir / f"chunk_{b_idx:04d}.parquet"
            chunk_files.append(chunk_file)
            chunk_key = str(b_idx)

            if self.config.resume and chunk_key in manifest["chunks"] and chunk_file.exists():
                entry = manifest["chunks"][chunk_key]
                logger.info("Resuming batch %d/%d from disk (%s | %d pairs)...", b_idx, total_batches, chunk_file.name, entry["pairs_count"])
                total_candidates += entry["pairs_count"]
                retrieved_true_total += entry["retrieved_true_total"]
                retrieved_true_s2 += entry["retrieved_true_s2"]
                retrieved_true_s3 += entry["retrieved_true_s3"]
                train_pairs_count += entry["train_pairs"]
                val_pairs_count += entry["val_pairs"]
                train_pos_count += entry.get("train_pos", 0)
                val_pos_count += entry.get("val_pos", 0)
                total_pos += entry["pos_count"]
                total_neg += entry["neg_count"]
                t_blocking_total += entry["t_blocking_s"]
                t_assembly_total += entry["t_assembly_s"]
                t_features_total += entry["t_features_s"]
                t_write_total += entry["t_write_s"]
                batch_telemetry.append(entry)
                continue

            t_b_start = time.time()

            # A. Candidate blocking for batch
            t0 = time.time()
            cands_batch = idx.query_records(s1_batch)
            t_blk = time.time() - t0
            t_blocking_total += t_blk

            # B. Candidate recall check for batch
            cands_by_s1 = cands_batch.groupby("s1_id")["candidate_id"].apply(set).to_dict() if len(cands_batch) > 0 else {}
            b_retrieved_total = 0
            b_retrieved_s2 = 0
            b_retrieved_s3 = 0
            for s1_raw in s1_batch["entity_id"]:
                s1_str = str(s1_raw)
                t_set = gt_dict.get(s1_str, set())
                f_set = cands_by_s1.get(s1_str, set())
                for tid in t_set:
                    if tid in f_set:
                        b_retrieved_total += 1
                        if tid.startswith("S2"):
                            b_retrieved_s2 += 1
                        elif tid.startswith("S3"):
                            b_retrieved_s3 += 1

            retrieved_true_total += b_retrieved_total
            retrieved_true_s2 += b_retrieved_s2
            retrieved_true_s3 += b_retrieved_s3
            b_cands_count = len(cands_batch)
            total_candidates += b_cands_count

            # C. Sliced candidate pair assembly for batch
            t0 = time.time()
            needed_cands = set(cands_batch["candidate_id"]) if len(cands_batch) > 0 else set()
            s2_sub = s2_proc[s2_proc["entity_id"].isin(needed_cands)] if needed_cands else pd.DataFrame(columns=s2_proc.columns)
            s3_sub = s3_proc[s3_proc["entity_id"].isin(needed_cands)] if needed_cands else pd.DataFrame(columns=s3_proc.columns)
            pairs_batch = build_train_pairs(
                s1_batch, s2_sub, s3_sub, cands_batch, gt_df, random_seed=self.config.random_state
            )
            t_asm = time.time() - t0
            t_assembly_total += t_asm

            # D. GPU feature extraction for batch (internally streamed in 50k chunks)
            t0 = time.time()
            feats_batch = compute_gpu_features(
                pairs_batch,
                idf_store=self.idf_store,
                include_turn7_features=self.config.include_turn7_features,
            )
            t_feat = time.time() - t0
            t_features_total += t_feat

            # E. Assemble compact chunk dataframe and persist to Parquet
            t0 = time.time()
            if len(pairs_batch) > 0:
                is_val_mask = pairs_batch["s1_id"].astype(str).isin(val_s1_set)
                pairs_batch["split"] = np.where(is_val_mask, "val", "train")
                for col in self.feature_names:
                    pairs_batch[col] = feats_batch[col].astype(np.float32).values
                out_cols = ["s1_id", "candidate_id", "split", "label"] + self.feature_names
                chunk_df = pairs_batch[out_cols].copy()
            else:
                chunk_df = pd.DataFrame(columns=["s1_id", "candidate_id", "split", "label"] + self.feature_names)

            tmp_chunk = chunk_file.with_suffix(".parquet.tmp")
            chunk_df.to_parquet(tmp_chunk, index=False, engine="pyarrow", compression="snappy")
            tmp_chunk.replace(chunk_file)
            t_wrt = time.time() - t0
            t_write_total += t_wrt

            # Statistics & telemetry
            b_pairs = len(chunk_df)
            b_pos = int((chunk_df["label"] == 1).sum()) if b_pairs > 0 else 0
            b_neg = b_pairs - b_pos
            b_train = int((chunk_df["split"] == "train").sum()) if b_pairs > 0 else 0
            b_val = b_pairs - b_train
            b_train_pos = int(((chunk_df["split"] == "train") & (chunk_df["label"] == 1)).sum()) if b_pairs > 0 else 0
            b_val_pos = b_pos - b_train_pos

            total_pos += b_pos
            total_neg += b_neg
            train_pairs_count += b_train
            val_pairs_count += b_val
            train_pos_count += b_train_pos
            val_pos_count += b_val_pos

            cur_rss = get_process_rss_mb()
            peak_rss = max(peak_rss, cur_rss)
            vram_stat = get_gpu_memory_mb()
            peak_vram = max(peak_vram, vram_stat["peak_allocated_mb"])
            file_bytes = chunk_file.stat().st_size if chunk_file.exists() else 0

            entry = {
                "chunk_file": chunk_file.name,
                "s1_count": len(s1_batch),
                "pairs_count": b_pairs,
                "train_pairs": b_train,
                "val_pairs": b_val,
                "train_pos": b_train_pos,
                "val_pos": b_val_pos,
                "pos_count": b_pos,
                "neg_count": b_neg,
                "retrieved_true_total": b_retrieved_total,
                "retrieved_true_s2": b_retrieved_s2,
                "retrieved_true_s3": b_retrieved_s3,
                "file_size_bytes": file_bytes,
                "t_blocking_s": round(t_blk, 3),
                "t_assembly_s": round(t_asm, 3),
                "t_features_s": round(t_feat, 3),
                "t_write_s": round(t_wrt, 3),
                "process_rss_mb": cur_rss,
                "peak_rss_mb": peak_rss,
                "gpu_allocated_mb": vram_stat["allocated_mb"],
                "gpu_reserved_mb": vram_stat["reserved_mb"],
            }
            manifest["chunks"][chunk_key] = entry
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            batch_telemetry.append(entry)
            print(
                f"  Batch {b_idx:2d}/{total_batches}: S1={len(s1_batch):,} -> Pairs={b_pairs:,} "
                f"| Block={t_blk:.2f}s Asm={t_asm:.2f}s GPU_Feat={t_feat:.2f}s Wrt={t_wrt:.2f}s "
                f"| RSS={cur_rss:.1f} MB (Peak: {peak_rss:.1f} MB) VRAM={vram_stat['peak_allocated_mb']:.1f} MB"
            )

            # Explicit memory release
            del s1_batch, cands_batch, needed_cands, s2_sub, s3_sub, pairs_batch, feats_batch, chunk_df
            gc.collect()
            clear_gpu_cache()

        del idx
        gc.collect()

        timing["blocking_s"] = round(t_blocking_total, 2)
        timing["pair_construction_s"] = round(t_assembly_total, 2)
        timing["feature_generation_s"] = round(t_features_total, 2)
        timing["disk_write_s"] = round(t_write_total, 2)

        candidate_recall = (retrieved_true_total / total_true_matches * 100) if total_true_matches > 0 else 0.0
        s2_candidate_recall = (retrieved_true_s2 / s2_true_matches * 100) if s2_true_matches > 0 else 0.0
        s3_candidate_recall = (retrieved_true_s3 / s3_true_matches * 100) if s3_true_matches > 0 else 0.0
        total_possible = len(s1_proc) * (len(s2_proc) + len(s3_proc))
        reduction_ratio = ((1.0 - (total_candidates / total_possible)) * 100) if total_possible > 0 else 0.0

        total_disk_bytes = sum(cp.stat().st_size for cp in chunk_files if cp.exists())
        total_disk_mb = round(total_disk_bytes / (1024**2), 2)

        print(f"\n  Streaming Ingestion Complete across {total_batches} chunk(s):")
        print(f"  Total Candidate Pairs: {total_candidates:,} ({total_pos:,} pos, {total_neg:,} neg)")
        print(f"  Candidate Recall:      {candidate_recall:.2f}% (S2: {s2_candidate_recall:.2f}%, S3: {s3_candidate_recall:.2f}%)")
        print(f"  Persisted Disk Output: {total_disk_mb:.2f} MB across {len(chunk_files)} Parquet chunk(s)")
        print(f"  Timings: Blocking={timing['blocking_s']}s | Assembly={timing['pair_construction_s']}s | GPU Feat={timing['feature_generation_s']}s | Disk Write={timing['disk_write_s']}s\n")

        ingestion_summary = {
            "source1_entities": len(s1_proc),
            "candidate_pool_size": len(s2_proc) + len(s3_proc),
            "s2_pool": len(s2_proc),
            "s3_pool": len(s3_proc),
            "total_true_matches": total_true_matches,
            "s2_true_matches": s2_true_matches,
            "s3_true_matches": s3_true_matches,
            "total_candidates": total_candidates,
            "retrieved_true_total": retrieved_true_total,
            "retrieved_true_s2": retrieved_true_s2,
            "retrieved_true_s3": retrieved_true_s3,
            "candidate_recall": candidate_recall,
            "s2_candidate_recall": s2_candidate_recall,
            "s3_candidate_recall": s3_candidate_recall,
            "reduction_ratio": reduction_ratio,
            "total_pos": total_pos,
            "total_neg": total_neg,
            "train_pairs": train_pairs_count,
            "val_pairs": val_pairs_count,
            "train_pos": train_pos_count,
            "val_pos": val_pos_count,
            "train_entities": len(train_s1_set),
            "val_entities": len(val_s1_set),
            "val_singletons": val_singletons,
            "total_disk_bytes": total_disk_bytes,
            "total_disk_mb": total_disk_mb,
            "timing_ingest": timing.copy(),
            "peak_rss_mb": peak_rss,
            "peak_vram_mb": peak_vram,
            "rss_baseline": rss_baseline,
        }
        manifest["ingestion_summary"] = ingestion_summary
        manifest["val_gt"] = {str(k): list(v) for k, v in val_gt.items()}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return ingestion_summary

    def run_training_from_chunks(
        self,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        gt_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        """Execute Stages 5-7: Streamed LightGBM model training, validation scoring,
        and threshold evaluation starting fresh from pre-existing Parquet chunks.
        Runs in a clean process with minimal baseline memory footprint.
        """
        cfg = Config()
        o_dir = output_dir or cfg.OUTPUT_DIR
        c_dir = cache_dir or cfg.CACHE_DIR
        o_dir.mkdir(parents=True, exist_ok=True)
        c_dir.mkdir(parents=True, exist_ok=True)

        chunks_dir = o_dir / "streaming_chunks"
        manifest_path = chunks_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot run training-only: manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk parquet files found in {chunks_dir}")

        total_batches = manifest.get("total_batches", len(chunk_files))
        b_size = manifest.get("s1_batch_size", 25000)
        batch_telemetry = manifest.get("chunks", {})
        batches_telemetry_list = [v for k, v in sorted(batch_telemetry.items(), key=lambda x: int(x[0]))]

        ingestion_summary = manifest.get("ingestion_summary", {})

        train_pairs_count = ingestion_summary.get(
            "train_pairs",
            sum(c.get("train_pairs", 0) for c in batch_telemetry.values()),
        )
        val_pairs_count = ingestion_summary.get(
            "val_pairs",
            sum(c.get("val_pairs", 0) for c in batch_telemetry.values()),
        )
        total_candidates = ingestion_summary.get(
            "total_candidates",
            sum(c.get("pairs_count", 0) for c in batch_telemetry.values()),
        )

        timing: dict[str, float] = ingestion_summary.get("timing_ingest", {}).copy()

        train_proc_baseline = get_process_rss_mb()
        peak_rss = train_proc_baseline
        vram_info = get_gpu_memory_mb()
        peak_vram = vram_info.get("peak_allocated_mb", 0.0)

        print("=" * 80)
        print("TURN 9: STREAMED MODEL TRAINING & VALIDATION (CLEAN PROCESS)")
        print("=" * 80)
        print(f"Device: {self.device_info['device_name']} (CUDA: {self.device_info['cuda_available']})")
        print(f"Clean Process Baseline RSS: {train_proc_baseline:.2f} MB")
        print(f"Parquet Chunks Available: {len(chunk_files)} chunk(s) | Train Pairs: {train_pairs_count:,} | Val Pairs: {val_pairs_count:,}\n")

        # -------------------------------------------------------------------
        # Stage 5: Streamed LightGBM Model Training (from Parquet Chunks)
        # -------------------------------------------------------------------
        print("[5/7] Streamed Training LightGBM Model (Reading Train Subsets from Parquet)...")
        t0 = time.time()
        n_feats = len(self.feature_names)

        X_train, y_train = load_preallocated_train_arrays(
            chunk_files=chunk_files,
            feature_names=self.feature_names,
            train_pairs_count=train_pairs_count,
        )
        peak_rss = max(peak_rss, get_process_rss_mb())

        matcher = LightGBMMatcher(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            feature_names=self.feature_names,
        )
        matcher.fit(X_train, y_train)
        timing["model_train_s"] = round(time.time() - t0, 3)
        self.matcher = matcher
        del X_train, y_train
        gc.collect()
        peak_rss = max(peak_rss, get_process_rss_mb())
        print(f"  Model Fitted in {timing['model_train_s']}s on {train_pairs_count:,} pairs | RSS: {get_process_rss_mb():.2f} MB\n")

        # -------------------------------------------------------------------
        # Stage 6: Streamed Validation Scoring & Threshold Sweep
        # -------------------------------------------------------------------
        print("[6/7] Threshold Sweep & Entity Evaluation (from Validation Parquet Subsets)...")
        t0 = time.time()
        val_dfs = [
            pd.read_parquet(cp, filters=[("split", "==", "val")], columns=["s1_id", "candidate_id", "label"] + self.feature_names)
            for cp in chunk_files if cp.exists()
        ]
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame(columns=["s1_id", "candidate_id", "label"] + self.feature_names)
        del val_dfs
        gc.collect()
        peak_rss = max(peak_rss, get_process_rss_mb())

        # Restore val_gt
        if "val_gt" in manifest:
            val_gt = {k: set(v) for k, v in manifest["val_gt"].items()}
        else:
            if gt_df is None:
                gt_df = pd.read_csv(cfg.TRAIN_GROUND_TRUTH, sep="\t", keep_default_na=False)
            gt_dict = ground_truth_to_dict(gt_df)
            val_s1_set = set(val_df["s1_id"].astype(str))
            val_gt = {eid: gt_dict[eid] for eid in val_s1_set if eid in gt_dict}

        val_proba = matcher.predict_proba(val_df[self.feature_names])
        best_thresh, best_macro_f05, sweep_history = find_optimal_threshold(
            val_df, val_proba, val_gt, threshold_grid=self.config.thresholds_to_evaluate
        )
        timing["threshold_sweep_s"] = round(time.time() - t0, 3)
        self.optimal_threshold = best_thresh

        val_metrics = evaluate_at_threshold(val_df, val_proba, best_thresh, val_gt)
        val_metrics["runtime_s"] = timing["model_train_s"]

        val_preds = predict_matches(
            val_df, val_proba, threshold=best_thresh, s1_entities=val_gt.keys()
        )

        error_analysis = analyze_validation_errors(
            val_gt=val_gt,
            val_predictions=val_preds,
            val_pairs=val_df,
            val_proba=val_proba,
            threshold=best_thresh,
        )
        timing["error_analysis_s"] = round(time.time() - t0, 3)

        del val_df, val_proba
        gc.collect()

        print("\nThreshold Sweep Results:")
        print(f"{'Threshold':<10} | {'Macro F0.5':<12}")
        print("-" * 26)
        for thresh_val, score_val in sorted(sweep_history.items()):
            print(f"{thresh_val:<10.2f} | {score_val:<12.4f}")
        print("-" * 26)
        print(f"Optimal Threshold: {best_thresh:.2f} -> Validation Macro F0.5: {best_macro_f05:.4f}\n")

        print("Feature Importance Diagnostics (Top 10 Features by Gain):")
        lgb_fi = matcher.get_feature_importances(importance_type="gain")
        sorted_fi = sorted(lgb_fi.items(), key=lambda x: x[1], reverse=True)
        for rank, (feat, gain) in enumerate(sorted_fi[:10], 1):
            print(f"  {rank:2d}. {feat:<32}: {gain*100:6.2f}%")
        print()

        total_disk_bytes = sum(cp.stat().st_size for cp in chunk_files if cp.exists())
        total_disk_mb = round(total_disk_bytes / (1024**2), 2)
        total_time = round(sum(timing.values()), 2)

        streaming_telemetry = {
            "total_candidate_pairs": total_candidates,
            "total_feature_rows": total_candidates,
            "train_pairs": train_pairs_count,
            "val_pairs": val_pairs_count,
            "total_runtime_s": total_time,
            "candidate_generation_runtime_s": timing.get("blocking_s", 0.0),
            "pair_assembly_runtime_s": timing.get("pair_construction_s", 0.0),
            "gpu_feature_runtime_s": timing.get("feature_generation_s", 0.0),
            "disk_write_runtime_s": timing.get("disk_write_s", 0.0),
            "model_train_runtime_s": timing["model_train_s"],
            "threshold_sweep_runtime_s": timing["threshold_sweep_s"],
            "peak_ram_mb": peak_rss,
            "clean_process_baseline_rss_mb": train_proc_baseline,
            "peak_vram_mb": peak_vram,
            "number_of_chunks": len(chunk_files),
            "total_chunk_size_bytes": total_disk_bytes,
            "total_chunk_size_mb": total_disk_mb,
            "s1_batch_size": b_size,
            "total_batches": total_batches,
            "batches_telemetry": batches_telemetry_list,
        }

        report: dict[str, Any] = {
            "experiment_name": "Turn 9 Memory-Safe Streaming Pipeline Validation",
            "device_info": self.device_info,
            "sample_size": {
                "source1_entities": ingestion_summary.get("source1_entities", len(val_gt) * 5),
                "candidate_pool_size": ingestion_summary.get("candidate_pool_size", 0),
                "s2_pool": ingestion_summary.get("s2_pool", 0),
                "s3_pool": ingestion_summary.get("s3_pool", 0),
                "total_true_matches": ingestion_summary.get("total_true_matches", 0),
            },
            "blocking_diagnostics": {
                "candidate_count": total_candidates,
                "total_true_matches": ingestion_summary.get("total_true_matches", 0),
                "retrieved_true_matches": ingestion_summary.get("retrieved_true_total", 0),
                "candidate_recall_pct": ingestion_summary.get("candidate_recall", 0.0),
                "s2_candidate_recall_pct": ingestion_summary.get("s2_candidate_recall", 0.0),
                "s3_candidate_recall_pct": ingestion_summary.get("s3_candidate_recall", 0.0),
                "reduction_ratio_pct": ingestion_summary.get("reduction_ratio", 0.0),
                "mean_candidates_per_s1": round(total_candidates / max(ingestion_summary.get("source1_entities", 1), 1), 2),
                "blocking_runtime_s": timing.get("blocking_s", 0.0),
            },
            "split_summary": {
                "train_entities": ingestion_summary.get("train_entities", 0),
                "val_entities": len(val_gt),
                "train_pairs": train_pairs_count,
                "val_pairs": val_pairs_count,
                "train_pos": ingestion_summary.get("train_pos", 0),
                "val_pos": ingestion_summary.get("val_pos", 0),
                "val_singletons": sum(1 for m in val_gt.values() if len(m) == 0),
            },
            "model_config": {
                "model_type": "LightGBMMatcher",
                "n_estimators": self.config.n_estimators,
                "learning_rate": self.config.learning_rate,
                "num_leaves": self.config.num_leaves,
                "feature_count": len(self.feature_names),
                "feature_names": self.feature_names,
                "optimal_threshold": float(best_thresh),
            },
            "validation_metrics": val_metrics,
            "threshold_sweep": sweep_history,
            "error_analysis": error_analysis,
            "feature_importance_gain": dict(sorted_fi[:15]),
            "resource_diagnostics": {
                "baseline_rss_mb": train_proc_baseline,
                "peak_rss_mb": peak_rss,
                "delta_rss_mb": round(peak_rss - train_proc_baseline, 2),
                "peak_vram_mb": peak_vram,
            },
            "timing_breakdown": timing,
            "streaming_telemetry": streaming_telemetry,
        }

        # Save artifacts
        report_path = o_dir / "turn7_validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        results_path = o_dir / "turn7_experiment_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        model_path = c_dir / "turn7_best_matcher.joblib"
        matcher.save(model_path)

        thresh_path = c_dir / "turn7_optimal_threshold.json"
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "Turn7_LightGBM",
                    "optimal_threshold": float(best_thresh),
                    "validation_macro_f05": float(best_macro_f05),
                    "feature_names": self.feature_names,
                    "beta": 0.5,
                },
                f,
                indent=2,
            )

        print("=" * 85)
        print("STREAMING PIPELINE EXECUTION SUMMARY (TRAINING PROCESS)")
        print("=" * 85)
        print(f"Total Candidate Pairs:        {total_candidates:,}")
        print(f"Validation Macro F0.5:        {best_macro_f05:.4f} (Threshold = {best_thresh:.2f})")
        print(f"Mean Entity Precision:        {val_metrics['mean_entity_precision']:.4f}")
        print(f"Mean Entity Recall:           {val_metrics['mean_entity_recall']:.4f}")
        print(f"Clean Process Peak RAM (RSS): {peak_rss:.2f} MB")
        print(f"Peak GPU VRAM:                {peak_vram:.2f} MB")
        print(f"Persisted Chunks on Disk:     {len(chunk_files)} file(s) ({total_disk_mb:.2f} MB)")
        print(f"Artifacts successfully saved to: {o_dir}")
        print("=" * 85 + "\n")
        return report

    def run_sharded_ingestion(
        self,
        s1: pd.DataFrame,
        s2: Optional[pd.DataFrame] = None,
        s3: Optional[pd.DataFrame] = None,
        gt_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        shard_size: int = 200_000,
        overwrite_shards: bool = False,
        s2_path: Optional["Path | str"] = None,
        s3_path: Optional["Path | str"] = None,
    ) -> dict[str, Any]:
        """Execute Stages 1-4 using a disk-backed ShardedReferenceIndex.

        Memory-efficient mode: supply ``s2_path`` / ``s3_path`` (TSV file paths)
        instead of pre-loaded DataFrames. The method will stream those files
        through preprocessing in 200 k-row chunks and write Parquet shards
        **without ever holding the full S2/S3 corpus in RAM**.

        Enables candidate generation against a reference corpus (S2+S3) that is too
        large to fit in RAM (e.g. 10.32M records on a 12.7 GB Colab runtime).

        Strategy
        --------
        1. Preprocess and partition the full S2+S3 corpus into N Parquet shards
           (one-time, reused across S1 batches via disk cache).
        2. Build a global token-frequency vocabulary by streaming over shards once.
        3. For each S1 batch, iterate shard-by-shard: build a temporary in-memory
           GPUBlockingIndex per shard, query, accumulate per-S1 candidate scores,
           release shard from RAM, then apply global top-k.
        4. Candidate pair assembly, GPU feature extraction, and Parquet writing proceed
           identically to the standard run_streaming_ingestion() path.

        Parameters
        ----------
        s1:
            Source 1 DataFrame (query side, always required).
        s2, s3:
            Source DataFrames. S2+S3 form the fixed reference universe.
            Either pass pre-loaded DataFrames OR supply ``s2_path`` / ``s3_path``
            to stream directly from disk (recommended for 30 GB-constrained environments).
        s2_path, s3_path:
            Paths to raw TSV files. When provided, the method streams and preprocesses
            them in-place; ``s2`` / ``s3`` DataFrame args are ignored for shard building.
        gt_df:
            Ground truth DataFrame.
        output_dir:
            Output directory for Parquet chunks and manifest.
        cache_dir:
            Cache directory.
        shard_size:
            Number of reference records per disk shard (default: 200_000).
            At ~3.08 KB/record, each shard uses ~620 MB RAM when indexed.
        overwrite_shards:
            If True, re-build shard Parquet files even if they exist on disk.

        Returns
        -------
        dict with ingestion summary (same schema as run_streaming_ingestion).
        """
        cfg = Config()
        o_dir = output_dir or cfg.OUTPUT_DIR
        c_dir = cache_dir or cfg.CACHE_DIR
        o_dir.mkdir(parents=True, exist_ok=True)
        c_dir.mkdir(parents=True, exist_ok=True)

        chunks_dir = o_dir / "streaming_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = chunks_dir / "manifest.json"

        # Shard storage: lives alongside the Parquet chunks so output_dir contains everything
        shard_dir = o_dir / "ref_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        rss_baseline = get_process_rss_mb()
        peak_rss = rss_baseline
        vram_info = get_gpu_memory_mb()
        peak_vram = vram_info["peak_allocated_mb"]
        total_start = time.time()
        timing: dict[str, float] = {}
        rss_samples: list[float] = []

        print("=" * 80)
        print("SHARDED INGESTION (STAGES 1-4) — COLAB MEMORY-SAFE MODE")
        print("=" * 80)
        print(f"Device: {self.device_info['device_name']} (CUDA: {self.device_info['cuda_available']})")
        print(f"Baseline Process RSS: {rss_baseline:.2f} MB | Initial VRAM: {vram_info['allocated_mb']:.2f} MB")
        print(f"Target shard_size: {shard_size:,} records/shard | S1 batch: {self.config.s1_batch_size:,} | Resume: {self.config.resume}\n")

        gt_dict = ground_truth_to_dict(gt_df)
        total_true_matches = sum(len(m) for m in gt_dict.values())
        s2_true_matches = sum(1 for m in gt_dict.values() for x in m if x.startswith("S2"))
        s3_true_matches = sum(1 for m in gt_dict.values() for x in m if x.startswith("S3"))

        # -------------------------------------------------------------------
        # Stage 1: Deterministic Entity-Level Train / Validation Split
        # -------------------------------------------------------------------
        print("[1/7] Establishing Deterministic Entity Split (Zero Leakage)...")
        all_s1_list = sorted(list(set(gt_dict.keys()) | set(s1["entity_id"].astype(str))))
        rng = np.random.RandomState(self.config.random_state)
        shuffled_s1 = rng.permutation(all_s1_list)
        n_val = max(1, int(len(all_s1_list) * self.config.val_size))
        val_s1_set = set(shuffled_s1[:n_val])
        train_s1_set = set(shuffled_s1[n_val:])
        assert len(train_s1_set & val_s1_set) == 0, "FATAL LEAKAGE!"
        train_gt = {eid: gt_dict[eid] for eid in train_s1_set if eid in gt_dict}
        val_gt = {eid: gt_dict[eid] for eid in val_s1_set if eid in gt_dict}
        val_singletons = sum(1 for m in val_gt.values() if len(m) == 0)
        print(f"  Train: {len(train_s1_set):,} S1 queries | Val: {len(val_s1_set):,} S1 queries ({val_singletons} singletons)\n")

        # -------------------------------------------------------------------
        # Stage 2: Preprocess S1 only; shard S2+S3 to disk (reuse if exists)
        # -------------------------------------------------------------------
        print("[2/7] Preprocessing S1 + Building Sharded Reference Index (S2+S3)...")
        t0 = time.time()
        s1_proc = s1 if "name_norm" in s1.columns else preprocess_records(s1)
        peak_rss = max(peak_rss, get_process_rss_mb())

        sharded_idx = ShardedReferenceIndex(
            shard_dir=shard_dir,
            config=self.config.blocking_config,
            shard_size=shard_size,
        )

        # --- Memory-safe shard build ---
        # Prefer streaming from TSV files (zero full-DF peak RAM) over DF path.
        tsv_paths = [p for p in (s2_path, s3_path) if p is not None]
        if tsv_paths:
            # Stream S2/S3 directly from disk — S2/S3 DataFrames never loaded.
            shard_meta = sharded_idx.build_shards_from_files(
                *tsv_paths,
                overwrite=overwrite_shards,
                chunk_size=shard_size,
                n_jobs=4,
            )
        else:
            # Fallback: build from pre-loaded DataFrames (legacy / small-scale use)
            shard_meta = sharded_idx.build_shards(s2, s3, overwrite=overwrite_shards)
            # Free reference DataFrames immediately after sharding
            del s2, s3
            gc.collect()

        peak_rss = max(peak_rss, get_process_rss_mb())
        print(
            f"  Reference Shards: {shard_meta['n_shards']} shards x ~{shard_size:,} records "
            f"= {shard_meta['total_records']:,} total ref records | RSS: {get_process_rss_mb():.2f} MB"
        )

        # Build/load global vocabulary (streams shards + S1, stays in RAM as Counter)
        sharded_idx.build_global_vocab(s1_proc, overwrite=overwrite_shards)
        peak_rss = max(peak_rss, get_process_rss_mb())
        print(f"  Global Vocab: {len(sharded_idx.token_freq):,} tokens | RSS: {get_process_rss_mb():.2f} MB")

        # Fit IDF store (used by GPU feature extraction) — stream shards one at a time
        print("  Fitting IDF store (streaming over shards)...")
        from src.gpu.gpu_features import TokenIDFStore  # noqa: PLC0415
        self.idf_store = TokenIDFStore()
        self.idf_store.start_incremental()
        # Accumulate S1 first
        self.idf_store.fit_incremental(s1_proc)
        # Then stream each shard individually — peak RAM = one shard at a time
        for fname in shard_meta["shard_files"]:
            sp = shard_dir / fname
            if sp.exists():
                tmp = pd.read_parquet(sp, columns=["name_norm"])
                self.idf_store.fit_incremental(tmp)
                del tmp
        self.idf_store.finalize()
        gc.collect()


        timing["preprocessing_s"] = round(time.time() - t0, 2)
        peak_rss = max(peak_rss, get_process_rss_mb())
        print(f"  IDF Vocab: {len(self.idf_store.idf_table):,} tokens fitted in {timing['preprocessing_s']}s | RSS: {get_process_rss_mb():.2f} MB\n")

        # -------------------------------------------------------------------
        # Stage 3: Not needed — index is sharded on disk
        # -------------------------------------------------------------------
        print("[3/7] Sharded index ready on disk (no monolithic in-memory build needed).")
        print(f"  {shard_meta['n_shards']} shards in {shard_dir}\n")

        # -------------------------------------------------------------------
        # Stage 4: Streaming S1 batches × shard iteration
        # -------------------------------------------------------------------
        print(f"[4/7] Streaming S1 Batches (batch={self.config.s1_batch_size:,}) × {shard_meta['n_shards']} shards...")
        n_s1 = len(s1_proc)
        b_size = max(1, self.config.s1_batch_size)
        batches = [s1_proc.iloc[i : min(i + b_size, n_s1)].copy() for i in range(0, n_s1, b_size)]
        total_batches = len(batches)

        manifest: dict[str, Any] = {
            "total_batches": total_batches,
            "s1_batch_size": b_size,
            "chunks": {},
            "sharded": True,
            "shard_dir": str(shard_dir),
            "n_shards": shard_meta["n_shards"],
            "ref_total_records": shard_meta["total_records"],
            "s2_records": shard_meta.get("s2_records", 0),
            "s3_records": shard_meta.get("s3_records", 0),
        }
        if self.config.resume and manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                logger.info("Loaded resume manifest (%d existing chunks).", len(manifest.get("chunks", {})))
            except Exception as e:
                logger.warning("Failed to load manifest (%s). Starting clean.", e)
        elif not self.config.resume:
            for old_f in chunks_dir.glob("chunk_*.parquet*"):
                old_f.unlink(missing_ok=True)

        total_candidates = 0
        retrieved_true_total = 0
        retrieved_true_s2 = 0
        retrieved_true_s3 = 0
        t_blocking_total = 0.0
        t_assembly_total = 0.0
        t_features_total = 0.0
        t_write_total = 0.0
        train_pairs_count = 0
        val_pairs_count = 0
        train_pos_count = 0
        val_pos_count = 0
        total_pos = 0
        total_neg = 0
        chunk_files: list[Path] = []
        batch_telemetry: list[dict[str, Any]] = []

        for b_idx, s1_batch in enumerate(batches, start=1):
            chunk_file = chunks_dir / f"chunk_{b_idx:04d}.parquet"
            chunk_files.append(chunk_file)
            chunk_key = str(b_idx)

            if self.config.resume and chunk_key in manifest.get("chunks", {}) and chunk_file.exists():
                entry = manifest["chunks"][chunk_key]
                logger.info("Resuming batch %d/%d from disk (%d pairs)...", b_idx, total_batches, entry["pairs_count"])
                total_candidates += entry["pairs_count"]
                retrieved_true_total += entry["retrieved_true_total"]
                retrieved_true_s2 += entry["retrieved_true_s2"]
                retrieved_true_s3 += entry["retrieved_true_s3"]
                train_pairs_count += entry["train_pairs"]
                val_pairs_count += entry["val_pairs"]
                train_pos_count += entry.get("train_pos", 0)
                val_pos_count += entry.get("val_pos", 0)
                total_pos += entry["pos_count"]
                total_neg += entry["neg_count"]
                t_blocking_total += entry["t_blocking_s"]
                t_assembly_total += entry["t_assembly_s"]
                t_features_total += entry["t_features_s"]
                t_write_total += entry["t_write_s"]
                batch_telemetry.append(entry)
                continue

            t_b_start = time.time()

            # A. Sharded blocking: iterate over shard files, accumulate candidates
            t0 = time.time()
            cands_batch = sharded_idx.query_s1_batch(s1_batch, rss_monitor=rss_samples)
            t_blk = time.time() - t0
            t_blocking_total += t_blk

            # B. Candidate recall check
            cands_by_s1 = (
                cands_batch.groupby("s1_id")["candidate_id"].apply(set).to_dict()
                if len(cands_batch) > 0
                else {}
            )
            b_retrieved_total = 0
            b_retrieved_s2 = 0
            b_retrieved_s3 = 0
            for s1_raw in s1_batch["entity_id"]:
                s1_str = str(s1_raw)
                t_set = gt_dict.get(s1_str, set())
                f_set = cands_by_s1.get(s1_str, set())
                for tid in t_set:
                    if tid in f_set:
                        b_retrieved_total += 1
                        if tid.startswith("S2"):
                            b_retrieved_s2 += 1
                        elif tid.startswith("S3"):
                            b_retrieved_s3 += 1
            retrieved_true_total += b_retrieved_total
            retrieved_true_s2 += b_retrieved_s2
            retrieved_true_s3 += b_retrieved_s3
            b_cands_count = len(cands_batch)
            total_candidates += b_cands_count

            # C. Sliced candidate pair assembly — fetch only the candidate records needed
            t0 = time.time()
            needed_cands = set(cands_batch["candidate_id"]) if b_cands_count > 0 else set()
            # Fetch candidate records from shard Parquet files (no full S2/S3 in RAM)
            s2_sub_rows: list[dict] = []
            s3_sub_rows: list[dict] = []
            if needed_cands:
                for fname in shard_meta["shard_files"]:
                    sp = shard_dir / fname
                    if not sp.exists():
                        continue
                    shard_df = pd.read_parquet(sp)
                    mask = shard_df["entity_id"].astype(str).isin(needed_cands)
                    if mask.any():
                        matched = shard_df[mask]
                        for _, mrow in matched.iterrows():
                            eid = str(mrow["entity_id"])
                            if eid.startswith("S2"):
                                s2_sub_rows.append(mrow.to_dict())
                            else:
                                s3_sub_rows.append(mrow.to_dict())
                    del shard_df, mask
                    gc.collect()

            s2_sub = pd.DataFrame(s2_sub_rows) if s2_sub_rows else pd.DataFrame()
            s3_sub = pd.DataFrame(s3_sub_rows) if s3_sub_rows else pd.DataFrame()
            # Ensure required columns exist even for empty frames
            _PAIR_COLS = ["entity_id", "name_norm", "name_compact", "name_no_suffix",
                          "address_norm", "address_is_missing", "country_norm",
                          "name_script", "is_domain", "domain_stem", "legal_suffix"]
            if len(s2_sub) == 0:
                s2_sub = pd.DataFrame(columns=_PAIR_COLS)
            if len(s3_sub) == 0:
                s3_sub = pd.DataFrame(columns=_PAIR_COLS)

            pairs_batch = build_train_pairs(
                s1_batch, s2_sub, s3_sub, cands_batch, gt_df,
                random_seed=self.config.random_state,
            )
            t_asm = time.time() - t0
            t_assembly_total += t_asm

            # D. GPU feature extraction for batch
            t0 = time.time()
            feats_batch = compute_gpu_features(
                pairs_batch,
                idf_store=self.idf_store,
                include_turn7_features=self.config.include_turn7_features,
            )
            t_feat = time.time() - t0
            t_features_total += t_feat

            # E. Assemble compact chunk DataFrame and persist
            t0 = time.time()
            if len(pairs_batch) > 0:
                is_val_mask = pairs_batch["s1_id"].astype(str).isin(val_s1_set)
                pairs_batch["split"] = np.where(is_val_mask, "val", "train")
                for col in self.feature_names:
                    pairs_batch[col] = feats_batch[col].astype(np.float32).values
                out_cols = ["s1_id", "candidate_id", "split", "label"] + self.feature_names
                chunk_df = pairs_batch[out_cols].copy()
            else:
                chunk_df = pd.DataFrame(
                    columns=["s1_id", "candidate_id", "split", "label"] + self.feature_names
                )

            tmp_chunk = chunk_file.with_suffix(".parquet.tmp")
            chunk_df.to_parquet(tmp_chunk, index=False, engine="pyarrow", compression="snappy")
            tmp_chunk.replace(chunk_file)
            t_wrt = time.time() - t0
            t_write_total += t_wrt

            b_pairs = len(chunk_df)
            b_pos = int((chunk_df["label"] == 1).sum()) if b_pairs > 0 else 0
            b_neg = b_pairs - b_pos
            b_train = int((chunk_df["split"] == "train").sum()) if b_pairs > 0 else 0
            b_val = b_pairs - b_train
            b_train_pos = (
                int(((chunk_df["split"] == "train") & (chunk_df["label"] == 1)).sum())
                if b_pairs > 0
                else 0
            )
            b_val_pos = b_pos - b_train_pos

            total_pos += b_pos
            total_neg += b_neg
            train_pairs_count += b_train
            val_pairs_count += b_val
            train_pos_count += b_train_pos
            val_pos_count += b_val_pos

            cur_rss = get_process_rss_mb()
            peak_rss = max(peak_rss, cur_rss)
            if rss_samples:
                peak_rss = max(peak_rss, max(rss_samples))
                rss_samples.clear()
            vram_stat = get_gpu_memory_mb()
            peak_vram = max(peak_vram, vram_stat["peak_allocated_mb"])
            file_bytes = chunk_file.stat().st_size if chunk_file.exists() else 0

            entry = {
                "chunk_file": chunk_file.name,
                "s1_count": len(s1_batch),
                "pairs_count": b_pairs,
                "train_pairs": b_train,
                "val_pairs": b_val,
                "train_pos": b_train_pos,
                "val_pos": b_val_pos,
                "pos_count": b_pos,
                "neg_count": b_neg,
                "retrieved_true_total": b_retrieved_total,
                "retrieved_true_s2": b_retrieved_s2,
                "retrieved_true_s3": b_retrieved_s3,
                "file_size_bytes": file_bytes,
                "t_blocking_s": round(t_blk, 3),
                "t_assembly_s": round(t_asm, 3),
                "t_features_s": round(t_feat, 3),
                "t_write_s": round(t_wrt, 3),
                "process_rss_mb": cur_rss,
                "peak_rss_mb": peak_rss,
                "gpu_allocated_mb": vram_stat["allocated_mb"],
                "gpu_reserved_mb": vram_stat["reserved_mb"],
            }
            manifest["chunks"][chunk_key] = entry
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            batch_telemetry.append(entry)
            print(
                f"  Batch {b_idx:2d}/{total_batches}: S1={len(s1_batch):,} -> Pairs={b_pairs:,} "
                f"| Block={t_blk:.2f}s Asm={t_asm:.2f}s GPU={t_feat:.2f}s Wrt={t_wrt:.2f}s "
                f"| RSS={cur_rss:.1f}MB (Peak:{peak_rss:.1f}MB) VRAM={vram_stat['peak_allocated_mb']:.1f}MB"
            )

            del s1_batch, cands_batch, needed_cands, s2_sub, s3_sub
            del pairs_batch, feats_batch, chunk_df
            gc.collect()
            clear_gpu_cache()

        # Cleanup sharded_idx (vocab stays on disk)
        del sharded_idx
        gc.collect()

        timing["blocking_s"] = round(t_blocking_total, 2)
        timing["pair_construction_s"] = round(t_assembly_total, 2)
        timing["feature_generation_s"] = round(t_features_total, 2)
        timing["disk_write_s"] = round(t_write_total, 2)

        candidate_recall = (retrieved_true_total / total_true_matches * 100) if total_true_matches > 0 else 0.0
        s2_candidate_recall = (retrieved_true_s2 / s2_true_matches * 100) if s2_true_matches > 0 else 0.0
        s3_candidate_recall = (retrieved_true_s3 / s3_true_matches * 100) if s3_true_matches > 0 else 0.0
        ref_total = shard_meta["total_records"]
        total_possible = len(s1_proc) * ref_total
        reduction_ratio = ((1.0 - (total_candidates / total_possible)) * 100) if total_possible > 0 else 0.0

        total_disk_bytes = sum(cp.stat().st_size for cp in chunk_files if cp.exists())
        total_disk_mb = round(total_disk_bytes / (1024**2), 2)

        print(f"\n  Sharded Ingestion Complete: {total_batches} S1 batches × {shard_meta['n_shards']} shards")
        print(f"  Total Candidate Pairs:   {total_candidates:,} ({total_pos:,} pos, {total_neg:,} neg)")
        print(f"  Candidate Recall:        {candidate_recall:.2f}% (S2: {s2_candidate_recall:.2f}%, S3: {s3_candidate_recall:.2f}%)")
        print(f"  Disk Output:             {total_disk_mb:.2f} MB across {len(chunk_files)} Parquet chunk(s)")
        print(f"  Peak RSS (all batches):  {peak_rss:.2f} MB")
        print(f"  Timings: Block={timing['blocking_s']}s Asm={timing['pair_construction_s']}s GPU={timing['feature_generation_s']}s Write={timing['disk_write_s']}s\n")

        ingestion_summary = {
            "source1_entities": len(s1_proc),
            "candidate_pool_size": ref_total,
            "s2_pool": shard_meta.get("s2_records", shard_meta.get("total_records", 0)),
            "s3_pool": shard_meta.get("s3_records", 0),
            "total_true_matches": total_true_matches,
            "s2_true_matches": s2_true_matches,
            "s3_true_matches": s3_true_matches,
            "total_candidates": total_candidates,
            "retrieved_true_total": retrieved_true_total,
            "retrieved_true_s2": retrieved_true_s2,
            "retrieved_true_s3": retrieved_true_s3,
            "candidate_recall": candidate_recall,
            "s2_candidate_recall": s2_candidate_recall,
            "s3_candidate_recall": s3_candidate_recall,
            "reduction_ratio": reduction_ratio,
            "total_pos": total_pos,
            "total_neg": total_neg,
            "train_pairs": train_pairs_count,
            "val_pairs": val_pairs_count,
            "train_pos": train_pos_count,
            "val_pos": val_pos_count,
            "train_entities": len(train_s1_set),
            "val_entities": len(val_s1_set),
            "val_singletons": val_singletons,
            "total_disk_bytes": total_disk_bytes,
            "total_disk_mb": total_disk_mb,
            "timing_ingest": timing.copy(),
            "peak_rss_mb": peak_rss,
            "peak_vram_mb": peak_vram,
            "rss_baseline": rss_baseline,
            "sharded": True,
            "n_shards": shard_meta["n_shards"],
            "shard_size": shard_size,
        }
        manifest["ingestion_summary"] = ingestion_summary
        manifest["val_gt"] = {str(k): list(v) for k, v in val_gt.items()}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return ingestion_summary

    def run_streaming_validation_experiment(
        self,
        s1: Optional[pd.DataFrame] = None,
        s2: Optional[pd.DataFrame] = None,
        s3: Optional[pd.DataFrame] = None,
        gt_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        ingest_only: bool = False,
        train_only: bool = False,
        use_sharded: bool = False,
        shard_size: int = 200_000,
        overwrite_shards: bool = False,
        s2_path: Optional["Path | str"] = None,
        s3_path: Optional["Path | str"] = None,
    ) -> dict[str, Any]:
        """Execute memory-safe streaming validation experiment.

        Supports:
        - ingest_only / train_only: process-boundary isolation.
        - use_sharded: route candidate generation through ShardedReferenceIndex
          for Colab-safe handling of the full 10.32M reference corpus.
        - s2_path / s3_path: stream reference TSV directly to shards; S2/S3 DFs
          never loaded into RAM (recommended for 30 GB Kaggle environments).
        """
        if train_only:
            return self.run_training_from_chunks(output_dir=output_dir, cache_dir=cache_dir, gt_df=gt_df)

        if use_sharded:
            ingestion_summary = self.run_sharded_ingestion(
                s1=s1, s2=s2, s3=s3, gt_df=gt_df,
                output_dir=output_dir, cache_dir=cache_dir,
                shard_size=shard_size, overwrite_shards=overwrite_shards,
                s2_path=s2_path, s3_path=s3_path,
            )
        else:
            ingestion_summary = self.run_streaming_ingestion(
                s1=s1, s2=s2, s3=s3, gt_df=gt_df, output_dir=output_dir, cache_dir=cache_dir
            )

        if ingest_only:
            return ingestion_summary

        return self.run_training_from_chunks(output_dir=output_dir, cache_dir=cache_dir, gt_df=gt_df)



