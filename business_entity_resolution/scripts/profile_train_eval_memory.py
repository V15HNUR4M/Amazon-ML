"""
profile_train_eval_memory.py — Phase A: Measure actual peak host RAM at each stage
of LightGBM training/evaluation in the Turn 9 streaming pipeline.

Uses the same psutil RSS pattern already established in gpu_utils.py.

Run:
    python scripts/profile_train_eval_memory.py --scale 5k
    python scripts/profile_train_eval_memory.py --scale 50k
    python scripts/profile_train_eval_memory.py --scale 100k
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gpu.gpu_utils import get_gpu_memory_mb, get_process_rss_mb
from src.model import LightGBMMatcher
from src.gpu.gpu_features import TURN7_FEATURE_NAMES
from src.threshold import find_optimal_threshold, evaluate_at_threshold


def rss() -> float:
    return get_process_rss_mb()


def vram() -> float:
    return get_gpu_memory_mb()["allocated_mb"]


SCALE_CONFIG = {
    "5k":   {"s1": 5000,   "chunks_dir": "output/turn9_benchmark_5k/streaming_chunks"},
    "50k":  {"s1": 50000,  "chunks_dir": "output/turn9_benchmark_50k/streaming_chunks"},
    "100k": {"s1": 100000, "chunks_dir": "output/turn9_benchmark_100k/streaming_chunks"},
}

PHASEC_CONFIG = {
    "5k":   {"s1": 5000,   "chunks_dir": "output/turn9_phaseC_5k/streaming_chunks"},
    "50k":  {"s1": 50000,  "chunks_dir": "output/turn9_benchmark_50k/streaming_chunks"},
    "100k": {"s1": 100000, "chunks_dir": "output/turn9_phaseC_100k/streaming_chunks"},
}

FEATURE_NAMES = TURN7_FEATURE_NAMES  # 34 canonical features


def measure_stage(label: str, rss_before: float) -> tuple[float, float]:
    """Return (rss_after, delta_rss) and print stage measurement."""
    gc.collect()
    r = rss()
    delta = r - rss_before
    print(f"  [{label}] RSS: {r:.1f} MB  (delta: {delta:+.1f} MB)")
    return r, delta


def profile_scale(scale: str, project_root: Path) -> dict[str, Any]:
    cfg = SCALE_CONFIG[scale]
    chunks_dir = project_root / cfg["chunks_dir"]
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))

    if not chunk_files:
        print(f"ERROR: No Parquet chunks found in {chunks_dir}")
        print("  Run the Turn 9 benchmark first:")
        print(f"  python scripts/run_turn7_validation.py --sample-source1 {cfg['s1']} --distractors 5000 "
              f"--output-dir output/turn9_benchmark_{scale}")
        return {}

    print("=" * 70)
    print(f"PHASE A — MEMORY PROFILING: {scale.upper()} scale ({len(chunk_files)} Parquet chunks)")
    print("=" * 70)
    results: dict[str, Any] = {"scale": scale, "num_chunks": len(chunk_files)}

    # ------------------------------------------------------------------
    # STAGE 0: Baseline
    # ------------------------------------------------------------------
    gc.collect()
    r0 = rss()
    v0 = vram()
    print(f"\n  [Baseline] RSS: {r0:.1f} MB | VRAM: {v0:.1f} MB")
    results["baseline_rss_mb"] = r0
    results["baseline_vram_mb"] = v0

    # ------------------------------------------------------------------
    # STAGE 1: Read Parquet chunks for TRAINING (train subset only)
    # Mimics exactly what gpu_pipeline.py does at Stage 5 (lines 835-841)
    # ------------------------------------------------------------------
    print("\n--- STAGE 1: Parquet file load (train subset) ---")
    r_before = rss()
    t0 = time.time()
    train_dfs = [
        pd.read_parquet(cp, filters=[("split", "==", "train")], columns=FEATURE_NAMES + ["label"])
        for cp in chunk_files if cp.exists()
    ]
    r_after_indiv, d1a = measure_stage("After loading individual train DataFrames", r_before)
    results["stage1a_parquet_load_individual_rss_mb"] = r_after_indiv
    results["stage1a_delta_mb"] = d1a

    # Check what dtypes are present (string/object cols carried in?)
    sample_df = train_dfs[0]
    obj_cols = [c for c in sample_df.columns if sample_df[c].dtype == object]
    print(f"  [Dtype check] Object/string columns in loaded train DF: {obj_cols if obj_cols else 'None'}")
    print(f"  [Dtype check] Feature dtypes: {dict(sample_df[FEATURE_NAMES[:3]].dtypes)}")
    results["object_cols_in_train_df"] = obj_cols

    # ------------------------------------------------------------------
    # STAGE 2: pd.concat across all chunks  [OLD PATTERN — now replaced]
    # ------------------------------------------------------------------
    print("\n--- STAGE 2 (OLD PATTERN): pd.concat of all train chunks ---")
    r_before = rss()
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(columns=FEATURE_NAMES + ["label"])
    r_after_concat, d2 = measure_stage("After pd.concat (full train DataFrame in memory)", r_before)
    results["stage2_concat_rss_mb"] = r_after_concat
    results["stage2_delta_mb"] = d2
    n_train = len(train_df)
    print(f"  [Info] train_df shape: {train_df.shape} | memory_usage: {train_df.memory_usage(deep=True).sum() / 1e6:.1f} MB (reported by pandas)")

    # Free individual chunk DataFrames
    del train_dfs
    gc.collect()
    r_after_del, _ = measure_stage("After del train_dfs (individual DFs freed)", r_after_concat)
    results["stage2b_after_del_individual_rss_mb"] = r_after_del

    # ------------------------------------------------------------------
    # STAGE 3: DataFrame -> NumPy array extraction (inside LightGBMMatcher._extract_feature_matrix)
    # We replicate this exactly so we can measure the peak
    # ------------------------------------------------------------------
    print("\n--- STAGE 3 (OLD PATTERN): DataFrame.to_numpy(dtype=float32) extraction ---")
    r_before = rss()
    X_train = train_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_train = np.asarray(train_df["label"], dtype=np.int32)
    r_after_numpy, d3 = measure_stage("After to_numpy() — both train_df AND numpy array live simultaneously", r_before)
    results["stage3_numpy_extraction_rss_mb"] = r_after_numpy
    results["stage3_delta_mb"] = d3
    results["old_pattern_peak_rss_mb"] = r_after_numpy
    print(f"  [Info] X_train shape: {X_train.shape} | dtype: {X_train.dtype} | size: {X_train.nbytes / 1e6:.1f} MB")
    print(f"  [Info] y_train shape: {y_train.shape} | dtype: {y_train.dtype} | size: {y_train.nbytes / 1e6:.1f} MB")
    print(f"  [Info] Pure float32 matrix footprint at full 2.2M (76.8M train rows): "
          f"{76_800_000 * 34 * 4 / 1e9:.2f} GB")

    # Now free train_df to observe how much RSS drops
    del train_df
    gc.collect()
    r_after_del_df, _ = measure_stage("After del train_df (is the DataFrame copy freed?)", r_after_numpy)
    results["stage3b_after_del_train_df_rss_mb"] = r_after_del_df
    duplication_overhead_mb = r_after_numpy - r_after_del_df
    print(f"  [Duplication analysis] Overhead while train_df + X_train both live: ~{duplication_overhead_mb:.1f} MB")
    results["stage3_simultaneous_copy_overhead_mb"] = duplication_overhead_mb

    # ------------------------------------------------------------------
    # STAGE 3-NEW: Sequential pre-allocated pattern  [NEW PATTERN — what gpu_pipeline.py now does]
    # Pre-allocate X_train once, fill chunk-by-chunk, only 1 chunk live at a time.
    # ------------------------------------------------------------------
    print("\n--- STAGE 3-NEW (FIXED PATTERN): Pre-allocated sequential chunk loading ---")
    # First free the X_train from old pattern, then re-profile new pattern
    del X_train, y_train
    gc.collect()
    r_before_new = rss()
    n_feats = len(FEATURE_NAMES)
    # Pre-allocate (same total size as before, but NO simultaneous chunk copies)
    X_train_new = np.empty((n_train, n_feats), dtype=np.float32)
    y_train_new = np.empty(n_train, dtype=np.int32)
    r_after_prealloc, _ = measure_stage("After np.empty pre-allocation (X_train + y_train)", r_before_new)
    results["stage3new_prealloc_rss_mb"] = r_after_prealloc

    row_offset = 0
    chunk_rss_during = []
    for cp in chunk_files:
        if not cp.exists():
            continue
        chunk_df = pd.read_parquet(cp, filters=[("split", "==", "train")], columns=FEATURE_NAMES + ["label"])
        n_rows = len(chunk_df)
        rss_with_chunk = rss()
        chunk_rss_during.append(rss_with_chunk)
        if n_rows > 0:
            X_train_new[row_offset:row_offset + n_rows] = chunk_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
            y_train_new[row_offset:row_offset + n_rows] = chunk_df["label"].to_numpy(dtype=np.int32)
        row_offset += n_rows
        del chunk_df
        gc.collect()

    r_after_new_fill, _ = measure_stage("After sequential fill (X_train_new + y_train_new, no concurrent chunks)", r_before_new)
    new_pattern_peak = max(chunk_rss_during) if chunk_rss_during else r_after_new_fill
    results["stage3new_fill_rss_mb"] = r_after_new_fill
    results["new_pattern_peak_rss_mb"] = new_pattern_peak
    print(f"  [Info] Peak RSS during chunk iteration: {new_pattern_peak:.1f} MB")
    print(f"  [Comparison] OLD pattern peak: {results['old_pattern_peak_rss_mb']:.1f} MB  |  NEW pattern peak: {new_pattern_peak:.1f} MB")
    saving_mb = results["old_pattern_peak_rss_mb"] - new_pattern_peak
    print(f"  [Saving] Memory savings from new pattern: {saving_mb:.1f} MB  ({saving_mb / results['old_pattern_peak_rss_mb'] * 100:.1f}% reduction)")
    results["training_stage_memory_savings_mb"] = saving_mb

    # Use the new arrays for LightGBM training (what the pipeline actually does now)
    X_train, y_train = X_train_new, y_train_new
    del X_train_new, y_train_new

    # ------------------------------------------------------------------
    # STAGE 4: LightGBM Dataset construction
    # ------------------------------------------------------------------
    print("\n--- STAGE 4: LightGBM Dataset construction + model training ---")
    import lightgbm as lgb
    r_before = rss()
    # Build LGBMClassifier exactly as LightGBMMatcher.fit() does
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        importance_type="gain",
        verbose=-1,
        n_jobs=-1,
    )
    r_before_fit = rss()
    t_train_start = time.time()
    clf.fit(X_train, y_train)
    t_train = time.time() - t_train_start
    r_after_fit, d4 = measure_stage(f"After LGBMClassifier.fit() [{t_train:.2f}s]", r_before_fit)
    results["stage4_lgbm_fit_rss_mb"] = r_after_fit
    results["stage4_delta_mb"] = d4
    results["stage4_train_time_s"] = round(t_train, 3)

    # Free training data — does the model internally reference X_train?
    del X_train, y_train
    gc.collect()
    r_after_del_train, _ = measure_stage("After del X_train, y_train (does model free them?)", r_after_fit)
    results["stage4b_after_del_train_arrays_rss_mb"] = r_after_del_train

    # ------------------------------------------------------------------
    # STAGE 5: Load validation data from Parquet
    # ------------------------------------------------------------------
    print("\n--- STAGE 5: Parquet file load (val subset) ---")
    r_before = rss()
    val_dfs = [
        pd.read_parquet(cp, filters=[("split", "==", "val")], columns=["s1_id", "candidate_id", "label"] + FEATURE_NAMES)
        for cp in chunk_files if cp.exists()
    ]
    r_after_val_indiv, _ = measure_stage("After loading individual val DataFrames", r_before)
    results["stage5a_val_parquet_load_rss_mb"] = r_after_val_indiv

    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    del val_dfs
    gc.collect()
    r_after_val_concat, d5 = measure_stage("After pd.concat val chunks + del individual dfs", r_after_val_indiv)
    results["stage5b_val_concat_rss_mb"] = r_after_val_concat
    results["stage5b_delta_mb"] = d5
    n_val = len(val_df)
    print(f"  [Info] val_df shape: {val_df.shape} | memory_usage: {val_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # ------------------------------------------------------------------
    # STAGE 6: predict_proba
    # ------------------------------------------------------------------
    print("\n--- STAGE 6: model.predict_proba (val features -> probabilities) ---")
    r_before = rss()
    X_val = val_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    r_after_xval, _ = measure_stage("After X_val = val_df[features].to_numpy()", r_before)
    val_proba = clf.predict_proba(X_val)[:, 1].astype(np.float32)
    r_after_proba, d6 = measure_stage("After predict_proba (val_df + X_val + proba all live)", r_after_xval)
    results["stage6_predict_proba_rss_mb"] = r_after_proba
    results["stage6_delta_mb"] = d6
    print(f"  [Info] val_proba shape: {val_proba.shape} | size: {val_proba.nbytes / 1e6:.1f} MB")
    print(f"  [Info] X_val shape: {X_val.shape} | size: {X_val.nbytes / 1e6:.1f} MB")

    # Free X_val immediately (no longer needed after proba computed)
    del X_val
    gc.collect()
    r_after_del_xval, _ = measure_stage("After del X_val", r_after_proba)
    results["stage6b_after_del_xval_rss_mb"] = r_after_del_xval

    # ------------------------------------------------------------------
    # STAGE 7: Threshold sweep
    # Does each threshold iteration re-allocate label/pred arrays?
    # ------------------------------------------------------------------
    print("\n--- STAGE 7: Threshold sweep (10 thresholds) ---")

    # Read val_gt from the manifest JSON for a proper ground truth dict
    manifest_path = chunks_dir / "manifest.json"
    import json
    val_gt: dict = {}
    if manifest_path.exists():
        # We can't reconstruct val_gt from Parquet alone without loading gt_df.
        # We'll approximate using pair labels to form a synthetic val_gt.
        print("  [Note] Building synthetic val_gt from val_df labels for threshold sweep profiling")
        for s1_id, grp in val_df.groupby("s1_id"):
            matched = set(grp.loc[grp["label"] == 1, "candidate_id"].astype(str))
            val_gt[str(s1_id)] = matched

    r_before = rss()
    thresholds = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    sweep_history: dict[float, float] = {}
    for thresh in thresholds:
        from src.prediction import predict_matches
        preds = predict_matches(val_df, val_proba, threshold=thresh, s1_entities=val_gt.keys())
        from src.evaluation import macro_f05
        score = macro_f05(val_gt, preds)
        sweep_history[thresh] = score

    r_after_sweep, d7 = measure_stage("After complete threshold sweep (10 thresholds)", r_before)
    results["stage7_threshold_sweep_rss_mb"] = r_after_sweep
    results["stage7_delta_mb"] = d7
    print(f"  [Info] val_df + val_proba held throughout sweep: no per-threshold re-allocation")
    print(f"  Best threshold in sweep: {max(sweep_history, key=lambda k: sweep_history[k]):.2f} -> F0.5={max(sweep_history.values()):.4f}")

    # ------------------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------------------
    del val_df, val_proba
    gc.collect()
    r_final, _ = measure_stage("After del val_df, val_proba (final cleanup)", r_after_sweep)
    results["stage_final_rss_mb"] = r_final

    # ------------------------------------------------------------------
    # Peak RSS summary
    # ------------------------------------------------------------------
    peak_rss = max(
        r0, r_after_indiv, r_after_concat, r_after_numpy, r_after_fit,
        r_after_val_concat, r_after_proba, r_after_sweep
    )
    results["measured_peak_rss_mb"] = peak_rss

    print("\n" + "=" * 70)
    print(f"PHASE A SUMMARY — {scale.upper()}")
    print("=" * 70)
    print(f"  Baseline RSS:                         {r0:.1f} MB")
    print(f"  After Parquet load (train indiv):     {r_after_indiv:.1f} MB")
    print(f"  After pd.concat (train full):         {r_after_concat:.1f} MB")
    print(f"  After to_numpy [train_df + array]:    {r_after_numpy:.1f} MB  ← simultaneous copy peak")
    print(f"  After del train_df (freed):           {r_after_del_df:.1f} MB")
    print(f"  Simultaneous copy overhead:           {duplication_overhead_mb:.1f} MB")
    print(f"  After LGBMClassifier.fit():           {r_after_fit:.1f} MB")
    print(f"  After del train arrays:               {r_after_del_train:.1f} MB")
    print(f"  After Parquet load (val concat):      {r_after_val_concat:.1f} MB")
    print(f"  After predict_proba:                  {r_after_proba:.1f} MB")
    print(f"  After del X_val:                      {r_after_del_xval:.1f} MB")
    print(f"  After threshold sweep:                {r_after_sweep:.1f} MB")
    print(f"  After final cleanup:                  {r_final:.1f} MB")
    print(f"  ───────────────────────────────────────────────────────")
    print(f"  MEASURED PEAK RSS (train/eval path):  {peak_rss:.1f} MB")
    print(f"  Train rows (n): {n_train:,}  |  Val rows (n): {n_val:,}")
    print(f"  Object/string cols in train DF:       {obj_cols if obj_cols else 'None'}")
    print("=" * 70 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A memory profiling for LightGBM train/eval stages.")
    parser.add_argument(
        "--scale", choices=["5k", "50k", "100k", "all"],
        default="all", help="Benchmark scale to profile (default: all)"
    )
    parser.add_argument(
        "--post-fix", action="store_true",
        help="Profile Phase C (post-fix) outputs instead of Turn 9 benchmark outputs"
    )
    args = parser.parse_args()
    if args.post_fix:
        import scripts.profile_train_eval_memory as _self
        _self.SCALE_CONFIG.update(PHASEC_CONFIG)

    scales = ["5k", "50k", "100k"] if args.scale == "all" else [args.scale]
    all_results = {}

    for scale in scales:
        result = profile_scale(scale, PROJECT_ROOT)
        if result:
            all_results[scale] = result

    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("CROSS-SCALE COMPARISON")
        print("=" * 70)
        header = f"{'Metric':<45} | {'5k':>8} | {'50k':>8} | {'100k':>8}"
        print(header)
        print("-" * 70)
        keys = [
            ("baseline_rss_mb", "Baseline RSS (MB)"),
            ("stage2_concat_rss_mb", "After pd.concat train (MB)"),
            ("stage3_numpy_extraction_rss_mb", "After to_numpy [simultaneous] (MB)"),
            ("stage3_simultaneous_copy_overhead_mb", "Simultaneous copy overhead (MB)"),
            ("stage4_lgbm_fit_rss_mb", "After LGBMClassifier.fit() (MB)"),
            ("stage5b_val_concat_rss_mb", "After val concat (MB)"),
            ("stage6_predict_proba_rss_mb", "After predict_proba (MB)"),
            ("stage7_threshold_sweep_rss_mb", "After threshold sweep (MB)"),
            ("measured_peak_rss_mb", "MEASURED PEAK RSS (MB)"),
        ]
        for key, label in keys:
            vals = {sc: all_results.get(sc, {}).get(key, "—") for sc in ["5k", "50k", "100k"]}
            def fmt(v): return f"{v:.1f}" if isinstance(v, float) else str(v)
            print(f"  {label:<43} | {fmt(vals['5k']):>8} | {fmt(vals['50k']):>8} | {fmt(vals['100k']):>8}")
        print("=" * 70)

        # 2.2M extrapolation
        # Scaling factor: 100k uses N_train rows, estimate 2.2M would use ~22x more train rows
        # (empirically: 100k S1 -> 3.5M train rows; 2.2M S1 -> ~76.8M train rows => 21.9x)
        scale_factor = 76_800_000 / all_results.get("100k", {}).get("stage3_numpy_extraction_rss_mb", 1)
        print("\nFull 2.2M extrapolation (based on 100k measured data):")
        if "100k" in all_results:
            r100k = all_results["100k"]
            baseline = r100k.get("baseline_rss_mb", 0)
            peak_100k = r100k.get("measured_peak_rss_mb", 0)
            # Candidate pool index stays ~constant (it's all 3.5M S2+S3 records at full scale)
            # Training matrix: 76.8M rows * 34 * 4 bytes = 10.4 GB
            X_full_gb = 76_800_000 * 34 * 4 / 1e9
            simultaneous_overhead = r100k.get("stage3_simultaneous_copy_overhead_mb", 0)
            print(f"  X_train matrix (76.8M rows x 34 x float32):  {X_full_gb:.2f} GB")
            print(f"  Simultaneous train_df + X_train overhead at 100k: {simultaneous_overhead:.1f} MB")
            overhead_ratio = simultaneous_overhead / max(r100k.get("stage2_concat_rss_mb", 1) - baseline, 1)
            print(f"  Overhead ratio (overhead / train_df size):    {overhead_ratio:.1%}")
            print(f"  Estimated simultaneous overhead at full scale: {X_full_gb * (1 + overhead_ratio):.2f} GB")
            print(f"  If overhead is eliminated: X_train alone requires {X_full_gb:.2f} GB")
        print("=" * 70)


if __name__ == "__main__":
    main()
