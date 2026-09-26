"""
benchmark_blocking.py — Benchmark blocking, pair building, and feature extraction.

Rigorously measures:
- Candidate recall (overall, S2, S3, by country, singletons)
- Blocker ablation:
  1. Blocker A only (Exact name, compact, no suffix, domain stem)
  2. Blocker A + B (+ selective token blocks)
  3. Blocker A + B + C (+ address number and street blocks)
  4. Blocker A + B + C + D (+ composite name/prefix + address number)
- Candidate reduction ratio
- Candidate distribution per Source 1 entity (mean, median, p90, p95, max)
- Training pair construction (positives vs hard negatives)
- Feature extraction performance and matrix shape
- Wall-clock runtime and peak memory (via tracemalloc)

Usage:
------
python scripts/benchmark_blocking.py --sample-source1 100
python scripts/benchmark_blocking.py --sample-source1 200 --distractors 1000
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
import tracemalloc
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.blocking import (
    BlockingConfig,
    BlockingIndex,
    blocking_recall,
    blocking_reduction_ratio,
    build_candidates,
    detailed_blocking_diagnostics,
)
from src.config import Config
from src.features import compute_features
from src.pair_builder import build_train_pairs
from src.preprocessing import preprocess_records


def load_or_create_benchmark_sample(
    sample_s1_count: int = 100,
    distractor_count: int = 1000,
    cache_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load or build a representative benchmark dataset.

    Consists of:
    - Sample Source 1 records
    - Ground truth for those Source 1 records
    - Candidate pool containing all true matches for the S1 sample
    - Plus realistic distractor records from Source 2 and Source 3
    """
    cfg = Config()
    c_dir = cache_dir or cfg.CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    cache_file = c_dir / f"benchmark_sample_s1_{sample_s1_count}_dist_{distractor_count}.pkl"

    if cache_file.exists():
        print(f"Loading cached benchmark sample from {cache_file}...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["s1"], data["s2"], data["s3"], data["gt"]

    print(f"Building benchmark sample ({sample_s1_count} S1 queries + true matches + {distractor_count} distractors)...")

    # 1. Load Ground Truth sample first
    gt = pd.read_csv(cfg.TRAIN_GROUND_TRUTH, sep="\t", nrows=sample_s1_count, keep_default_na=False)
    needed_s1: set[str] = set(gt["source1_entity_id"])

    # Stream S1 chunks to collect the corresponding S1 records
    s1_rows: list[dict] = []
    for chunk in pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", chunksize=100000, keep_default_na=False):
        matches = chunk[chunk["entity_id"].isin(needed_s1)]
        if len(matches) > 0:
            s1_rows.extend(matches.to_dict(orient="records"))
            needed_s1 -= set(matches["entity_id"])
        if len(needed_s1) == 0:
            break
    s1 = pd.DataFrame(s1_rows).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

    # 2. Collect all true match IDs from S2 and S3
    needed_s2: set[str] = set()
    needed_s3: set[str] = set()
    for _, r in gt.iterrows():
        for m in str(r.get("matched_entity_ids", "")).split(","):
            m = m.strip()
            if m.startswith("S2"):
                needed_s2.add(m)
            elif m.startswith("S3"):
                needed_s3.add(m)

    print(f"  Target matches needed: {len(needed_s2)} in S2, {len(needed_s3)} in S3")

    # 3. Stream S2 chunks to retrieve true matches + distractors
    s2_rows: list[dict] = []
    distractors_s2_needed = distractor_count // 2
    for chunk in pd.read_csv(cfg.TRAIN_SOURCE2, sep="\t", chunksize=100000, keep_default_na=False):
        # Extract true matches
        matches = chunk[chunk["entity_id"].isin(needed_s2)]
        if len(matches) > 0:
            s2_rows.extend(matches.to_dict(orient="records"))
            needed_s2 -= set(matches["entity_id"])
        # Take initial distractors
        if distractors_s2_needed > 0:
            dist = chunk[~chunk["entity_id"].isin(needed_s2)].head(distractors_s2_needed)
            s2_rows.extend(dist.to_dict(orient="records"))
            distractors_s2_needed -= len(dist)
        if len(needed_s2) == 0 and distractors_s2_needed <= 0:
            break

    # 4. Stream S3 chunks to retrieve true matches + distractors
    s3_rows: list[dict] = []
    distractors_s3_needed = distractor_count // 2
    for chunk in pd.read_csv(cfg.TRAIN_SOURCE3, sep="\t", chunksize=100000, keep_default_na=False):
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

    # 5. Preprocess records
    print("  Preprocessing benchmark sample records...")
    s1_proc = preprocess_records(s1)
    s2_proc = preprocess_records(s2_df)
    s3_proc = preprocess_records(s3_df)

    sample_data = {"s1": s1_proc, "s2": s2_proc, "s3": s3_proc, "gt": gt}
    with open(cache_file, "wb") as f:
        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Benchmark sample saved to {cache_file}")
    return s1_proc, s2_proc, s3_proc, gt


def run_blocking_benchmark(
    sample_s1: int = 100,
    distractors: int = 1000,
    max_cands_per_block: int = 100,
    top_k: int = 50,
) -> None:
    print("=" * 80)
    print("BLOCKING & FEATURE BENCHMARK — DEVELOPMENT SAMPLE EVALUATION")
    print("=" * 80)
    print(f"Settings: Sample S1={sample_s1}, Distractors={distractors}, MaxBlock={max_cands_per_block}, TopK={top_k}\n")

    s1, s2, s3, gt = load_or_create_benchmark_sample(
        sample_s1_count=sample_s1,
        distractor_count=distractors,
    )

    print(f"Loaded records: S1={len(s1)}, S2 pool={len(s2)}, S3 pool={len(s3)}, GT rows={len(gt)}")
    total_pool = len(s2) + len(s3)

    # -----------------------------------------------------------------------
    # Part 1: Blocker Ablation Study
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: BLOCKER ABLATION STUDY (CANDIDATE RECALL & EFFICIENCY)")
    print("=" * 80)

    ablation_stages = [
        (
            "Stage 1: Blocker A only (Exact name/compact/no-suf/domain)",
            BlockingConfig(
                enable_blocker_a=True,
                enable_blocker_b=False,
                enable_blocker_c=False,
                enable_blocker_d=False,
                max_candidates_per_block=max_cands_per_block,
                top_k_per_s1=top_k,
            ),
        ),
        (
            "Stage 2: Blocker A + B (+ selective token blocks)",
            BlockingConfig(
                enable_blocker_a=True,
                enable_blocker_b=True,
                enable_blocker_c=False,
                enable_blocker_d=False,
                max_candidates_per_block=max_cands_per_block,
                top_k_per_s1=top_k,
            ),
        ),
        (
            "Stage 3: Blocker A + B + C (+ address number & street blocks)",
            BlockingConfig(
                enable_blocker_a=True,
                enable_blocker_b=True,
                enable_blocker_c=True,
                enable_blocker_d=False,
                max_candidates_per_block=max_cands_per_block,
                top_k_per_s1=top_k,
            ),
        ),
        (
            "Stage 4: Blocker A + B + C + D (+ composite name/prefix + addr num)",
            BlockingConfig(
                enable_blocker_a=True,
                enable_blocker_b=True,
                enable_blocker_c=True,
                enable_blocker_d=True,
                max_candidates_per_block=max_cands_per_block,
                top_k_per_s1=top_k,
            ),
        ),
    ]

    print(
        f"\n{'Stage':<42} | {'Recall':<8} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'Cands':<8} | {'Red. Ratio':<12} | {'Time (s)':<8}"
    )
    print("-" * 105)

    final_candidates = None
    final_diag = None

    for label, b_cfg in ablation_stages:
        tracemalloc.start()
        t0 = time.perf_counter()

        cands = build_candidates(s1, s2, s3, config=b_cfg)

        t1 = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = t1 - t0
        diag = detailed_blocking_diagnostics(
            cands,
            gt,
            source1=s1,
            source2_count=len(s2),
            source3_count=len(s3),
        )

        final_candidates = cands
        final_diag = diag

        rec = diag["overall_candidate_recall"]
        s2_rec = diag["s2_candidate_recall"]
        s3_rec = diag["s3_candidate_recall"]
        n_cands = len(cands)
        rr = diag["candidate_reduction_ratio"] or 0.0

        print(
            f"{label:<42} | {rec:<8.4f} | {s2_rec:<8.4f} | {s3_rec:<8.4f} | {n_cands:<8} | {rr:<12.6f} | {elapsed:<8.3f}"
        )

    # -----------------------------------------------------------------------
    # Part 2: Detailed Candidate Diagnostics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: FINAL BLOCKING DIAGNOSTICS & CANDIDATE DISTRIBUTION")
    print("=" * 80)
    if final_diag:
        print(f"  Total Source 1 Entities : {final_diag['total_s1_entities']}")
        print(f"  Total True Matches      : {final_diag['total_true_matches']}")
        print(f"  Retrieved True Matches  : {final_diag['retrieved_true_matches']}")
        print(f"  Overall Candidate Recall: {final_diag['overall_candidate_recall']:.4f} ({final_diag['overall_candidate_recall']*100:.2f}%)")
        print(f"  Source 2 Recall         : {final_diag['s2_candidate_recall']:.4f} ({final_diag['s2_candidate_recall']*100:.2f}%)")
        print(f"  Source 3 Recall         : {final_diag['s3_candidate_recall']:.4f} ({final_diag['s3_candidate_recall']*100:.2f}%)")
        print(f"  Singleton Entities      : {final_diag['singleton_count']} (Clean zero-candidate singletons: {final_diag['singleton_clean_count']})")
        print(f"  Candidate Reduction Rate: {final_diag['candidate_reduction_ratio']:.6f}")
        print(f"  Candidates/S1 Mean      : {final_diag['candidates_per_s1_mean']:.2f}")
        print(f"  Candidates/S1 Median    : {final_diag['candidates_per_s1_median']:.1f}")
        print(f"  Candidates/S1 P90       : {final_diag['candidates_per_s1_p90']:.1f}")
        print(f"  Candidates/S1 P95       : {final_diag['candidates_per_s1_p95']:.1f}")
        print(f"  Candidates/S1 P99       : {final_diag['candidates_per_s1_p99']:.1f}")
        print(f"  Candidates/S1 Max       : {final_diag['candidates_per_s1_max']}")

    # -----------------------------------------------------------------------
    # Part 3: Training Pair Construction
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: TRAINING PAIR CONSTRUCTION (POSITIVES & HARD NEGATIVES)")
    print("=" * 80)
    t0 = time.perf_counter()
    train_pairs = build_train_pairs(s1, s2, s3, final_candidates, gt, sample_neg_ratio=5.0)
    t1 = time.perf_counter()

    pos_count = (train_pairs["label"] == 1).sum() if len(train_pairs) > 0 else 0
    neg_count = (train_pairs["label"] == 0).sum() if len(train_pairs) > 0 else 0
    unique_s1 = train_pairs["s1_id"].nunique() if len(train_pairs) > 0 else 0

    print(f"  Total Pairs Assembled   : {len(train_pairs)} in {t1-t0:.3f}s")
    print(f"  True Positives (label=1): {pos_count}")
    print(f"  Hard Negatives (label=0): {neg_count} (Ratio: {neg_count/max(pos_count, 1):.1f}:1)")
    print(f"  Unique S1 Groups Preserved: {unique_s1} (GroupKFold leakage-safe)")

    # -----------------------------------------------------------------------
    # Part 4: Pairwise Feature Extraction
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 4: PAIRWISE FEATURE EXTRACTION")
    print("=" * 80)
    tracemalloc.start()
    t0 = time.perf_counter()

    feature_matrix = compute_features(train_pairs)

    t1 = time.perf_counter()
    _, feat_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    feat_time = t1 - t0
    feat_throughput = len(train_pairs) / feat_time if feat_time > 0 else 0

    print(f"  Feature Matrix Dimensions: {feature_matrix.shape[0]} pairs × {feature_matrix.shape[1]} features")
    print(f"  Computation Time         : {feat_time:.3f}s ({feat_throughput:.0f} pairs/sec)")
    print(f"  Peak Memory Allocation   : {feat_peak / (1024*1024):.2f} MB")
    print(f"  Features Computed        :\n    {list(feature_matrix.columns)}")

    print("\nSample Feature Vector (First Pair):")
    if len(feature_matrix) > 0:
        sample_row = feature_matrix.iloc[0].to_dict()
        pair_meta = train_pairs.iloc[0]
        print(f"    Pair: {pair_meta['s1_id']} <-> {pair_meta['candidate_id']} (Label = {pair_meta['label']})")
        for k, v in list(sample_row.items())[:12]:
            print(f"      {k:<28}: {v}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE — STAGES READY FOR MODEL TRAINING IN COLAB")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark blocking and feature extraction.")
    parser.add_argument("--sample-source1", type=int, default=100, help="Number of S1 entities to evaluate.")
    parser.add_argument("--distractors", type=int, default=1000, help="Number of distractor candidate records.")
    parser.add_argument("--max-candidates-per-block", type=int, default=100, help="Max posting list per block.")
    parser.add_argument("--top-k", type=int, default=50, help="Max candidates per S1 entity.")
    args = parser.parse_args()

    run_blocking_benchmark(
        sample_s1=args.sample_source1,
        distractors=args.distractors,
        max_cands_per_block=args.max_candidates_per_block,
        top_k=args.top_k,
    )
