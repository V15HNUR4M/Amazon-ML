import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Load reports
with open("output/turn7_validation_report.json", "r") as f:
    r5k = json.load(f)

with open("output/turn8_5_100k/turn7_validation_report.json", "r") as f:
    r100k = json.load(f)

print("="*80)
print("1. REPORT METRICS COMPARISON (5k vs 100k)")
print("="*80)
print(f"Candidate Blocking Recall: 5k={r5k['blocking_diagnostics']['candidate_recall_pct']}% vs 100k={r100k['blocking_diagnostics']['candidate_recall_pct']}% (Delta: {r100k['blocking_diagnostics']['candidate_recall_pct'] - r5k['blocking_diagnostics']['candidate_recall_pct']:+.2f}%)")
print(f"  - S2 Recall: 5k={r5k['blocking_diagnostics']['s2_candidate_recall_pct']}% vs 100k={r100k['blocking_diagnostics']['s2_candidate_recall_pct']}%")
print(f"  - S3 Recall: 5k={r5k['blocking_diagnostics']['s3_candidate_recall_pct']}% vs 100k={r100k['blocking_diagnostics']['s3_candidate_recall_pct']}%")
print(f"Pair Precision: 5k={r5k['validation_metrics']['pair_precision']*100:.2f}% vs 100k={r100k['validation_metrics']['pair_precision']*100:.2f}% (Delta: {r100k['validation_metrics']['pair_precision']*100 - r5k['validation_metrics']['pair_precision']*100:+.2f}%)")
print(f"Pair Recall:    5k={r5k['validation_metrics']['pair_recall']*100:.2f}% vs 100k={r100k['validation_metrics']['pair_recall']*100:.2f}% (Delta: {r100k['validation_metrics']['pair_recall']*100 - r5k['validation_metrics']['pair_recall']*100:+.2f}%)")
print(f"Pair F1:        5k={r5k['validation_metrics']['pair_f1']:.4f} vs 100k={r100k['validation_metrics']['pair_f1']:.4f}")
print(f"Entity Precision: 5k={r5k['validation_metrics']['mean_entity_precision']:.4f} vs 100k={r100k['validation_metrics']['mean_entity_precision']:.4f}")
print(f"Entity Recall:    5k={r5k['validation_metrics']['mean_entity_recall']:.4f} vs 100k={r100k['validation_metrics']['mean_entity_recall']:.4f}")
print(f"Macro F0.5:       5k={r5k['validation_metrics']['macro_f05']:.4f} vs 100k={r100k['validation_metrics']['macro_f05']:.4f} (Delta: {r100k['validation_metrics']['macro_f05'] - r5k['validation_metrics']['macro_f05']:+.4f})")
print(f"Optimal Thresh:   5k={r5k['validation_metrics']['threshold']} vs 100k={r100k['validation_metrics']['threshold']}")

print("\nMiss Breakdown:")
print(f"Val True Matches:       5k={r5k['error_analysis']['total_true_matches']} vs 100k={r100k['error_analysis']['total_true_matches']}")
print(f"Blocking Misses:        5k={r5k['error_analysis']['blocking_misses_count']} ({r5k['error_analysis']['blocking_miss_rate_pct']:.2f}%) vs 100k={r100k['error_analysis']['blocking_misses_count']} ({r100k['error_analysis']['blocking_miss_rate_pct']:.2f}%)")
print(f"Classifier Misses:      5k={r5k['error_analysis']['classifier_misses_count']} ({r5k['error_analysis']['classifier_miss_rate_pct']:.2f}%) vs 100k={r100k['error_analysis']['classifier_misses_count']} ({r100k['error_analysis']['classifier_miss_rate_pct']:.2f}%)")
print(f"False Positives:        5k={r5k['error_analysis']['false_positives']} vs 100k={r100k['error_analysis']['false_positives']}")
print(f"FP Rate (per val ent):  5k={r5k['error_analysis']['false_positives']/1000:.4f} vs 100k={r100k['error_analysis']['false_positives']/20000:.4f}")

print("\nThreshold Sweeps:")
thresh_keys = sorted(r5k['threshold_sweep'].keys(), key=float)
print(f"{'Thresh':<8} | {'5k Macro F0.5':<14} | {'100k Macro F0.5':<14} | {'Delta':<10}")
for k in thresh_keys:
    v5 = r5k['threshold_sweep'][k]
    v100 = r100k['threshold_sweep'].get(k, 0.0)
    print(f"{float(k):<8.2f} | {v5:<14.4f} | {v100:<14.4f} | {v100 - v5:+.4f}")

# Load samples to inspect distribution
print("\n" + "="*80)
print("2. DATASET DISTRIBUTION & PROPERTIES COMPARISON")
print("="*80)
with open("cache/benchmark_sample_s1_5000_dist_5000.pkl", "rb") as f:
    d5k = pickle.load(f)

with open("cache/benchmark_sample_s1_100000_dist_5000.pkl", "rb") as f:
    d100k = pickle.load(f)

s1_5k, gt_5k = d5k["s1"], d5k["gt"]
s1_100k, gt_100k = d100k["s1"], d100k["gt"]

def analyze_gt(gt_df):
    matches_per_s1 = []
    has_s2_cnt = 0
    has_s3_cnt = 0
    has_both_cnt = 0
    zero_cnt = 0
    for _, r in gt_df.iterrows():
        m_str = str(r.get("matched_entity_ids", "")).strip()
        m_list = [x.strip() for x in m_str.split(",") if x.strip()]
        matches_per_s1.append(len(m_list))
        s2 = any(x.startswith("S2") for x in m_list)
        s3 = any(x.startswith("S3") for x in m_list)
        if len(m_list) == 0:
            zero_cnt += 1
        if s2: has_s2_cnt += 1
        if s3: has_s3_cnt += 1
        if s2 and s3: has_both_cnt += 1
    return {
        "n": len(gt_df),
        "mean_matches": np.mean(matches_per_s1),
        "median_matches": np.median(matches_per_s1),
        "zero_matches_pct": zero_cnt / len(gt_df) * 100,
        "has_s2_pct": has_s2_cnt / len(gt_df) * 100,
        "has_s3_pct": has_s3_cnt / len(gt_df) * 100,
        "has_both_pct": has_both_cnt / len(gt_df) * 100,
    }

gt_stats_5k = analyze_gt(gt_5k)
gt_stats_100k = analyze_gt(gt_100k)

print(f"{'Property':<30} | {'5k Sample':<15} | {'100k Sample':<15}")
print("-" * 65)
for k in gt_stats_5k:
    print(f"{k:<30} | {gt_stats_5k[k]:<15.2f} | {gt_stats_100k[k]:<15.2f}")

def analyze_s1_features(s1_df):
    has_addr = (~s1_df["address"].isna()) & (s1_df["address"].astype(str).str.strip().str.len() > 0)
    has_ctry = (~s1_df["country"].isna()) & (s1_df["country"].astype(str).str.strip().str.len() > 0)
    name_len = s1_df["business_name"].astype(str).str.len()
    top_ctry = s1_df["country"].value_counts(normalize=True).head(3).to_dict()
    return {
        "addr_presence_pct": has_addr.mean() * 100,
        "ctry_presence_pct": has_ctry.mean() * 100,
        "mean_name_len": name_len.mean(),
        "top_ctry": top_ctry,
    }

s1_feat_5k = analyze_s1_features(s1_5k)
s1_feat_100k = analyze_s1_features(s1_100k)

print(f"\nS1 Feature Properties:")
print(f"Address Presence: 5k={s1_feat_5k['addr_presence_pct']:.2f}% vs 100k={s1_feat_100k['addr_presence_pct']:.2f}%")
print(f"Country Presence: 5k={s1_feat_5k['ctry_presence_pct']:.2f}% vs 100k={s1_feat_100k['ctry_presence_pct']:.2f}%")
print(f"Mean Name Length: 5k={s1_feat_5k['mean_name_len']:.2f} vs 100k={s1_feat_100k['mean_name_len']:.2f}")

