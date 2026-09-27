import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"e:\Projects\amazon ML\Architecture\business_entity_resolution")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gpu.gpu_blocking import GPUBlockingConfig, GPUBlockingIndex, generate_gpu_blocking_keys
from src.preprocessing import preprocess_records

print("Loading cached 5k and 100k samples...")
with open("cache/benchmark_sample_s1_5000_dist_5000.pkl", "rb") as f:
    d5k = pickle.load(f)

with open("cache/benchmark_sample_s1_100000_dist_5000.pkl", "rb") as f:
    d100k = pickle.load(f)

s1_5k, s2_5k, s3_5k, gt_5k = d5k["s1"], d5k["s2"], d5k["s3"], d5k["gt"]
s1_100k, s2_100k, s3_100k, gt_100k = d100k["s1"], d100k["s2"], d100k["s3"], d100k["gt"]

print(f"5k Sample:  S1={len(s1_5k)}, S2={len(s2_5k)}, S3={len(s3_5k)}, GT={len(gt_5k)}")
print(f"100k Sample: S1={len(s1_100k)}, S2={len(s2_100k)}, S3={len(s3_100k)}, GT={len(gt_100k)}")

# Ground truth dictionaries
gt_dict_5k = {}
for _, r in gt_5k.iterrows():
    s1_id = str(r["source1_entity_id"])
    m_str = str(r.get("matched_entity_ids", "")).strip()
    m_set = {x.strip() for x in m_str.split(",") if x.strip()}
    gt_dict_5k[s1_id] = m_set

gt_dict_100k = {}
for _, r in gt_100k.iterrows():
    s1_id = str(r["source1_entity_id"])
    m_str = str(r.get("matched_entity_ids", "")).strip()
    m_set = {x.strip() for x in m_str.split(",") if x.strip()}
    gt_dict_100k[s1_id] = m_set

# Let's inspect candidate distributions
# Let's run blocking index on 5k sample
print("\nIndexing 5k sample candidates...")
idx_5k = GPUBlockingIndex(GPUBlockingConfig(top_k_per_s1=60))
idx_5k.build_vocabulary(s1_5k, s2_5k, s3_5k)
idx_5k.add_records(s2_5k)
idx_5k.add_records(s3_5k)

cand_counts_5k = []
zero_cands_5k = 0
hit_cap_5k = 0
untruncated_counts_5k = []

for _, r in s1_5k.iterrows():
    keys_by_b = generate_gpu_blocking_keys(r, idx_5k.config, idx_5k.token_freq)
    scores = defaultdict(float)
    for b_tag, klist in keys_by_b.items():
        base_w = {"A": 4, "D": 3, "B": 2, "C": 2}.get(b_tag, 1)
        for k in klist:
            plist = idx_5k.index.get(k)
            if not plist or len(plist) >= idx_5k.config.max_candidates_per_block:
                continue
            freq_discount = 1.0 if len(plist) <= 10 else (0.8 if len(plist) <= 50 else 0.5)
            w = base_w * freq_discount
            for cid in plist:
                scores[cid] += w
    c_count = len(scores)
    untruncated_counts_5k.append(c_count)
    if c_count == 0:
        zero_cands_5k += 1
    if c_count >= 60:
        hit_cap_5k += 1
    cand_counts_5k.append(min(c_count, 60))

cand_counts_5k = np.array(cand_counts_5k)
untruncated_counts_5k = np.array(untruncated_counts_5k)

print(f"\n--- 5k Candidate Distribution (N={len(s1_5k)}) ---")
print(f"Entities with 0 candidates: {zero_cands_5k} ({zero_cands_5k/len(s1_5k)*100:.2f}%)")
print(f"Entities hitting top_k=60 cap: {hit_cap_5k} ({hit_cap_5k/len(s1_5k)*100:.2f}%)")
print(f"Mean candidates per S1: {cand_counts_5k.mean():.2f} (untruncated mean: {untruncated_counts_5k.mean():.2f})")
print(f"Median: {np.median(cand_counts_5k):.1f} | P25: {np.percentile(cand_counts_5k, 25):.1f} | P75: {np.percentile(cand_counts_5k, 75):.1f} | P90: {np.percentile(cand_counts_5k, 90):.1f}")
print(f"Max candidates before cap: {untruncated_counts_5k.max()}")

# Now let's test on 100k sample
# We'll index the 100k candidate pool and evaluate a sample of 5,000 S1 queries to see untruncated distribution and top_k impact
print("\nIndexing 100k candidate pool (350k records)...")
idx_100k = GPUBlockingIndex(GPUBlockingConfig(top_k_per_s1=60))
idx_100k.build_vocabulary(s1_100k.head(10000), s2_100k, s3_100k)
idx_100k.add_records(s2_100k)
idx_100k.add_records(s3_100k)

print("Evaluating 5,000 queries against 100k index with varying top_k...")
test_s1_sample = s1_100k.head(5000)
untruncated_counts_100k = []
cand_counts_100k_60 = []
zero_cands_100k = 0
hit_cap_100k_60 = 0

recall_at_topk = {20: 0, 40: 0, 60: 0, 80: 0, 100: 0, 150: 0, 999999: 0}
total_trues_in_sample = 0

for _, r in test_s1_sample.iterrows():
    s1_id = str(r["entity_id"])
    true_set = gt_dict_100k.get(s1_id, set())
    total_trues_in_sample += len(true_set)

    keys_by_b = generate_gpu_blocking_keys(r, idx_100k.config, idx_100k.token_freq)
    scores = defaultdict(float)
    for b_tag, klist in keys_by_b.items():
        base_w = {"A": 4, "D": 3, "B": 2, "C": 2}.get(b_tag, 1)
        for k in klist:
            plist = idx_100k.index.get(k)
            if not plist or len(plist) >= idx_100k.config.max_candidates_per_block:
                continue
            freq_discount = 1.0 if len(plist) <= 10 else (0.8 if len(plist) <= 50 else 0.5)
            w = base_w * freq_discount
            for cid in plist:
                scores[cid] += w

    c_count = len(scores)
    untruncated_counts_100k.append(c_count)
    if c_count == 0:
        zero_cands_100k += 1
    if c_count >= 60:
        hit_cap_100k_60 += 1
    cand_counts_100k_60.append(min(c_count, 60))

    sorted_cands = [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    for k_val in recall_at_topk:
        k_cands = set(sorted_cands[:k_val])
        recall_at_topk[k_val] += len(true_set & k_cands)

cand_counts_100k_60 = np.array(cand_counts_100k_60)
untruncated_counts_100k = np.array(untruncated_counts_100k)

print(f"\n--- 100k Candidate Distribution (Sample N=5,000 queries against 350k pool) ---")
print(f"Entities with 0 candidates: {zero_cands_100k} ({zero_cands_100k/len(test_s1_sample)*100:.2f}%)")
print(f"Entities hitting top_k=60 cap: {hit_cap_100k_60} ({hit_cap_100k_60/len(test_s1_sample)*100:.2f}%)")
print(f"Mean candidates per S1 (capped at 60): {cand_counts_100k_60.mean():.2f}")
print(f"Untruncated Mean candidates per S1:   {untruncated_counts_100k.mean():.2f}")
print(f"Median: {np.median(cand_counts_100k_60):.1f} | P25: {np.percentile(cand_counts_100k_60, 25):.1f} | P75: {np.percentile(cand_counts_100k_60, 75):.1f} | P90: {np.percentile(cand_counts_100k_60, 90):.1f}")
print(f"Max candidates before cap: {untruncated_counts_100k.max()}")

print("\n--- Recall vs Top-K Cutoff on 100k Sample ---")
print(f"Total True Matches: {total_trues_in_sample}")
for k_val in sorted(recall_at_topk.keys()):
    rec = recall_at_topk[k_val] / total_trues_in_sample * 100
    tag = " (Current Turn 7 cap)" if k_val == 60 else (" (Untruncated Index Ceiling)" if k_val == 999999 else "")
    k_str = f"top_k={k_val}" if k_val != 999999 else "top_k=ALL"
    print(f"  {k_str:<12}: Recall = {rec:6.2f}% ({recall_at_topk[k_val]}/{total_trues_in_sample}){tag}")
