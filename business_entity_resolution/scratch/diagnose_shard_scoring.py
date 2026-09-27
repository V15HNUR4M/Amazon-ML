"""
scratch/diagnose_shard_scoring.py — Inspect posting lists, caps, and scores in a real 200k shard.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
import time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from src.config import Config
from src.preprocessing import preprocess_records
from src.gpu.gpu_blocking import (
    GPUBlockingConfig,
    GPUBlockingIndex,
    generate_gpu_blocking_keys,
)

def run_diagnostic():
    cfg = Config()
    t0 = time.time()
    
    print("1. Loading 200,000 records from train_source2.tsv (Shard 0 simulation)...")
    s2_chunk = pd.read_csv(cfg.TRAIN_SOURCE2, sep="\t", nrows=200_000, keep_default_na=False, dtype=str)
    s2_proc = preprocess_records(s2_chunk)
    s2_ids_in_shard = set(s2_proc["entity_id"].astype(str))
    print(f"Shard 0 ready: {len(s2_proc)} records in {time.time() - t0:.1f}s.")
    
    print("2. Finding S1 entities in train_ground_truth.tsv whose true match is in this Shard 0...")
    gt = pd.read_csv(cfg.TRAIN_GROUND_TRUTH, sep="\t", keep_default_na=False)
    s1_col = "source1_entity_id" if "source1_entity_id" in gt.columns else "s1_id"
    
    s1_to_s2_matches = {}
    for _, r in gt.iterrows():
        s1_id = str(r[s1_col])
        m_str = str(r.get("matched_entity_ids", "")).strip()
        m_list = [x.strip() for x in m_str.split(",") if x.strip()]
        in_shard = [m for m in m_list if m in s2_ids_in_shard]
        if in_shard:
            s1_to_s2_matches[s1_id] = in_shard
            if len(s1_to_s2_matches) >= 300:
                break
                
    print(f"Found {len(s1_to_s2_matches)} S1 entities whose true S2 match is inside Shard 0.")
    
    # Load these S1 entities from train_source1.tsv
    needed_s1_ids = set(s1_to_s2_matches.keys())
    s1_rows = []
    for chunk in pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", chunksize=100_000, keep_default_na=False, dtype=str):
        matched = chunk[chunk["entity_id"].astype(str).isin(needed_s1_ids)]
        if len(matched) > 0:
            s1_rows.append(matched)
        if sum(len(x) for x in s1_rows) == len(needed_s1_ids):
            break
    s1_df = pd.concat(s1_rows, ignore_index=True)
    s1_proc = preprocess_records(s1_df)
    print(f"Loaded and preprocessed {len(s1_proc)} S1 entities.")
    
    # Build GPUBlockingIndex on Shard 0
    b_cfg = GPUBlockingConfig(top_k_per_s1=60, max_candidates_per_block=150)
    index = GPUBlockingIndex(config=b_cfg)
    # Build vocab on shard
    index.build_vocabulary(s1_proc, s2_proc)
    index.add_records(s2_proc)
    
    # Inspect posting lists
    print("\n--- Posting List Inspection on Shard 0 ---")
    capped_keys = [k for k, plist in index.index.items() if len(plist) >= 150]
    print(f"Total Unique Keys in Shard 0 Index: {len(index.index):,}")
    print(f"Keys hitting cap (len >= 150): {len(capped_keys):,} ({len(capped_keys)/len(index.index)*100:.2f}%)")
    
    # Breakdown capped keys by prefix
    capped_by_type = Counter(k.split(":")[0] for k in capped_keys)
    print("Capped keys by blocker type:")
    for ktype, cnt in capped_by_type.most_common(10):
        print(f"  {ktype}: {cnt:,}")
        
    # Now query the S1 entities
    print("\n3. Querying Shard 0 for the S1 entities...")
    retrieved_true = 0
    total_true = 0
    miss_reasons = Counter()
    miss_details = []
    
    blocker_weights = {"A": 4, "D": 3, "B": 2, "C": 2}
    cap = b_cfg.max_candidates_per_block
    
    for _, s1_row in s1_proc.iterrows():
        s1_id = str(s1_row["entity_id"])
        true_s2_ids = s1_to_s2_matches[s1_id]
        
        # Keys for this S1
        keys_by_blocker = generate_gpu_blocking_keys(s1_row, b_cfg, index.token_freq)
        
        # Let's inspect which keys match in index
        candidate_scores = defaultdict(float)
        skipped_keys_due_to_cap = []
        active_keys = []
        
        for blocker_tag, key_list in keys_by_blocker.items():
            base_w = blocker_weights.get(blocker_tag, 1)
            for k in key_list:
                plist = index.index.get(k)
                if not plist:
                    continue
                if len(plist) >= cap:
                    skipped_keys_due_to_cap.append(k)
                    continue
                active_keys.append(k)
                freq_discount = 1.0 if len(plist) <= 10 else (0.8 if len(plist) <= 50 else 0.5)
                score_inc = base_w * freq_discount
                for cid in plist:
                    candidate_scores[cid] += score_inc
                    
        # Sort candidates
        sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_60_cands = {cid for cid, _ in sorted_cands[:60]}
        all_scored_cands = set(candidate_scores.keys())
        
        for tid in true_s2_ids:
            total_true += 1
            if tid in top_60_cands:
                retrieved_true += 1
            else:
                # Why missed?
                # Check true row keys
                true_row = s2_proc[s2_proc["entity_id"] == tid].iloc[0]
                true_keys_by_blocker = generate_gpu_blocking_keys(true_row, b_cfg, index.token_freq)
                
                s1_all_keys = {k: b for b, kl in keys_by_blocker.items() for k in kl}
                tr_all_keys = {k: b for b, kl in true_keys_by_blocker.items() for k in kl}
                shared = set(s1_all_keys.keys()) & set(tr_all_keys.keys())
                
                if not shared:
                    reason = "NO_SHARED_KEYS"
                elif tid in all_scored_cands:
                    # Scored, but ranked outside top 60!
                    score = candidate_scores[tid]
                    rank = [c for c, _ in sorted_cands].index(tid) + 1
                    reason = f"RANKED_OUT_OF_TOP60 (rank={rank}, score={score:.2f}, top60_cutoff={sorted_cands[59][1]:.2f}, total_candidates={len(sorted_cands)})"
                else:
                    # Shared keys existed, but none added tid to candidate_scores!
                    # Why? Because shared keys were capped or tid was rejected at add_records!
                    shared_capped = [k for k in shared if k in skipped_keys_due_to_cap]
                    shared_in_index = [k for k in shared if k in index.index]
                    reason = f"SHARED_KEYS_DROPPED_BY_CAP (shared={list(shared)}, capped={shared_capped})"
                    
                miss_reasons[reason.split(" ")[0]] += 1
                miss_details.append({
                    "s1_id": s1_id,
                    "target_id": tid,
                    "reason": reason,
                    "shared_keys": list(shared),
                })
                
    print(f"\nShard 0 Results:")
    print(f"Total True Matches Evaluated: {total_true}")
    print(f"Retrieved in Top 60: {retrieved_true} ({retrieved_true/total_true*100:.2f}%)")
    print(f"Missed: {total_true - retrieved_true} ({(total_true - retrieved_true)/total_true*100:.2f}%)")
    print("\nMiss Reasons Breakdown:")
    for r, cnt in miss_reasons.most_common():
        print(f"  {r}: {cnt} ({(cnt/total_true)*100:.2f}%)")
        
    print("\nSample Miss Details:")
    for m in miss_details[:10]:
        print(f"  S1={m['s1_id']} -> TR={m['target_id']}: {m['reason']}")

if __name__ == "__main__":
    run_diagnostic()
