"""
scratch/test_multi_shard_ranking.py — Test if adding shards dilutes true matches out of top-k.
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

def run_test():
    cfg = Config()
    t0 = time.time()
    
    print("1. Loading 500,000 records from train_source2.tsv (simulating first 2.5 shards)...")
    s2_chunk = pd.read_csv(cfg.TRAIN_SOURCE2, sep="\t", nrows=500_000, keep_default_na=False, dtype=str)
    
    # Split into 5 mini-shards of 100,000 records each
    shards_raw = [s2_chunk.iloc[i*100_000 : (i+1)*100_000].copy() for i in range(5)]
    shards = [preprocess_records(s) for s in shards_raw]
    del s2_chunk, shards_raw
    print(f"5 shards of 100k records preprocessed in {time.time() - t0:.1f}s.")
    
    # True matches in Shard 0
    s2_shard0_ids = set(shards[0]["entity_id"].astype(str))
    
    print("2. Finding S1 entities whose true match is in Shard 0...")
    gt = pd.read_csv(cfg.TRAIN_GROUND_TRUTH, sep="\t", keep_default_na=False)
    s1_col = "source1_entity_id" if "source1_entity_id" in gt.columns else "s1_id"
    
    s1_to_s2 = {}
    for _, r in gt.iterrows():
        s1_id = str(r[s1_col])
        m_str = str(r.get("matched_entity_ids", "")).strip()
        m_list = [x.strip() for x in m_str.split(",") if x.strip()]
        in_s0 = [m for m in m_list if m in s2_shard0_ids]
        if in_s0:
            s1_to_s2[s1_id] = in_s0
            if len(s1_to_s2) >= 100:
                break
                
    # Load these S1 entities
    needed_s1 = set(s1_to_s2.keys())
    s1_rows = []
    for chunk in pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", chunksize=100_000, keep_default_na=False, dtype=str):
        matched = chunk[chunk["entity_id"].astype(str).isin(needed_s1)]
        if len(matched) > 0:
            s1_rows.append(matched)
        if sum(len(x) for x in s1_rows) == len(needed_s1):
            break
    s1_df = pd.concat(s1_rows, ignore_index=True)
    s1_proc = preprocess_records(s1_df)
    print(f"Loaded {len(s1_proc)} S1 entities.")
    
    b_cfg = GPUBlockingConfig(top_k_per_s1=60, max_candidates_per_block=150)
    blocker_weights = {"A": 4, "D": 3, "B": 2, "C": 2}
    cap = b_cfg.max_candidates_per_block
    
    # Index each shard
    shard_indices = []
    for i, sh_df in enumerate(shards):
        idx = GPUBlockingIndex(config=b_cfg)
        idx.build_vocabulary(s1_proc, sh_df)
        idx.add_records(sh_df)
        shard_indices.append(idx)
        print(f"Indexed shard {i} with {len(idx.index):,} keys.")
        
    # Pre-compute S1 keys
    s1_records = s1_proc.to_dict(orient="records")
    s1_keys = {}
    for rec in s1_records:
        eid = str(rec["entity_id"])
        s1_keys[eid] = generate_gpu_blocking_keys(rec, b_cfg, shard_indices[0].token_freq)
        
    # Track candidate scores as we accumulate shards: Shard 0 alone, then +Shard 1, +Shard 2, +Shard 3, +Shard 4
    global_scores = defaultdict(lambda: defaultdict(float))
    
    for num_shards in range(1, 6):
        curr_idx = shard_indices[num_shards - 1]
        
        for eid, keys_by_blocker in s1_keys.items():
            for blocker_tag, key_list in keys_by_blocker.items():
                base_w = blocker_weights.get(blocker_tag, 1)
                for k in key_list:
                    plist = curr_idx.index.get(k)
                    if not plist or len(plist) >= cap:
                        continue
                    freq_discount = 1.0 if len(plist) <= 10 else (0.8 if len(plist) <= 50 else 0.5)
                    score_inc = base_w * freq_discount
                    for cid in plist:
                        global_scores[eid][cid] += score_inc
                        
        # Evaluate recall and rank on the 100 S1 queries
        retrieved_top60 = 0
        total_eval = 0
        cand_counts = []
        ranks = []
        
        for eid, true_ids in s1_to_s2.items():
            scores = global_scores[eid]
            cand_counts.append(len(scores))
            sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top60 = {cid for cid, _ in sorted_cands[:60]}
            cand_order = [c for c, _ in sorted_cands]
            
            for tid in true_ids:
                total_eval += 1
                if tid in top60:
                    retrieved_top60 += 1
                if tid in cand_order:
                    ranks.append(cand_order.index(tid) + 1)
                else:
                    ranks.append(999999)
                    
        print(f"\n--- After {num_shards} Shard(s) ({num_shards * 100}k reference records) ---")
        print(f"  Avg candidates accumulated per S1: {np.mean(cand_counts):.1f} (Max: {np.max(cand_counts)})")
        print(f"  Recall in Top 60: {retrieved_top60}/{total_eval} ({retrieved_top60/total_eval*100:.2f}%)")
        print(f"  True match ranks: Mean={np.mean(ranks):.1f}, Median={np.median(ranks)}, P90={np.quantile(ranks, 0.90):.1f}")
        displaced = sum(1 for r in ranks if 60 < r < 999999)
        print(f"  True matches present in pool but displaced beyond Top 60: {displaced}")

if __name__ == "__main__":
    run_test()
