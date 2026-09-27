"""
scratch/inspect_displaced.py — Inspect the exact scores and keys of displaced true matches.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from src.config import Config
from src.preprocessing import preprocess_records
from src.gpu.gpu_blocking import GPUBlockingConfig, GPUBlockingIndex, generate_gpu_blocking_keys

def run():
    cfg = Config()
    s2_chunk = pd.read_csv(cfg.TRAIN_SOURCE2, sep="\t", nrows=500_000, keep_default_na=False, dtype=str)
    shards_raw = [s2_chunk.iloc[i*100_000 : (i+1)*100_000].copy() for i in range(5)]
    shards = [preprocess_records(s) for s in shards_raw]
    s2_shard0_ids = set(shards[0]["entity_id"].astype(str))
    
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
    
    b_cfg = GPUBlockingConfig(top_k_per_s1=60, max_candidates_per_block=150)
    blocker_weights = {"A": 4, "D": 3, "B": 2, "C": 2}
    cap = b_cfg.max_candidates_per_block
    
    shard_indices = []
    for i, sh_df in enumerate(shards):
        idx = GPUBlockingIndex(config=b_cfg)
        idx.build_vocabulary(s1_proc, sh_df)
        idx.add_records(sh_df)
        shard_indices.append(idx)
        
    s1_records = s1_proc.to_dict(orient="records")
    s1_keys = {str(r["entity_id"]): generate_gpu_blocking_keys(r, b_cfg, shard_indices[0].token_freq) for r in s1_records}
    
    # Run across 5 shards and record candidate scores with key provenance
    # global_scores: {eid -> {cid -> score}}
    # cand_provenance: {eid -> {cid -> [matching keys]}}
    global_scores = defaultdict(lambda: defaultdict(float))
    cand_provenance = defaultdict(lambda: defaultdict(list))
    
    for curr_idx in shard_indices:
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
                        cand_provenance[eid][cid].append(k)
                        
    # Find displaced true matches
    print("="*80)
    print("ANALYSIS OF DISPLACED TRUE MATCHES")
    print("="*80)
    
    s1_map = {str(r["entity_id"]): r for _, r in s1_proc.iterrows()}
    s2_map = {str(r["entity_id"]): r for _, r in shards[0].iterrows()}
    
    for eid, true_ids in s1_to_s2.items():
        scores = global_scores[eid]
        sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top60 = {cid for cid, _ in sorted_cands[:60]}
        cand_order = [c for c, _ in sorted_cands]
        cutoff_score = sorted_cands[59][1] if len(sorted_cands) >= 60 else 0.0
        
        for tid in true_ids:
            if tid in cand_order and tid not in top60:
                rank = cand_order.index(tid) + 1
                tr_score = scores[tid]
                tr_keys = cand_provenance[eid][tid]
                s1_rec = s1_map[eid]
                tr_rec = s2_map[tid]
                
                print(f"\n[DISPLACED] S1={eid} -> TrueMatch={tid} | Rank={rank}/{len(sorted_cands)} | Score={tr_score:.2f} (Cutoff={cutoff_score:.2f})")
                print(f"  S1: '{s1_rec.get('business_name')}' | Addr: '{s1_rec.get('business_address')}'")
                print(f"  TR: '{tr_rec.get('business_name')}' | Addr: '{tr_rec.get('business_address')}'")
                print(f"  Keys matching True Match: {tr_keys}")
                
                # Check what candidates are in Top 5
                print(f"  Top 3 Candidates that beat it:")
                for c_cid, c_sc in sorted_cands[:3]:
                    print(f"    - {c_cid}: score={c_sc:.2f}, keys={cand_provenance[eid][c_cid][:3]}")
                    
                # Check score distribution of candidates that beat it
                scores_above = [s for _, s in sorted_cands[:60]]
                print(f"  Top 60 score distribution: min={min(scores_above):.2f}, median={np.median(scores_above):.2f}, max={max(scores_above):.2f}")
                print(f"  Count of candidates tied at cutoff score ({cutoff_score:.2f}): {scores_above.count(cutoff_score)}")

if __name__ == "__main__":
    run()
