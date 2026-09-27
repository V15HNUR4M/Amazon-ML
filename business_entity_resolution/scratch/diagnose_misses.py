"""
scratch/diagnose_misses.py — Root cause investigation for blocking recall drop.
"""
from __future__ import annotations

import time
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")
from src.config import Config
from src.preprocessing import preprocess_records
from src.gpu.gpu_blocking import (
    GPUBlockingConfig,
    generate_gpu_blocking_keys,
    strip_accents,
    extract_core_domain,
)

def run_diagnosis():
    cfg = Config()
    t0 = time.time()
    print("1. Loading S1 sample (5,000 entities with random_seed=42)...")
    s1_full = pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t", keep_default_na=False, low_memory=False)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(s1_full), size=5000, replace=False)
    idx.sort()
    s1_5k = s1_full.iloc[idx].reset_index(drop=True)
    del s1_full
    
    # We take the first 500 S1 entities for deep miss diagnosis
    s1_subset = s1_5k.iloc[:500].copy()
    s1_ids = set(s1_subset["entity_id"].astype(str))
    print(f"Selected {len(s1_subset)} S1 entities for diagnosis.")
    
    print("2. Loading Ground Truth for subset...")
    gt = pd.read_csv(cfg.TRAIN_GROUND_TRUTH, sep="\t", keep_default_na=False)
    s1_col = "source1_entity_id" if "source1_entity_id" in gt.columns else "s1_id"
    gt_sub = gt[gt[s1_col].astype(str).isin(s1_ids)].copy()
    
    gt_dict = {}
    needed_s2_ids = set()
    needed_s3_ids = set()
    for _, r in gt_sub.iterrows():
        s1_id = str(r[s1_col])
        m_str = str(r.get("matched_entity_ids", "")).strip()
        m_list = [x.strip() for x in m_str.split(",") if x.strip()]
        gt_dict[s1_id] = m_list
        for mid in m_list:
            if mid.startswith("S2"):
                needed_s2_ids.add(mid)
            elif mid.startswith("S3"):
                needed_s3_ids.add(mid)
                
    total_true = sum(len(v) for v in gt_dict.values())
    print(f"Total true matches to trace: {total_true} (S2: {len(needed_s2_ids)}, S3: {len(needed_s3_ids)})")
    
    print("3. Scanning S2 and S3 for true match records...")
    s2_records = {}
    for chunk in pd.read_csv(cfg.TRAIN_SOURCE2, sep="\t", chunksize=200_000, keep_default_na=False, dtype=str):
        matched = chunk[chunk["entity_id"].astype(str).isin(needed_s2_ids)]
        for _, row in matched.iterrows():
            s2_records[str(row["entity_id"])] = row.to_dict()
        if len(s2_records) == len(needed_s2_ids):
            break
            
    s3_records = {}
    for chunk in pd.read_csv(cfg.TRAIN_SOURCE3, sep="\t", chunksize=200_000, keep_default_na=False, dtype=str):
        matched = chunk[chunk["entity_id"].astype(str).isin(needed_s3_ids)]
        for _, row in matched.iterrows():
            s3_records[str(row["entity_id"])] = row.to_dict()
        if len(s3_records) == len(needed_s3_ids):
            break
            
    print(f"Found {len(s2_records)} S2 records and {len(s3_records)} S3 records in {time.time() - t0:.1f}s.")
    
    # Preprocess S1 and true match records
    s1_proc = preprocess_records(s1_subset)
    all_ref_df = pd.DataFrame(list(s2_records.values()) + list(s3_records.values()))
    ref_proc = preprocess_records(all_ref_df)
    ref_dict = {str(r["entity_id"]): r for _, r in ref_proc.iterrows()}
    s1_proc_dict = {str(r["entity_id"]): r for _, r in s1_proc.iterrows()}
    
    b_cfg = GPUBlockingConfig(top_k_per_s1=60, max_candidates_per_block=150)
    
    # 4. Analyze blocking key intersection between S1 and its true matches
    print("\n" + "="*80)
    print("4. KEY OVERLAP ANALYSIS BETWEEN S1 AND TRUE MATCHES")
    print("="*80)
    
    zero_key_overlap_pairs = []
    has_overlap_pairs = []
    overlap_by_strategy = Counter()
    
    for s1_id, true_list in gt_dict.items():
        s1_row = s1_proc_dict[s1_id]
        s1_keys = generate_gpu_blocking_keys(s1_row, b_cfg, token_freq=None)
        
        all_s1_keys = set()
        keys_by_strat_s1 = {}
        for strat, klist in s1_keys.items():
            all_s1_keys.update(klist)
            for k in klist:
                keys_by_strat_s1[k] = strat
                
        for tid in true_list:
            if tid not in ref_dict:
                continue
            true_row = ref_dict[tid]
            true_keys = generate_gpu_blocking_keys(true_row, b_cfg, token_freq=None)
            all_true_keys = set()
            for klist in true_keys.values():
                all_true_keys.update(klist)
                
            shared = all_s1_keys & all_true_keys
            if not shared:
                zero_key_overlap_pairs.append({
                    "s1_id": s1_id,
                    "target_id": tid,
                    "s1_name": s1_row.get("business_name"),
                    "target_name": true_row.get("business_name"),
                    "s1_addr": s1_row.get("business_address"),
                    "target_addr": true_row.get("business_address"),
                    "s1_keys": list(all_s1_keys),
                    "true_keys": list(all_true_keys),
                })
            else:
                strats_hit = {keys_by_strat_s1[k] for k in shared}
                for s in strats_hit:
                    overlap_by_strategy[s] += 1
                has_overlap_pairs.append({
                    "s1_id": s1_id,
                    "target_id": tid,
                    "shared_keys": list(shared),
                    "shared_strats": list(strats_hit),
                })
                
    total_eval_pairs = len(zero_key_overlap_pairs) + len(has_overlap_pairs)
    print(f"Total Evaluated True Match Pairs: {total_eval_pairs}")
    print(f"Pairs with AT LEAST ONE shared blocking key: {len(has_overlap_pairs)} ({len(has_overlap_pairs)/total_eval_pairs*100:.2f}%)")
    print(f"Pairs with ZERO shared blocking keys (Theoretical Blocker Ceiling): {len(zero_key_overlap_pairs)} ({len(zero_key_overlap_pairs)/total_eval_pairs*100:.2f}%)")
    print(f"Overlap by strategy when shared:")
    for s, cnt in overlap_by_strategy.most_common():
        print(f"  Strategy {s}: {cnt} pairs ({cnt/total_eval_pairs*100:.2f}%)")
        
    print("\nSample of ZERO KEY OVERLAP pairs (Blocker cannot possibly retrieve these):")
    for p in zero_key_overlap_pairs[:5]:
        s1_n = str(p['s1_name']).encode('ascii', 'replace').decode('ascii')
        tr_n = str(p['target_name']).encode('ascii', 'replace').decode('ascii')
        s1_a = str(p['s1_addr']).encode('ascii', 'replace').decode('ascii')
        tr_a = str(p['target_addr']).encode('ascii', 'replace').decode('ascii')
        print(f"  S1: '{s1_n}' | Addr: '{s1_a}'")
        print(f"  TR: '{tr_n}' | Addr: '{tr_a}'")
        print(f"  S1 keys: {p['s1_keys'][:3]}")
        print(f"  TR keys: {p['true_keys'][:3]}")
        print("-" * 60)

if __name__ == "__main__":
    run_diagnosis()
