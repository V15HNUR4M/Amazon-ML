import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pickle
import pandas as pd
from collections import Counter
from src.evaluation import ground_truth_to_dict
from src.blocking import generate_record_blocking_keys, BlockingConfig

def inspect():
    with open('cache/benchmark_sample_s1_5000_dist_5000.pkl', 'rb') as f:
        data = pickle.load(f)
    s1, s2, s3, gt = data['s1'], data['s2'], data['s3'], data['gt']
    gt_dict = ground_truth_to_dict(gt)
    cands = pd.read_pickle('cache/scale_check_candidates_s1_5000.pkl')
    cand_by_s1 = cands.groupby('s1_id')['candidate_id'].apply(set).to_dict()

    cand_rec = {}
    for r in s2.to_dict('records'):
        cand_rec[str(r['entity_id'])] = r
    for r in s3.to_dict('records'):
        cand_rec[str(r['entity_id'])] = r
    s1_rec = {str(r['entity_id']): r for r in s1.to_dict('records')}

    missed = []
    for s, ts in gt_dict.items():
        found = cand_by_s1.get(s, set())
        for t in ts:
            if t not in found:
                missed.append((s1_rec.get(s), cand_rec.get(t)))

    print(f"Total missed true matches: {len(missed)}")
    
    cfg = BlockingConfig()
    miss_reasons = Counter()
    
    for i, (m1, m2) in enumerate(missed):
        if not m1 or not m2:
            continue
        k1 = generate_record_blocking_keys(m1, cfg)
        k2 = generate_record_blocking_keys(m2, cfg)
        
        all_k1 = set(k for kl in k1.values() for k in kl)
        all_k2 = set(k for kl in k2.values() for k in kl)
        common_keys = all_k1 & all_k2
        
        if len(common_keys) == 0:
            miss_reasons["no_shared_blocking_keys"] += 1
        else:
            miss_reasons["key_shared_but_dropped_or_capped"] += 1

        if i < 20:
            s1_name = m1.get('business_name', '')
            c_name = m2.get('business_name', '')
            print(f"--- Miss {i+1} ---")
            print(f"  S1: '{s1_name}' | norm: '{m1.get('name_norm')}' | comp: '{m1.get('name_compact')}' | addr: '{m1.get('address_norm')}' | ctry: '{m1.get('country_norm')}'")
            print(f"  C:  '{c_name}' | norm: '{m2.get('name_norm')}' | comp: '{m2.get('name_compact')}' | dom: '{m2.get('domain_stem')}' | addr: '{m2.get('address_norm')}' | ctry: '{m2.get('country_norm')}'")
            print(f"  Common keys: {common_keys}")
            
    print(f"\nMiss reasons breakdown: {dict(miss_reasons)}")

if __name__ == '__main__':
    inspect()
