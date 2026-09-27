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

print("Loading 100k sample...")
with open("cache/benchmark_sample_s1_100000_dist_5000.pkl", "rb") as f:
    d100k = pickle.load(f)

s1 = d100k["s1"].head(3000)
s2 = d100k["s2"]
s3 = d100k["s3"]
gt = d100k["gt"]

gt_dict = {}
for _, r in gt.iterrows():
    s1_id = str(r["source1_entity_id"])
    m_str = str(r.get("matched_entity_ids", "")).strip()
    m_set = {x.strip() for x in m_str.split(",") if x.strip()}
    gt_dict[s1_id] = m_set

for cap_val in [150, 300, 500]:
    cfg = GPUBlockingConfig(max_candidates_per_block=cap_val, top_k_per_s1=60)
    idx = GPUBlockingIndex(cfg)
    idx.build_vocabulary(s1, s2, s3)
    idx.add_records(s2)
    idx.add_records(s3)

    retrieved = 0
    total_true = 0
    for _, r in s1.iterrows():
        s1_id = str(r["entity_id"])
        true_set = gt_dict.get(s1_id, set())
        total_true += len(true_set)
        cands = set(idx.query_entity(r))
        retrieved += len(true_set & cands)

    rec = retrieved / total_true * 100
    print(f"max_candidates_per_block = {cap_val:3d} (top_k=60): Recall = {rec:.2f}% ({retrieved}/{total_true})")
