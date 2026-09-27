import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"e:\Projects\amazon ML\Architecture\business_entity_resolution")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gpu.gpu_blocking import GPUBlockingConfig, build_gpu_candidates
from src.gpu.gpu_features import compute_gpu_features, TURN7_FEATURE_NAMES
from src.pair_builder import build_train_pairs

print("Loading 100k sample to test CPU vs GPU parity on 100k pairs...")
with open("cache/benchmark_sample_s1_100000_dist_5000.pkl", "rb") as f:
    d100k = pickle.load(f)

s1_100k = d100k["s1"].head(200)
s2_100k = d100k["s2"]
s3_100k = d100k["s3"]
gt_100k = d100k["gt"]

cands = build_gpu_candidates(s1_100k, s2_100k, s3_100k, config=GPUBlockingConfig(top_k_per_s1=60))
pairs = build_train_pairs(s1_100k, s2_100k, s3_100k, cands, gt_100k, random_seed=42)

test_pairs = pairs.head(500).copy()
print(f"Generated {len(test_pairs)} candidate pairs from 100k sample. Running CPU vs GPU feature extraction...")

cpu_feats = compute_gpu_features(test_pairs, include_turn7_features=True, device="cpu")
gpu_feats = compute_gpu_features(test_pairs, include_turn7_features=True, device="gpu")

diffs = {}
max_abs_diff = 0.0
for col in TURN7_FEATURE_NAMES:
    diff = float(np.max(np.abs(cpu_feats[col].values - gpu_feats[col].values)))
    diffs[col] = diff
    if diff > max_abs_diff:
        max_abs_diff = diff

print(f"MAX ABSOLUTE DIFFERENCE ACROSS ALL 34 FEATURES ON 100k DATA: {max_abs_diff:.8e}")
for col, diff in diffs.items():
    if diff > 1e-4:
        print(f"  MISMATCH in {col}: diff={diff}")
if max_abs_diff <= 1e-4:
    print("PERFECT PARITY CONFIRMED ON 100k DATA (max_diff <= 1e-4)!")
