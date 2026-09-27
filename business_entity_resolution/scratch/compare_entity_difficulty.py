import pickle
import numpy as np
import pandas as pd

with open("cache/benchmark_sample_s1_5000_dist_5000.pkl", "rb") as f:
    d5k = pickle.load(f)

with open("cache/benchmark_sample_s1_100000_dist_5000.pkl", "rb") as f:
    d100k = pickle.load(f)

s1_5k, gt_5k = d5k["s1"], d5k["gt"]
s1_100k, gt_100k = d100k["s1"], d100k["gt"]

def get_stats(df, gt):
    # Address missing
    addr_miss = df["address_is_missing"].mean() * 100 if "address_is_missing" in df.columns else (df["business_address"].fillna("").str.strip() == "").mean() * 100
    # Single token names
    name_tokens = df["name_norm"].str.split().str.len() if "name_norm" in df.columns else df["business_name"].str.split().str.len()
    single_token = (name_tokens == 1).mean() * 100
    # Short names (len <= 5)
    short_names = (df["business_name"].astype(str).str.len() <= 5).mean() * 100
    # Domain names
    is_dom = df["is_domain"].mean() * 100 if "is_domain" in df.columns else 0.0
    # Matches count per S1
    match_counts = []
    for _, r in gt.iterrows():
        m = str(r.get("matched_entity_ids", "")).strip()
        cnt = len([x for x in m.split(",") if x.strip()])
        match_counts.append(cnt)
    match_counts = np.array(match_counts)

    return {
        "n": len(df),
        "addr_missing_pct": addr_miss,
        "single_token_name_pct": single_token,
        "short_name_pct": short_names,
        "domain_name_pct": is_dom,
        "mean_matches": match_counts.mean(),
        "zero_matches_pct": (match_counts == 0).mean() * 100,
        "gt_1_matches_pct": (match_counts > 1).mean() * 100,
        "gt_3_matches_pct": (match_counts > 3).mean() * 100,
        "max_matches": match_counts.max(),
    }

st5 = get_stats(s1_5k, gt_5k)
st100 = get_stats(s1_100k, gt_100k)

print(f"{'Feature/Stat':<28} | {'5k Dataset':<15} | {'100k Dataset':<15}")
print("-" * 65)
for k in st5:
    print(f"{k:<28} | {st5[k]:<15.2f} | {st100[k]:<15.2f}")
