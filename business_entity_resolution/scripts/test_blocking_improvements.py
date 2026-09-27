import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pickle
import re
import time
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

from src.evaluation import ground_truth_to_dict
from src.blocking import BlockingConfig, NAME_STOPWORDS, ADDRESS_STOPWORDS

# Extended TLD pattern matching international, multi-part, and modern TLDs
TLD_EXTENDED_PATTERN = re.compile(
    r"\.(?:co\.uk|co\.in|org\.uk|gov\.in|net\.in|ac\.in|com\.au|co\.nz|co\.za|"
    r"com|org|net|in|io|co|biz|info|us|gov|edu|online|store|shop|tech|ai|de|uk|eu|"
    r"ca|fr|ch|nl|se|no|es|it|ru|jp|cn|br|mx|app|club|global|ltd|xyz|site|website|c0m)\b",
    re.IGNORECASE
)

DOMAIN_CLEAN_REGEX = re.compile(r"^(?:https?://)?(?:www\d*\.)?", re.IGNORECASE)

def extract_core_domain(raw_name: str) -> tuple[str, str]:
    """Extract clean domain stem and compact core from a domain string.
    
    Returns (domain_stem, domain_compact).
    """
    if not raw_name:
        return "", ""
    s = raw_name.strip().lower()
    # Strip URL protocol and www
    s = DOMAIN_CLEAN_REGEX.sub("", s)
    # Strip path, port, query params
    s = re.split(r"[:/?#\s]", s)[0]
    # Strip pipe if any
    s = s.split("|")[-1].strip()
    # Strip TLD
    s = TLD_EXTENDED_PATTERN.sub("", s)
    # Strip trailing dots or dashes
    s = s.strip(".-_ ")
    # Core compact: remove all punctuation
    compact = re.sub(r"[^a-z0-9]", "", s)
    # Replaced hyphens/dots with space
    stem = re.sub(r"[-_.]+", " ", s).strip()
    return stem, compact

def generate_enhanced_keys(row: dict, cfg: BlockingConfig, token_freq: Counter = None) -> dict[str, list[str]]:
    keys = {"A": [], "B": [], "C": [], "D": [], "E": []}
    
    # 1. Names
    name_norm = str(row.get("name_norm") or "").strip()
    name_compact = str(row.get("name_compact") or "").strip()
    name_no_suffix = str(row.get("name_no_suffix") or "").strip()
    domain_stem = str(row.get("domain_stem") or "").strip()
    raw_name = str(row.get("business_name") or "").strip()
    is_domain = bool(row.get("is_domain", False))
    
    # Check domain core
    core_stem, core_compact = extract_core_domain(raw_name)
    if not core_stem and domain_stem:
        core_stem, core_compact = extract_core_domain(domain_stem)
        
    # Blocker A: Exact & compact
    if name_norm and len(name_norm) >= cfg.min_compact_len:
        keys["A"].append(f"a_exact:{name_norm}")
    if name_compact and len(name_compact) >= cfg.min_compact_len:
        keys["A"].append(f"a_compact:{name_compact}")
    if name_no_suffix:
        nosuf_norm = name_no_suffix.strip()
        if len(nosuf_norm) >= cfg.min_compact_len:
            keys["A"].append(f"a_exact:{nosuf_norm}")
            nosuf_comp = "".join(nosuf_norm.split())
            keys["A"].append(f"a_compact:{nosuf_comp}")
            
    # Domain core in Blocker A
    if core_compact and len(core_compact) >= 3:
        keys["A"].append(f"a_compact:{core_compact}")
        keys["A"].append(f"a_domain:{core_compact}")
    if core_stem and len(core_stem) >= 3 and core_stem != core_compact:
        keys["A"].append(f"a_exact:{core_stem}")
        
    # Blocker B: Tokens, stopword bigrams, and rare tokens
    tokens = name_norm.split() if name_norm else []
    # All tokens (even stopwords) for leading bigrams
    if len(tokens) >= 2:
        keys["B"].append(f"b_lead_bi:{tokens[0]}_{tokens[1]}")
        # Compact 2-token (recovers "@Firstseven" vs "First Seven Exports")
        if len(tokens[0]) + len(tokens[1]) >= 5:
            keys["A"].append(f"a_compact:{tokens[0]}{tokens[1]}")
            
    clean_tokens = [t for t in tokens if len(t) >= cfg.min_token_len and t not in cfg.name_stopwords]
    for t in clean_tokens[:4]: # extended to top 4 clean tokens
        keys["B"].append(f"b_tok:{t}")
    if len(clean_tokens) >= 2:
        keys["B"].append(f"b_bi:{clean_tokens[0]}_{clean_tokens[1]}")
    if len(clean_tokens) >= 3:
        keys["B"].append(f"b_bi:{clean_tokens[0]}_{clean_tokens[2]}")
        
    # Rare token blocking: if token frequency is low, index it with high priority
    if token_freq is not None:
        for t in clean_tokens:
            if 1 < token_freq.get(t, 0) <= 20: # Rare across candidate corpus
                keys["B"].append(f"b_rare:{t}")
                
    # Blocker C & D: Address & Composite
    addr_norm = str(row.get("address_norm") or "").strip()
    addr_is_missing = bool(row.get("address_is_missing", False)) or not addr_norm
    
    if not addr_is_missing:
        addr_tokens = addr_norm.split()
        nums = [t for t in addr_tokens if re.match(r"^\d+$", t)]
        # Postal code detection (5-digit US or 6-digit India)
        zip_codes = [t for t in nums if len(t) in (5, 6)]
        
        info_words = [
            t for t in addr_tokens
            if len(t) >= 4 and not re.match(r"^\d+$", t) and t not in cfg.address_stopwords
        ]
        
        # Blocker C: Number + Word (all numbers up to 3, all words up to 3)
        for num in nums[:3]:
            for w in info_words[:3]:
                keys["C"].append(f"c_num_w:{num}_{w}")
                
        # Zip code blocking with name token or distinctive address word
        for z in zip_codes[:2]:
            if clean_tokens:
                keys["D"].append(f"d_zip_name:{z}_{clean_tokens[0]}")
            for w in info_words[:2]:
                keys["C"].append(f"c_zip_w:{z}_{w}")
                
        # Blocker D: Name + Number
        if nums:
            num = nums[0]
            if clean_tokens:
                keys["D"].append(f"d_name_num:{clean_tokens[0]}_{num}")
            if name_compact and len(name_compact) >= 5:
                keys["D"].append(f"d_pfx_num:{name_compact[:5]}_{num}")
                
        # Address-only bigram if no numbers present (Miss 13 case)
        if not nums and len(info_words) >= 2:
            keys["C"].append(f"c_addr_bi:{info_words[0]}_{info_words[1]}")
            if len(info_words) >= 3:
                keys["C"].append(f"c_addr_bi:{info_words[0]}_{info_words[2]}")

    return keys

def run_experiment():
    print("Loading 5k validation benchmark sample...")
    with open('cache/benchmark_sample_s1_5000_dist_5000.pkl', 'rb') as f:
        data = pickle.load(f)
    s1, s2, s3, gt = data['s1'], data['s2'], data['s3'], data['gt']
    gt_dict = ground_truth_to_dict(gt)
    total_true = sum(len(ts) for ts in gt_dict.values())
    
    cfg = BlockingConfig()
    
    # Precompute token frequency across all candidates + S1
    print("Computing candidate token frequency for rare token blocker...")
    token_counter = Counter()
    for df in (s1, s2, s3):
        for name in df["name_norm"]:
            if name:
                for t in str(name).split():
                    if len(t) >= 3 and t not in NAME_STOPWORDS:
                        token_counter[t] += 1
                        
    print(f"Vocabulary size: {len(token_counter)} distinctive tokens")
    
    # Build Inverted Index
    print("Building enhanced candidate inverted index...")
    t0 = time.time()
    index = defaultdict(list)
    cap = 150 # slightly higher cap to prevent premature dropping of valuable keys
    
    cands_df = pd.concat([s2, s3], ignore_index=True)
    cands_records = cands_df.to_dict('records')
    
    for r in cands_records:
        cid = str(r['entity_id'])
        keys_dict = generate_enhanced_keys(r, cfg, token_counter)
        for _, klist in keys_dict.items():
            for k in klist:
                plist = index[k]
                if len(plist) < cap:
                    plist.append(cid)
                    
    index_time = time.time() - t0
    print(f"Indexed {len(cands_records)} candidate records with {len(index)} distinct keys in {index_time:.2f}s")
    
    # Query S1 entities
    print("Querying candidate index for 5,000 S1 queries...")
    t0 = time.time()
    s1_records = s1.to_dict('records')
    
    weights = {"A": 4, "D": 3, "B": 2, "C": 2, "E": 1}
    top_k = 60 # evaluated at top_k = 60
    
    retrieved_true = 0
    retrieved_s2 = 0
    retrieved_s3 = 0
    total_pairs = 0
    
    s2_true_total = sum(1 for ts in gt_dict.values() for t in ts if t.startswith("S2"))
    s3_true_total = sum(1 for ts in gt_dict.values() for t in ts if t.startswith("S3"))
    
    for r in s1_records:
        s1_id = str(r['entity_id'])
        true_set = gt_dict.get(s1_id, set())
        
        keys_dict = generate_enhanced_keys(r, cfg, token_counter)
        scores = defaultdict(int)
        
        for tag, klist in keys_dict.items():
            w = weights.get(tag, 1)
            for k in klist:
                plist = index.get(k)
                if not plist or len(plist) >= cap:
                    continue
                # Weight inversely by posting list length (IDF-like key weighting)
                key_w = w * (1.0 if len(plist) <= 10 else (0.8 if len(plist) <= 50 else 0.5))
                for cid in plist:
                    scores[cid] += key_w
                    
        if not scores:
            continue
            
        ranked = [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        ranked_set = set(ranked)
        total_pairs += len(ranked)
        
        for t in true_set:
            if t in ranked_set:
                retrieved_true += 1
                if t.startswith("S2"):
                    retrieved_s2 += 1
                elif t.startswith("S3"):
                    retrieved_s3 += 1
                    
    query_time = time.time() - t0
    recall = retrieved_true / total_true * 100
    s2_recall = retrieved_s2 / s2_true_total * 100
    s3_recall = retrieved_s3 / s3_true_total * 100
    total_possible = len(s1) * len(cands_df)
    rr = (1.0 - (total_pairs / total_possible)) * 100
    
    print("\n" + "=" * 60)
    print("ENHANCED BLOCKING RESULTS (5k S1 Entities):")
    print("=" * 60)
    print(f"Overall Candidate Recall: {recall:.2f}% (Baseline Turn 5.5: 93.40%)")
    print(f"  - S2 Recall:           {s2_recall:.2f}% (Baseline Turn 5.5: 92.76%)")
    print(f"  - S3 Recall:           {s3_recall:.2f}% (Baseline Turn 5.5: 93.98%)")
    print(f"Candidate Pairs Total:   {total_pairs:,} (Baseline Turn 5.5: 166,296)")
    print(f"Reduction Ratio:         {rr:.4f}% (Baseline Turn 5.5: 99.8505%)")
    print(f"Total True Retrieved:    {retrieved_true}/{total_true} (Misses: {total_true - retrieved_true} vs baseline 1,139)")
    print(f"Runtime:                 Index {index_time:.2f}s | Query {query_time:.2f}s")
    print("=" * 60)

if __name__ == '__main__':
    run_experiment()
