"""
gpu_features.py — GPU-accelerated and accuracy-optimized pairwise feature extraction.

Architecture:
1. Full Backward Compatibility: Implements all 21 baseline Turn 5.5 features in the exact
   same canonical order and with matching value semantics.
2. 13 Turn 7 Accuracy Features:
   - Name: token count diff, token containment, IDF-weighted Jaccard, character 3-gram Jaccard,
     acronym match.
   - Address: street number conflict indicator (e.g. 102 vs 104), postal code agreement,
     character 3-gram Jaccard.
   - Domain: core domain exact match, core domain 3-gram Jaccard.
   - Country: one-sided missing country indicator.
   - Source: source_is_s3 indicator, source_s3 * domain_stem interaction.
3. High Performance Vectorization & Two-Tier Filtering:
   - Pre-allocated 2D float32 NumPy feature matrices.
   - Precomputed token IDF tables (O(1) lookups).
   - Fast Levenshtein distance with identical-string / length-difference fast paths.
   - Character 3-gram hashing for high throughput.
   - PyTorch CUDA acceleration when available; vectorized CPU fallback.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.gpu.gpu_blocking import extract_core_domain, strip_accents
from src.gpu.gpu_utils import (
    DeviceContext,
    clear_gpu_cache,
    get_device_info,
    get_gpu_memory_mb,
    get_process_rss_mb,
    is_cuda_available,
    is_cupy_available,
)
from src.preprocessing import preprocess_records

logger = logging.getLogger(__name__)

# Canonical list of 21 baseline pairwise features
BASELINE_FEATURE_NAMES: list[str] = [
    # Name features (10)
    "name_exact_match",
    "name_compact_exact_match",
    "name_no_suffix_match",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_levenshtein_sim",
    "name_compact_levenshtein_sim",
    "legal_suffix_match",
    "domain_stem_match",
    "script_match",
    # Address features (6)
    "address_exact_match",
    "address_token_jaccard",
    "address_token_overlap_count",
    "address_number_match",
    "address_levenshtein_sim",
    "address_missing_either",
    # Country features (2)
    "country_exact_match",
    "country_missing_either",
    # Meta features (3)
    "source_is_s2",
    "name_len_diff_ratio",
    "address_len_diff_ratio",
]

# 13 Turn 7 Accuracy Features
TURN7_NEW_FEATURE_NAMES: list[str] = [
    # Name enhancements (5)
    "name_token_count_diff",
    "name_token_containment",
    "name_idf_jaccard",
    "name_char_3gram_jaccard",
    "name_acronym_match",
    # Address enhancements (3)
    "address_number_conflict",
    "address_postal_match",
    "address_char_3gram_jaccard",
    # Domain enhancements (2)
    "domain_core_exact_match",
    "domain_core_jaccard",
    # Country enhancements (1)
    "country_one_sided_missing",
    # Source-aware enhancements (2)
    "source_is_s3",
    "source_s3_domain_stem_interaction",
]

TURN7_FEATURE_NAMES: list[str] = BASELINE_FEATURE_NAMES + TURN7_NEW_FEATURE_NAMES


# ---------------------------------------------------------------------------
# String Similarity & Token Helpers
# ---------------------------------------------------------------------------


def fast_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein edit distance with fast paths."""
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    if not m:
        return n
    if not n:
        return m
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m

    # Single pre-allocated DP row
    dp = list(range(n + 1))
    for i in range(m):
        c1 = s1[i]
        prev_diag = dp[0]
        dp[0] = i + 1
        for j in range(n):
            prev_dp = dp[j + 1]
            cost = prev_diag if c1 == s2[j] else min(prev_diag, dp[j], prev_dp) + 1
            prev_diag = prev_dp
            dp[j + 1] = cost
    return dp[n]


def fast_normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity in [0.0, 1.0] with early exits."""
    if s1 == s2:
        return 1.0 if s1 else 1.0
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    max_len = max(m, n)
    len_diff = abs(m - n)
    # If length difference alone makes similarity < 0.2
    if (len_diff / max_len) >= 0.8:
        return max(0.0, 1.0 - (len_diff / max_len))

    dist = fast_levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))


def char_ngrams(s: str, n: int = 3) -> set[str]:
    """Generate set of character n-grams from string."""
    if not s or len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity in [0.0, 1.0]."""
    if s1 == s2:
        return 1.0 if s1 else 1.0
    if not s1 or not s2:
        return 0.0
    g1 = char_ngrams(s1, n)
    g2 = char_ngrams(s2, n)
    if not g1 or not g2:
        return 0.0
    inter = len(g1 & g2)
    union = len(g1 | g2)
    return inter / union if union > 0 else 0.0


def token_jaccard(t1: list[str], t2: list[str]) -> float:
    """Token Jaccard similarity given pre-split token lists."""
    s1, s2 = set(t1), set(t2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def idf_weighted_jaccard(
    t1: list[str],
    t2: list[str],
    idf_table: dict[str, float],
    default_idf: float = 5.0,
) -> float:
    """IDF-weighted token Jaccard similarity."""
    s1, s2 = set(t1), set(t2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    inter = s1 & s2
    union = s1 | s2

    inter_w = sum(idf_table.get(t, default_idf) for t in inter)
    union_w = sum(idf_table.get(t, default_idf) for t in union)

    return (inter_w / union_w) if union_w > 0 else 0.0


def check_domain_stem_match(name_a: str, stem_b: str, is_domain_b: bool) -> float:
    """Check if domain stem from Candidate matches tokens in S1."""
    if not is_domain_b or not stem_b:
        return 0.0
    stem_clean = stem_b.lower()
    toks_a = name_a.lower().split()
    if stem_clean in toks_a:
        return 1.0
    name_str = "".join(toks_a)
    if stem_clean in name_str or name_str in stem_clean:
        return 0.75
    return 0.0


def check_number_overlap_and_conflict(
    addr_a: str, addr_b: str
) -> tuple[int, float, float]:
    """Analyze street numbers in addresses.

    Returns:
    - number_match: 1 if match, 0 if conflict, -1 if no numbers
    - conflict_flag: 1.0 if both have numbers and NONE match, 0.0 otherwise
    - postal_match: 1.0 if postal codes match, 0.0 if mismatch, -1.0 if missing
    """
    toks_a = [t for t in addr_a.split() if re.match(r"^\d+$", t)]
    toks_b = [t for t in addr_b.split() if re.match(r"^\d+$", t)]

    if not toks_a or not toks_b:
        return -1, 0.0, -1.0

    set_a = set(toks_a)
    set_b = set(toks_b)
    has_match = bool(set_a & set_b)

    number_match = 1 if has_match else 0
    conflict = 1.0 if not has_match else 0.0

    # Postal codes: 5-digit US or 6-digit India
    zips_a = {t for t in toks_a if len(t) in (5, 6)}
    zips_b = {t for t in toks_b if len(t) in (5, 6)}

    if zips_a and zips_b:
        postal_match = 1.0 if bool(zips_a & zips_b) else 0.0
    else:
        postal_match = -1.0

    return number_match, conflict, postal_match


# ---------------------------------------------------------------------------
# Precomputed IDF Statistics Helper
# ---------------------------------------------------------------------------


class TokenIDFStore:
    """In-memory precomputed token inverse document frequency store."""

    def __init__(self) -> None:
        self.idf_table: dict[str, float] = {}
        self.total_docs: int = 1
        self.default_idf: float = 5.0

    def fit(self, *dfs: pd.DataFrame) -> TokenIDFStore:
        """Precompute token IDF table from source DataFrames."""
        doc_freq: Counter[str] = Counter()
        doc_count = 0

        for df in dfs:
            if df is None or len(df) == 0:
                continue
            name_col = "name_norm" if "name_norm" in df.columns else "business_name"
            for val in df[name_col].dropna():
                doc_count += 1
                unique_tokens = set(str(val).split())
                for t in unique_tokens:
                    if len(t) >= 2:
                        doc_freq[t] += 1

        self.total_docs = max(doc_count, 1)
        self.idf_table = {
            t: math.log(1.0 + (self.total_docs + 1) / (count + 1))
            for t, count in doc_freq.items()
        }
        self.default_idf = math.log(1.0 + (self.total_docs + 1))
        logger.info(
            "Computed IDF store across %d records (%d unique tokens)",
            self.total_docs,
            len(self.idf_table),
        )
        return self


# ---------------------------------------------------------------------------
# Vectorized / GPU Pairwise Feature Extraction
# ---------------------------------------------------------------------------


def compute_gpu_features(
    pairs: pd.DataFrame,
    source1: Optional[pd.DataFrame] = None,
    source2: Optional[pd.DataFrame] = None,
    source3: Optional[pd.DataFrame] = None,
    idf_store: Optional[TokenIDFStore] = None,
    include_turn7_features: bool = True,
    device: str = "auto",
) -> pd.DataFrame:
    """Compute high-performance pairwise numerical feature matrix for candidate pairs.

    Parameters
    ----------
    pairs:
        DataFrame containing at minimum 's1_id' and 'candidate_id'.
        If preprocessed fields are present (e.g. from build_train_pairs),
        they are utilized directly.
    source1, source2, source3:
        Optional source DataFrames used if pairs only contains ID columns.
    idf_store:
        Optional precomputed TokenIDFStore.
    include_turn7_features:
        If True, computes all 34 features (21 baseline + 13 Turn 7).
        If False, computes only the exact 21 baseline features.
    device:
        'auto', 'cuda', or 'cpu'.

    Returns
    -------
    pd.DataFrame
        Float32 numerical feature matrix with identical index to `pairs`.
    """
    if pairs is None or len(pairs) == 0:
        feat_names = TURN7_FEATURE_NAMES if include_turn7_features else BASELINE_FEATURE_NAMES
        return pd.DataFrame(columns=feat_names)

    df = pairs

    # Enrich pairs from source tables if necessary
    if "s1_name_norm" not in df.columns:
        if source1 is None or source2 is None or source3 is None:
            raise ValueError(
                "When pairs DataFrame lacks joined fields, source1, source2, and source3 must be provided."
            )
        s1_proc = source1 if "name_norm" in source1.columns else preprocess_records(source1)
        s2_proc = source2 if "name_norm" in source2.columns else preprocess_records(source2)
        s3_proc = source3 if "name_norm" in source3.columns else preprocess_records(source3)

        s1_dict = s1_proc.set_index("entity_id").to_dict(orient="index")
        cand_dict: dict[str, dict] = {}
        for eid, row in s2_proc.set_index("entity_id").to_dict(orient="index").items():
            cand_dict[str(eid)] = row
        for eid, row in s3_proc.set_index("entity_id").to_dict(orient="index").items():
            cand_dict[str(eid)] = row

        enriched_rows: list[dict] = []
        for _, r in df.iterrows():
            s1_id = str(r["s1_id"])
            c_id = str(r["candidate_id"])
            s1_rec = s1_dict.get(s1_id, {})
            c_rec = cand_dict.get(c_id, {})
            enriched_rows.append(
                {
                    "s1_name_norm": s1_rec.get("name_norm"),
                    "s1_name_compact": s1_rec.get("name_compact"),
                    "s1_name_no_suffix": s1_rec.get("name_no_suffix"),
                    "s1_address_norm": s1_rec.get("address_norm"),
                    "s1_address_is_missing": s1_rec.get("address_is_missing", False),
                    "s1_country_norm": s1_rec.get("country_norm"),
                    "s1_script": s1_rec.get("name_script", "unknown"),
                    "s1_is_domain": s1_rec.get("is_domain", False),
                    "s1_domain_stem": s1_rec.get("domain_stem"),
                    "s1_legal_suffix": s1_rec.get("legal_suffix"),
                    "cand_name_norm": c_rec.get("name_norm"),
                    "cand_name_compact": c_rec.get("name_compact"),
                    "cand_name_no_suffix": c_rec.get("name_no_suffix"),
                    "cand_address_norm": c_rec.get("address_norm"),
                    "cand_address_is_missing": c_rec.get("address_is_missing", False),
                    "cand_country_norm": c_rec.get("country_norm"),
                    "cand_script": c_rec.get("name_script", "unknown"),
                    "cand_is_domain": c_rec.get("is_domain", False),
                    "cand_domain_stem": c_rec.get("domain_stem"),
                    "cand_legal_suffix": c_rec.get("legal_suffix"),
                    "cand_source": "S2" if c_id.startswith("S2") else "S3",
                }
            )
        df = pd.DataFrame(enriched_rows, index=pairs.index)

    n_pairs = len(df)
    feature_names = TURN7_FEATURE_NAMES if include_turn7_features else BASELINE_FEATURE_NAMES
    use_gpu = is_cuda_available() if device == "auto" else (device in ("gpu", "cuda"))

    if use_gpu:
        try:
            return _compute_gpu_features_cuda(
                df, idf_store=idf_store, include_turn7_features=include_turn7_features, feature_names=feature_names, original_index=pairs.index
            )
        except Exception as e:
            logger.warning("GPU feature extraction failed (%s), falling back to CPU implementation.", e)
            return _compute_gpu_features_cpu(
                df, idf_store=idf_store, include_turn7_features=include_turn7_features, feature_names=feature_names, original_index=pairs.index
            )

    return _compute_gpu_features_cpu(
        df, idf_store=idf_store, include_turn7_features=include_turn7_features, feature_names=feature_names, original_index=pairs.index
    )


# ---------------------------------------------------------------------------
# GPU CUDA Kernels & Execution
# ---------------------------------------------------------------------------

_CUDA_KERNELS_SRC = r'''
extern "C" {

__global__
void levenshtein_u32_kernel(const unsigned int* s1_data, const int* s1_offsets, const int* s1_lens,
                            const unsigned int* s2_data, const int* s2_offsets, const int* s2_lens,
                            float* out_sim, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    int len1 = s1_lens[idx];
    int len2 = s2_lens[idx];
    if (len1 == 0 && len2 == 0) {
        out_sim[idx] = 1.0f;
        return;
    }
    if (len1 == 0 || len2 == 0) {
        out_sim[idx] = 0.0f;
        return;
    }
    int max_len = (len1 > len2) ? len1 : len2;
    int len_diff = (len1 > len2) ? (len1 - len2) : (len2 - len1);
    if ((float)len_diff / (float)max_len >= 0.8f) {
        float s = 1.0f - (float)len_diff / (float)max_len;
        out_sim[idx] = (s > 0.0f) ? s : 0.0f;
        return;
    }

    const unsigned int* str1 = s1_data + s1_offsets[idx];
    const unsigned int* str2 = s2_data + s2_offsets[idx];

    if (len1 < len2) {
        const unsigned int* tmp_s = str1; str1 = str2; str2 = tmp_s;
        int tmp_l = len1; len1 = len2; len2 = tmp_l;
    }

    if (len1 == len2) {
        bool match = true;
        for (int i = 0; i < len1; ++i) {
            if (str1[i] != str2[i]) { match = false; break; }
        }
        if (match) {
            out_sim[idx] = 1.0f;
            return;
        }
    }

    int dp[512];
    int n2 = (len2 < 511) ? len2 : 511;
    for (int j = 0; j <= n2; ++j) dp[j] = j;

    for (int i = 0; i < len1; ++i) {
        unsigned int c1 = str1[i];
        int prev_diag = dp[0];
        dp[0] = i + 1;
        for (int j = 0; j < n2; ++j) {
            int prev_dp = dp[j + 1];
            int cost = prev_diag;
            if (c1 != str2[j]) {
                int m1 = (prev_diag < dp[j]) ? prev_diag : dp[j];
                cost = ((m1 < prev_dp) ? m1 : prev_dp) + 1;
            }
            prev_diag = prev_dp;
            dp[j + 1] = cost;
        }
    }
    int dist = dp[n2];
    float sim = 1.0f - (float)dist / (float)max_len;
    out_sim[idx] = (sim > 0.0f) ? sim : 0.0f;
}

__global__
void char3gram_u32_kernel(const unsigned int* s1_data, const int* s1_offsets, const int* s1_lens,
                          const unsigned int* s2_data, const int* s2_offsets, const int* s2_lens,
                          float* out_sim, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    int len1 = s1_lens[idx];
    int len2 = s2_lens[idx];
    if (len1 == 0 && len2 == 0) {
        out_sim[idx] = 1.0f;
        return;
    }
    if (len1 == 0 || len2 == 0) {
        out_sim[idx] = 0.0f;
        return;
    }
    if (len1 < 3 && len2 < 3) {
        const unsigned int* str1 = s1_data + s1_offsets[idx];
        const unsigned int* str2 = s2_data + s2_offsets[idx];
        if (len1 == len2) {
            bool match = true;
            for (int i = 0; i < len1; ++i) {
                if (str1[i] != str2[i]) { match = false; break; }
            }
            out_sim[idx] = match ? 1.0f : 0.0f;
        } else {
            out_sim[idx] = 0.0f;
        }
        return;
    }
    if (len1 < 3 || len2 < 3) {
        out_sim[idx] = 0.0f;
        return;
    }

    const unsigned int* str1 = s1_data + s1_offsets[idx];
    const unsigned int* str2 = s2_data + s2_offsets[idx];

    if (len1 == len2) {
        bool match = true;
        for (int i = 0; i < len1; ++i) {
            if (str1[i] != str2[i]) { match = false; break; }
        }
        if (match) {
            out_sim[idx] = 1.0f;
            return;
        }
    }

    unsigned long long g1[256];
    int n_g1 = 0;
    int limit1 = (len1 - 2 < 256) ? (len1 - 2) : 256;
    for (int i = 0; i < limit1; ++i) {
        unsigned long long h = 14695981039346656037ULL;
        h = (h ^ str1[i]) * 1099511628211ULL;
        h = (h ^ str1[i+1]) * 1099511628211ULL;
        h = (h ^ str1[i+2]) * 1099511628211ULL;

        bool exists = false;
        for (int k = 0; k < n_g1; ++k) {
            if (g1[k] == h) { exists = true; break; }
        }
        if (!exists) {
            g1[n_g1++] = h;
        }
    }

    unsigned long long g2[256];
    int n_g2 = 0;
    int limit2 = (len2 - 2 < 256) ? (len2 - 2) : 256;
    for (int i = 0; i < limit2; ++i) {
        unsigned long long h = 14695981039346656037ULL;
        h = (h ^ str2[i]) * 1099511628211ULL;
        h = (h ^ str2[i+1]) * 1099511628211ULL;
        h = (h ^ str2[i+2]) * 1099511628211ULL;

        bool exists = false;
        for (int k = 0; k < n_g2; ++k) {
            if (g2[k] == h) { exists = true; break; }
        }
        if (!exists) {
            g2[n_g2++] = h;
        }
    }

    if (n_g1 == 0 && n_g2 == 0) {
        out_sim[idx] = 1.0f;
        return;
    }
    if (n_g1 == 0 || n_g2 == 0) {
        out_sim[idx] = 0.0f;
        return;
    }

    int inter = 0;
    for (int i = 0; i < n_g1; ++i) {
        unsigned long long h = g1[i];
        for (int j = 0; j < n_g2; ++j) {
            if (h == g2[j]) {
                inter++;
                break;
            }
        }
    }
    int un = n_g1 + n_g2 - inter;
    out_sim[idx] = (un > 0) ? ((float)inter / (float)un) : 0.0f;
}

}
'''

_CUDA_MODULE = None


def _get_cuda_kernels():
    global _CUDA_MODULE
    if _CUDA_MODULE is not None:
        return _CUDA_MODULE
    if not is_cupy_available():
        return None
    import cupy as cp

    mod = cp.RawModule(code=_CUDA_KERNELS_SRC)
    lev_k = mod.get_function("levenshtein_u32_kernel")
    c3_k = mod.get_function("char3gram_u32_kernel")
    _CUDA_MODULE = (lev_k, c3_k)
    return _CUDA_MODULE


def _encode_u32_gpu(strings: list[str]):
    import cupy as cp

    lens = np.array([len(s) for s in strings], dtype=np.int32)
    offsets = np.zeros(len(strings), dtype=np.int32)
    offsets[1:] = np.cumsum(lens[:-1])
    u32_flat = (
        np.concatenate([
            np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
            if len(s) > 0
            else np.empty(0, dtype=np.uint32)
            for s in strings
        ])
        if len(strings) > 0
        else np.empty(0, dtype=np.uint32)
    )
    return cp.asarray(u32_flat), cp.asarray(offsets), cp.asarray(lens)


def _compute_gpu_features_cuda(
    df: pd.DataFrame,
    idf_store: Optional[TokenIDFStore],
    include_turn7_features: bool,
    feature_names: list[str],
    original_index: Any,
) -> pd.DataFrame:
    """Compute pairwise features utilizing GPU CUDA kernels for string matching."""
    import cupy as cp

    kernels = _get_cuda_kernels()
    if kernels is None:
        return _compute_gpu_features_cpu(df, idf_store, include_turn7_features, feature_names, original_index)
    lev_k, c3_k = kernels

    n_pairs = len(df)
    n_feats = len(feature_names)
    feat_matrix = np.zeros((n_pairs, n_feats), dtype=np.float32)

    records = df.to_dict(orient="records")
    idf_map = idf_store.idf_table if idf_store is not None else {}
    def_idf = idf_store.default_idf if idf_store is not None else 5.0

    s1_names: list[str] = []
    c_names: list[str] = []
    s1_comps: list[str] = []
    c_comps: list[str] = []
    s1_addrs: list[str] = []
    c_addrs: list[str] = []
    s1_clean_comps: list[str] = []
    c_dom_comps: list[str] = []
    addr_valid_mask = np.zeros(n_pairs, dtype=bool)

    for i, r in enumerate(records):
        s1_name = str(r.get("s1_name_norm") or "").strip()
        c_name = str(r.get("cand_name_norm") or "").strip()
        s1_comp = str(r.get("s1_name_compact") or "").strip()
        c_comp = str(r.get("cand_name_compact") or "").strip()
        s1_nosuf = str(r.get("s1_name_no_suffix") or "").strip()
        c_nosuf = str(r.get("cand_name_no_suffix") or "").strip()

        s1_addr = str(r.get("s1_address_norm") or "").strip()
        c_addr = str(r.get("cand_address_norm") or "").strip()
        s1_addr_miss = bool(r.get("s1_address_is_missing", False)) or not s1_addr
        c_addr_miss = bool(r.get("cand_address_is_missing", False)) or not c_addr

        s1_ctry = str(r.get("s1_country_norm") or "").strip().lower()
        c_ctry = str(r.get("cand_country_norm") or "").strip().lower()

        s1_suf = r.get("s1_legal_suffix")
        c_suf = r.get("cand_legal_suffix")

        s1_script = str(r.get("s1_script") or "unknown")
        c_script = str(r.get("cand_script") or "unknown")

        c_src = str(r.get("cand_source") or "")
        cid = str(r.get("candidate_id") or "")
        is_s2 = 1.0 if (c_src == "S2" or cid.startswith("S2")) else 0.0
        is_s3 = 1.0 if (c_src == "S3" or cid.startswith("S3")) else 0.0

        name_exact = 1.0 if s1_name and s1_name == c_name else 0.0
        name_comp_exact = 1.0 if s1_comp and s1_comp == c_comp else 0.0
        name_nosuf_exact = 1.0 if s1_nosuf and s1_nosuf == c_nosuf else 0.0

        s1_tokens = s1_name.split() if s1_name else []
        c_tokens = c_name.split() if c_name else []
        s1_set = set(s1_tokens)
        c_set = set(c_tokens)

        shared_tokens = s1_set & c_set
        overlap_cnt = float(len(shared_tokens))
        union_cnt = len(s1_set | c_set)
        name_jaccard_val = (overlap_cnt / union_cnt) if union_cnt > 0 else (1.0 if not s1_tokens and not c_tokens else 0.0)

        if s1_suf and c_suf:
            suf_match = 1.0 if s1_suf == c_suf else 0.0
        elif s1_suf or c_suf:
            suf_match = 0.0
        else:
            suf_match = -1.0

        is_dom = bool(r.get("cand_is_domain", False))
        dom_stem = str(r.get("cand_domain_stem") or "")
        dom_match = check_domain_stem_match(s1_name, dom_stem, is_dom)

        script_val = 1.0 if s1_script != "unknown" and s1_script == c_script else 0.0

        addr_miss_either = 1.0 if (s1_addr_miss or c_addr_miss) else 0.0
        num_conflict = 0.0
        postal_match = -1.0

        if s1_addr_miss or c_addr_miss:
            addr_exact = 0.0
            addr_jaccard_val = 0.0
            addr_overlap_cnt = 0.0
            addr_num_match = -1.0
        else:
            addr_valid_mask[i] = True
            addr_exact = 1.0 if s1_addr == c_addr else 0.0
            a_toks = s1_addr.split()
            b_toks = c_addr.split()
            a_set = set(a_toks)
            b_set = set(b_toks)
            addr_overlap_cnt = float(len(a_set & b_set))
            u_len = len(a_set | b_set)
            addr_jaccard_val = (addr_overlap_cnt / u_len) if u_len > 0 else 0.0
            addr_num_match, num_conflict, postal_match = check_number_overlap_and_conflict(s1_addr, c_addr)

        s1_has_ctry = bool(s1_ctry and s1_ctry not in ("nan", "none", "null"))
        c_has_ctry = bool(c_ctry and c_ctry not in ("nan", "none", "null"))

        if s1_has_ctry and c_has_ctry:
            ctry_match = 1.0 if s1_ctry == c_ctry else 0.0
            ctry_miss_either = 0.0
            ctry_one_sided = 0.0
        else:
            ctry_match = -1.0
            ctry_miss_either = 1.0
            ctry_one_sided = 1.0 if (s1_has_ctry or c_has_ctry) else 0.0

        max_name_len = max(len(s1_name), len(c_name), 1)
        name_len_diff = abs(len(s1_name) - len(c_name)) / max_name_len

        max_addr_len = max(len(s1_addr), len(c_addr), 1)
        addr_len_diff = abs(len(s1_addr) - len(c_addr)) / max_addr_len

        feat_matrix[i, 0] = name_exact
        feat_matrix[i, 1] = name_comp_exact
        feat_matrix[i, 2] = name_nosuf_exact
        feat_matrix[i, 3] = name_jaccard_val
        feat_matrix[i, 4] = overlap_cnt
        feat_matrix[i, 7] = suf_match
        feat_matrix[i, 8] = dom_match
        feat_matrix[i, 9] = script_val
        feat_matrix[i, 10] = addr_exact
        feat_matrix[i, 11] = addr_jaccard_val
        feat_matrix[i, 12] = addr_overlap_cnt
        feat_matrix[i, 13] = float(addr_num_match)
        feat_matrix[i, 15] = addr_miss_either
        feat_matrix[i, 16] = ctry_match
        feat_matrix[i, 17] = ctry_miss_either
        feat_matrix[i, 18] = is_s2
        feat_matrix[i, 19] = name_len_diff
        feat_matrix[i, 20] = addr_len_diff

        if include_turn7_features:
            feat_matrix[i, 21] = float(abs(len(s1_tokens) - len(c_tokens)))
            if s1_set and c_set:
                feat_matrix[i, 22] = 1.0 if (s1_set <= c_set or c_set <= s1_set) else 0.0
            else:
                feat_matrix[i, 22] = 0.0
            feat_matrix[i, 23] = idf_weighted_jaccard(s1_tokens, c_tokens, idf_map, def_idf)

            s1_acronym = "".join(t[0] for t in s1_tokens if t)
            c_acronym = "".join(t[0] for t in c_tokens if t)
            acronym_match = 0.0
            if len(s1_acronym) >= 2 and (s1_acronym == c_comp or s1_acronym == c_name):
                acronym_match = 1.0
            elif len(c_acronym) >= 2 and (c_acronym == s1_comp or c_acronym == s1_name):
                acronym_match = 1.0
            elif len(s1_acronym) >= 2 and len(c_acronym) >= 2 and s1_acronym == c_acronym:
                acronym_match = 1.0
            feat_matrix[i, 25] = acronym_match

            feat_matrix[i, 26] = num_conflict
            feat_matrix[i, 27] = postal_match

            raw_c_name = str(r.get("cand_name_norm") or "")
            dom_core_stem, dom_core_comp = extract_core_domain(raw_c_name)
            if not dom_core_comp and dom_stem:
                ds_stem, ds_comp = extract_core_domain(dom_stem)
                if ds_comp:
                    dom_core_comp = ds_comp
                else:
                    dom_core_comp = re.sub(r"[^a-z0-9]", "", strip_accents(dom_stem.lower()))

            s1_comp_clean = strip_accents(s1_comp)
            if dom_core_comp and (dom_core_comp == s1_comp_clean or dom_core_comp in s1_comp_clean or s1_comp_clean in dom_core_comp):
                dom_core_exact = 1.0
            else:
                dom_core_exact = 0.0
            feat_matrix[i, 29] = dom_core_exact

            feat_matrix[i, 31] = ctry_one_sided
            feat_matrix[i, 32] = is_s3
            feat_matrix[i, 33] = is_s3 * dom_match

            s1_clean_comps.append(s1_comp_clean if dom_core_comp else "")
            c_dom_comps.append(dom_core_comp if dom_core_comp else "")

        s1_names.append(s1_name)
        c_names.append(c_name)
        s1_comps.append(s1_comp)
        c_comps.append(c_comp)
        s1_addrs.append(s1_addr if addr_valid_mask[i] else "")
        c_addrs.append(c_addr if addr_valid_mask[i] else "")

    # Execute GPU kernels in bounded batches of 50,000 pairs to preserve VRAM
    gpu_batch_size = 50_000
    for b_start in range(0, n_pairs, gpu_batch_size):
        b_end = min(b_start + gpu_batch_size, n_pairs)
        b_n = b_end - b_start

        d_s1_n_d, d_s1_n_o, d_s1_n_l = _encode_u32_gpu(s1_names[b_start:b_end])
        d_c_n_d, d_c_n_o, d_c_n_l = _encode_u32_gpu(c_names[b_start:b_end])
        d_s1_c_d, d_s1_c_o, d_s1_c_l = _encode_u32_gpu(s1_comps[b_start:b_end])
        d_c_c_d, d_c_c_o, d_c_c_l = _encode_u32_gpu(c_comps[b_start:b_end])
        d_s1_a_d, d_s1_a_o, d_s1_a_l = _encode_u32_gpu(s1_addrs[b_start:b_end])
        d_c_a_d, d_c_a_o, d_c_a_l = _encode_u32_gpu(c_addrs[b_start:b_end])

        block_size = 256
        grid_size = (b_n + block_size - 1) // block_size

        d_name_lev = cp.zeros(b_n, dtype=np.float32)
        d_comp_lev = cp.zeros(b_n, dtype=np.float32)
        d_addr_lev = cp.zeros(b_n, dtype=np.float32)

        lev_k((grid_size,), (block_size,), (d_s1_n_d, d_s1_n_o, d_s1_n_l, d_c_n_d, d_c_n_o, d_c_n_l, d_name_lev, b_n))
        lev_k((grid_size,), (block_size,), (d_s1_c_d, d_s1_c_o, d_s1_c_l, d_c_c_d, d_c_c_o, d_c_c_l, d_comp_lev, b_n))
        lev_k((grid_size,), (block_size,), (d_s1_a_d, d_s1_a_o, d_s1_a_l, d_c_a_d, d_c_a_o, d_c_a_l, d_addr_lev, b_n))

        if include_turn7_features:
            d_s1_dc_d, d_s1_dc_o, d_s1_dc_l = _encode_u32_gpu(s1_clean_comps[b_start:b_end])
            d_c_dc_d, d_c_dc_o, d_c_dc_l = _encode_u32_gpu(c_dom_comps[b_start:b_end])

            d_comp_c3 = cp.zeros(b_n, dtype=np.float32)
            d_addr_c3 = cp.zeros(b_n, dtype=np.float32)
            d_dom_c3 = cp.zeros(b_n, dtype=np.float32)

            c3_k((grid_size,), (block_size,), (d_s1_c_d, d_s1_c_o, d_s1_c_l, d_c_c_d, d_c_c_o, d_c_c_l, d_comp_c3, b_n))
            c3_k((grid_size,), (block_size,), (d_s1_a_d, d_s1_a_o, d_s1_a_l, d_c_a_d, d_c_a_o, d_c_a_l, d_addr_c3, b_n))
            c3_k((grid_size,), (block_size,), (d_s1_dc_d, d_s1_dc_o, d_s1_dc_l, d_c_dc_d, d_c_dc_o, d_c_dc_l, d_dom_c3, b_n))
            cp.cuda.Stream.null.synchronize()

            feat_matrix[b_start:b_end, 24] = d_comp_c3.get()
            feat_matrix[b_start:b_end, 28] = np.where(addr_valid_mask[b_start:b_end], d_addr_c3.get(), 0.0)
            feat_matrix[b_start:b_end, 30] = np.where([bool(x) for x in c_dom_comps[b_start:b_end]], d_dom_c3.get(), 0.0)
        else:
            cp.cuda.Stream.null.synchronize()

        feat_matrix[b_start:b_end, 5] = d_name_lev.get()
        feat_matrix[b_start:b_end, 6] = d_comp_lev.get()
        feat_matrix[b_start:b_end, 14] = np.where(addr_valid_mask[b_start:b_end], d_addr_lev.get(), 0.0)

    return pd.DataFrame(feat_matrix, columns=feature_names, index=original_index)


def _compute_gpu_features_cpu(
    df: pd.DataFrame,
    idf_store: Optional[TokenIDFStore],
    include_turn7_features: bool,
    feature_names: list[str],
    original_index: Any,
) -> pd.DataFrame:
    """CPU fallback implementation of feature extraction (Turn 7 exact logic)."""
    n_pairs = len(df)
    n_feats = len(feature_names)
    feat_matrix = np.zeros((n_pairs, n_feats), dtype=np.float32)

    records = df.to_dict(orient="records")
    idf_map = idf_store.idf_table if idf_store is not None else {}
    def_idf = idf_store.default_idf if idf_store is not None else 5.0

    for i, r in enumerate(records):
        s1_name = str(r.get("s1_name_norm") or "").strip()
        c_name = str(r.get("cand_name_norm") or "").strip()
        s1_comp = str(r.get("s1_name_compact") or "").strip()
        c_comp = str(r.get("cand_name_compact") or "").strip()
        s1_nosuf = str(r.get("s1_name_no_suffix") or "").strip()
        c_nosuf = str(r.get("cand_name_no_suffix") or "").strip()

        s1_addr = str(r.get("s1_address_norm") or "").strip()
        c_addr = str(r.get("cand_address_norm") or "").strip()
        s1_addr_miss = bool(r.get("s1_address_is_missing", False)) or not s1_addr
        c_addr_miss = bool(r.get("cand_address_is_missing", False)) or not c_addr

        s1_ctry = str(r.get("s1_country_norm") or "").strip().lower()
        c_ctry = str(r.get("cand_country_norm") or "").strip().lower()

        s1_suf = r.get("s1_legal_suffix")
        c_suf = r.get("cand_legal_suffix")

        s1_script = str(r.get("s1_script") or "unknown")
        c_script = str(r.get("cand_script") or "unknown")

        c_src = str(r.get("cand_source") or "")
        cid = str(r.get("candidate_id") or "")
        is_s2 = 1.0 if (c_src == "S2" or cid.startswith("S2")) else 0.0
        is_s3 = 1.0 if (c_src == "S3" or cid.startswith("S3")) else 0.0

        name_exact = 1.0 if s1_name and s1_name == c_name else 0.0
        name_comp_exact = 1.0 if s1_comp and s1_comp == c_comp else 0.0
        name_nosuf_exact = 1.0 if s1_nosuf and s1_nosuf == c_nosuf else 0.0

        s1_tokens = s1_name.split() if s1_name else []
        c_tokens = c_name.split() if c_name else []
        s1_set = set(s1_tokens)
        c_set = set(c_tokens)

        shared_tokens = s1_set & c_set
        overlap_cnt = float(len(shared_tokens))
        union_cnt = len(s1_set | c_set)
        name_jaccard_val = (overlap_cnt / union_cnt) if union_cnt > 0 else (1.0 if not s1_tokens and not c_tokens else 0.0)

        name_lev = fast_normalized_levenshtein(s1_name, c_name)
        comp_lev = fast_normalized_levenshtein(s1_comp, c_comp)

        if s1_suf and c_suf:
            suf_match = 1.0 if s1_suf == c_suf else 0.0
        elif s1_suf or c_suf:
            suf_match = 0.0
        else:
            suf_match = -1.0

        is_dom = bool(r.get("cand_is_domain", False))
        dom_stem = str(r.get("cand_domain_stem") or "")
        dom_match = check_domain_stem_match(s1_name, dom_stem, is_dom)

        script_val = 1.0 if s1_script != "unknown" and s1_script == c_script else 0.0

        addr_miss_either = 1.0 if (s1_addr_miss or c_addr_miss) else 0.0
        num_conflict = 0.0
        postal_match = -1.0

        if s1_addr_miss or c_addr_miss:
            addr_exact = 0.0
            addr_jaccard_val = 0.0
            addr_overlap_cnt = 0.0
            addr_num_match = -1.0
            addr_lev = 0.0
        else:
            addr_exact = 1.0 if s1_addr == c_addr else 0.0
            a_toks = s1_addr.split()
            b_toks = c_addr.split()
            a_set = set(a_toks)
            b_set = set(b_toks)
            addr_overlap_cnt = float(len(a_set & b_set))
            u_len = len(a_set | b_set)
            addr_jaccard_val = (addr_overlap_cnt / u_len) if u_len > 0 else 0.0
            addr_num_match, num_conflict, postal_match = check_number_overlap_and_conflict(s1_addr, c_addr)
            addr_lev = fast_normalized_levenshtein(s1_addr, c_addr)

        s1_has_ctry = bool(s1_ctry and s1_ctry not in ("nan", "none", "null"))
        c_has_ctry = bool(c_ctry and c_ctry not in ("nan", "none", "null"))

        if s1_has_ctry and c_has_ctry:
            ctry_match = 1.0 if s1_ctry == c_ctry else 0.0
            ctry_miss_either = 0.0
            ctry_one_sided = 0.0
        else:
            ctry_match = -1.0
            ctry_miss_either = 1.0
            ctry_one_sided = 1.0 if (s1_has_ctry or c_has_ctry) else 0.0

        max_name_len = max(len(s1_name), len(c_name), 1)
        name_len_diff = abs(len(s1_name) - len(c_name)) / max_name_len

        max_addr_len = max(len(s1_addr), len(c_addr), 1)
        addr_len_diff = abs(len(s1_addr) - len(c_addr)) / max_addr_len

        feat_matrix[i, 0] = name_exact
        feat_matrix[i, 1] = name_comp_exact
        feat_matrix[i, 2] = name_nosuf_exact
        feat_matrix[i, 3] = name_jaccard_val
        feat_matrix[i, 4] = overlap_cnt
        feat_matrix[i, 5] = name_lev
        feat_matrix[i, 6] = comp_lev
        feat_matrix[i, 7] = suf_match
        feat_matrix[i, 8] = dom_match
        feat_matrix[i, 9] = script_val
        feat_matrix[i, 10] = addr_exact
        feat_matrix[i, 11] = addr_jaccard_val
        feat_matrix[i, 12] = addr_overlap_cnt
        feat_matrix[i, 13] = float(addr_num_match)
        feat_matrix[i, 14] = addr_lev
        feat_matrix[i, 15] = addr_miss_either
        feat_matrix[i, 16] = ctry_match
        feat_matrix[i, 17] = ctry_miss_either
        feat_matrix[i, 18] = is_s2
        feat_matrix[i, 19] = name_len_diff
        feat_matrix[i, 20] = addr_len_diff

        if include_turn7_features:
            feat_matrix[i, 21] = float(abs(len(s1_tokens) - len(c_tokens)))
            if s1_set and c_set:
                feat_matrix[i, 22] = 1.0 if (s1_set <= c_set or c_set <= s1_set) else 0.0
            else:
                feat_matrix[i, 22] = 0.0

            feat_matrix[i, 23] = idf_weighted_jaccard(s1_tokens, c_tokens, idf_map, def_idf)
            feat_matrix[i, 24] = char_ngram_jaccard(s1_comp, c_comp, n=3)

            s1_acronym = "".join(t[0] for t in s1_tokens if t)
            c_acronym = "".join(t[0] for t in c_tokens if t)
            acronym_match = 0.0
            if len(s1_acronym) >= 2 and (s1_acronym == c_comp or s1_acronym == c_name):
                acronym_match = 1.0
            elif len(c_acronym) >= 2 and (c_acronym == s1_comp or c_acronym == s1_name):
                acronym_match = 1.0
            elif len(s1_acronym) >= 2 and len(c_acronym) >= 2 and s1_acronym == c_acronym:
                acronym_match = 1.0
            feat_matrix[i, 25] = acronym_match

            feat_matrix[i, 26] = num_conflict
            feat_matrix[i, 27] = postal_match
            feat_matrix[i, 28] = 0.0 if (s1_addr_miss or c_addr_miss) else char_ngram_jaccard(s1_addr, c_addr, n=3)

            raw_c_name = str(r.get("cand_name_norm") or "")
            dom_core_stem, dom_core_comp = extract_core_domain(raw_c_name)
            if not dom_core_comp and dom_stem:
                ds_stem, ds_comp = extract_core_domain(dom_stem)
                if ds_comp:
                    dom_core_comp = ds_comp
                else:
                    dom_core_comp = re.sub(r"[^a-z0-9]", "", strip_accents(dom_stem.lower()))

            s1_comp_clean = strip_accents(s1_comp)
            if dom_core_comp and (dom_core_comp == s1_comp_clean or dom_core_comp in s1_comp_clean or s1_comp_clean in dom_core_comp):
                dom_core_exact = 1.0
            else:
                dom_core_exact = 0.0
            feat_matrix[i, 29] = dom_core_exact
            feat_matrix[i, 30] = char_ngram_jaccard(s1_comp_clean, dom_core_comp, n=3) if dom_core_comp else 0.0
            feat_matrix[i, 31] = ctry_one_sided
            feat_matrix[i, 32] = is_s3
            feat_matrix[i, 33] = is_s3 * dom_match

    return pd.DataFrame(feat_matrix, columns=feature_names, index=original_index)

