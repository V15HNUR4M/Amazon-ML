"""
features.py — Pairwise feature engineering for entity-pair matching.

Design intent
-------------
Computes informative, interpretable, deterministic numeric feature vectors
for each candidate (Source 1, Candidate S2/S3) pair.

Feature categories:
1. Name features:
   - exact_match_name: binary indicator
   - exact_match_compact: binary indicator
   - exact_match_no_suffix: binary indicator
   - name_token_jaccard: token Jaccard similarity
   - name_token_overlap_count: count of shared tokens
   - name_levenshtein_sim: normalized character edit distance
   - name_compact_levenshtein_sim: normalized edit distance on compact string
   - legal_suffix_match: 1 if match, 0 if mismatch, -1 if either missing
   - domain_stem_match: domain stem overlap indicator
   - script_match: binary indicator of script agreement

2. Address features:
   - address_exact_match: binary indicator
   - address_token_jaccard: token Jaccard (order-independent comparison)
   - address_token_overlap_count: count of shared address tokens
   - address_number_match: 1 if numbers match, 0 if mismatch, -1 if no numbers
   - address_levenshtein_sim: character edit distance
   - address_missing_either: binary indicator if either address is missing

3. Country features:
   - country_exact_match: 1 if match, 0 if mismatch, -1 if missing
   - country_missing_either: binary indicator

4. Meta / Structural features:
   - source_is_s2: 1 if S2, 0 if S3
   - name_len_diff_ratio: relative difference in character length
   - address_len_diff_ratio: relative difference in address length
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_records

logger = logging.getLogger(__name__)

# Canonical list of 21 pairwise numerical features
DEFAULT_FEATURE_NAMES: list[str] = [
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


# ---------------------------------------------------------------------------
# Fast Levenshtein Distance & String Similarity Helpers
# ---------------------------------------------------------------------------


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr[j + 1] = min(insertions, deletions, substitutions)
        prev = curr

    return prev[len(s2)]


def normalized_levenshtein_sim(s1: Optional[str], s2: Optional[str]) -> float:
    """Compute normalized Levenshtein similarity in [0.0, 1.0]."""
    str1 = str(s1 or "")
    str2 = str(s2 or "")
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    dist = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return max(0.0, 1.0 - (dist / max_len))


def token_jaccard(s1: Optional[str], s2: Optional[str]) -> float:
    """Compute token Jaccard similarity between two strings."""
    t1 = set(str(s1 or "").split())
    t2 = set(str(s2 or "").split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def token_overlap_count(s1: Optional[str], s2: Optional[str]) -> int:
    """Compute count of common tokens between two strings."""
    t1 = set(str(s1 or "").split())
    t2 = set(str(s2 or "").split())
    return len(t1 & t2)


# ---------------------------------------------------------------------------
# Individual Feature Stubs (Preserved & Implemented for Compatibility)
# ---------------------------------------------------------------------------


def name_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two name strings."""
    return token_jaccard(a, b)


def name_edit_distance_norm(a: str, b: str) -> float:
    """Normalised character edit similarity between two name strings."""
    return normalized_levenshtein_sim(a, b)


def address_token_overlap(a: str, b: str) -> float:
    """Token-level overlap between two address strings."""
    return token_jaccard(a, b)


def country_exact_match(a: str, b: str) -> int:
    """Return 1 if country strings match and non-empty, 0 if different, -1 if missing."""
    s_a = str(a or "").strip().lower()
    s_b = str(b or "").strip().lower()
    if not s_a or not s_b or s_a in ("nan", "none", "null") or s_b in ("nan", "none", "null"):
        return -1
    return 1 if s_a == s_b else 0


# ---------------------------------------------------------------------------
# Domain & Legal Suffix Helpers
# ---------------------------------------------------------------------------


def check_domain_stem_match(
    name_a: Optional[str],
    stem_b: Optional[str],
    is_domain_b: bool,
) -> float:
    """Check if domain stem from Candidate matches tokens in S1."""
    if not is_domain_b or not stem_b:
        return 0.0
    stem_clean = str(stem_b).lower()
    toks_a = str(name_a or "").lower().split()
    if stem_clean in toks_a:
        return 1.0
    # Check substring
    name_str = "".join(toks_a)
    if stem_clean in name_str or name_str in stem_clean:
        return 0.75
    return 0.0


def check_number_overlap(addr_a: Optional[str], addr_b: Optional[str]) -> int:
    """Check overlap of street / building numbers between two addresses.

    Returns:
     1 if at least one shared number
     0 if both have numbers but none match
    -1 if either address has no numbers
    """
    toks_a = [t for t in str(addr_a or "").split() if re.match(r"^\d+$", t)]
    toks_b = [t for t in str(addr_b or "").split() if re.match(r"^\d+$", t)]
    if not toks_a or not toks_b:
        return -1
    set_a = set(toks_a)
    set_b = set(toks_b)
    return 1 if (set_a & set_b) else 0


# ---------------------------------------------------------------------------
# Main Feature Computation Entry Point
# ---------------------------------------------------------------------------


def compute_features(
    pairs: pd.DataFrame,
    source1: Optional[pd.DataFrame] = None,
    source2: Optional[pd.DataFrame] = None,
    source3: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute numerical pairwise feature matrix for candidate pairs.

    Parameters
    ----------
    pairs:
        DataFrame containing at minimum 's1_id' and 'candidate_id'.
        If pairs was created by :mod:`src.pair_builder`, preprocessed columns
        are already present and used directly for high performance.
    source1, source2, source3:
        Optional source DataFrames used if pairs only contains ID columns.

    Returns
    -------
    pd.DataFrame
        Numerical feature matrix with identical index to `pairs`.
    """
    if pairs is None or len(pairs) == 0:
        return pd.DataFrame()

    df = pairs

    # If joined fields are missing from pairs, join from sources
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

        # Build joined dicts
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

    # Compute features vectorized / row-by-row
    features: dict[str, list[Any]] = {
        # Name features
        "name_exact_match": [],
        "name_compact_exact_match": [],
        "name_no_suffix_match": [],
        "name_token_jaccard": [],
        "name_token_overlap_count": [],
        "name_levenshtein_sim": [],
        "name_compact_levenshtein_sim": [],
        "legal_suffix_match": [],
        "domain_stem_match": [],
        "script_match": [],
        # Address features
        "address_exact_match": [],
        "address_token_jaccard": [],
        "address_token_overlap_count": [],
        "address_number_match": [],
        "address_levenshtein_sim": [],
        "address_missing_either": [],
        # Country features
        "country_exact_match": [],
        "country_missing_either": [],
        # Meta features
        "source_is_s2": [],
        "name_len_diff_ratio": [],
        "address_len_diff_ratio": [],
    }

    records = df.to_dict(orient="records")

    for r in records:
        s1_name = str(r.get("s1_name_norm") or "")
        c_name = str(r.get("cand_name_norm") or "")
        s1_comp = str(r.get("s1_name_compact") or "")
        c_comp = str(r.get("cand_name_compact") or "")
        s1_nosuf = str(r.get("s1_name_no_suffix") or "")
        c_nosuf = str(r.get("cand_name_no_suffix") or "")

        s1_addr = str(r.get("s1_address_norm") or "")
        c_addr = str(r.get("cand_address_norm") or "")
        s1_addr_miss = bool(r.get("s1_address_is_missing", False)) or not s1_addr
        c_addr_miss = bool(r.get("cand_address_is_missing", False)) or not c_addr

        s1_ctry = str(r.get("s1_country_norm") or "")
        c_ctry = str(r.get("cand_country_norm") or "")

        s1_suf = r.get("s1_legal_suffix")
        c_suf = r.get("cand_legal_suffix")

        s1_script = str(r.get("s1_script") or "unknown")
        c_script = str(r.get("cand_script") or "unknown")

        c_src = str(r.get("cand_source") or "")

        # --- Name Features ---
        features["name_exact_match"].append(1 if s1_name and s1_name == c_name else 0)
        features["name_compact_exact_match"].append(1 if s1_comp and s1_comp == c_comp else 0)
        features["name_no_suffix_match"].append(
            1 if s1_nosuf and s1_nosuf == c_nosuf else 0
        )
        features["name_token_jaccard"].append(token_jaccard(s1_name, c_name))
        features["name_token_overlap_count"].append(token_overlap_count(s1_name, c_name))
        features["name_levenshtein_sim"].append(normalized_levenshtein_sim(s1_name, c_name))
        features["name_compact_levenshtein_sim"].append(
            normalized_levenshtein_sim(s1_comp, c_comp)
        )

        # Legal suffix match
        if s1_suf and c_suf:
            features["legal_suffix_match"].append(1 if s1_suf == c_suf else 0)
        elif s1_suf or c_suf:
            features["legal_suffix_match"].append(0)
        else:
            features["legal_suffix_match"].append(-1)

        # Domain stem match
        is_dom = bool(r.get("cand_is_domain", False))
        dom_stem = r.get("cand_domain_stem")
        features["domain_stem_match"].append(check_domain_stem_match(s1_name, dom_stem, is_dom))

        # Script match
        features["script_match"].append(
            1 if s1_script != "unknown" and s1_script == c_script else 0
        )

        # --- Address Features ---
        features["address_missing_either"].append(1 if (s1_addr_miss or c_addr_miss) else 0)
        if s1_addr_miss or c_addr_miss:
            features["address_exact_match"].append(0)
            features["address_token_jaccard"].append(0.0)
            features["address_token_overlap_count"].append(0)
            features["address_number_match"].append(-1)
            features["address_levenshtein_sim"].append(0.0)
        else:
            features["address_exact_match"].append(1 if s1_addr == c_addr else 0)
            features["address_token_jaccard"].append(token_jaccard(s1_addr, c_addr))
            features["address_token_overlap_count"].append(token_overlap_count(s1_addr, c_addr))
            features["address_number_match"].append(check_number_overlap(s1_addr, c_addr))
            features["address_levenshtein_sim"].append(normalized_levenshtein_sim(s1_addr, c_addr))

        # --- Country Features ---
        if s1_ctry and c_ctry and s1_ctry != "nan" and c_ctry != "nan":
            features["country_exact_match"].append(1 if s1_ctry.lower() == c_ctry.lower() else 0)
            features["country_missing_either"].append(0)
        else:
            features["country_exact_match"].append(-1)
            features["country_missing_either"].append(1)

        # --- Meta Features ---
        features["source_is_s2"].append(1 if c_src == "S2" or "S2-" in str(r.get("candidate_id", "")) else 0)

        # Length differences
        max_name_len = max(len(s1_name), len(c_name), 1)
        features["name_len_diff_ratio"].append(abs(len(s1_name) - len(c_name)) / max_name_len)

        max_addr_len = max(len(s1_addr), len(c_addr), 1)
        features["address_len_diff_ratio"].append(
            abs(len(s1_addr) - len(c_addr)) / max_addr_len
        )

    feat_df = pd.DataFrame(features, index=pairs.index)
    return feat_df
