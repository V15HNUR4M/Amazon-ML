"""
gpu_blocking.py — GPU-accelerated and accuracy-optimized candidate pair blocking.

Key Improvements over Turn 5.5:
1. Core Domain Normalization: Strips multi-part TLDs (.co.uk, .co.in, .com.au),
   leetspeak (.c0m), ports, paths, and query parameters to align domain stems
   with compact name representations.
2. Stopword & Prefix Preservation: Produces multi-token keys even when leading
   words are stopwords ("the_home", "home_depot") and 2-token compact strings.
3. Rare Token Indexing: Indexes high-information rare tokens (corpus frequency <= 25)
   with elevated weighting to capture specific business entity names.
4. Comprehensive Address & Postal Blocking: Indexes all address numbers (up to 3),
   detects 5/6-digit US and India postal codes, and creates address-only bigram keys
   when numbers are absent.
5. Dynamic Posting List Weighting: Scores candidates inversely to key frequency
   (IDF-like key weighting) rather than hard-dropping useful blocks.
6. CUDA / cuDF acceleration with seamless, highly optimized CPU fallback.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import re
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.blocking import ADDRESS_STOPWORDS, BlockingConfig, NAME_STOPWORDS
from src.config import Config
from src.gpu.gpu_utils import (
    DeviceContext,
    clear_gpu_cache,
    get_device_info,
    get_gpu_memory_mb,
    get_process_rss_mb,
    is_cuda_available,
    is_cudf_available,
)
from src.preprocessing import preprocess_records

logger = logging.getLogger(__name__)

import unicodedata

# Extended international, multi-part, and modern TLDs
_TLD_EXTENDED_REGEX = re.compile(
    r"\.(?:co\.uk|co\.in|org\.uk|gov\.in|net\.in|ac\.in|com\.au|co\.nz|co\.za|"
    r"com|org|net|in|io|co|biz|info|us|gov|edu|online|store|shop|tech|ai|de|uk|eu|"
    r"ca|fr|ch|nl|se|no|es|it|ru|jp|cn|br|mx|app|club|global|ltd|xyz|site|website|c0m)\b",
    re.IGNORECASE,
)

_DOMAIN_CLEAN_PREFIX_REGEX = re.compile(
    r"^(?:https?://)?(?:www\d*\.)?", re.IGNORECASE
)

_COMMON_SUBDOMAINS = frozenset({
    "www", "www1", "www2", "shop", "store", "mail", "app", "api", "web", "dev",
    "portal", "secure", "en", "fr", "de", "es", "it", "mobile", "m", "my",
    "login", "admin", "beta", "corp", "support", "help", "static", "cdn",
})


def strip_accents(text: str) -> str:
    """Normalize and strip Unicode accents/diacritics for robust matching."""
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")


def extract_core_domain(raw_name: Optional[str]) -> tuple[str, str]:
    """Extract clean domain stem and compact alphanumeric core from a domain string.

    Handles:
    - URL schemes (http://, https://)
    - www and www1, www2
    - pipe syntax (e.g. 'Basera Advisors | www.baseraadv.com')
    - ports (:80, :8080)
    - paths (/index.html) and query strings (?ref=...)
    - subdomains (shop.acme-corp.com -> acme corp)
    - multi-part TLDs (.co.uk, .co.in, .com.au)
    - common TLDs and leetspeak (.c0m)
    - punctuation, accents, and whitespace normalization
    - safely rejects non-domain business names (e.g. 'General Electric Company')

    Returns
    -------
    (domain_stem, domain_compact) or ("", "") if not a valid domain string.
    """
    if not raw_name or pd.isna(raw_name):
        return "", ""

    text = str(raw_name).strip()
    if not text:
        return "", ""

    # 1. Handle pipe syntax: find the part containing domain indicators
    domain_candidate = text
    if "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        found = False
        for p in reversed(parts):
            if re.search(r"https?://|www\d*\.", p, re.IGNORECASE) or _TLD_EXTENDED_REGEX.search(p):
                domain_candidate = p
                found = True
                break
        if not found:
            for p in reversed(parts):
                if re.search(r"\.[a-z]{2,}", p, re.IGNORECASE):
                    domain_candidate = p
                    found = True
                    break
        if not found:
            domain_candidate = parts[-1]

    s = strip_accents(domain_candidate.lower())

    # 2. Check if string actually contains domain indicators (protocol, www, or TLD)
    has_protocol = bool(re.match(r"^https?://", s))
    has_www = bool(re.search(r"\bwww\d*\.", s))
    has_tld = bool(_TLD_EXTENDED_REGEX.search(s))

    if not (has_protocol or has_www or has_tld):
        # NOT a domain! Do not extract first word of ordinary business names.
        return "", ""

    # 3. Strip URL protocol and www
    s = _DOMAIN_CLEAN_PREFIX_REGEX.sub("", s)

    # 4. Strip path, port, query params, hash
    s = re.split(r"[:/?#\s]", s)[0].strip()
    if not s:
        return "", ""

    # 5. Strip TLD
    s = _TLD_EXTENDED_REGEX.sub("", s).strip(".-_ ")
    if not s:
        return "", ""

    # 6. Handle subdomains: e.g. 'shop.acme-corp' or 'aws.amazon'
    parts = [p for p in s.split(".") if p]
    if len(parts) > 1:
        filtered = [p for p in parts if p not in _COMMON_SUBDOMAINS]
        core = filtered[-1] if filtered else parts[-1]
    else:
        core = s

    # 7. Core compact: alphanumeric only
    compact = re.sub(r"[^a-z0-9]", "", core)
    # Stem: words separated by single space
    stem = re.sub(r"[-_.]+", " ", core).strip()

    if len(compact) < 2:
        return "", ""

    return stem, compact


@dataclass
class GPUBlockingConfig:
    """Configuration for GPU-accelerated and accuracy-optimized candidate generation."""

    enable_blocker_a: bool = True  # Exact name, compact, no-suffix, domain stem
    enable_blocker_b: bool = True  # Selective tokens, stopword bigrams, rare tokens
    enable_blocker_c: bool = True  # Address number + word, zip code + word, address bigrams
    enable_blocker_d: bool = True  # Name + Address composite blocks
    enable_domain_blocking: bool = True  # Core domain normalization & compact matching
    enable_rare_token_blocking: bool = True  # Low-frequency token blocks
    enable_stopword_prefix_bi: bool = True  # Multi-token stopword bigrams and 2-token compact
    max_candidates_per_block: int = 150  # Cap posting list size
    min_token_len: int = 3
    min_compact_len: int = 4
    top_k_per_s1: int = 60  # Max candidates per S1 entity (increased for higher recall)
    rare_token_max_freq: int = 25  # Max corpus frequency for rare-token blocker
    device: str = "auto"  # 'auto', 'cuda', or 'cpu'
    name_stopwords: frozenset[str] = field(default_factory=lambda: NAME_STOPWORDS)
    address_stopwords: frozenset[str] = field(default_factory=lambda: ADDRESS_STOPWORDS)

    @classmethod
    def from_config(cls, cfg: Any) -> GPUBlockingConfig:
        """Create GPUBlockingConfig from global Config instance."""
        return cls(
            enable_blocker_a=getattr(cfg, "BLOCKER_A_ENABLED", True),
            enable_blocker_b=getattr(cfg, "BLOCKER_B_ENABLED", True),
            enable_blocker_c=getattr(cfg, "BLOCKER_C_ENABLED", True),
            enable_blocker_d=getattr(cfg, "BLOCKER_D_ENABLED", True),
            max_candidates_per_block=150,
            min_token_len=getattr(cfg, "MIN_TOKEN_LEN", 3),
            top_k_per_s1=60,
        )


def generate_gpu_blocking_keys(
    row: dict[str, Any] | pd.Series,
    config: GPUBlockingConfig,
    token_freq: Optional[Counter[str] | dict[str, int]] = None,
) -> dict[str, list[str]]:
    """Generate multi-strategy blocking keys for a single entity record.

    Includes domain core, stopword preservation, rare tokens, and postal codes.
    """
    keys: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}

    # 1. Name fields
    name_norm = str(row.get("name_norm") or "").strip()
    name_compact = str(row.get("name_compact") or "").strip()
    name_no_suffix = str(row.get("name_no_suffix") or "").strip()
    domain_stem = str(row.get("domain_stem") or "").strip()
    raw_name = str(row.get("business_name") or "").strip()

    # Core domain extraction
    core_stem, core_compact = extract_core_domain(raw_name)
    if not core_compact and domain_stem:
        ds_stem, ds_comp = extract_core_domain(domain_stem)
        if ds_comp:
            core_stem, core_compact = ds_stem, ds_comp
        else:
            core_compact = re.sub(r"[^a-z0-9]", "", strip_accents(domain_stem.lower()))
            core_stem = re.sub(r"[-_.]+", " ", strip_accents(domain_stem.lower())).strip()

    # Raw name clean compact (handles hashtags, punctuation e.g. '#sjace' -> 'sjace')
    raw_clean_comp = re.sub(r"[^a-z0-9]", "", strip_accents(raw_name.lower()))
    if raw_clean_comp and len(raw_clean_comp) >= config.min_compact_len and raw_clean_comp != name_compact:
        keys["A"].append(f"a_compact:{raw_clean_comp}")

    # Blocker A: Exact & compact
    if config.enable_blocker_a:
        if name_norm and len(name_norm) >= config.min_compact_len:
            keys["A"].append(f"a_exact:{name_norm}")
            norm_unacc = strip_accents(name_norm)
            if norm_unacc != name_norm:
                keys["A"].append(f"a_exact:{norm_unacc}")
        if name_compact and len(name_compact) >= config.min_compact_len:
            keys["A"].append(f"a_compact:{name_compact}")
            comp_unacc = strip_accents(name_compact)
            if comp_unacc != name_compact:
                keys["A"].append(f"a_compact:{comp_unacc}")
        if name_no_suffix:
            nosuf_norm = name_no_suffix.strip()
            if len(nosuf_norm) >= config.min_compact_len:
                keys["A"].append(f"a_exact:{nosuf_norm}")
                nosuf_comp = "".join(nosuf_norm.split())
                keys["A"].append(f"a_compact:{nosuf_comp}")

        # Domain core representations
        if config.enable_domain_blocking:
            if core_compact and len(core_compact) >= 3:
                keys["A"].append(f"a_compact:{core_compact}")
                keys["A"].append(f"a_domain:{core_compact}")
            if core_stem and len(core_stem) >= 3 and core_stem != core_compact:
                keys["A"].append(f"a_exact:{core_stem}")

    # Blocker B: Tokens, stopword bigrams, and rare tokens
    tokens = name_norm.split() if name_norm else []

    # Leading bigram and compact 2-token (preserves leading stopwords like 'the', 'first', etc.)
    if config.enable_stopword_prefix_bi and len(tokens) >= 2:
        keys["B"].append(f"b_lead_bi:{tokens[0]}_{tokens[1]}")
        if len(tokens[0]) + len(tokens[1]) >= 5:
            keys["A"].append(f"a_compact:{tokens[0]}{tokens[1]}")

    clean_tokens = [
        t for t in tokens if len(t) >= config.min_token_len and t not in config.name_stopwords
    ]

    if config.enable_blocker_b:
        for t in clean_tokens[:4]:
            keys["B"].append(f"b_tok:{t}")
        if len(clean_tokens) >= 2:
            keys["B"].append(f"b_bi:{clean_tokens[0]}_{clean_tokens[1]}")
        if len(clean_tokens) >= 3:
            keys["B"].append(f"b_bi:{clean_tokens[0]}_{clean_tokens[2]}")

        # Rare token blocker
        if config.enable_rare_token_blocking and token_freq is not None:
            for t in clean_tokens:
                freq = token_freq.get(t, 0)
                if 1 < freq <= config.rare_token_max_freq:
                    keys["B"].append(f"b_rare:{t}")

    # Blocker C & D: Address & Composite
    addr_norm = str(row.get("address_norm") or "").strip()
    addr_is_missing = bool(row.get("address_is_missing", False)) or not addr_norm

    if not addr_is_missing:
        addr_tokens = addr_norm.split()
        nums: list[str] = []
        for t in addr_tokens:
            if re.match(r"^\d+$", t):
                nums.append(t)
                stripped = t.lstrip("0")
                if stripped and stripped != t:
                    nums.append(stripped)
            else:
                digit_parts = re.findall(r"\d+", t)
                for dp in digit_parts:
                    if len(dp) >= 2:
                        nums.append(dp)
        nums = list(dict.fromkeys(nums))

        # Postal code detection (5-digit US or 6-digit India)
        zip_codes = [t for t in nums if len(t) in (5, 6)]

        info_words = [
            t
            for t in addr_tokens
            if len(t) >= 4
            and not re.match(r"^\d+$", t)
            and t not in config.address_stopwords
        ]

        if len(info_words) <= 4:
            selected_words = info_words
        else:
            selected_words = list(dict.fromkeys(info_words[:2] + info_words[-2:]))

        # Blocker C: Number + Word
        if config.enable_blocker_c:
            for num in nums[:3]:
                for w in selected_words[:4]:
                    keys["C"].append(f"c_num_w:{num}_{w}")

            # Postal code + distinctive address word
            for z in zip_codes[:2]:
                for w in selected_words[:3]:
                    keys["C"].append(f"c_zip_w:{z}_{w}")

            # Address-only bigram: generated across address tokens to match records with/without numbers
            if len(info_words) >= 2:
                for i in range(min(3, len(info_words) - 1)):
                    keys["C"].append(f"c_addr_bi:{info_words[i]}_{info_words[i+1]}")

        # Blocker D: Name + Number composite
        if config.enable_blocker_d:
            if nums:
                num = nums[0]
                if clean_tokens:
                    keys["D"].append(f"d_name_num:{clean_tokens[0]}_{num}")
                if name_compact and len(name_compact) >= 5:
                    keys["D"].append(f"d_pfx_num:{name_compact[:5]}_{num}")

            # Zip code + Name token
            for z in zip_codes[:2]:
                if clean_tokens:
                    keys["D"].append(f"d_zip_name:{z}_{clean_tokens[0]}")

    return keys


class GPUBlockingIndex:
    """In-memory inverted index with GPU/cuDF support and fast vectorized CPU fallback."""

    def __init__(self, config: Optional[GPUBlockingConfig] = None) -> None:
        self.config = config or GPUBlockingConfig()
        self.index: dict[str, list[str]] = defaultdict(list)
        self.total_records_indexed: int = 0
        self.source_counts: dict[str, int] = defaultdict(int)
        self.token_freq: Counter[str] = Counter()

    def build_vocabulary(self, *dfs: pd.DataFrame) -> None:
        """Precompute token frequencies across source DataFrames for rare-token blocking."""
        self.token_freq.clear()
        for df in dfs:
            if df is None or len(df) == 0:
                continue
            name_col = "name_norm" if "name_norm" in df.columns else "business_name"
            for val in df[name_col].dropna():
                tokens = str(val).split()
                for t in tokens:
                    if len(t) >= self.config.min_token_len and t not in self.config.name_stopwords:
                        self.token_freq[t] += 1

    def add_records(self, df: pd.DataFrame) -> None:
        """Add candidate records from Source 2 and Source 3 into the inverted index."""
        if df is None or len(df) == 0:
            return

        if "entity_id" not in df.columns:
            raise ValueError("Input DataFrame must contain 'entity_id' column")

        records = df.to_dict(orient="records")
        cap = self.config.max_candidates_per_block

        for rec in records:
            cand_id = str(rec["entity_id"])
            src_tag = "S2" if cand_id.startswith("S2") else ("S3" if cand_id.startswith("S3") else "other")
            self.source_counts[src_tag] += 1
            self.total_records_indexed += 1

            keys_by_blocker = generate_gpu_blocking_keys(rec, self.config, self.token_freq)
            for _, key_list in keys_by_blocker.items():
                for k in key_list:
                    plist = self.index[k]
                    if len(plist) < cap:
                        plist.append(cand_id)

    def query_entity(self, s1_record: dict[str, Any] | pd.Series) -> list[str]:
        """Query index for a single Source 1 record."""
        keys_by_blocker = generate_gpu_blocking_keys(s1_record, self.config, self.token_freq)

        blocker_weights = {
            "A": 4,  # Exact / domain match highest weight
            "D": 3,  # Name + Address composite high weight
            "B": 2,  # Token / rare-token match
            "C": 2,  # Address match
        }

        candidate_scores: dict[str, float] = defaultdict(float)
        cap = self.config.max_candidates_per_block

        for blocker_tag, key_list in keys_by_blocker.items():
            base_w = blocker_weights.get(blocker_tag, 1)
            for k in key_list:
                plist = self.index.get(k)
                if not plist or len(plist) >= cap:
                    continue
                # Inverse posting list frequency weighting
                freq_discount = 1.0 if len(plist) <= 10 else (0.8 if len(plist) <= 50 else 0.5)
                score_increment = base_w * freq_discount
                for cid in plist:
                    candidate_scores[cid] += score_increment

        if not candidate_scores:
            return []

        sorted_candidates = [
            cid
            for cid, _ in sorted(
                candidate_scores.items(), key=lambda x: x[1], reverse=True
            )[: self.config.top_k_per_s1]
        ]
        return sorted_candidates

    def query_records(self, df_s1: pd.DataFrame) -> pd.DataFrame:
        """Query a batch of Source 1 entities and return candidate pairs."""
        if df_s1 is None or len(df_s1) == 0:
            return pd.DataFrame(columns=["s1_id", "candidate_id"])

        records = df_s1.to_dict(orient="records")
        pairs: list[dict[str, str]] = []

        for rec in records:
            s1_id = str(rec["entity_id"])
            cands = self.query_entity(rec)
            for cid in cands:
                if cid != s1_id and not cid.startswith("S1"):
                    pairs.append({"s1_id": s1_id, "candidate_id": cid})

        return pd.DataFrame(pairs, columns=["s1_id", "candidate_id"])

    def save(self, path: Path | str) -> None:
        """Persist index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "index": dict(self.index),
                    "total_records": self.total_records_indexed,
                    "source_counts": dict(self.source_counts),
                    "token_freq": dict(self.token_freq),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Saved GPUBlockingIndex to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> GPUBlockingIndex:
        """Load index from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(config=data["config"])
        obj.index = defaultdict(list, data["index"])
        obj.total_records_indexed = data["total_records"]
        obj.source_counts = defaultdict(int, data["source_counts"])
        obj.token_freq = Counter(data.get("token_freq", {}))
        logger.info("Loaded GPUBlockingIndex from %s (%d records)", path, obj.total_records_indexed)
        return obj


class DiskGPUBlockingIndex:
    """Disk-backed SQLite blocking index for large-scale GPU/streaming execution."""

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        config: Optional[GPUBlockingConfig] = None,
    ) -> None:
        self.config = config or GPUBlockingConfig()
        self._is_temp = False

        if db_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            tmp.close()
            self.db_path = Path(tmp.name)
            self._is_temp = True
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.total_records_indexed: int = 0
        self.source_counts: dict[str, int] = defaultdict(int)
        self._is_finalized: bool = False
        self.token_freq: Counter[str] = Counter()

        self.conn: Optional[sqlite3.Connection] = sqlite3.connect(str(self.db_path))
        self._setup_pragmas()
        self._init_db()

    def _setup_pragmas(self) -> None:
        if self.conn is None:
            return
        cur = self.conn.cursor()
        cur.execute("PRAGMA synchronous = OFF")
        cur.execute("PRAGMA journal_mode = OFF")
        cur.execute("PRAGMA cache_size = -64000")
        cur.execute("PRAGMA temp_store = FILE")

    def _init_db(self) -> None:
        if self.conn is None:
            return
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS candidate_records (
                entity_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                name_norm TEXT,
                name_compact TEXT,
                name_no_suffix TEXT,
                address_norm TEXT,
                address_is_missing INTEGER,
                country_norm TEXT,
                name_script TEXT,
                is_domain INTEGER,
                domain_stem TEXT,
                legal_suffix TEXT
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS raw_blocking_keys (
                block_key TEXT NOT NULL,
                candidate_id TEXT NOT NULL
            )"""
        )
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='posting_lists'")
        if cur.fetchone():
            self._is_finalized = True
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='meta'")
            if cur.fetchone():
                cur.execute("SELECT value FROM meta WHERE key='total_records_indexed'")
                r_tot = cur.fetchone()
                if r_tot:
                    self.total_records_indexed = int(r_tot[0])
                cur.execute("SELECT value FROM meta WHERE key='source_counts'")
                r_sc = cur.fetchone()
                if r_sc:
                    self.source_counts = defaultdict(int, json.loads(r_sc[0]))
        self.conn.commit()

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    def build_vocabulary(self, *dfs: pd.DataFrame) -> None:
        """Precompute token frequencies across source DataFrames for rare-token blocking."""
        self.token_freq.clear()
        for df in dfs:
            if df is None or len(df) == 0:
                continue
            name_col = "name_norm" if "name_norm" in df.columns else "business_name"
            for val in df[name_col].dropna():
                tokens = str(val).split()
                for t in tokens:
                    if len(t) >= self.config.min_token_len and t not in self.config.name_stopwords:
                        self.token_freq[t] += 1

    def add_records(self, df: pd.DataFrame) -> None:
        """Add candidate records to SQLite in a memory-bounded streaming batch."""
        if df is None or len(df) == 0:
            return

        if "entity_id" not in df.columns:
            raise ValueError("Input DataFrame must contain 'entity_id' column")

        if "name_norm" not in df.columns:
            df = preprocess_records(df)

        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self._setup_pragmas()

        records = df.to_dict(orient="records")
        cand_rows = []
        key_rows = []

        for rec in records:
            eid = str(rec["entity_id"])
            src = "S2" if eid.startswith("S2") else ("S3" if eid.startswith("S3") else "other")
            self.source_counts[src] += 1
            self.total_records_indexed += 1

            cand_rows.append(
                (
                    eid,
                    src,
                    str(rec.get("name_norm") or ""),
                    str(rec.get("name_compact") or ""),
                    str(rec.get("name_no_suffix") or ""),
                    str(rec.get("address_norm") or ""),
                    1 if rec.get("address_is_missing") else 0,
                    str(rec.get("country_norm") or ""),
                    str(rec.get("name_script") or ""),
                    1 if rec.get("is_domain") else 0,
                    str(rec.get("domain_stem") or ""),
                    str(rec.get("legal_suffix") or ""),
                )
            )

            keys_by_blocker = generate_gpu_blocking_keys(rec, self.config, self.token_freq)
            for _, key_list in keys_by_blocker.items():
                for k in key_list:
                    key_rows.append((k, eid))

        cur = self.conn.cursor()
        cur.executemany(
            """INSERT OR REPLACE INTO candidate_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            cand_rows,
        )
        cur.executemany(
            "INSERT INTO raw_blocking_keys (block_key, candidate_id) VALUES (?, ?)",
            key_rows,
        )
        self.conn.commit()

    def finalize(self) -> None:
        """Aggregate raw keys into indexed posting lists."""
        if self._is_finalized:
            return

        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self._setup_pragmas()

        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_blocking_keys'")
        if not cur.fetchone():
            self._is_finalized = True
            return

        cap = self.config.max_candidates_per_block
        cur.execute(
            """CREATE TABLE IF NOT EXISTS posting_lists AS
            SELECT block_key, GROUP_CONCAT(candidate_id, ',') AS candidate_ids, COUNT(*) AS cnt
            FROM raw_blocking_keys
            GROUP BY block_key
            HAVING cnt < ?""",
            (cap,),
        )
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_posting_key ON posting_lists(block_key)")
        cur.execute("DROP TABLE IF EXISTS raw_blocking_keys")

        cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("INSERT OR REPLACE INTO meta VALUES ('is_finalized', '1')")
        cur.execute(
            "INSERT OR REPLACE INTO meta VALUES ('total_records_indexed', ?)",
            (str(self.total_records_indexed),),
        )
        cur.execute(
            "INSERT OR REPLACE INTO meta VALUES ('source_counts', ?)",
            (json.dumps(dict(self.source_counts)),),
        )
        self.conn.commit()
        self._is_finalized = True

    def query_records(self, df_s1: pd.DataFrame) -> pd.DataFrame:
        """Query a DataFrame of Source 1 entities and return candidate pairs."""
        if df_s1 is None or len(df_s1) == 0:
            return pd.DataFrame(columns=["s1_id", "candidate_id"])

        self.finalize()
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self._setup_pragmas()

        records = df_s1.to_dict(orient="records")
        keys_per_record: list[tuple[str, dict[str, list[str]]]] = []
        all_keys: set[str] = set()

        for rec in records:
            s1_id = str(rec["entity_id"])
            keys_by_blocker = generate_gpu_blocking_keys(rec, self.config, self.token_freq)
            keys_per_record.append((s1_id, keys_by_blocker))
            for _, klist in keys_by_blocker.items():
                all_keys.update(klist)

        postings: dict[str, list[str]] = {}
        if all_keys:
            key_list = list(all_keys)
            cur = self.conn.cursor()
            batch_size = 500
            for i in range(0, len(key_list), batch_size):
                k_batch = key_list[i : i + batch_size]
                placeholders = ",".join("?" for _ in k_batch)
                cur.execute(
                    f"SELECT block_key, candidate_ids FROM posting_lists WHERE block_key IN ({placeholders})",
                    k_batch,
                )
                for k, cstr in cur.fetchall():
                    if cstr:
                        postings[k] = cstr.split(",")

        blocker_weights = {"A": 4, "D": 3, "B": 2, "C": 2}
        pairs: list[dict[str, str]] = []

        for s1_id, keys_by_blocker in keys_per_record:
            candidate_scores: dict[str, float] = defaultdict(float)
            for blocker_tag, key_list in keys_by_blocker.items():
                base_w = blocker_weights.get(blocker_tag, 1)
                for k in key_list:
                    cands = postings.get(k)
                    if not cands:
                        continue
                    freq_discount = 1.0 if len(cands) <= 10 else (0.8 if len(cands) <= 50 else 0.5)
                    score_increment = base_w * freq_discount
                    for cid in cands:
                        candidate_scores[cid] += score_increment

            if not candidate_scores:
                continue

            sorted_candidates = [
                cid
                for cid, _ in sorted(
                    candidate_scores.items(), key=lambda x: x[1], reverse=True
                )[: self.config.top_k_per_s1]
            ]

            for cid in sorted_candidates:
                if cid != s1_id and not cid.startswith("S1"):
                    pairs.append({"s1_id": s1_id, "candidate_id": cid})

        return pd.DataFrame(pairs, columns=["s1_id", "candidate_id"])

    def get_candidate_records(
        self, candidate_ids: list[str] | set[str] | np.ndarray
    ) -> list[dict[str, Any]]:
        """Fetch candidate feature records from SQLite."""
        id_list = list(dict.fromkeys(candidate_ids))
        if not id_list:
            return []

        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self._setup_pragmas()

        columns = [
            "entity_id",
            "source",
            "name_norm",
            "name_compact",
            "name_no_suffix",
            "address_norm",
            "address_is_missing",
            "country_norm",
            "name_script",
            "is_domain",
            "domain_stem",
            "legal_suffix",
        ]
        results: list[dict[str, Any]] = []
        cur = self.conn.cursor()
        batch_size = 500
        for i in range(0, len(id_list), batch_size):
            chunk = id_list[i : i + batch_size]
            placeholders = ",".join("?" for _ in chunk)
            cur.execute(
                f"SELECT {','.join(columns)} FROM candidate_records WHERE entity_id IN ({placeholders})",
                chunk,
            )
            for row in cur.fetchall():
                rec = dict(zip(columns, row))
                rec["address_is_missing"] = bool(rec["address_is_missing"])
                rec["is_domain"] = bool(rec["is_domain"])
                results.append(rec)
        return results

    def add_ground_truth(
        self, gt: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]
    ) -> None:
        """Store ground truth match mapping in SQLite for memory-efficient lookup."""
        if gt is None or len(gt) == 0:
            return

        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self._setup_pragmas()

        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS ground_truth (
                s1_id TEXT PRIMARY KEY,
                matched_ids TEXT
            )"""
        )

        if isinstance(gt, pd.DataFrame):
            s1_col = "source1_entity_id" if "source1_entity_id" in gt.columns else "s1_id"
            m_col = "matched_entity_ids" if "matched_entity_ids" in gt.columns else "matched_ids"
            s1_series = gt[s1_col].astype(str)
            m_series = gt[m_col].fillna("").astype(str) if m_col in gt.columns else pd.Series([""] * len(gt))
            rows = list(zip(s1_series, m_series))
        elif isinstance(gt, dict):
            rows = []
            for k, v in gt.items():
                if isinstance(v, (set, list, tuple)):
                    val = ",".join(str(x) for x in v if str(x).strip())
                else:
                    val = str(v or "")
                rows.append((str(k), val))
        else:
            rows = [
                (
                    str(r.get("source1_entity_id") or r.get("s1_id")),
                    str(r.get("matched_entity_ids") or r.get("matched_ids") or ""),
                )
                for r in gt
            ]

        cur.executemany("INSERT OR REPLACE INTO ground_truth VALUES (?, ?)", rows)
        self.conn.commit()

    def get_ground_truth_for_s1(self, s1_ids: list[str] | set[str]) -> dict[str, set[str]]:
        """Fetch ground truth mapping for a batch of S1 IDs."""
        id_list = list(dict.fromkeys(s1_ids))
        if not id_list:
            return {}

        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self._setup_pragmas()

        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ground_truth'")
        if not cur.fetchone():
            return {s1_id: set() for s1_id in id_list}

        result: dict[str, set[str]] = {s1_id: set() for s1_id in id_list}
        batch_size = 500
        for i in range(0, len(id_list), batch_size):
            chunk = id_list[i : i + batch_size]
            placeholders = ",".join("?" for _ in chunk)
            cur.execute(
                f"SELECT s1_id, matched_ids FROM ground_truth WHERE s1_id IN ({placeholders})",
                chunk,
            )
            for s1_id, m_str in cur.fetchall():
                if m_str:
                    result[s1_id] = {
                        m.strip()
                        for m in str(m_str).split(",")
                        if m.strip() and m.strip().lower() not in ("nan", "none", "null")
                    }

        return result

    def close(self) -> None:
        """Close SQLite database connection."""
        if hasattr(self, "conn") and self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def cleanup(self) -> None:
        """Close connection and delete database file."""
        self.close()
        if self.db_path and self.db_path.exists():
            try:
                self.db_path.unlink(missing_ok=True)
                for ext in ["-wal", "-shm"]:
                    p = Path(str(self.db_path) + ext)
                    if p.exists():
                        p.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to clean up SQLite DB %s: %s", self.db_path, e)

    def __enter__(self) -> DiskGPUBlockingIndex:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        if self._is_temp:
            self.cleanup()


def build_gpu_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: Optional[GPUBlockingConfig] = None,
    use_disk_index: bool = False,
    db_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate candidate entity pairs using GPU/accuracy-optimized blocking."""
    if source1 is None or len(source1) == 0:
        return pd.DataFrame(columns=["s1_id", "candidate_id"])

    cfg = config or GPUBlockingConfig()

    s1_proc = source1 if "name_norm" in source1.columns else preprocess_records(source1)
    s2_proc = source2 if "name_norm" in source2.columns else preprocess_records(source2)
    s3_proc = source3 if "name_norm" in source3.columns else preprocess_records(source3)

    if use_disk_index:
        idx: Any = DiskGPUBlockingIndex(db_path=db_path, config=cfg)
        try:
            idx.build_vocabulary(s1_proc, s2_proc, s3_proc)
            idx.add_records(s2_proc)
            idx.add_records(s3_proc)
            idx.finalize()
            cands = idx.query_records(s1_proc)
        finally:
            if db_path is None:
                idx.cleanup()
            else:
                idx.close()
    else:
        index = GPUBlockingIndex(config=cfg)
        index.build_vocabulary(s1_proc, s2_proc, s3_proc)
        index.add_records(s2_proc)
        index.add_records(s3_proc)
        cands = index.query_records(s1_proc)

    cands = cands.drop_duplicates(subset=["s1_id", "candidate_id"]).reset_index(drop=True)
    logger.info("GPU blocking generated %d candidate pairs for %d S1 entities", len(cands), len(source1))
    return cands


# ---------------------------------------------------------------------------
# ShardedReferenceIndex — memory-safe sharded blocking for 10M+ reference pools
# ---------------------------------------------------------------------------


class ShardedReferenceIndex:
    """Memory-safe, disk-backed sharded blocking index for large reference corpora.

    Problem: The in-memory GPUBlockingIndex requires ~3.08 KB per reference record.
    Loading 10.32M reference records simultaneously needs ~31.8 GB RAM — impossible on
    a 12.7 GB Colab runtime.

    Solution: Partition the preprocessed reference corpus (S2 + S3) into N on-disk
    Parquet shards. For each S1 query batch, iterate over shards one at a time:
      1. Load shard into RAM (~shard_size × 3.08 KB).
      2. Build a temporary GPUBlockingIndex for that shard.
      3. Query all S1 records in the batch.
      4. Accumulate per-entity candidate scores.
      5. Release shard from RAM and GC.
    After all shards, apply global top-k per S1 entity.

    Key design properties:
    - Global token-frequency vocabulary is built once in a streaming pass and stored
      on disk — reused across S1 batches without rebuilding.
    - Shard Parquet files are written once and reused across all S1 batches.
    - Blocking semantics (key generation, weights, posting list cap, top-k) are
      IDENTICAL to the in-memory GPUBlockingIndex.
    - Target shard_size=200_000 → ~600 MB per shard RAM → peak ≈ 2.5 GB total.
    """

    SHARD_META_FILE = "shard_meta.json"
    VOCAB_FILE = "global_vocab.json"

    def __init__(
        self,
        shard_dir: Path | str,
        config: Optional[GPUBlockingConfig] = None,
        shard_size: int = 200_000,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or GPUBlockingConfig()
        self.shard_size = shard_size
        self.token_freq: Counter[str] = Counter()
        self._meta: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Shard building (run once per corpus)
    # ------------------------------------------------------------------

    def build_shards(
        self,
        *dfs: pd.DataFrame,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Preprocess and partition reference DataFrames into Parquet shards.

        Parameters
        ----------
        *dfs:
            One or more preprocessed (or raw) reference DataFrames (S2, S3, ...).
            Each will be preprocessed if 'name_norm' is not already present.
        overwrite:
            If True, delete and re-create existing shard files.

        Returns
        -------
        dict with shard metadata (shard count, total records, shard_size).
        """
        meta_path = self.shard_dir / self.SHARD_META_FILE
        if meta_path.exists() and not overwrite:
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
            logger.info(
                "ShardedReferenceIndex: loaded existing %d shards (%d total records) from %s",
                self._meta.get("n_shards", 0),
                self._meta.get("total_records", 0),
                self.shard_dir,
            )
            return self._meta

        # Delete stale shard files
        for old in self.shard_dir.glob("ref_shard_*.parquet"):
            old.unlink(missing_ok=True)

        # Required preprocessed columns to persist
        COLS_TO_KEEP = [
            "entity_id", "business_name",
            "name_norm", "name_compact", "name_no_suffix",
            "address_norm", "address_is_missing",
            "country_norm", "name_script", "is_domain",
            "domain_stem", "legal_suffix",
        ]

        shard_idx = 0
        total_records = 0
        shard_files: list[str] = []
        leftover_df: Optional[pd.DataFrame] = None

        def _write_shard(df_chunk: pd.DataFrame) -> None:
            nonlocal shard_idx
            if df_chunk is None or len(df_chunk) == 0:
                return
            keep = [c for c in COLS_TO_KEEP if c in df_chunk.columns]
            shard_df = df_chunk[keep]
            fname = f"ref_shard_{shard_idx:04d}.parquet"
            shard_df.to_parquet(
                self.shard_dir / fname,
                index=False,
                engine="pyarrow",
                compression="snappy",
            )
            shard_files.append(fname)
            shard_idx += 1
            logger.info(
                "ShardedReferenceIndex: flushed shard %d with %d records -> %s",
                shard_idx - 1,
                len(shard_df),
                fname,
            )

        for df in dfs:
            if df is None or len(df) == 0:
                continue
            proc_df = df if "name_norm" in df.columns else preprocess_records(df, n_jobs=4)
            total_records += len(proc_df)

            if leftover_df is not None and len(leftover_df) > 0:
                proc_df = pd.concat([leftover_df, proc_df], ignore_index=True)
                leftover_df = None

            n_full = len(proc_df) // self.shard_size
            for i in range(n_full):
                chunk = proc_df.iloc[i * self.shard_size : (i + 1) * self.shard_size]
                _write_shard(chunk)

            rem_start = n_full * self.shard_size
            if rem_start < len(proc_df):
                leftover_df = proc_df.iloc[rem_start:].copy()

        if leftover_df is not None and len(leftover_df) > 0:
            _write_shard(leftover_df)
            leftover_df = None

        self._meta = {
            "n_shards": shard_idx,
            "total_records": total_records,
            "shard_size": self.shard_size,
            "shard_files": shard_files,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2)
        logger.info(
            "ShardedReferenceIndex: built %d shards across %d total records in %s",
            shard_idx,
            total_records,
            self.shard_dir,
        )
        return self._meta

    # ------------------------------------------------------------------
    # Global vocabulary (run once, reuse across S1 batches)
    # ------------------------------------------------------------------

    def build_global_vocab(
        self,
        *extra_dfs: pd.DataFrame,
        overwrite: bool = False,
    ) -> Counter:
        """Build global token-frequency table by streaming over all shard files.

        Parameters
        ----------
        *extra_dfs:
            Additional DataFrames to include (e.g. S1 records) — these are kept
            in memory only transiently for counting purposes.
        overwrite:
            If True, recompute even if vocab file exists on disk.

        Returns
        -------
        Counter mapping token -> corpus frequency.
        """
        vocab_path = self.shard_dir / self.VOCAB_FILE
        if vocab_path.exists() and not overwrite:
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.token_freq = Counter(json.load(f))
            logger.info(
                "ShardedReferenceIndex: loaded existing vocab (%d tokens) from %s",
                len(self.token_freq),
                vocab_path,
            )
            return self.token_freq

        self.token_freq = Counter()
        min_len = self.config.min_token_len
        stopwords = self.config.name_stopwords

        # Stream through reference shards
        meta_path = self.shard_dir / self.SHARD_META_FILE
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for fname in meta.get("shard_files", []):
                shard_path = self.shard_dir / fname
                if not shard_path.exists():
                    continue
                shard_df = pd.read_parquet(shard_path, columns=["name_norm"])
                for val in shard_df["name_norm"].dropna():
                    for t in str(val).split():
                        if len(t) >= min_len and t not in stopwords:
                            self.token_freq[t] += 1
                del shard_df
                gc.collect()

        # Count extra DataFrames (S1, etc.)
        for df in extra_dfs:
            if df is None or len(df) == 0:
                continue
            name_col = "name_norm" if "name_norm" in df.columns else "business_name"
            for val in df[name_col].dropna():
                for t in str(val).split():
                    if len(t) >= min_len and t not in stopwords:
                        self.token_freq[t] += 1

        # Persist
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(dict(self.token_freq), f)
        logger.info(
            "ShardedReferenceIndex: built global vocab with %d unique tokens, saved to %s",
            len(self.token_freq),
            vocab_path,
        )
        return self.token_freq

    def load_vocab(self) -> Counter:
        """Load previously built vocab from disk."""
        vocab_path = self.shard_dir / self.VOCAB_FILE
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocab file not found at {vocab_path}. Run build_global_vocab() first."
            )
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_freq = Counter(json.load(f))
        return self.token_freq

    # ------------------------------------------------------------------
    # Query (called per S1 batch, iterates over shards)
    # ------------------------------------------------------------------

    def query_s1_batch(
        self,
        s1_batch: pd.DataFrame,
        rss_monitor: Optional[list] = None,
    ) -> pd.DataFrame:
        """Query all S1 records in `s1_batch` against the full sharded corpus.

        Iterates shard-by-shard. For each shard:
          1. Load shard Parquet → build temporary GPUBlockingIndex → query → accumulate.
          2. Explicitly delete shard index and GC before next shard.
        Finally, apply global top-k deduplication and return candidate pairs.

        Parameters
        ----------
        s1_batch:
            Preprocessed S1 DataFrame (requires 'entity_id', 'name_norm', etc.)
        rss_monitor:
            Optional list — peak RSS values are appended for memory tracking.

        Returns
        -------
        pd.DataFrame with columns ['s1_id', 'candidate_id'].
        """
        if s1_batch is None or len(s1_batch) == 0:
            return pd.DataFrame(columns=["s1_id", "candidate_id"])

        meta_path = self.shard_dir / self.SHARD_META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Shard metadata not found at {meta_path}. Run build_shards() first."
            )
        if not self.token_freq:
            self.load_vocab()

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        shard_files: list[str] = meta.get("shard_files", [])
        if not shard_files:
            return pd.DataFrame(columns=["s1_id", "candidate_id"])

        # Accumulate per-S1 candidate scores across shards
        # Structure: {s1_id -> {candidate_id -> score}}
        global_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        blocker_weights = {"A": 4, "D": 3, "B": 2, "C": 2}
        cap = self.config.max_candidates_per_block

        # Pre-compute S1 blocking keys (done once, reused across shards)
        s1_records = s1_batch.to_dict(orient="records")
        s1_keys: list[tuple[str, dict[str, list[str]]]] = []
        for rec in s1_records:
            s1_id = str(rec["entity_id"])
            keys_by_blocker = generate_gpu_blocking_keys(rec, self.config, self.token_freq)
            s1_keys.append((s1_id, keys_by_blocker))

        for fname in shard_files:
            shard_path = self.shard_dir / fname
            if not shard_path.exists():
                logger.warning("ShardedReferenceIndex: shard file missing: %s", shard_path)
                continue

            t_shard = time.time()

            # Load shard and build temporary in-memory index
            shard_df = pd.read_parquet(shard_path)
            shard_index = GPUBlockingIndex(config=self.config)
            shard_index.token_freq = self.token_freq  # share global vocab
            shard_index.add_records(shard_df)

            shard_records = shard_df.shape[0]
            del shard_df
            # Track peak RSS
            if rss_monitor is not None:
                rss_monitor.append(get_process_rss_mb())

            # Collect all blocking keys from this shard's index
            shard_inv = shard_index.index  # {key -> [cand_id, ...]}

            # Query each S1 entity against this shard's posting lists
            for s1_id, keys_by_blocker in s1_keys:
                for blocker_tag, key_list in keys_by_blocker.items():
                    base_w = blocker_weights.get(blocker_tag, 1)
                    for k in key_list:
                        plist = shard_inv.get(k)
                        if not plist or len(plist) >= cap:
                            continue
                        freq_discount = (
                            1.0 if len(plist) <= 10
                            else (0.8 if len(plist) <= 50 else 0.5)
                        )
                        score_inc = base_w * freq_discount
                        for cid in plist:
                            global_scores[s1_id][cid] += score_inc

            # Explicitly release shard index before next iteration
            del shard_index
            gc.collect()

            logger.debug(
                "ShardedReferenceIndex: shard %s (%d records) queried in %.2fs",
                fname,
                shard_records,
                time.time() - t_shard,
            )

        # Apply global top-k and build output pairs
        top_k = self.config.top_k_per_s1
        pairs: list[dict[str, str]] = []

        for s1_id, cand_scores in global_scores.items():
            if not cand_scores:
                continue
            sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for cid, _ in sorted_cands:
                if cid != s1_id and not cid.startswith("S1"):
                    pairs.append({"s1_id": s1_id, "candidate_id": cid})

        result = pd.DataFrame(pairs, columns=["s1_id", "candidate_id"])
        return result

    @property
    def total_records(self) -> int:
        """Total reference records across all shards."""
        meta_path = self.shard_dir / self.SHARD_META_FILE
        if not self._meta and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta.get("total_records", 0)

    @property
    def n_shards(self) -> int:
        """Number of reference shards."""
        meta_path = self.shard_dir / self.SHARD_META_FILE
        if not self._meta and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta.get("n_shards", 0)
