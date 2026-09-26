"""
blocking.py — Multi-strategy candidate-pair generation for entity resolution.

Design intent
-------------
Reduces the O(n²) comparison space to a tractable, high-recall candidate set.
Uses a union of complementary, independently configurable blocking strategies:

- Blocker A (Exact Name Blocks):
  Exact normalized Unicode name, compact alphanumeric string (no spaces),
  name without legal suffix, and domain stem.
- Blocker B (Selective Token-Based Blocks):
  Distinctive name tokens and token bigrams, filtered against frequency thresholds
  and generic stopwords (e.g. 'inc', 'llc', 'pvt', 'company', 'the').
- Blocker C (Address-Based Blocks):
  Order-independent combinations of street numbers and informative address/city tokens.
- Blocker D (Composite Blocks):
  Name prefix / distinctive token combined with address number.

Candidate explosion control:
- High-frequency / generic keys are filtered.
- Posting lists per block are bounded by `max_candidates_per_block`.
- Candidates per Source 1 entity are ranked and capped by `top_k_per_s1`.
- Intermediate index is serializable to disk for Google Colab sessions.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation import ground_truth_to_dict
from src.preprocessing import (
    preprocess_address,
    preprocess_name,
    preprocess_records,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic Stopwords (Name and Address) to Prevent Explosion
# ---------------------------------------------------------------------------

NAME_STOPWORDS: frozenset[str] = frozenset(
    {
        # English generic words
        "the",
        "and",
        "of",
        "for",
        "in",
        "at",
        "to",
        "a",
        "an",
        "on",
        # Common entity generic nouns
        "company",
        "co",
        "corporation",
        "corp",
        "incorporated",
        "inc",
        "limited",
        "ltd",
        "llc",
        "llp",
        "pllc",
        "private",
        "pvt",
        "group",
        "services",
        "service",
        "enterprises",
        "enterprise",
        "solutions",
        "solution",
        "international",
        "global",
        "holdings",
        "holding",
        "consulting",
        "technologies",
        "technology",
        "industries",
        "industry",
        "systems",
        "system",
        "associates",
        "agency",
        "center",
        "centre",
        "care",
        "management",
        "foundation",
        "trust",
        "club",
        "store",
        "shop",
        "mart",
        "market",
        "products",
        "ventures",
        "properties",
        "india",
        "us",
        "usa",
        # French suffixes / stopwords
        "sarl",
        "sas",
        "sasu",
        "sci",
        "eurl",
        "sa",
        "snc",
        "de",
        "du",
        "des",
        "la",
        "le",
        "les",
        "et",
        # Indic transliterated / native tokens
        "प्राइवेट",
        "लिमिटेड",
        "प्रा",
        "लि",
        "कंपनी",
        "एंटरप्राइजेज",
        "பிரைவேட்",
        "லிமிடெட்",
        "પ્રાઇવેટ",
        "લિમિટેડ",
    }
)

ADDRESS_STOPWORDS: frozenset[str] = frozenset(
    {
        "street",
        "road",
        "avenue",
        "drive",
        "lane",
        "court",
        "boulevard",
        "place",
        "way",
        "highway",
        "parkway",
        "circle",
        "suite",
        "ste",
        "apartment",
        "apt",
        "unit",
        "floor",
        "fl",
        "building",
        "bldg",
        "pobox",
        "box",
        "po",
        "near",
        "opp",
        "opposite",
        "behind",
        "block",
        "sector",
        "plot",
        "door",
        "no",
        "kh",
        "cross",
        "main",
        "null",
        "none",
        "nan",
        "rue",
        "avenue",
        "boulevard",
    }
)


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------


@dataclass
class BlockingConfig:
    """Configurable settings for candidate generation."""

    enable_blocker_a: bool = True  # Exact name, compact, no-suffix, domain stem
    enable_blocker_b: bool = True  # Selective token blocks
    enable_blocker_c: bool = True  # Address number + street/city words
    enable_blocker_d: bool = True  # Name + Address composite blocks
    max_candidates_per_block: int = 100  # Cap posting list size
    min_token_len: int = 3
    min_compact_len: int = 4
    top_k_per_s1: int = 50  # Max candidates per S1 entity
    name_stopwords: frozenset[str] = field(default_factory=lambda: NAME_STOPWORDS)
    address_stopwords: frozenset[str] = field(default_factory=lambda: ADDRESS_STOPWORDS)

    @classmethod
    def from_config(cls, cfg: Any) -> BlockingConfig:
        """Create BlockingConfig from global Config instance."""
        return cls(
            enable_blocker_a=getattr(cfg, "BLOCKER_A_ENABLED", True),
            enable_blocker_b=getattr(cfg, "BLOCKER_B_ENABLED", True),
            enable_blocker_c=getattr(cfg, "BLOCKER_C_ENABLED", True),
            enable_blocker_d=getattr(cfg, "BLOCKER_D_ENABLED", True),
            max_candidates_per_block=getattr(cfg, "MAX_CANDIDATES_PER_BLOCK", 100),
            min_token_len=getattr(cfg, "MIN_TOKEN_LEN", 3),
            top_k_per_s1=getattr(cfg, "BLOCKING_TOP_K", 50) or 50,
        )


# ---------------------------------------------------------------------------
# Key Generation Functions
# ---------------------------------------------------------------------------


def generate_record_blocking_keys(
    row: dict[str, Any] | pd.Series,
    config: BlockingConfig,
) -> dict[str, list[str]]:
    """Generate blocking keys tagged by blocker strategy (A, B, C, D).

    Accepts either raw record fields or preprocessed fields.
    """
    # 1. Resolve normalized name representations
    raw_name = row.get("business_name")
    name_norm = row.get("name_norm")
    name_compact = row.get("name_compact")
    name_no_suffix = row.get("name_no_suffix")
    domain_stem = row.get("domain_stem")
    is_domain = row.get("is_domain", False)

    if (
        name_norm is None
        or pd.isna(name_norm)
        or name_compact is None
        or pd.isna(name_compact)
    ):
        p_name = preprocess_name(raw_name)
        name_norm = p_name["punct_norm"] or ""
        name_compact = p_name["compact"] or ""
        name_no_suffix = p_name["name_no_suffix"]
        domain_stem = p_name["domain_stem"]
        is_domain = p_name["is_domain"]

    # Ensure all name variables are safe strings
    name_norm = "" if name_norm is None or pd.isna(name_norm) else str(name_norm).strip()
    name_compact = "" if name_compact is None or pd.isna(name_compact) else str(name_compact).strip()
    name_no_suffix = "" if name_no_suffix is None or pd.isna(name_no_suffix) else str(name_no_suffix).strip()
    domain_stem = "" if domain_stem is None or pd.isna(domain_stem) else str(domain_stem).strip()
    is_domain = bool(is_domain)

    # 2. Resolve address representations
    addr_norm = row.get("address_norm")
    addr_is_missing = row.get("address_is_missing")
    if (addr_norm is None or pd.isna(addr_norm)) and addr_is_missing is None:
        raw_addr = row.get("business_address") or row.get("full_address")
        p_addr = preprocess_address(raw_addr)
        addr_norm = p_addr["normalized"]
        addr_is_missing = p_addr["is_missing"]
    else:
        addr_is_missing = bool(addr_is_missing)

    addr_norm = "" if addr_norm is None or pd.isna(addr_norm) else str(addr_norm).strip()
    if not addr_norm:
        addr_is_missing = True

    keys: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}

    # -----------------------------------------------------------------------
    # Blocker A: Exact Normalized Name representations
    # -----------------------------------------------------------------------
    if config.enable_blocker_a:
        if name_norm and len(name_norm) >= config.min_compact_len:
            keys["A"].append(f"a_exact:{name_norm}")
        if name_compact and len(name_compact) >= config.min_compact_len:
            keys["A"].append(f"a_compact:{name_compact}")
        if name_no_suffix:
            nosuf_norm = str(name_no_suffix).strip()
            if len(nosuf_norm) >= config.min_compact_len:
                keys["A"].append(f"a_exact:{nosuf_norm}")
                nosuf_comp = "".join(nosuf_norm.split())
                keys["A"].append(f"a_compact:{nosuf_comp}")
        if is_domain and domain_stem and len(domain_stem) >= 3:
            keys["A"].append(f"a_domain:{domain_stem}")
            keys["A"].append(f"a_compact:{domain_stem}")

    # -----------------------------------------------------------------------
    # Blocker B: Selective Token-Based Name Blocks
    # -----------------------------------------------------------------------
    name_tokens = name_norm.split() if name_norm else []
    clean_name_tokens = [
        t
        for t in name_tokens
        if len(t) >= config.min_token_len and t not in config.name_stopwords
    ]

    if config.enable_blocker_b:
        for t in clean_name_tokens[:3]:
            keys["B"].append(f"b_tok:{t}")
        if len(clean_name_tokens) >= 2:
            keys["B"].append(f"b_bi:{clean_name_tokens[0]}_{clean_name_tokens[1]}")

    # -----------------------------------------------------------------------
    # Blocker C & D: Address & Composite Blocks
    # -----------------------------------------------------------------------
    if not addr_is_missing and addr_norm:
        addr_tokens = addr_norm.split()
        nums = [t for t in addr_tokens if re.match(r"^\d+$", t)]
        informative_addr_words = [
            t
            for t in addr_tokens
            if len(t) >= 4
            and not re.match(r"^\d+$", t)
            and t not in config.address_stopwords
        ]

        # Blocker C: Address Number + Street/City Token
        if config.enable_blocker_c and nums:
            for num in nums[:2]:
                for word in informative_addr_words[:2]:
                    keys["C"].append(f"c_num_w:{num}_{word}")

        # Blocker D: Name Token / Prefix + Address Number
        if config.enable_blocker_d and nums:
            num = nums[0]
            if clean_name_tokens:
                keys["D"].append(f"d_name_num:{clean_name_tokens[0]}_{num}")
            if name_compact and len(name_compact) >= 5:
                keys["D"].append(f"d_pfx_num:{name_compact[:5]}_{num}")

    return keys


# ---------------------------------------------------------------------------
# Inverted Index for Candidate Generation
# ---------------------------------------------------------------------------


class BlockingIndex:
    """Inverted index mapping blocking keys to candidate entity IDs."""

    def __init__(self, config: Optional[BlockingConfig] = None) -> None:
        self.config = config or BlockingConfig()
        # key -> list of candidate_ids
        self.index: dict[str, list[str]] = defaultdict(list)
        # Track posting list sizes
        self.total_records_indexed: int = 0
        self.source_counts: dict[str, int] = defaultdict(int)

    def add_records(self, df: pd.DataFrame) -> None:
        """Add candidate records (from Source 2 or Source 3) to the index.

        Supports streaming chunk additions to minimize RAM footprint.
        """
        if df is None or len(df) == 0:
            return

        # Ensure required columns are present
        if "entity_id" not in df.columns:
            raise ValueError("Input DataFrame must contain 'entity_id' column")

        # Convert to records or iterate
        records = df.to_dict(orient="records")
        cap = self.config.max_candidates_per_block

        for rec in records:
            cand_id = str(rec["entity_id"])
            src_tag = "S2" if cand_id.startswith("S2") else ("S3" if cand_id.startswith("S3") else "other")
            self.source_counts[src_tag] += 1
            self.total_records_indexed += 1

            keys_by_blocker = generate_record_blocking_keys(rec, self.config)
            for blocker_tag, key_list in keys_by_blocker.items():
                for k in key_list:
                    plist = self.index[k]
                    # Cap posting list to avoid memory explosion from generic keys
                    if len(plist) < cap:
                        plist.append(cand_id)

    def query_entity(self, s1_record: dict[str, Any] | pd.Series) -> list[str]:
        """Query the index for a single Source 1 entity.

        Returns a deduplicated, ranked list of candidate IDs.
        """
        keys_by_blocker = generate_record_blocking_keys(s1_record, self.config)

        # Weight contributions from different blockers
        blocker_weights = {
            "A": 3,  # Exact match highest weight
            "D": 2,  # Name + Address composite high weight
            "B": 1,  # Token match
            "C": 1,  # Address match
        }

        candidate_scores: dict[str, int] = defaultdict(int)
        cap = self.config.max_candidates_per_block

        for blocker_tag, key_list in keys_by_blocker.items():
            w = blocker_weights.get(blocker_tag, 1)
            for k in key_list:
                plist = self.index.get(k)
                if not plist:
                    continue
                # Skip overly generic blocks
                if len(plist) >= cap:
                    continue
                for cid in plist:
                    candidate_scores[cid] += w

        if not candidate_scores:
            return []

        # Sort by score descending and cap at top_k_per_s1
        sorted_candidates = [
            cid
            for cid, _ in sorted(
                candidate_scores.items(), key=lambda x: x[1], reverse=True
            )[: self.config.top_k_per_s1]
        ]

        return sorted_candidates

    def query_records(self, df_s1: pd.DataFrame) -> pd.DataFrame:
        """Query a DataFrame of Source 1 entities and return candidate pairs."""
        if df_s1 is None or len(df_s1) == 0:
            return pd.DataFrame(columns=["s1_id", "candidate_id"])

        records = df_s1.to_dict(orient="records")
        pairs: list[dict[str, str]] = []

        for rec in records:
            s1_id = str(rec["entity_id"])
            cands = self.query_entity(rec)
            for cid in cands:
                # Ensure no self matches or S1-to-S1 candidate pairs
                if cid != s1_id and not cid.startswith("S1"):
                    pairs.append({"s1_id": s1_id, "candidate_id": cid})

        return pd.DataFrame(pairs, columns=["s1_id", "candidate_id"])

    def save(self, path: Path | str) -> None:
        """Persist index to disk for reuse across Colab sessions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "index": dict(self.index),
                    "total_records": self.total_records_indexed,
                    "source_counts": dict(self.source_counts),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Saved BlockingIndex to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> BlockingIndex:
        """Load index from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(config=data["config"])
        obj.index = defaultdict(list, data["index"])
        obj.total_records_indexed = data["total_records"]
        obj.source_counts = defaultdict(int, data["source_counts"])
        logger.info("Loaded BlockingIndex from %s (%d records)", path, obj.total_records_indexed)
        return obj


# ---------------------------------------------------------------------------
# Disk-Backed Inverted Index for Candidate Generation (SQLite)
# ---------------------------------------------------------------------------


class DiskBlockingIndex:
    """Disk-backed inverted index and candidate store using SQLite.

    Solves the Turn 6 RAM scalability problem for the full TEST candidate pool
    (S2: ~4.88M records, S3: ~4.8M records = ~9.7M candidates total).

    Key design principles:
    1. Zero-RAM growth: stores candidate records and posting lists on disk.
    2. Exact ML equivalence: identical 4-stage blocking keys, posting list cap
       (max_candidates_per_block=100), blocker weights (A=3, D=2, B=1, C=1),
       and top_k_per_s1=50 candidate ranking.
    3. Batched chunk inserts: chunks of preprocessed S2/S3 records are inserted
       into SQLite sequentially, freeing DataFrame memory immediately.
    4. Fast batch queries: collects distinct keys for an S1 entity batch and
       fetches posting lists in batched SQL queries using B-tree index.
    5. Clean lifecycle: supports persistent caching or automatic cleanup.
    """

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        config: Optional[BlockingConfig] = None,
    ) -> None:
        self.config = config or BlockingConfig()
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

        self.conn: Optional[sqlite3.Connection] = sqlite3.connect(str(self.db_path))
        self._setup_pragmas()
        self._init_db()

    def _setup_pragmas(self) -> None:
        if self.conn is None:
            return
        cur = self.conn.cursor()
        cur.execute("PRAGMA synchronous = OFF")
        cur.execute("PRAGMA journal_mode = OFF")
        cur.execute("PRAGMA cache_size = -64000")  # 64 MB RAM cache cap
        cur.execute("PRAGMA temp_store = FILE")     # Disk-backed temporary tables

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
        # Check if already finalized from a previous run
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

    def add_records(self, df: pd.DataFrame) -> None:
        """Add candidate records to SQLite database in a memory-safe batch."""
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

            keys_by_blocker = generate_record_blocking_keys(rec, self.config)
            for _, key_list in keys_by_blocker.items():
                for k in key_list:
                    key_rows.append((k, eid))

        cur = self.conn.cursor()
        cur.executemany(
            """INSERT OR REPLACE INTO candidate_records (
                entity_id, source, name_norm, name_compact, name_no_suffix,
                address_norm, address_is_missing, country_norm, name_script,
                is_domain, domain_stem, legal_suffix
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            cand_rows,
        )
        cur.executemany(
            "INSERT INTO raw_blocking_keys (block_key, candidate_id) VALUES (?, ?)",
            key_rows,
        )
        self.conn.commit()

    def finalize(self) -> None:
        """Finalize posting lists: aggregate raw keys, filter generic blocks, index."""
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

    def query_entity(self, s1_record: dict[str, Any] | pd.Series) -> list[str]:
        """Query the index for a single Source 1 entity."""
        self.finalize()
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self._setup_pragmas()

        keys_by_blocker = generate_record_blocking_keys(s1_record, self.config)
        all_keys: list[str] = []
        for _, klist in keys_by_blocker.items():
            all_keys.extend(klist)

        postings: dict[str, list[str]] = {}
        if all_keys:
            cur = self.conn.cursor()
            placeholders = ",".join("?" for _ in all_keys)
            cur.execute(
                f"SELECT block_key, candidate_ids FROM posting_lists WHERE block_key IN ({placeholders})",
                all_keys,
            )
            for k, cstr in cur.fetchall():
                if cstr:
                    postings[k] = cstr.split(",")

        blocker_weights = {"A": 3, "D": 2, "B": 1, "C": 1}
        candidate_scores: dict[str, int] = defaultdict(int)
        for blocker_tag, key_list in keys_by_blocker.items():
            w = blocker_weights.get(blocker_tag, 1)
            for k in key_list:
                cands = postings.get(k)
                if not cands:
                    continue
                for cid in cands:
                    candidate_scores[cid] += w

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
            keys_by_blocker = generate_record_blocking_keys(rec, self.config)
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

        blocker_weights = {"A": 3, "D": 2, "B": 1, "C": 1}
        pairs: list[dict[str, str]] = []

        for s1_id, keys_by_blocker in keys_per_record:
            candidate_scores: dict[str, int] = defaultdict(int)
            for blocker_tag, key_list in keys_by_blocker.items():
                w = blocker_weights.get(blocker_tag, 1)
                for k in key_list:
                    cands = postings.get(k)
                    if not cands:
                        continue
                    for cid in cands:
                        candidate_scores[cid] += w

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
        """Fetch candidate feature records for feature computation."""
        id_list = list(dict.fromkeys(candidate_ids))
        if not id_list:
            return []

        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self._setup_pragmas()

        columns = [
            "entity_id", "source", "name_norm", "name_compact", "name_no_suffix",
            "address_norm", "address_is_missing", "country_norm", "name_script",
            "is_domain", "domain_stem", "legal_suffix"
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
        """Store ground truth match mapping in SQLite for memory-efficient chunked lookup."""
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
        self._has_ground_truth = True

    def get_ground_truth_for_s1(self, s1_ids: list[str] | set[str]) -> dict[str, set[str]]:
        """Fetch ground truth mapping for a batch of S1 IDs from SQLite."""
        id_list = list(dict.fromkeys(s1_ids))
        if not id_list:
            return {}

        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self._setup_pragmas()

        cur = self.conn.cursor()
        if not getattr(self, "_has_ground_truth", False):
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ground_truth'")
            if not cur.fetchone():
                return {s1_id: set() for s1_id in id_list}
            self._has_ground_truth = True

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

    def get(self, entity_id: str, default: Any = None) -> Optional[dict[str, Any]]:
        """Dictionary-compatible lookup for a single entity_id."""
        records = self.get_candidate_records([entity_id])
        return records[0] if records else default

    def __len__(self) -> int:
        return self.total_records_indexed

    def close(self) -> None:
        """Close SQLite database connection."""
        if hasattr(self, "conn") and self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def cleanup(self) -> None:
        """Close database and delete database file from disk."""
        self.close()
        if self.db_path and self.db_path.exists():
            try:
                self.db_path.unlink(missing_ok=True)
                for ext in ["-wal", "-shm"]:
                    p = Path(str(self.db_path) + ext)
                    if p.exists():
                        p.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to remove temporary DB file %s: %s", self.db_path, e)

    def __enter__(self) -> DiskBlockingIndex:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        if self._is_temp:
            self.cleanup()

    @classmethod
    def load(cls, path: Path | str, config: Optional[BlockingConfig] = None) -> DiskBlockingIndex:
        """Load an existing disk-backed blocking index."""
        return cls(db_path=path, config=config)


# ---------------------------------------------------------------------------
# Main Public Blocking API
# ---------------------------------------------------------------------------


def build_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: Optional[Config | BlockingConfig] = None,
    use_disk_index: bool = False,
    db_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate candidate (Source 1, Source 2/3) entity pairs via multi-strategy blocking.

    Parameters
    ----------
    source1, source2, source3:
        Source DataFrames. Can be raw or preprocessed.
    config:
        Config object or BlockingConfig instance.
    use_disk_index:
        If True, use SQLite-backed DiskBlockingIndex instead of in-memory BlockingIndex.
    db_path:
        Optional path to SQLite database when use_disk_index=True.

    Returns
    -------
    pd.DataFrame with columns:
        s1_id        — Source 1 entity_id
        candidate_id — Source 2 or Source 3 entity_id
    """
    if source1 is None or len(source1) == 0:
        return pd.DataFrame(columns=["s1_id", "candidate_id"])

    # Adapt config
    if isinstance(config, BlockingConfig):
        b_cfg = config
    elif isinstance(config, Config):
        b_cfg = BlockingConfig.from_config(config)
    else:
        b_cfg = BlockingConfig()

    # Preprocess if normalized columns are not already present
    s1_proc = source1 if "name_norm" in source1.columns else preprocess_records(source1)
    s2_proc = source2 if "name_norm" in source2.columns else preprocess_records(source2)
    s3_proc = source3 if "name_norm" in source3.columns else preprocess_records(source3)

    if use_disk_index:
        idx: Any = DiskBlockingIndex(db_path=db_path, config=b_cfg)
        try:
            idx.add_records(s2_proc)
            idx.add_records(s3_proc)
            idx.finalize()
            candidates_df = idx.query_records(s1_proc)
        finally:
            if db_path is None:
                idx.cleanup()
            else:
                idx.close()
    else:
        # Inverted index (in-memory)
        index = BlockingIndex(config=b_cfg)
        index.add_records(s2_proc)
        index.add_records(s3_proc)
        candidates_df = index.query_records(s1_proc)

    # Deduplicate candidate pairs
    candidates_df = candidates_df.drop_duplicates(subset=["s1_id", "candidate_id"]).reset_index(
        drop=True
    )

    logger.info(
        "Blocking generated %d candidate pairs for %d S1 entities",
        len(candidates_df),
        len(source1),
    )
    return candidates_df


# ---------------------------------------------------------------------------
# Blocking Quality Metrics & Diagnostics
# ---------------------------------------------------------------------------


def blocking_recall(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame | dict[str, set[str]],
) -> float:
    """Compute the recall of the candidate set relative to ground truth.

    Recall = |Candidates ∩ True Matches| / |True Matches|

    Parameters
    ----------
    candidates:
        DataFrame with columns 's1_id', 'candidate_id'.
    ground_truth:
        Either ground-truth DataFrame or dict (from `ground_truth_to_dict`).

    Returns
    -------
    float in [0.0, 1.0]
    """
    if isinstance(ground_truth, pd.DataFrame):
        gt_dict = ground_truth_to_dict(ground_truth)
    else:
        gt_dict = ground_truth

    # Count total true matches across non-singleton entities
    total_true_matches = sum(len(m_set) for m_set in gt_dict.values())
    if total_true_matches == 0:
        return 1.0

    if candidates is None or len(candidates) == 0:
        return 0.0

    # Build candidate lookup
    cand_lookup: dict[str, set[str]] = defaultdict(set)
    for _, row in candidates.iterrows():
        cand_lookup[str(row["s1_id"])].add(str(row["candidate_id"]))

    retrieved_true_matches = 0
    for s1_id, true_set in gt_dict.items():
        found_set = cand_lookup.get(s1_id, set())
        retrieved_true_matches += len(true_set & found_set)

    return retrieved_true_matches / total_true_matches


def blocking_reduction_ratio(
    candidates: pd.DataFrame,
    source1: pd.DataFrame | int,
    source2: pd.DataFrame | int,
    source3: pd.DataFrame | int,
) -> float:
    """Compute fraction of all possible pairs eliminated by blocking.

    RR = 1.0 - (|Candidates| / (|S1| * (|S2| + |S3|)))
    """
    n_cands = len(candidates) if isinstance(candidates, pd.DataFrame) else int(candidates)
    n_s1 = len(source1) if isinstance(source1, pd.DataFrame) else int(source1)
    n_s2 = len(source2) if isinstance(source2, pd.DataFrame) else int(source2)
    n_s3 = len(source3) if isinstance(source3, pd.DataFrame) else int(source3)

    total_possible_pairs = n_s1 * (n_s2 + n_s3)
    if total_possible_pairs == 0:
        return 1.0

    return 1.0 - (n_cands / total_possible_pairs)


def detailed_blocking_diagnostics(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame | dict[str, set[str]],
    source1: Optional[pd.DataFrame] = None,
    source2_count: Optional[int] = None,
    source3_count: Optional[int] = None,
) -> dict[str, Any]:
    """Compute comprehensive diagnostic metrics for candidate generation."""
    if isinstance(ground_truth, pd.DataFrame):
        gt_dict = ground_truth_to_dict(ground_truth)
    else:
        gt_dict = ground_truth

    # Build candidate lookup
    cand_lookup: dict[str, set[str]] = defaultdict(set)
    if candidates is not None and len(candidates) > 0:
        for _, row in candidates.iterrows():
            cand_lookup[str(row["s1_id"])].add(str(row["candidate_id"]))

    total_s1 = len(gt_dict)
    total_true = 0
    true_s2 = 0
    true_s3 = 0
    retrieved_true = 0
    retrieved_s2 = 0
    retrieved_s3 = 0
    singletons = 0
    singleton_clean = 0

    cands_per_s1: list[int] = []

    for s1_id, true_set in gt_dict.items():
        found = cand_lookup.get(s1_id, set())
        cands_per_s1.append(len(found))

        if not true_set:
            singletons += 1
            if len(found) == 0:
                singleton_clean += 1
            continue

        for m in true_set:
            total_true += 1
            is_s2 = m.startswith("S2")
            if is_s2:
                true_s2 += 1
            else:
                true_s3 += 1

            if m in found:
                retrieved_true += 1
                if is_s2:
                    retrieved_s2 += 1
                else:
                    retrieved_s3 += 1

    overall_recall = retrieved_true / total_true if total_true > 0 else 1.0
    s2_recall = retrieved_s2 / true_s2 if true_s2 > 0 else 1.0
    s3_recall = retrieved_s3 / true_s3 if true_s3 > 0 else 1.0

    cands_arr = np.array(cands_per_s1) if cands_per_s1 else np.array([0])
    p90 = float(np.percentile(cands_arr, 90))
    p95 = float(np.percentile(cands_arr, 95))
    p99 = float(np.percentile(cands_arr, 99))

    # Calculate reduction ratio if pool counts provided
    rr = None
    if source1 is not None and source2_count is not None and source3_count is not None:
        rr = blocking_reduction_ratio(
            len(candidates) if candidates is not None else 0,
            len(source1),
            source2_count,
            source3_count,
        )

    return {
        "total_s1_entities": total_s1,
        "total_true_matches": total_true,
        "retrieved_true_matches": retrieved_true,
        "overall_candidate_recall": overall_recall,
        "s2_candidate_recall": s2_recall,
        "s3_candidate_recall": s3_recall,
        "singleton_count": singletons,
        "singleton_clean_count": singleton_clean,
        "total_candidates": len(candidates) if candidates is not None else 0,
        "candidate_reduction_ratio": rr,
        "candidates_per_s1_mean": float(np.mean(cands_arr)),
        "candidates_per_s1_median": float(np.median(cands_arr)),
        "candidates_per_s1_p90": p90,
        "candidates_per_s1_p95": p95,
        "candidates_per_s1_p99": p99,
        "candidates_per_s1_max": int(np.max(cands_arr)),
        "candidates_per_s1_min": int(np.min(cands_arr)),
    }
