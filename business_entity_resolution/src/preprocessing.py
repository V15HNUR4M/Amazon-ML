"""
preprocessing.py — Text normalisation for business entity records.

Design intent
-------------
Provides reusable, memory-conscious, Unicode-aware text preprocessing for
business entity resolution.  Supports multiple representations of names,
addresses, legal suffixes, domain-like names, and country fields.

Key Design Principles:
1. Preserve information: create multiple representations rather than
   destroying the original data (e.g. raw, unicode_norm, punct_norm,
   compact, tokens, order-independent).
2. Unicode preservation: Devanagari, Tamil, Kannada, Gujarati, and French
   accents carry essential matching signal and are NOT stripped or transliterated.
3. Configurable legal suffixes: suffix canonicalisation and extraction are
   separated from the base name, providing suffix features without data loss.
4. Separate address normalisation: preserves street numbers and alphanumeric
   tokens while offering both order-preserving and order-independent representations.
5. Colab-conscious memory management: supports row-level, chunk-level, and
   file-streaming chunked processing to operate safely within Colab RAM limits.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Generator, Iterator, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Missing-value literals
# ---------------------------------------------------------------------------

NULL_STRINGS: frozenset[str] = frozenset(
    {
        "",
        "null",
        "<null>",
        "none",
        "<none>",
        "nan",
        "<nan>",
        "n/a",
        "na",
        "undefined",
        "nil",
        ".",
    }
)


def is_missing(val: Any) -> bool:
    """Return True if *val* represents a missing, null, or empty value."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str):
        return val.strip().lower() in NULL_STRINGS
    return False


# ---------------------------------------------------------------------------
# Unicode Script Detection
# ---------------------------------------------------------------------------


def detect_script(text: Optional[str]) -> str:
    """Identify the predominant Unicode script of *text*.

    Supported return values:
    * 'latin': ASCII Latin + Latin Extended (including French accents)
    * 'devanagari': Hindi, Marathi, Sanskrit, etc.
    * 'tamil': Tamil script
    * 'kannada': Kannada script
    * 'gujarati': Gujarati script
    * 'other_indic': Telugu, Bengali, Malayalam, Gurmukhi, Oriya
    * 'mixed': multiple distinct scripts present in significant proportions
    * 'other': Cyrillic, Arabic, CJK, etc.
    * 'unknown': no letters or missing text

    Parameters
    ----------
    text:
        Input string.

    Returns
    -------
    str
        Script classification tag.
    """
    if is_missing(text) or not text:
        return "unknown"

    counts: dict[str, int] = {
        "latin": 0,
        "devanagari": 0,
        "tamil": 0,
        "kannada": 0,
        "gujarati": 0,
        "other_indic": 0,
        "other": 0,
    }
    total_letters = 0

    for char in text:
        cat = unicodedata.category(char)
        # Check letters (L) and combining marks (M)
        if not cat.startswith(("L", "M")):
            continue

        total_letters += 1
        cp = ord(char)

        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            counts["latin"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["tamil"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["kannada"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["gujarati"] += 1
        elif (
            (0x0980 <= cp <= 0x09FF)
            or (0x0A00 <= cp <= 0x0A7F)
            or (0x0B00 <= cp <= 0x0B7F)
            or (0x0C00 <= cp <= 0x0C7F)
            or (0x0D00 <= cp <= 0x0D7F)
        ):
            counts["other_indic"] += 1
        else:
            counts["other"] += 1

    if total_letters == 0:
        return "unknown"

    dominant_script, dominant_count = max(counts.items(), key=lambda x: x[1])

    # If any secondary script has significant presence (>= 2 chars and >= 5% or >= 3 chars), classify as mixed
    secondary = [
        s
        for s, c in counts.items()
        if s != dominant_script and c >= 2 and ((c / total_letters) >= 0.05 or c >= 3)
    ]
    if secondary:
        return "mixed"

    if dominant_count / total_letters >= 0.7:
        return dominant_script

    return "mixed"


# ---------------------------------------------------------------------------
# Unicode-safe punctuation normalization
# ---------------------------------------------------------------------------

# Fast ASCII table cache: keep alphanumeric characters (letters and numbers)
_ASCII_KEEP: dict[str, bool] = {
    chr(i): (unicodedata.category(chr(i))[0] in ("L", "M", "N")) for i in range(128)
}


def clean_punctuation(text: str) -> str:
    """Safely replace punctuation and symbols with spaces.

    Preserves letters (L), combining marks (M), and numbers (N) across all
    Unicode scripts (Devanagari, Tamil, Kannada, Gujarati, French accents).
    Collapses repeated whitespace and strips ends.
    """
    res: list[str] = []
    for ch in text:
        is_keep = _ASCII_KEEP.get(ch)
        if is_keep is None:
            # Non-ASCII Unicode character: check category
            is_keep = unicodedata.category(ch)[0] in ("L", "M", "N")
        res.append(ch if is_keep else " ")
    return " ".join("".join(res).split())


# ---------------------------------------------------------------------------
# Legal Suffix Configuration and Extraction
# ---------------------------------------------------------------------------

DEFAULT_LEGAL_SUFFIXES: list[tuple[str, tuple[str, ...]]] = [
    # Multi-token patterns first (to prevent partial matching)
    (
        "pvt_ltd",
        (
            "pvt ltd",
            "pvt limited",
            "private limited",
            "private ltd",
            "prvt ltd",
            "pvtltd",
            "private limted",
            "प्राइवेट लिमिटेड",
            "प्रा लि",
            "प्रा. लि.",
            "प्रा.लि.",
            "பிரைவேட் லிமிடெட்",
            "પ્રાઇવેટ લિમિટેડ",
            "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
        ),
    ),
    (
        "llp",
        (
            "limited liability partnership",
            "llp",
            "l l p",
            "एलएलपी",
            "எல்எல்பி",
        ),
    ),
    (
        "llc",
        (
            "limited liability company",
            "limited liability co",
            "llc",
            "l l c",
        ),
    ),
    (
        "pllc",
        (
            "professional limited liability company",
            "pllc",
            "p l l c",
        ),
    ),
    (
        "sarl",
        (
            "societe a responsabilite limitee",
            "sarl",
            "s a r l",
        ),
    ),
    (
        "sasu",
        (
            "sasu",
            "s a s u",
        ),
    ),
    (
        "sas",
        (
            "sas",
            "s a s",
        ),
    ),
    (
        "eurl",
        (
            "eurl",
            "e u r l",
        ),
    ),
    (
        "sci",
        (
            "sci",
            "s c i",
        ),
    ),
    (
        "inc",
        (
            "incorporated",
            "inc",
        ),
    ),
    (
        "corp",
        (
            "corporation",
            "corp",
        ),
    ),
    (
        "ltd",
        (
            "limited",
            "ltd",
            "limted",
            "लिमिटेड",
            "लि",
            "लि.",
            "லிமிடெட்",
            "લિમિટેડ",
            "ಲಿಮಿಟೆಡ್",
        ),
    ),
    (
        "co",
        (
            "company",
            "co",
            "कंपनी",
        ),
    ),
    (
        "sa",
        (
            "societe anonyme",
            "sa",
            "s a",
        ),
    ),
    (
        "snc",
        (
            "snc",
            "s n c",
        ),
    ),
    (
        "pc",
        (
            "pc",
            "p c",
        ),
    ),
    (
        "lp",
        (
            "lp",
            "l p",
        ),
    ),
]


def extract_legal_suffix(
    name_str: Optional[str],
    suffixes: Optional[list[tuple[str, tuple[str, ...]]]] = None,
) -> tuple[Optional[str], bool, Optional[str]]:
    """Extract and canonicalise legal suffix from a business name.

    Handles:
    * Suffix at end (e.g. 'Orelee's Barbershop LLC')
    * Suffix at start (e.g. 'LLC Moncada Learning Center', 'SCI Ptit Àmicale')
    * Split suffix (e.g. 'Pvt. EFS Print Ventures Ltd.')
    * Suffix in middle (e.g. '-- Holloway Peak Inc Seafood')
    * Indic script suffixes (Devanagari, Tamil, etc.)
    * French legal structures (SARL, SAS, SASU, EURL, SCI)

    Parameters
    ----------
    name_str:
        Business name string.
    suffixes:
        Optional custom list of (canonical_name, pattern_tuple) pairs.
        Defaults to `DEFAULT_LEGAL_SUFFIXES`.

    Returns
    -------
    tuple of (canonical_suffix, has_suffix, name_without_suffix)
    """
    if is_missing(name_str) or not name_str:
        return None, False, None

    cfg_suffixes = suffixes if suffixes is not None else DEFAULT_LEGAL_SUFFIXES
    clean_target = name_str.strip()

    # 1. Check split prefix/suffix: e.g. 'Pvt. ... Ltd.' or 'Private ... Limited'
    split_pvt_ltd = re.match(
        r"^\s*(?:pvt\.?|private)\s+(.*?)\s+(?:ltd\.?|limited)\.?\s*$",
        clean_target,
        re.IGNORECASE,
    )
    if split_pvt_ltd:
        inner = split_pvt_ltd.group(1).strip(" ,.-_")
        if inner:
            return "pvt_ltd", True, inner

    # 2. Check suffix at end of name
    for canon, patterns in cfg_suffixes:
        for pat in patterns:
            escaped = re.escape(pat)
            end_match = re.search(rf"[\s,]+{escaped}\.?\s*$", clean_target, re.IGNORECASE)
            if end_match:
                remainder = clean_target[: end_match.start()].strip(" ,.-_")
                if remainder:
                    return canon, True, remainder

    # 3. Check suffix at beginning of name
    for canon, patterns in cfg_suffixes:
        for pat in patterns:
            escaped = re.escape(pat)
            start_match = re.search(rf"^\s*{escaped}\.?[\s,]+", clean_target, re.IGNORECASE)
            if start_match:
                remainder = clean_target[start_match.end() :].strip(" ,.-_")
                if remainder:
                    return canon, True, remainder

    # 4. Check suffix in middle with whitespace boundaries
    for canon, patterns in cfg_suffixes:
        for pat in patterns:
            escaped = re.escape(pat)
            mid_match = re.search(
                rf"(?:^|[\s,]+)({escaped}\.?)(?:[\s,]+|$)",
                clean_target,
                re.IGNORECASE,
            )
            if mid_match:
                remainder = (
                    clean_target[: mid_match.start()]
                    + " "
                    + clean_target[mid_match.end() :]
                ).strip(" ,.-_")
                remainder = " ".join(remainder.split())
                if remainder:
                    return canon, True, remainder

    return None, False, clean_target


# ---------------------------------------------------------------------------
# Domain / Website-Like Name Processing
# ---------------------------------------------------------------------------

_TLD_REGEX: str = (
    r"(?:com|org|net|in|co\.in|co|io|fr|biz|info|us|gov|edu|online|store|shop|tech|ai|de|uk|eu)"
)

_DOMAIN_PATTERN: re.Pattern = re.compile(
    rf"(?:https?://)?(?:www\.)?([a-zA-Z0-9][a-zA-Z0-9.-]*\.(?:{_TLD_REGEX}))\b(?:/[^\s|,]*)?",
    re.IGNORECASE,
)

_URL_PIPE_PATTERN: re.Pattern = re.compile(
    r"\s*\|\s*(?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9.-]*\.(?:"
    + _TLD_REGEX
    + r")\S*",
    re.IGNORECASE,
)


def parse_domain_name(name_str: Optional[str]) -> dict[str, Any]:
    """Identify and parse domain-like business names.

    Derives information strictly from the input string with zero network calls:
    * Strips protocol (http://, https://)
    * Strips www. prefix
    * Strips TLD suffix (.com, .org, .in, etc.)
    * Extracts the meaningful domain stem
    * Tokenises the domain stem

    Parameters
    ----------
    name_str:
        Input business name string.

    Returns
    -------
    dict with keys:
        'is_domain': bool
        'domain_stem': Optional[str]
        'domain_tokens': list[str]
        'name_without_domain_url': Optional[str]
    """
    if is_missing(name_str) or not name_str:
        return {
            "is_domain": False,
            "domain_stem": None,
            "domain_tokens": [],
            "name_without_domain_url": None,
        }

    m = _DOMAIN_PATTERN.search(name_str)
    if not m:
        return {
            "is_domain": False,
            "domain_stem": None,
            "domain_tokens": [],
            "name_without_domain_url": name_str,
        }

    full_domain = m.group(1).lower()
    if full_domain.startswith("www."):
        full_domain = full_domain[4:]

    # Remove TLD
    stem = re.sub(rf"\.(?:{_TLD_REGEX})$", "", full_domain, flags=re.IGNORECASE)
    parts = stem.split(".")
    # If subdomain present, take the primary domain part
    if len(parts) > 1 and parts[-1] not in ("co", "com"):
        stem = parts[-1]
    else:
        stem = parts[0]

    # Clean stem into tokens (splitting on hyphens, underscores, dots)
    stem_clean = re.sub(r"[-_.]", " ", stem).strip()
    tokens = [t for t in stem_clean.split() if t]

    # If the input has a pipe like 'M/s Basera Advisors | www.baseraadv.com', clean that
    name_without_url = _URL_PIPE_PATTERN.sub("", name_str).strip(" |,.-_")
    if not name_without_url:
        name_without_url = stem_clean

    return {
        "is_domain": True,
        "domain_stem": stem,
        "domain_tokens": tokens,
        "name_without_domain_url": name_without_url,
    }


# ---------------------------------------------------------------------------
# Business Name Preprocessing
# ---------------------------------------------------------------------------


def preprocess_name(
    text: Optional[str],
    legal_suffixes: Optional[list[tuple[str, tuple[str, ...]]]] = None,
) -> dict[str, Any]:
    """Create multiple structured representations of a business name.

    Preserves original data while producing:
    * raw: original value (or None if missing)
    * is_missing: bool
    * unicode_norm: NFKC-normalized
    * lower: lowercased NFKC
    * ampersand_norm: '&' converted to ' and '
    * punct_norm: punctuation-cleaned, Unicode preserved
    * compact: alphanumeric characters with no spaces
    * alphanumeric: alphanumeric words with single spaces
    * tokens: list of tokens
    * token_set: frozenset of tokens
    * legal_suffix: canonical suffix or None
    * has_legal_suffix: bool
    * name_no_suffix: name with legal suffix stripped
    * is_domain: bool
    * domain_stem: extracted domain stem or None
    * domain_tokens: list of tokens from domain
    * script_type: detected script

    Parameters
    ----------
    text:
        Raw business name string.
    legal_suffixes:
        Optional custom legal suffix definitions.

    Returns
    -------
    dict[str, Any]
    """
    if is_missing(text) or not text:
        return {
            "raw": None,
            "is_missing": True,
            "unicode_norm": None,
            "lower": None,
            "ampersand_norm": None,
            "punct_norm": None,
            "compact": None,
            "alphanumeric": None,
            "tokens": [],
            "token_set": frozenset(),
            "legal_suffix": None,
            "has_legal_suffix": False,
            "name_no_suffix": None,
            "is_domain": False,
            "domain_stem": None,
            "domain_tokens": [],
            "script_type": "unknown",
        }

    raw = str(text)

    # 1. Unicode normalization (NFKC)
    u_norm = unicodedata.normalize("NFKC", raw).strip()

    # 2. Lowercase
    lowered = u_norm.lower()

    # 3. Script detection
    script = detect_script(u_norm)

    # 4. Domain check and URL pipe handling
    domain_info = parse_domain_name(u_norm)
    clean_target = domain_info["name_without_domain_url"] or u_norm

    # 5. Legal suffix extraction
    canon_suffix, has_suffix, name_no_suffix_raw = extract_legal_suffix(
        clean_target, suffixes=legal_suffixes
    )

    # 6. Ampersand normalization: '&' -> ' and '
    amp_norm = re.sub(r"&", " and ", lowered)
    amp_norm = " ".join(amp_norm.split())

    # 7. Safe punctuation normalization
    clean_str = clean_punctuation(amp_norm)

    # 8. Tokenization
    tokens = clean_str.split()
    token_set = frozenset(tokens)

    # 9. Compact representation (alphanumeric, no spaces)
    compact = "".join(tokens)

    # 10. Clean name without suffix
    name_no_suffix = None
    if name_no_suffix_raw:
        nns_lower = unicodedata.normalize("NFKC", name_no_suffix_raw).lower()
        nns_amp = re.sub(r"&", " and ", nns_lower)
        name_no_suffix = clean_punctuation(nns_amp)

    return {
        "raw": raw,
        "is_missing": False,
        "unicode_norm": u_norm,
        "lower": lowered,
        "ampersand_norm": amp_norm,
        "punct_norm": clean_str,
        "compact": compact,
        "alphanumeric": clean_str,
        "tokens": tokens,
        "token_set": token_set,
        "legal_suffix": canon_suffix,
        "has_legal_suffix": has_suffix,
        "name_no_suffix": name_no_suffix,
        "is_domain": domain_info["is_domain"],
        "domain_stem": domain_info["domain_stem"],
        "domain_tokens": domain_info["domain_tokens"],
        "script_type": script,
    }


# ---------------------------------------------------------------------------
# Address Preprocessing
# ---------------------------------------------------------------------------

DEFAULT_ADDRESS_ABBREVIATIONS: dict[str, str] = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bld": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "cir": "circle",
    "hwy": "highway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "pobox": "pobox",
}


def preprocess_address(
    text: Optional[str],
    abbreviations: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Create structured representations of a business address.

    Preserves numbers, street modifiers, and Unicode scripts while providing:
    * raw: original address string or None
    * is_missing: bool
    * unicode_norm: NFKC-normalized
    * lower: lowercased NFKC
    * punct_norm: punctuation-cleaned, numbers preserved
    * tokens: list of tokens in preserved original order
    * token_set: frozenset of tokens
    * sorted_tokens: tuple of tokens sorted alphabetically
    * normalized: clean address string in original order
    * order_independent: clean address string sorted by tokens
    * script_type: detected script

    Parameters
    ----------
    text:
        Raw address string.
    abbreviations:
        Optional custom dictionary of abbreviations to expand.
        Defaults to `DEFAULT_ADDRESS_ABBREVIATIONS`.

    Returns
    -------
    dict[str, Any]
    """
    if is_missing(text) or not text:
        return {
            "raw": None,
            "is_missing": True,
            "unicode_norm": None,
            "lower": None,
            "punct_norm": None,
            "tokens": [],
            "token_set": frozenset(),
            "sorted_tokens": (),
            "normalized": None,
            "order_independent": None,
            "script_type": "unknown",
        }

    raw = str(text)

    # 1. Unicode normalization (NFKC)
    u_norm = unicodedata.normalize("NFKC", raw).strip()

    # 2. Lowercase
    lowered = u_norm.lower()

    # 3. Handle 'p.o. box' / 'po box'
    lowered = re.sub(r"\bp\.?\s*o\.?\s*box\b", "pobox", lowered)

    # 4. Safe punctuation cleaning (letters, combining marks, numbers preserved)
    cleaned = clean_punctuation(lowered)

    # 5. Tokenization and abbreviation expansion
    abbr_map = (
        abbreviations if abbreviations is not None else DEFAULT_ADDRESS_ABBREVIATIONS
    )
    raw_tokens = cleaned.split()

    tokens: list[str] = []
    for t in raw_tokens:
        # Filter out standalone artifact null literals embedded in noisy addresses
        if t in ("null", "none", "nan"):
            continue
        t_norm = abbr_map.get(t, t)
        tokens.append(t_norm)

    if not tokens:
        return {
            "raw": raw,
            "is_missing": True,
            "unicode_norm": u_norm,
            "lower": lowered,
            "punct_norm": None,
            "tokens": [],
            "token_set": frozenset(),
            "sorted_tokens": (),
            "normalized": None,
            "order_independent": None,
            "script_type": detect_script(u_norm),
        }

    token_set = frozenset(tokens)
    sorted_tokens = tuple(sorted(tokens))
    norm_str = " ".join(tokens)
    order_indep = " ".join(sorted_tokens)

    return {
        "raw": raw,
        "is_missing": False,
        "unicode_norm": u_norm,
        "lower": lowered,
        "punct_norm": cleaned,
        "tokens": tokens,
        "token_set": token_set,
        "sorted_tokens": sorted_tokens,
        "normalized": norm_str,
        "order_independent": order_indep,
        "script_type": detect_script(u_norm),
    }


# ---------------------------------------------------------------------------
# Country Normalization
# ---------------------------------------------------------------------------

COUNTRY_CANONICAL_MAP: dict[str, str] = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "in": "India",
    "ind": "India",
    "india": "India",
    "fr": "France",
    "fra": "France",
    "france": "France",
    "uk": "United Kingdom",
    "gbr": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "de": "Germany",
    "deu": "Germany",
    "germany": "Germany",
    "ca": "Canada",
    "can": "Canada",
    "canada": "Canada",
}


def normalize_country(text: Optional[str]) -> str:
    """Normalise country name/code.

    Standardises common formatting differences for known countries
    (including US, India, and France which appears in TEST) while
    safely preserving arbitrary countries.

    Parameters
    ----------
    text:
        Raw country string.

    Returns
    -------
    str
        Canonicalised country name or empty string if input is missing.
    """
    if is_missing(text) or not text:
        return ""

    u = unicodedata.normalize("NFKC", str(text)).strip()
    key = u.lower()

    if key in COUNTRY_CANONICAL_MAP:
        return COUNTRY_CANONICAL_MAP[key]

    # For unknown/arbitrary countries, preserve whitespace-cleaned and title-cased
    if len(u) <= 3:
        return u.upper()
    return u.title()


# ---------------------------------------------------------------------------
# Backward-compatible convenience functions
# ---------------------------------------------------------------------------


def normalize_name(text: Optional[str]) -> str:
    """Convenience function returning the clean normalized business name string."""
    if is_missing(text) or not text:
        return ""
    res = preprocess_name(text)
    return res["punct_norm"] or ""


def normalize_address(text: Optional[str]) -> str:
    """Convenience function returning the clean normalized address string."""
    if is_missing(text) or not text:
        return ""
    res = preprocess_address(text)
    return res["normalized"] or ""


# ---------------------------------------------------------------------------
# Record-level / Chunk-level Preprocessing
# ---------------------------------------------------------------------------


def preprocess_records(
    df: pd.DataFrame,
    in_place: bool = False,
) -> pd.DataFrame:
    """Apply field normalisations to a DataFrame or chunk of records.

    Memory-conscious design: operates on NumPy arrays and list comprehensions
    for speed, avoiding expensive row-by-row iterrows / apply calls.

    Columns added:
    * `name_norm`: clean lowercase normalized name
    * `name_compact`: compact alphanumeric representation (no spaces)
    * `name_no_suffix`: name with legal suffix stripped
    * `legal_suffix`: canonical suffix (or None)
    * `has_legal_suffix`: boolean flag
    * `is_domain`: boolean flag
    * `domain_stem`: extracted domain stem (or None)
    * `name_script`: script tag ('latin', 'devanagari', etc.)
    * `address_norm`: normalized address string (or None)
    * `address_is_missing`: boolean flag
    * `address_order_indep`: sorted token string for order-independent matching
    * `address_script`: script tag
    * `country_norm`: normalized country string

    Parameters
    ----------
    df:
        DataFrame with at minimum columns:
        entity_id, business_name, business_address, country.
    in_place:
        If True, mutate *df* directly to conserve RAM in Colab.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized columns added.
    """
    if df is None:
        raise ValueError("Input DataFrame df cannot be None")

    target = df if in_place else df.copy()

    # Process business names
    if "business_name" in target.columns:
        name_vals = target["business_name"].fillna("").astype(str).values
        name_results = [preprocess_name(n) for n in name_vals]

        target["name_norm"] = [r["punct_norm"] for r in name_results]
        target["name_compact"] = [r["compact"] for r in name_results]
        target["name_no_suffix"] = [r["name_no_suffix"] for r in name_results]
        target["legal_suffix"] = [r["legal_suffix"] for r in name_results]
        target["has_legal_suffix"] = [r["has_legal_suffix"] for r in name_results]
        target["is_domain"] = [r["is_domain"] for r in name_results]
        target["domain_stem"] = [r["domain_stem"] for r in name_results]
        target["name_script"] = [r["script_type"] for r in name_results]

    # Process addresses
    if "business_address" in target.columns:
        addr_vals = target["business_address"].fillna("").astype(str).values
        addr_results = [preprocess_address(a) for a in addr_vals]

        target["address_norm"] = [r["normalized"] for r in addr_results]
        target["address_is_missing"] = [r["is_missing"] for r in addr_results]
        target["address_order_indep"] = [r["order_independent"] for r in addr_results]
        target["address_script"] = [r["script_type"] for r in addr_results]

    # Process countries
    if "country" in target.columns:
        c_vals = target["country"].fillna("").astype(str).values
        target["country_norm"] = [normalize_country(c) for c in c_vals]

    return target


# ---------------------------------------------------------------------------
# Streaming Chunked Preprocessing
# ---------------------------------------------------------------------------


def preprocess_file_chunked(
    input_path: Path | str,
    output_path: Optional[Path | str] = None,
    chunksize: int = 50000,
    sep: str = "\t",
) -> Generator[pd.DataFrame, None, None]:
    """Process a large TSV file incrementally in chunks.

    Adheres to the Colab memory constraint:
    TSV -> chunk -> preprocess chunk -> write/cache chunk -> release chunk

    Parameters
    ----------
    input_path:
        Path to input TSV file.
    output_path:
        Optional path to write preprocessed TSV chunks.
        If specified, chunks are appended incrementally.
    chunksize:
        Number of rows per chunk (default 50,000).
    sep:
        Delimiter (default tab '\\t').

    Yields
    ------
    pd.DataFrame
        Preprocessed chunk DataFrame.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    first_chunk = True
    reader = pd.read_csv(
        input_path,
        sep=sep,
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False,
    )

    for chunk in reader:
        processed_chunk = preprocess_records(chunk, in_place=True)

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if first_chunk else "a"
            header = first_chunk
            processed_chunk.to_csv(
                out_p,
                sep=sep,
                mode=mode,
                header=header,
                index=False,
            )
            first_chunk = False

        yield processed_chunk
