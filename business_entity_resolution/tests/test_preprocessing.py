"""
test_preprocessing.py — Comprehensive behavioral tests for preprocessing.py.

Covers:
- Case normalization (uppercase, lowercase, mixed case)
- Punctuation removal and safe normalization
- Repeated whitespace collapsing
- Unicode and accented characters (French accents: é, à, ü, ç, etc.)
- Non-Latin scripts (Devanagari, Tamil, Kannada, Gujarati)
- Ampersand and symbol normalization (& -> and)
- Legal suffix detection and canonicalization (US, India, France, split, prefix, Indic)
- Website and domain-like business names
- Missing and null-like values (None, NaN, '', 'null', '<null>', 'none')
- Address number and alphanumeric preservation
- Order-independent address token representation
- Country normalization (US, India, France, arbitrary countries)
- Script detection
- DataFrame / chunk-level record preprocessing
"""

from __future__ import annotations

import math
import pandas as pd
import pytest

from src.preprocessing import (
    clean_punctuation,
    detect_script,
    extract_legal_suffix,
    is_missing,
    normalize_address,
    normalize_country,
    normalize_name,
    parse_domain_name,
    preprocess_address,
    preprocess_name,
    preprocess_records,
)


# ---------------------------------------------------------------------------
# 1. Null / Missing Value Detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "val",
    [
        None,
        float("nan"),
        "",
        "   ",
        "null",
        "NULL",
        "Null",
        "<null>",
        "<NULL>",
        "none",
        "NONE",
        "nan",
        "NaN",
        "N/A",
        "n/a",
        "na",
        "NA",
        "undefined",
        ".",
    ],
)
def test_is_missing_true(val):
    assert is_missing(val) is True


@pytest.mark.parametrize(
    "val",
    [
        "Acme",
        "123 Main St",
        "US",
        "0",
        "None of the above LLC",
    ],
)
def test_is_missing_false(val):
    assert is_missing(val) is False


# ---------------------------------------------------------------------------
# 2. Script Detection
# ---------------------------------------------------------------------------


def test_detect_script_latin():
    assert detect_script("Orelee's Barbershop") == "latin"
    assert detect_script("SCI Ptit Àmicale") == "latin"
    assert detect_script("B+ Retail Inc") == "latin"


def test_detect_script_devanagari():
    assert detect_script("राम मार्केटिंग प्राइवेट लिमिटेड") == "devanagari"
    assert detect_script("आदित्य प्रॉपर्टीज एलएलपी") == "devanagari"


def test_detect_script_tamil():
    assert detect_script("குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்") == "tamil"


def test_detect_script_kannada():
    assert detect_script("ಕರ್ನಾಟಕ") == "kannada"


def test_detect_script_gujarati():
    assert detect_script("શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ") == "gujarati"


def test_detect_script_mixed():
    assert detect_script("राम Marketing Private Limited") == "mixed"


def test_detect_script_missing_or_numbers():
    assert detect_script(None) == "unknown"
    assert detect_script("") == "unknown"
    assert detect_script("12345 67890") == "unknown"


# ---------------------------------------------------------------------------
# 3. Business Name Normalization: Case, Punctuation, Whitespace, & Unicode
# ---------------------------------------------------------------------------


def test_name_case_normalization():
    res1 = preprocess_name("ACME ENTERPRISES")
    res2 = preprocess_name("acme enterprises")
    res3 = preprocess_name("AcMe EnTeRpRiSeS")

    assert res1["lower"] == "acme enterprises"
    assert res1["punct_norm"] == res2["punct_norm"] == res3["punct_norm"] == "acme enterprises"


def test_name_repeated_whitespace():
    res = preprocess_name("   Acme     Enterprises    LLC   ")
    assert res["punct_norm"] == "acme enterprises llc"
    assert res["tokens"] == ["acme", "enterprises", "llc"]


def test_name_punctuation_cleaning():
    res = preprocess_name("-- Holloway Peak, Inc. (Seafood) <<")
    assert "holloway peak" in res["punct_norm"]
    assert "-" not in res["punct_norm"]
    assert "<" not in res["punct_norm"]
    assert "(" not in res["punct_norm"]


def test_name_ampersand_normalization():
    res = preprocess_name("Thermal & Fils SASU")
    assert "thermal and fils" in res["punct_norm"]
    assert res["legal_suffix"] == "sasu"


def test_name_compact_representation():
    res = preprocess_name("Orelee's Barbershop")
    assert res["compact"] == "oreleesbarbershop"
    assert res["tokens"] == ["orelee", "s", "barbershop"] or "orelees" in res["compact"]


def test_name_preserves_french_accents():
    res = preprocess_name("Marina École France Sarl")
    assert "école" in res["punct_norm"]
    assert res["legal_suffix"] == "sarl"
    assert res["script_type"] == "latin"


def test_name_preserves_indic_scripts():
    res = preprocess_name("सन कंस्ट्रक्शंस प्राइवेट लिमिटेड")
    assert res["script_type"] == "devanagari"
    assert "सन" in res["punct_norm"]
    assert "कंस्ट्रक्शंस" in res["punct_norm"]
    assert res["legal_suffix"] == "pvt_ltd"
    assert res["has_legal_suffix"] is True
    assert res["name_no_suffix"] == "सन कंस्ट्रक्शंस"


def test_name_missing_values():
    for val in [None, "", "null", "<NULL>", float("nan")]:
        res = preprocess_name(val)
        assert res["is_missing"] is True
        assert res["punct_norm"] is None
        assert res["tokens"] == []
        assert normalize_name(val) == ""


# ---------------------------------------------------------------------------
# 4. Legal Suffix Handling
# ---------------------------------------------------------------------------


def test_legal_suffix_us_standard():
    canon, has_suf, clean = extract_legal_suffix("Orelee's Barbershop LLC")
    assert canon == "llc"
    assert has_suf is True
    assert clean == "Orelee's Barbershop"

    canon, has_suf, clean = extract_legal_suffix("Alpha Corp.")
    assert canon == "corp"
    assert has_suf is True
    assert clean == "Alpha"

    canon, has_suf, clean = extract_legal_suffix("Summit Inc")
    assert canon == "inc"
    assert has_suf is True
    assert clean == "Summit"


def test_legal_suffix_at_start():
    canon, has_suf, clean = extract_legal_suffix("LLC Moncada Learning Center")
    assert canon == "llc"
    assert has_suf is True
    assert clean == "Moncada Learning Center"

    canon, has_suf, clean = extract_legal_suffix("SCI Ptit Àmicale")
    assert canon == "sci"
    assert has_suf is True
    assert clean == "Ptit Àmicale"


def test_legal_suffix_split():
    canon, has_suf, clean = extract_legal_suffix("Pvt. EFS Print Ventures Ltd.")
    assert canon == "pvt_ltd"
    assert has_suf is True
    assert clean == "EFS Print Ventures"

    canon, has_suf, clean = extract_legal_suffix("Private Ambernath Solar Limited")
    assert canon == "pvt_ltd"
    assert has_suf is True
    assert clean == "Ambernath Solar"


def test_legal_suffix_in_middle():
    canon, has_suf, clean = extract_legal_suffix("-- Holloway Peak Inc Seafood")
    assert canon == "inc"
    assert has_suf is True
    assert "Holloway Peak" in clean
    assert "Seafood" in clean


def test_legal_suffix_french():
    canon, has_suf, clean = extract_legal_suffix("ZNB Club SARL")
    assert canon == "sarl"
    assert clean == "ZNB Club"

    canon, has_suf, clean = extract_legal_suffix("Thermal & Fils SASU")
    assert canon == "sasu"
    assert clean == "Thermal & Fils"


def test_legal_suffix_devanagari_and_indic():
    canon, has_suf, clean = extract_legal_suffix("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert canon == "pvt_ltd"
    assert clean == "राम मार्केटिंग"

    canon, has_suf, clean = extract_legal_suffix("ग्लोबल इन्वेस्टमेंट प्रा. लि.")
    assert canon == "pvt_ltd"
    assert clean == "ग्लोबल इन्वेस्टमेंट"

    canon, has_suf, clean = extract_legal_suffix("आदित्य प्रॉपर्टीज एलएलपी")
    assert canon == "llp"
    assert clean == "आदित्य प्रॉपर्टीज"


def test_legal_suffix_absent():
    canon, has_suf, clean = extract_legal_suffix("Christ Chapel")
    assert canon is None
    assert has_suf is False
    assert clean == "Christ Chapel"


# ---------------------------------------------------------------------------
# 5. Website / Domain-Like Business Names
# ---------------------------------------------------------------------------


def test_domain_name_pure():
    res = parse_domain_name("wilfordhancock.com")
    assert res["is_domain"] is True
    assert res["domain_stem"] == "wilfordhancock"
    assert "wilfordhancock" in res["domain_tokens"]


def test_domain_name_url_with_protocol():
    res = parse_domain_name("https://www.example-company.com/about")
    assert res["is_domain"] is True
    assert res["domain_stem"] == "example-company"
    assert res["domain_tokens"] == ["example", "company"]


def test_domain_name_pipe_syntax():
    res = parse_domain_name("M/s Basera Advisors  Group | www.baseraadv.com")
    assert res["is_domain"] is True
    assert res["domain_stem"] == "baseraadv"
    assert "M/s Basera Advisors" in res["name_without_domain_url"]


def test_domain_name_not_domain():
    res = parse_domain_name("Standard Business Solutions")
    assert res["is_domain"] is False
    assert res["domain_stem"] is None
    assert res["domain_tokens"] == []


# ---------------------------------------------------------------------------
# 6. Address Preprocessing
# ---------------------------------------------------------------------------


def test_address_preserves_numbers():
    res = preprocess_address("1795 Westchester Drive, High Point, NC")
    assert "1795" in res["tokens"]
    assert "drive" in res["tokens"]
    assert "nc" in res["tokens"]


def test_address_abbreviation_expansion():
    res = preprocess_address("105 ELM ST, MORGANTON, NC")
    assert "street" in res["tokens"]
    assert "st" not in res["tokens"]

    res_blvd = preprocess_address("175 Blvd du Président")
    assert "boulevard" in res_blvd["tokens"]


def test_address_filters_embedded_null_literals():
    res = preprocess_address("067 PRODUCTION CT, NULL, INDEPENDENCE, KY")
    assert "null" not in res["tokens"]
    assert "court" in res["tokens"]
    assert "independence" in res["tokens"]

    res_bracket = preprocess_address("##16978 Moore Rd, <NULL>, Andalusia, Alabama")
    assert "null" not in res_bracket["tokens"]
    assert "road" in res_bracket["tokens"]
    assert "16978" in res_bracket["tokens"]


def test_address_order_independent_representation():
    a1 = preprocess_address("630 45th Terrace, Kansas City, MO")
    a2 = preprocess_address("KANSAS CITY, MO, 630 45TH TERRACE, null")

    # Tokens set should match despite reordering
    assert a1["token_set"] == a2["token_set"]
    # Order-independent sorted string should be identical
    assert a1["order_independent"] == a2["order_independent"]


def test_address_preserves_kannada_and_indic():
    addr = "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ"
    res = preprocess_address(addr)
    assert res["is_missing"] is False
    assert "ಕರ್ನಾಟಕ" in res["normalized"]
    assert "183" in res["tokens"]
    assert res["script_type"] == "mixed" or "kannada" in res["script_type"]


def test_address_missing_values():
    for val in [None, "", "null", "<null>", "NULL", float("nan"), "none"]:
        res = preprocess_address(val)
        assert res["is_missing"] is True
        assert res["normalized"] is None
        assert res["tokens"] == []
        assert normalize_address(val) == ""


# ---------------------------------------------------------------------------
# 7. Country Normalization
# ---------------------------------------------------------------------------


def test_country_us_variants():
    for val in ["US", "us", "USA", "usa", "United States", " united states "]:
        assert normalize_country(val) == "US"


def test_country_india_variants():
    for val in ["India", "india", "INDIA", "IN", "ind"]:
        assert normalize_country(val) == "India"


def test_country_france():
    for val in ["France", "france", "FR", "fra"]:
        assert normalize_country(val) == "France"


def test_country_arbitrary():
    assert normalize_country("Germany") == "Germany"
    assert normalize_country("brazil") == "Brazil"
    assert normalize_country("JPN") == "JPN"


def test_country_missing():
    for val in [None, "", "null", "<null>", float("nan")]:
        assert normalize_country(val) == ""


# ---------------------------------------------------------------------------
# 8. Record-Level DataFrame Preprocessing
# ---------------------------------------------------------------------------


def test_preprocess_records_dataframe():
    df = pd.DataFrame(
        {
            "entity_id": ["E1", "E2", "E3", "E4"],
            "business_name": [
                "Orelee's Barbershop LLC",
                "राम मार्केटिंग प्राइवेट लिमिटेड",
                "wilfordhancock.com",
                None,
            ],
            "business_address": [
                "1795 Westchester Drive, High Point, NC",
                "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
                None,
                "<null>",
            ],
            "country": ["US", "India", "US", "France"],
        }
    )

    processed = preprocess_records(df)

    # Check that new columns are generated
    assert "name_norm" in processed.columns
    assert "name_compact" in processed.columns
    assert "name_no_suffix" in processed.columns
    assert "legal_suffix" in processed.columns
    assert "has_legal_suffix" in processed.columns
    assert "is_domain" in processed.columns
    assert "name_script" in processed.columns
    assert "address_norm" in processed.columns
    assert "address_is_missing" in processed.columns
    assert "address_order_indep" in processed.columns
    assert "country_norm" in processed.columns

    # Check row 0 (US with LLC)
    assert processed.loc[0, "legal_suffix"] == "llc"
    assert bool(processed.loc[0, "has_legal_suffix"]) is True
    assert processed.loc[0, "country_norm"] == "US"
    assert bool(processed.loc[0, "address_is_missing"]) is False

    # Check row 1 (Devanagari)
    assert processed.loc[1, "name_script"] == "devanagari"
    assert processed.loc[1, "legal_suffix"] == "pvt_ltd"
    assert processed.loc[1, "country_norm"] == "India"

    # Check row 2 (Domain)
    assert bool(processed.loc[2, "is_domain"]) is True
    assert processed.loc[2, "domain_stem"] == "wilfordhancock"
    assert bool(processed.loc[2, "address_is_missing"]) is True

    # Check row 3 (Missing name and address, France country)
    assert bool(processed.loc[3, "address_is_missing"]) is True
    assert processed.loc[3, "country_norm"] == "France"
