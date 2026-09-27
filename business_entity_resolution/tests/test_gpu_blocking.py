"""
Unit tests for GPU-accelerated and accuracy-optimized candidate blocking.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.gpu.gpu_blocking import (
    DiskGPUBlockingIndex,
    GPUBlockingConfig,
    GPUBlockingIndex,
    build_gpu_candidates,
    extract_core_domain,
    generate_gpu_blocking_keys,
)


def test_extract_core_domain() -> None:
    # Standard domain
    stem, comp = extract_core_domain("https://www.acme-corp.com/index.html")
    assert comp == "acmecorp"
    assert "acme" in stem

    # Multi-part TLD
    stem, comp = extract_core_domain("acme-corp.co.uk")
    assert comp == "acmecorp"

    # Leetspeak / custom
    stem, comp = extract_core_domain("kbmresearch.c0m")
    assert comp == "kbmresearch"

    # With ports and query params
    stem, comp = extract_core_domain("http://my-service.org:8080?user=123")
    assert comp == "myservice"

    # Pipe syntax
    stem, comp = extract_core_domain("M/s Basera Advisors Group | www.baseraadv.com")
    assert comp == "baseraadv"

    # Subdomain stripping
    stem, comp = extract_core_domain("shop.acme-corp.com")
    assert comp == "acmecorp"

    # Accented domain names
    stem, comp = extract_core_domain("HÍSTORICALCOMMITTEE.COM")
    assert comp == "historicalcommittee"

    # Ordinary business names should NOT be identified as domains
    stem, comp = extract_core_domain("General Electric Company")
    assert comp == ""
    assert stem == ""

    stem, comp = extract_core_domain("102 Main Street")
    assert comp == ""


def test_generate_gpu_blocking_keys() -> None:
    cfg = GPUBlockingConfig()
    record = {
        "entity_id": "S1-100",
        "business_name": "The Home Depot Inc",
        "name_norm": "the home depot inc",
        "name_compact": "thehomedepotinc",
        "name_no_suffix": "the home depot",
        "address_norm": "102 main street 94103 san francisco ca",
        "address_is_missing": False,
        "country_norm": "US",
    }
    keys = generate_gpu_blocking_keys(record, cfg)

    assert "A" in keys and len(keys["A"]) > 0
    assert "B" in keys and len(keys["B"]) > 0
    assert "C" in keys and len(keys["C"]) > 0
    assert "D" in keys and len(keys["D"]) > 0

    # Stopword prefix preservation
    all_b = " ".join(keys["B"])
    assert "b_lead_bi:the_home" in all_b or "b_tok:home" in all_b

    # Postal code key
    all_c = " ".join(keys["C"])
    assert "c_zip_w:94103" in all_c or "c_num_w:102" in all_c


def test_gpu_blocking_index_synthetic() -> None:
    cfg = GPUBlockingConfig(top_k_per_s1=10)
    index = GPUBlockingIndex(config=cfg)

    s2 = pd.DataFrame([
        {
            "entity_id": "S2-1",
            "business_name": "Acme Widgets Corp",
            "name_norm": "acme widgets corp",
            "name_compact": "acmewidgetscorp",
            "name_no_suffix": "acme widgets",
            "address_norm": "123 elm street",
            "address_is_missing": False,
            "country_norm": "US",
        }
    ])
    s3 = pd.DataFrame([
        {
            "entity_id": "S3-1",
            "business_name": "acmewidgets.com",
            "name_norm": "acmewidgets com",
            "name_compact": "acmewidgetscom",
            "name_no_suffix": "acmewidgets",
            "domain_stem": "acmewidgets",
            "is_domain": True,
            "address_norm": "123 elm st",
            "address_is_missing": False,
            "country_norm": "US",
        }
    ])

    index.build_vocabulary(s2, s3)
    index.add_records(s2)
    index.add_records(s3)

    s1 = pd.DataFrame([
        {
            "entity_id": "S1-1",
            "business_name": "Acme Widgets Inc",
            "name_norm": "acme widgets inc",
            "name_compact": "acmewidgetsinc",
            "name_no_suffix": "acme widgets",
            "address_norm": "123 elm street",
            "address_is_missing": False,
            "country_norm": "US",
        }
    ])

    cands = index.query_records(s1)
    cand_ids = set(cands["candidate_id"])
    assert "S2-1" in cand_ids
    assert "S3-1" in cand_ids


def test_disk_gpu_blocking_index_lifecycle() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        cfg = GPUBlockingConfig(top_k_per_s1=5)
        with DiskGPUBlockingIndex(db_path=db_path, config=cfg) as disk_idx:
            df = pd.DataFrame([
                {
                    "entity_id": "S2-10",
                    "business_name": "Beta Labs LLC",
                    "name_norm": "beta labs llc",
                    "name_compact": "betalabsllc",
                    "name_no_suffix": "beta labs",
                    "address_norm": "500 market st",
                    "address_is_missing": False,
                    "country_norm": "US",
                }
            ])
            disk_idx.add_records(df)
            disk_idx.finalize()

            s1 = pd.DataFrame([
                {
                    "entity_id": "S1-10",
                    "business_name": "Beta Labs",
                    "name_norm": "beta labs",
                    "name_compact": "betalabs",
                    "name_no_suffix": "beta labs",
                    "address_norm": "500 market street",
                    "address_is_missing": False,
                    "country_norm": "US",
                }
            ])
            cands = disk_idx.query_records(s1)
            assert len(cands) == 1
            assert cands.iloc[0]["candidate_id"] == "S2-10"
    finally:
        if db_path.exists():
            db_path.unlink(missing_ok=True)
