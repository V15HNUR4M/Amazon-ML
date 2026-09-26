"""
benchmark_preprocessing.py — Benchmark preprocessing performance and inspect examples.

Measures:
- Wall-clock runtime
- Memory allocation (via tracemalloc)
- Processing throughput (rows / second)
- Before / after transformations on real dataset samples

Usage:
------
python scripts/benchmark_preprocessing.py --sample-size 1000
python scripts/benchmark_preprocessing.py --sample-size 5000
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import Config
from src.preprocessing import (
    detect_script,
    extract_legal_suffix,
    normalize_country,
    parse_domain_name,
    preprocess_address,
    preprocess_name,
    preprocess_records,
)


def run_benchmark(sample_size: int = 1000) -> None:
    cfg = Config()

    print("=" * 80)
    print(f"PREPROCESSING BENCHMARK — SAMPLE SIZE: {sample_size} ROWS PER SOURCE")
    print("=" * 80)

    sources = [
        ("Train Source 1 (US + India)", cfg.TRAIN_SOURCE1),
        ("Train Source 2 (Missing Addr + Indic)", cfg.TRAIN_SOURCE2),
        ("Train Source 3 (Domains + Noise)", cfg.TRAIN_SOURCE3),
        ("Test Source 1 (US + India + France)", cfg.TEST_SOURCE1),
    ]

    total_rows = 0
    total_time = 0.0

    print(f"\n{'Source':<38} | {'Rows':<8} | {'Time (s)':<10} | {'Throughput (r/s)':<18} | {'Peak RAM (MB)':<12}")
    print("-" * 94)

    processed_dfs: dict[str, pd.DataFrame] = {}

    for label, path in sources:
        if not path.exists():
            print(f"Skipping {label}: file not found at {path}")
            continue

        df_sample = pd.read_csv(
            path,
            sep="\t",
            nrows=sample_size,
            dtype=str,
            keep_default_na=False,
        )

        tracemalloc.start()
        t0 = time.perf_counter()

        processed = preprocess_records(df_sample, in_place=False)

        t1 = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = t1 - t0
        rows = len(df_sample)
        throughput = rows / elapsed if elapsed > 0 else 0
        peak_mb = peak_mem / (1024 * 1024)

        total_rows += rows
        total_time += elapsed
        processed_dfs[label] = processed

        print(
            f"{label:<38} | {rows:<8} | {elapsed:<10.4f} | {throughput:<18.0f} | {peak_mb:<12.2f}"
        )

    print("-" * 94)
    overall_throughput = total_rows / total_time if total_time > 0 else 0
    print(
        f"{'TOTAL / OVERALL':<38} | {total_rows:<8} | {total_time:<10.4f} | {overall_throughput:<18.0f} |"
    )

    print("\n" + "=" * 80)
    print("REPRESENTATIVE BEFORE / AFTER EXAMPLES FROM REAL DATA")
    print("=" * 80)

    # 1. Non-Latin text examples
    print("\n[A] Non-Latin Script Examples (Devanagari, Tamil, Kannada, Accents):")
    print("-" * 80)
    s2_df = processed_dfs.get("Train Source 2 (Missing Addr + Indic)")
    if s2_df is not None:
        non_latin = s2_df[s2_df["name_script"] != "latin"].head(4)
        for _, row in non_latin.iterrows():
            print(f"  Entity ID    : {row['entity_id']}")
            print(f"  Raw Name     : {row['business_name']}")
            print(f"  Script       : {row['name_script']}")
            print(f"  Norm Name    : {row['name_norm']}")
            print(f"  Compact      : {row['name_compact']}")
            print(f"  Legal Suffix : {row['legal_suffix']} (has_suffix={row['has_legal_suffix']})")
            print(f"  No-Suffix    : {row['name_no_suffix']}")
            print(f"  Address Norm : {row['address_norm']}")
            print()

    # 2. Missing address examples
    print("\n[B] Missing Address Examples (NaN, literal null, empty):")
    print("-" * 80)
    s3_df = processed_dfs.get("Train Source 3 (Domains + Noise)")
    if s3_df is not None:
        missing_addr = s3_df[s3_df["address_is_missing"]].head(3)
        for _, row in missing_addr.iterrows():
            print(f"  Entity ID      : {row['entity_id']}")
            print(f"  Raw Name       : {row['business_name']}")
            print(f"  Raw Address    : {row['business_address']!r}")
            print(f"  Addr Is Missing: {row['address_is_missing']}")
            print(f"  Address Norm   : {row['address_norm']!r}")
            print()

    # 3. Domain-like name examples
    print("\n[C] Domain-Like Name Examples (Websites in Source 3):")
    print("-" * 80)
    if s3_df is not None:
        domain_rows = s3_df[s3_df["is_domain"]].head(4)
        for _, row in domain_rows.iterrows():
            print(f"  Entity ID    : {row['entity_id']}")
            print(f"  Raw Name     : {row['business_name']}")
            print(f"  Is Domain    : {row['is_domain']}")
            print(f"  Domain Stem  : {row['domain_stem']}")
            print(f"  Norm Name    : {row['name_norm']}")
            print(f"  Compact      : {row['name_compact']}")
            print()

    # 4. Legal suffix variations (split, start, middle, end)
    print("\n[D] Legal Suffix Variations (End, Start, Split, Middle):")
    print("-" * 80)
    examples = [
        "Orelee's Barbershop LLC",
        "LLC Moncada Learning Center",
        "Pvt. EFS Print Ventures Ltd.",
        "Private Ambernath Solar Limited",
        "-- Holloway Peak Inc Seafood",
        "Thermal & Fils SASU",
        "ZNB Club SARL",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
    ]
    for ex in examples:
        canon, has_suf, clean = extract_legal_suffix(ex)
        print(f"  {ex:<35} -> Suffix: {str(canon):<10} | Clean: {clean}")

    # 5. French records in Test Source 1
    print("\n[E] French Records in Test Source 1:")
    print("-" * 80)
    test_df = processed_dfs.get("Test Source 1 (US + India + France)")
    if test_df is not None:
        french_rows = test_df[test_df["country_norm"] == "France"].head(3)
        for _, row in french_rows.iterrows():
            print(f"  Entity ID    : {row['entity_id']}")
            print(f"  Raw Name     : {row['business_name']}")
            print(f"  Norm Name    : {row['name_norm']}")
            print(f"  Suffix       : {row['legal_suffix']}")
            print(f"  Raw Address  : {row['business_address']}")
            print(f"  Address Norm : {row['address_norm']}")
            print(f"  Country Norm : {row['country_norm']}")
            print()

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark business entity preprocessing.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of rows per source to benchmark (default: 1000).",
    )
    args = parser.parse_args()
    run_benchmark(sample_size=args.sample_size)
