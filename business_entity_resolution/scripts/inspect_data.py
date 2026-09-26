"""
inspect_data.py — Real dataset inspection for the Business Entity Resolution challenge.

Processes ONE file at a time to stay memory-efficient with 2.4 GB of data.

Run from the project root:
    python scripts/inspect_data.py
"""

from __future__ import annotations

import io
import sys

# Force UTF-8 output so non-Latin characters don't crash on Windows cp1252
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='replace',
        line_buffering=True,
    )

from pathlib import Path

# Make src/ importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import Config

SEP = "=" * 72
SEP2 = "-" * 72


def header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def subheader(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


def load_tsv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  *** FILE NOT FOUND: {path} ***")
        return None
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    return df


def _safe(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "<null>"
    return str(val)


def inspect_source_file(path: Path, label: str) -> dict:
    """Load a source file, compute stats, return a summary dict. Free memory after."""
    print(f"  Loading {label} ... ", end="", flush=True)
    df = load_tsv(path)
    if df is None:
        return {}

    n_rows = len(df)
    n_cols = len(df.columns)
    cols = list(df.columns)
    print(f"{n_rows:,} rows")

    # Missing values
    missing = {}
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col in df.columns:
            n = df[col].isna().sum()
            missing[col] = (n, n / n_rows * 100)

    # Duplicates
    id_col = "entity_id"
    n_unique = df[id_col].nunique() if id_col in df.columns else None
    n_dup = n_rows - n_unique if n_unique is not None else None

    # String statistics
    str_stats = {}
    for col in ["business_name", "business_address"]:
        if col in df.columns:
            lengths = df[col].dropna().str.len()
            if len(lengths) > 0:
                str_stats[col] = {
                    "count": len(lengths),
                    "min": int(lengths.min()),
                    "median": float(lengths.median()),
                    "mean": float(lengths.mean()),
                    "max": int(lengths.max()),
                    "le3": int((lengths <= 3).sum()),
                }

    # Country distribution
    country_vc = {}
    if "country" in df.columns:
        country_vc = df["country"].fillna("<null>").value_counts().to_dict()

    # Sample records (first 7)
    sample_rows = []
    for _, row in df.head(7).iterrows():
        sample_rows.append({c: _safe(row.get(c)) for c in cols})

    del df
    return {
        "label": label,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "cols": cols,
        "missing": missing,
        "n_unique_ids": n_unique,
        "n_dup_ids": n_dup,
        "str_stats": str_stats,
        "country_vc": country_vc,
        "sample_rows": sample_rows,
    }


def inspect_ground_truth(path: Path) -> dict:
    print(f"  Loading ground_truth ... ", end="", flush=True)
    df = load_tsv(path)
    if df is None:
        return {}

    n_rows = len(df)
    print(f"{n_rows:,} rows")

    # Missing
    missing = {}
    for col in ["source1_entity_id", "matched_entity_ids"]:
        if col in df.columns:
            n = df[col].isna().sum()
            missing[col] = (n, n / n_rows * 100)

    # Duplicates on source1_entity_id
    n_unique = df["source1_entity_id"].nunique()
    n_dup = n_rows - n_unique

    # Match cardinality
    def count_matches(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0
        return len(str(val).split(","))

    df["n_matches"] = df["matched_entity_ids"].apply(count_matches)
    cardinality_vc = df["n_matches"].value_counts().sort_index().to_dict()
    n_zero = (df["n_matches"] == 0).sum()
    n_one  = (df["n_matches"] == 1).sum()
    n_multi = (df["n_matches"] > 1).sum()
    max_matches = int(df["n_matches"].max())

    # Source 2 vs 3 distribution
    has_s2 = []
    has_s3 = []
    s2_total = 0
    s3_total = 0
    for val in df["matched_entity_ids"]:
        if pd.isna(val) or str(val).strip() == "":
            has_s2.append(False); has_s3.append(False)
            continue
        ids = [x.strip() for x in str(val).split(",")]
        hs2 = any(i.startswith("S2-") for i in ids)
        hs3 = any(i.startswith("S3-") for i in ids)
        has_s2.append(hs2); has_s3.append(hs3)
        s2_total += sum(1 for i in ids if i.startswith("S2-"))
        s3_total += sum(1 for i in ids if i.startswith("S3-"))

    src_df = pd.DataFrame({"has_s2": has_s2, "has_s3": has_s3})
    only_s2 = int(((src_df["has_s2"]) & (~src_df["has_s3"])).sum())
    only_s3 = int(((~src_df["has_s2"]) & (src_df["has_s3"])).sum())
    both    = int(((src_df["has_s2"]) & (src_df["has_s3"])).sum())
    neither = int(((~src_df["has_s2"]) & (~src_df["has_s3"])).sum())

    # Sample rows
    sample_rows = []
    for _, row in df.head(7).iterrows():
        sample_rows.append({
            "source1_entity_id": _safe(row.get("source1_entity_id")),
            "matched_entity_ids": _safe(row.get("matched_entity_ids")),
        })

    del df

    return {
        "n_rows": n_rows,
        "n_cols": 2,
        "cols": ["source1_entity_id", "matched_entity_ids"],
        "missing": missing,
        "n_unique_ids": n_unique,
        "n_dup_ids": n_dup,
        "cardinality_vc": cardinality_vc,
        "n_zero": int(n_zero),
        "n_one": int(n_one),
        "n_multi": int(n_multi),
        "max_matches": max_matches,
        "only_s2": only_s2,
        "only_s3": only_s3,
        "both_s2_s3": both,
        "neither": neither,
        "s2_total": s2_total,
        "s3_total": s3_total,
        "sample_rows": sample_rows,
    }


def get_noise_examples(cfg: Config, n: int = 8) -> list[dict]:
    """Load GT + all train sources to build noise examples. One pass each."""
    print("  Loading files for noise examples ...", flush=True)
    gt = load_tsv(cfg.TRAIN_GROUND_TRUTH)
    s1 = load_tsv(cfg.TRAIN_SOURCE1)
    s2 = load_tsv(cfg.TRAIN_SOURCE2)
    s3 = load_tsv(cfg.TRAIN_SOURCE3)

    if any(x is None for x in [gt, s1, s2, s3]):
        return []

    def build_lookup(df: pd.DataFrame) -> dict:
        return df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")

    lk1 = build_lookup(s1)
    lk2 = build_lookup(s2)
    lk3 = build_lookup(s3)
    del s1, s2, s3

    examples = []
    for _, row in gt.iterrows():
        if len(examples) >= n:
            break
        s1_id = row["source1_entity_id"]
        matched_raw = row["matched_entity_ids"]
        if pd.isna(matched_raw) or str(matched_raw).strip() == "":
            continue
        if s1_id not in lk1:
            continue

        s1_rec = lk1[s1_id]
        ex = {
            "s1_id": s1_id,
            "s1": {k: _safe(v) for k, v in s1_rec.items()},
            "matches": [],
        }
        for mid in str(matched_raw).split(","):
            mid = mid.strip()
            if mid.startswith("S2-"):
                rec = lk2.get(mid)
                src = "S2"
            elif mid.startswith("S3-"):
                rec = lk3.get(mid)
                src = "S3"
            else:
                continue
            if rec:
                ex["matches"].append({"id": mid, "src": src, "rec": {k: _safe(v) for k, v in rec.items()}})
        if ex["matches"]:
            examples.append(ex)

    del gt, lk1, lk2, lk3
    return examples


def print_source_report(info: dict) -> None:
    label = info.get("label", "?")
    n = info.get("n_rows", 0)
    subheader(label)
    print(f"  rows    : {n:,}")
    print(f"  columns : {info.get('n_cols')}")
    print(f"  names   : {info.get('cols')}")

    print(f"\n  -- Missing values --")
    for col, (n_miss, pct) in info.get("missing", {}).items():
        print(f"  {col:<25} missing={n_miss:>8,}  ({pct:5.2f}%)")

    print(f"\n  -- Duplicates (entity_id) --")
    print(f"  unique IDs : {info.get('n_unique_ids'):,}")
    print(f"  duplicated : {info.get('n_dup_ids'):,}")

    print(f"\n  -- String statistics --")
    for col, st in info.get("str_stats", {}).items():
        print(f"  {col}")
        print(f"    count (non-null) : {st['count']:,}")
        print(f"    min len          : {st['min']}")
        print(f"    median len       : {st['median']:.1f}")
        print(f"    mean len         : {st['mean']:.1f}")
        print(f"    max len          : {st['max']}")
        print(f"    len ≤ 3          : {st['le3']:,}  ({st['le3']/n*100:.2f}%)")

    print(f"\n  -- Country distribution --")
    total = n
    for country, cnt in sorted(info.get("country_vc", {}).items(), key=lambda x: -x[1]):
        print(f"  {str(country):<20} {cnt:>8,}  ({cnt/total*100:5.2f}%)")

    print(f"\n  -- Sample rows --")
    for i, row in enumerate(info.get("sample_rows", []), 1):
        print(f"  [{i}] " + " | ".join(f"{k}={v[:60]!r}" for k, v in row.items()))


def main() -> None:
    print(SEP)
    print("  Amazon ML Hackathon — Business Entity Resolution")
    print("  Dataset Inspection Report  (memory-efficient, one file at a time)")
    print(SEP)

    cfg = Config()

    # ---- A: File Discovery ----
    header("A. FILE DISCOVERY")
    files = {
        "train_source1": cfg.TRAIN_SOURCE1,
        "train_source2": cfg.TRAIN_SOURCE2,
        "train_source3": cfg.TRAIN_SOURCE3,
        "train_ground_truth": cfg.TRAIN_GROUND_TRUTH,
        "test_source1": cfg.TEST_SOURCE1,
        "test_source2": cfg.TEST_SOURCE2,
        "test_source3": cfg.TEST_SOURCE3,
    }
    for name, path in files.items():
        status = "FOUND" if path.exists() else "MISSING"
        size_kb = f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else "n/a"
        print(f"  [{status}] {name:<25} {size_kb:>12}   {path}")

    # ---- B-G: Source file stats (one at a time) ----
    header("B–G. PER-FILE STATS (one file at a time)")

    source_infos = {}
    print()
    for name, path in [
        ("train_source1", cfg.TRAIN_SOURCE1),
        ("train_source2", cfg.TRAIN_SOURCE2),
        ("train_source3", cfg.TRAIN_SOURCE3),
        ("test_source1",  cfg.TEST_SOURCE1),
        ("test_source2",  cfg.TEST_SOURCE2),
        ("test_source3",  cfg.TEST_SOURCE3),
    ]:
        info = inspect_source_file(path, name)
        source_infos[name] = info
        print_source_report(info)

    # ---- H-I: Ground truth ----
    header("H–I. GROUND TRUTH ANALYSIS")
    print()
    gt_info = inspect_ground_truth(cfg.TRAIN_GROUND_TRUTH)
    n = gt_info["n_rows"]

    subheader("Ground truth dimensions")
    print(f"  rows    : {n:,}")
    print(f"  columns : {gt_info['cols']}")
    for col, (nm, pct) in gt_info["missing"].items():
        print(f"  {col:<25} missing={nm:>8,}  ({pct:5.2f}%)")
    print(f"  unique source1_entity_id : {gt_info['n_unique_ids']:,}")
    print(f"  duplicated               : {gt_info['n_dup_ids']:,}")

    subheader("H. Match cardinality (train)")
    print(f"\n  {'# matches':<12} {'count':>8}  {'%':>8}")
    print(f"  {'-'*34}")
    for k, v in gt_info["cardinality_vc"].items():
        print(f"  {k:<12} {v:>8,}  ({v/n*100:7.2f}%)")

    print()
    print(f"  Singletons  (0 matches): {gt_info['n_zero']:,}  ({gt_info['n_zero']/n*100:.2f}%)")
    print(f"  Exactly one (1 match)  : {gt_info['n_one']:,}  ({gt_info['n_one']/n*100:.2f}%)")
    print(f"  Multiple    (>1 match) : {gt_info['n_multi']:,}  ({gt_info['n_multi']/n*100:.2f}%)")
    print(f"  Max matches (any S1)   : {gt_info['max_matches']}")

    subheader("I. Source 2 vs Source 3 distribution")
    print(f"\n  Total S1 entities          : {n:,}")
    print(f"  Only S2 matches            : {gt_info['only_s2']:,}  ({gt_info['only_s2']/n*100:.2f}%)")
    print(f"  Only S3 matches            : {gt_info['only_s3']:,}  ({gt_info['only_s3']/n*100:.2f}%)")
    print(f"  Both S2 and S3 matches     : {gt_info['both_s2_s3']:,}  ({gt_info['both_s2_s3']/n*100:.2f}%)")
    print(f"  No matches (singletons)    : {gt_info['neither']:,}  ({gt_info['neither']/n*100:.2f}%)")
    print(f"\n  Total S2 match references  : {gt_info['s2_total']:,}")
    print(f"  Total S3 match references  : {gt_info['s3_total']:,}")

    subheader("Ground truth sample rows")
    for row in gt_info["sample_rows"]:
        print(f"  {row['source1_entity_id']:<20} -> {row['matched_entity_ids'][:100]}")

    # ---- J: Noise examples ----
    header("J. ACTUAL NOISE EXAMPLES (matched pairs side by side)")
    print()
    examples = get_noise_examples(cfg, n=8)

    if not examples:
        print("  No examples found.")
    else:
        for i, ex in enumerate(examples, 1):
            print(f"\n  === Example {i} === (S1: {ex['s1_id']})")
            print(f"  SOURCE 1:")
            for k, v in ex["s1"].items():
                print(f"    {k:<20} : {v}")
            for m in ex["matches"]:
                print(f"  {m['src']} ({m['id']}):")
                for k, v in m["rec"].items():
                    print(f"    {k:<20} : {v}")
            print(f"  {'-'*60}")

    header("INSPECTION COMPLETE")
    print("  All sections done. No modeling decisions were made.")
    print(SEP)


if __name__ == "__main__":
    main()
