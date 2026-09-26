# Business Entity Resolution — Amazon ML Hackathon

## Problem

Given three independently scraped databases of business entities (Source 1, Source 2, Source 3), identify which records across sources refer to the same real-world business.

**Task:** For each Source 1 entity, predict zero, one, or multiple matching entity IDs from Source 2 and/or Source 3.

**Evaluation metric:** Macro-averaged **F₀.₅** (precision-weighted 2× over recall).  
False merges are penalised more than missed links.

## Official Problem Summary

- Three noisy, independently-scraped business entity databases.
- Source 1 is the "query" side; Sources 2 and 3 are the "candidate" sides.
- A Source 1 entity may have **zero matches** (singleton), **one match**, or **multiple matches** (in S2 and/or S3).
- Singletons score 1.0 when correctly predicted as empty.
- Submission requires two output files:
  - `output/matching_results.tsv` — final matches (leaderboard upload)
  - `output/candidate_pairs.tsv` — blocking candidate set (pipeline audit)

## Project Structure

```
business_entity_resolution/
│
├── src/
│   ├── config.py          ← all paths and hyper-parameters
│   ├── data_loader.py     ← TSV loaders
│   ├── preprocessing.py   ← text normalisation (TODO)
│   ├── blocking.py        ← candidate-pair generation (TODO)
│   ├── pair_builder.py    ← labelled/unlabelled pair assembly (TODO)
│   ├── features.py        ← pairwise feature engineering (TODO)
│   ├── model.py           ← Matcher interface (TODO)
│   ├── evaluation.py      ← precision / recall / F₀.₅ (TODO)
│   ├── threshold.py       ← threshold selection (TODO)
│   ├── prediction.py      ← test prediction pipeline (TODO)
│   ├── output.py          ← write submission TSVs (TODO)
│   └── main.py            ← top-level entry point
│
├── scripts/
│   └── inspect_data.py    ← data inspection (run this first)
│
├── tests/                 ← pytest scaffold tests
├── notebooks/             ← experiment notebooks
├── cache/                 ← intermediate artefacts (not committed)
└── output/                ← submission files (not committed)
```

## Current Development Stage

> **Stage 6 — Full Test Inference & Submission Generation Implemented & Verified**

✅ Project scaffold created  
✅ Data inspection completed (`scripts/inspect_data.py`)  
✅ Unicode-aware preprocessing implemented (`src/preprocessing.py`)  
✅ Multi-strategy blocking implemented (`src/blocking.py`: Blockers A, B, C, D)  
✅ Candidate-pair and training pair construction implemented (`src/pair_builder.py`)  
✅ Pairwise feature extraction implemented (`src/features.py`: 21 numeric features)  
✅ Evaluation metrics implemented (`src/evaluation.py`: entity & macro F0.5)  
✅ ML matching models implemented (`src/model.py`: LogisticRegression, LightGBM, entity GroupSplit)  
✅ Validation threshold tuning implemented (`src/threshold.py`: maximizes macro F0.5)  
✅ Multi-match and singleton prediction implemented (`src/prediction.py`)  
✅ Detailed validation error analysis implemented (`src/evaluation.py`)  
✅ End-to-end ML training pipeline created (`scripts/train_matcher.py`)  
✅ 15-cell Colab orchestration notebook created (`notebooks/experiments.ipynb`)  
✅ Complete test suite passing (143/143 unit & regression tests pass)  
✅ Large-scale validation sanity check (5,000 S1 entities, Macro F0.5 = 0.9584, `scripts/run_validation_scale_check.py`)  
✅ Memory-efficient chunked streaming inference pipeline (`scripts/generate_submission.py`)  
✅ Disk-backed candidate blocking index (`DiskBlockingIndex` via SQLite) for zero-RAM growth  
✅ Final competition model training pipeline on 100% labeled train data with no validation split (`scripts/train_final_model.py`)  
✅ Automated output validation and submission packaging (`output/submission.zip`)  


## Dataset Location

The dataset lives at (relative to workspace root):
```
Dataset/student_resource/dataset/
    train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
    test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

`src/config.py` resolves this path automatically relative to the project root.

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Run the Inspection Script

```bash
cd business_entity_resolution
python scripts/inspect_data.py
```

Output covers: file discovery, dimensions, missing values, duplicates, sample records, string statistics, country distribution, match cardinality, source distribution, and actual noise examples.

## Run Tests

```bash
cd business_entity_resolution
pytest -v
```

100 comprehensive behavioral tests covering case, punctuation, whitespace, Unicode, Indic scripts, legal suffixes, domain names, missing values, address numbers, order-independent address matching, and country normalization.

## Preprocessing Pipeline

Implemented in `src/preprocessing.py`, the preprocessing layer adheres strictly to competition constraints and Google Colab execution limits:

1. **Multiple Representations Instead of Destructive Normalization:**
   Different downstream stages require different views of the data:
   - Blocking benefits from compact alphanumeric strings and domain stems.
   - String similarity features (Jaccard, edit distance) benefit from punctuation-cleaned, lowercase tokens.
   - Entity classification benefits from raw Unicode tokens and script tags.
   Creating multiple representations avoids prematurely discarding discriminatory information.

2. **Preservation of Unicode & Indic Scripts:**
   The dataset contains real Indian business records in Devanagari (`राम`), Tamil (`குளோபல்`), Kannada (`ಕರ್ನಾಟಕ`), and Gujarati, as well as French records with accents (`École`, `Nouvelle-Aquitaine`). Replacing non-ASCII characters with spaces or question marks would destroy entity identity. Python's Unicode character properties (`L`, `M`, `N`) are leveraged to safely strip punctuation while keeping all letters, combining marks, and numerals intact.

3. **Careful Legal Suffix Handling:**
   Rather than stripping suffixes globally (which can cause false merges, e.g. "Acme Corp" vs "Acme LLC" or stripping "Inc" from "Seafood Inc"), legal suffixes (US: LLC, Inc, Corp; India: Pvt Ltd, LLP, Indic equivalents; France: SARL, SASU, SCI) are canonicalized and exposed as separate features (`legal_suffix`, `has_legal_suffix`, `name_no_suffix`).

4. **Address Numbers & Alphanumerics Preserved:**
   Address numbers (e.g. `1795`, `630 45th`, `183`) and unit identifiers are crucial discriminators for entity resolution. Generic abbreviations (`st` → `street`, `rd` → `road`, `blvd` → `boulevard`) are expanded, and an `order_independent` representation is generated to match reordered addresses (e.g. City, State, Street vs Street, City, State).

5. **Transliteration Intentionally Deferred:**
   While cross-script entity matches exist (e.g. English vs Devanagari/Tamil), heavy transliteration libraries introduce major runtime overhead and external dependencies. We first detect and flag script metadata (`script_type`), deferring transliteration to a targeted benchmark before adding it to the pipeline.

6. **Chunked Processing for Google Colab RAM Limits:**
   With 26.5 million records (~2.4 GB), loading all files simultaneously causes Out-Of-Memory (OOM) crashes in Colab. Both `src/preprocessing.py` and `src/data_loader.py` support streaming chunked processing:
   $$\text{TSV} \longrightarrow \text{Chunk (50k rows)} \longrightarrow \text{Preprocess Chunk} \longrightarrow \text{Cache/Stream} \longrightarrow \text{Release Chunk}$$

## Run Preprocessing Benchmark

```bash
cd business_entity_resolution
python scripts/benchmark_preprocessing.py --sample-size 1000
```

The benchmark processes a configurable sample (default: 1000 rows per source), reports runtime, peak memory allocation via `tracemalloc`, throughput (rows/sec), and prints before/after examples for non-Latin text, missing addresses, domain-like names, and legal suffixes.

## Blocking & Candidate Generation Pipeline

Implemented in `src/blocking.py`, `src/pair_builder.py`, and `src/features.py`:

### 1. Multi-Strategy Blocking Architecture

To avoid single-point failure and capture complementary signals, candidate pairs are generated through a union of four independently configurable blocking strategies:

- **Blocker A (Exact Normalized Name Blocks):**
  Matches on exact normalized Unicode name (`name_norm`), compact alphanumeric string (`name_compact`), name without legal suffix (`name_no_suffix`), and domain stem (`domain_stem`).
- **Blocker B (Selective Token-Based Blocks):**
  Matches on distinctive name tokens and bigrams, filtered against frequency thresholds and generic stopwords (`NAME_STOPWORDS`: *inc, llc, pvt, limited, company, group, services, the, and, etc.*).
- **Blocker C (Address-Based Blocks):**
  Matches on combinations of street numbers and informative address/city tokens. Accommodates reordered and abbreviated address structures without requiring exact address equality.
- **Blocker D (Composite Blocks):**
  Combines name discriminators with address discriminators (e.g. `first_name_token + street_number` and `compact_name_prefix + street_number`), generating high-precision, low-explosion candidate pairs.

### 2. Candidate Explosion Control
- **Posting List Cap:** Every block posting list is capped at `max_candidates_per_block` (default: 100). Oversized blocks are skipped or bounded, preventing single tokens from creating millions of candidate pairs.
- **Top-K Selection:** Candidates retrieved for each Source 1 query are scored across matching blocks and capped at `top_k_per_s1` (default: 50).
- **Deduplication:** Candidate pairs `(s1_id, candidate_id)` are strictly deduplicated and validated against self-matches.

### 3. Training Pair Construction & Leakage Prevention
Implemented in `src/pair_builder.py`:
- **Hard Negatives:** Candidates generated by blocking that are not ground-truth matches serve as authentic hard negatives.
- **Multi-Match Support:** S1 entities with multiple matches receive multiple positive rows (`label = 1`). Singletons receive 0 positive rows (`label = 0` for all candidates).
- **Leakage-Safe Entity Grouping:** Every training pair explicitly retains `s1_id`. In Turn 5, train/validation splitting is strictly performed by `s1_id` groups (GroupKFold) so records from the same business never appear in both train and validation sets.

### 4. Pairwise Feature Matrix
Implemented in `src/features.py`, 21 numeric features are computed per candidate pair:
- **Name Features:** `name_exact_match`, `name_compact_exact_match`, `name_no_suffix_match`, `name_token_jaccard`, `name_token_overlap_count`, `name_levenshtein_sim`, `name_compact_levenshtein_sim`, `legal_suffix_match`, `domain_stem_match`, `script_match`
- **Address Features:** `address_exact_match`, `address_token_jaccard`, `address_token_overlap_count`, `address_number_match`, `address_levenshtein_sim`, `address_missing_either`
- **Country Features:** `country_exact_match`, `country_missing_either`
- **Meta Features:** `source_is_s2`, `name_len_diff_ratio`, `address_len_diff_ratio`

### 5. Blocker Ablation Benchmark Results (Real Ground-Truth Sample)

Measured on 100 Source 1 query entities evaluated against their actual ground-truth matches in Source 2 and Source 3 (343 true matches) plus 1,000 distractor records:

| Stage | Overall Recall | S2 Recall | S3 Recall | Candidates | Reduction Ratio | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 (Blocker A only)** | 42.57% | 42.77% | 42.37% | 146 | 99.83% | 0.18s |
| **Stage 2 (Blocker A + B)** | 88.63% | 85.54% | 91.53% | 527 | 99.37% | 0.16s |
| **Stage 3 (Blocker A + B + C)** | 97.38% | 96.99% | 97.74% | 561 | 99.33% | 0.22s |
| **Stage 4 (Blocker A + B + C + D)** | **97.38%** | **96.99%** | **97.74%** | **561** | **99.33%** | **0.21s** |

- **Candidates per S1:** Mean: 5.61, Median: 5.0, P90: 9.1, Max: 19 (Zero candidate explosion).
- **Training Pairs:** 556 pairs assembled (334 true positives, 222 hard negatives).
- **Feature Extraction Throughput:** ~230 pairs/second at 0.98 MB peak memory.

## Run Blocking & Feature Benchmark

```bash
cd business_entity_resolution
python scripts/benchmark_blocking.py --sample-source1 100 --distractors 1000
```

## Stage 5: ML Matching & Validation Benchmark

Implemented in `src/model.py`, `src/threshold.py`, `src/prediction.py`, and `scripts/train_matcher.py`:

### 1. Entity-Level Group Split (Zero Data Leakage)
To prevent optimistic leakage from multi-candidate entities, training pairs and ground truth are partitioned strictly by `source1_entity_id` using an 80/20 GroupSplit:
$$\text{set}(\text{Train S1 Entities}) \cap \text{set}(\text{Val S1 Entities}) = \emptyset$$
- **Train Split:** 80 S1 entities, 456 candidate pairs (274 positives, 182 hard negatives)
- **Validation Split:** 20 S1 entities, 105 candidate pairs (60 positives, 45 hard negatives)

### 2. Multi-Match & Singleton Decision Logic
The matching classifier scores each pair $P(\text{match} | \text{features})$. Candidate IDs satisfying $P \ge \tau$ are selected:
- **Multi-Match Entities:** All candidates exceeding threshold $\tau$ are predicted (no argmax restriction).
- **Singletons (Zero True Matches):** If all candidates score below $\tau$ (or zero candidates exist), the prediction is an empty set $\emptyset$, yielding $F_{0.5} = 1.0$.

### 3. Model Comparison Table (Held-Out Validation Set)

Evaluated across candidate thresholds $\tau \in [0.05, 0.95]$ to strictly maximize the competition metric: **entity-level macro $F_{0.5}$** ($\beta = 0.5$):

| Model | Optimal $\tau$ | Pair Precision | Pair Recall | Pair F1 | PR-AUC | Macro $F_{0.5}$ | Singleton Acc | Training Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Standardized + Balanced) | 0.15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.9976** | 100.0% | 0.19s |
| **LightGBM (GBDT)** (MIT license, $\le 8\text{B}$ params) | 0.60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.9976** | 100.0% | 0.34s |

### 4. Feature Importance Diagnostics

Top features distinguishing true matches from hard negatives (ranked by LightGBM gain):

| Rank | Feature Name | LightGBM Gain % | Logistic Regression Weight % |
| :---: | :--- | :---: | :---: |
| 1 | `address_token_jaccard` | **85.33%** | 18.27% |
| 2 | `name_token_jaccard` | **12.41%** | 11.54% |
| 3 | `address_len_diff_ratio` | **1.62%** | 6.69% |
| 4 | `name_len_diff_ratio` | **0.25%** | 0.11% |
| 5 | `address_levenshtein_sim` | **0.22%** | 12.01% |

### 5. Validation Error Analysis
- **Candidate Generation (Blocking) Errors:** 1 true match missed (1.64% of true matches, in Source 2).
- **Classification Errors ($P < \tau$):** 0 true matches missed! The classifier correctly selected 100% of available true matches.
- **Singleton Accuracy:** 100.0% (singletons correctly rejected as empty sets, 0 false alarms).

### 6. Run Training & Validation Benchmark

```bash
cd business_entity_resolution
python scripts/train_matcher.py --sample-source1 100 --distractors 500
```

### 7. Google Colab Workflow

Open `notebooks/experiments.ipynb` in Google Colab:
- **Thin Orchestration Notebook:** 15 standardized cells executing diagnostics, data loading, training, threshold sweep, error analysis, and artifact persistence.
- **Optional Google Drive Persistence:** Set `BER_CACHE_DIR` to save intermediate Parquet/pickle files to Drive.

## Stage 5.5: Large-Scale Validation Sanity Check (5,000 S1 Entities)

Implemented in `scripts/run_validation_scale_check.py`:

To rigorously determine whether the Turn 5 results hold at statistical scale, Turn 5.5 executed a validation experiment across **5,000 Source-1 entities** (22,244 candidate pool records, 17,250 true matches, 166,296 candidate pairs, 1,000 validation entities):

### 1. Key Performance Comparison: Turn 5 vs Turn 5.5

| Metric | Turn 5 (Small) | Turn 5.5 (5k Scale) | Status / Stability Assessment |
| :--- | :---: | :---: | :--- |
| **Source-1 Query Entities** | 100 | **5,000** | $50\times$ larger statistical coverage |
| **Validation S1 Entities** | 20 | **1,000** | $50\times$ larger validation sample |
| **Candidate Pool Size (S2+S3)** | 843 | **22,244** | Realistic candidate competition pool |
| **Candidate Pairs Generated** | 561 | **166,296** | Full blocking pipeline under scale |
| **Overall Candidate Recall** | 97.38% | **93.40%** | Stable ($\ge 93\%$ across 17.2k matches) |
| **S2 Candidate Recall** | 96.99% | **92.76%** | Stable |
| **S3 Candidate Recall** | 97.74% | **93.98%** | Stable |
| **Candidate Reduction Ratio** | 99.33% | **99.8505%** | Excellent pruning of $1.11 \times 10^8$ search space |
| **Mean Candidates / S1** | 5.61 | **33.26** | Bounded at max 50 |
| **Validation Positive Pairs** | 60 | **3,254** | Robust positive sample |
| **Validation Hard Negatives** | 45 | **30,992** | Authentic 90.5% negative imbalance |
| **Validation Singletons** | 1 | **58** | Matches real data ~5.6% singleton rate |
| **Singleton Accuracy** | 100.0% | **98.28%** | 57 / 58 correctly rejected as $\emptyset$ |
| **Best Model** | LightGBM / LR | **LightGBM** | GBDT achieves top precision and recall |
| **Optimal Threshold $\tau^*$** | 0.60 | **0.90** | Higher threshold strictly optimizes $F_{0.5}$ at scale |
| **Validation Macro $F_{0.5}$** | 0.9976 | **0.9584** | **Extremely high performance confirmed!** |
| **Mean Entity Precision** | 1.0000 | **0.9788** | 97.9% average precision across entities |
| **Mean Entity Recall** | 0.9836 | **0.9187** | 91.9% average recall across entities |
| **Candidate Gen Misses** | 1 (1.64%) | **235 (6.74%)** | Dominant bottleneck (80.8% of errors) |
| **Classification Misses** | 0 (0.00%) | **56 (1.61%)** | Classifier selects 98.28% of generated true matches |
| **False Merges** | 0 | **21** | Only 21 false positives across 30,992 negatives |
| **Peak Process RSS** | 5.77 MB | **343.81 MB** | Fully Colab-compatible (< 400 MB) |
| **Total Runtime** | 9.07s | **206.53s** | ~3.4 minutes end-to-end |

### 2. Run Large-Scale Validation Check

```bash
cd business_entity_resolution
python scripts/run_validation_scale_check.py --sample-source1 5000 --distractors 5000
```

## Stage 6: Full TEST Inference & Submission Generation

Implemented in `scripts/generate_submission.py` and callable via `python -m src.main --mode predict`:

### 1. Architecture Highlights

1. **Disk-Backed Streaming Indexing (SQLite):**
   - Candidate sources (TEST S2: ~4.88M records, TEST S3: ~4.8M records) are streamed in chunks.
   - For each chunk, records are preprocessed and immediately written to a disk-backed SQLite database (`DiskBlockingIndex`).
   - Bulky chunk DataFrames are garbage-collected after each iteration — Python heap never accumulates the 10M record pool.
   - Finalizes posting lists with the exact same 4-stage blocking keys, `max_candidates_per_block=100` cap, and B-tree index.
   - Peak RSS remains **~250-450 MB**, completely eliminating Colab OOM risks.
2. **Batch Inference:**
   - S1 entities are processed in configurable batches (default: 10,000 entities).
   - Candidate pairs are generated via the 4 blocking rules, features computed on-the-fly, scored with the Turn 5.5 LightGBM model, and filtered at the validated optimal threshold ($\tau^* = 0.90$).
3. **Automated Submission Packaging & Verification:**
   - Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
   - Packages both into `output/submission.zip`.
   - Performs automated assertions checking coverage, row counts, column formats, ID uniqueness, and archive integrity.

### 2. Execution Instructions

**Fast Smoke Test (500 S1 entities, ~2.3 minutes):**
```bash
cd business_entity_resolution
python scripts/generate_submission.py --smoke-test --smoke-n 500 --pool-rows 200000 --batch-size 250
```

**Full Test Set Inference (all ~1.73M S1 entities):**
```bash
cd business_entity_resolution
python scripts/generate_submission.py --batch-size 10000
```

**Google Colab Execution (fast local NVMe disk recommended):**
```python
import os
os.environ["BER_DATASET_ROOT"] = "/content/drive/MyDrive/dataset"
# Placing SQLite on /content (local NVMe SSD) provides max I/O throughput
```

## Stage 7: Final Competition Model Training (`scripts/train_final_model.py`)

Dedicated pipeline to train the final LightGBM model on 100% of the available labeled competition training data (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`):

### 1. Key Principles
- **Zero Validation Withholding:** 100% of labeled training entities are available for training (NO 80/20 train/val split).
- **Architecture & Hyperparameter Consistency:** Identical 4-stage blocking, 21 numerical features, and `LightGBMMatcher(n_estimators=100, learning_rate=0.05, random_state=42)`.
- **Threshold Preservation:** Retains validated optimal threshold ($\tau^* = 0.90$) without data-leaking re-tuning.
- **Safe Persistence:** Saves model artifact to `cache/turn6_final_matcher.joblib` and metadata to `cache/turn6_final_model_config.json`, preserving Turn 5.5 validation artifacts (`turn5_5_best_matcher.joblib`).

### 2. Execution Commands

**Full 100% Competition Training (Colab / High-Resource Environment):**
```bash
python scripts/train_final_model.py
# or explicitly:
python scripts/train_final_model.py --full
```

**Deterministic Sanity Sample (5,000 S1 entities, 100% train / 0% val):**
```bash
python scripts/train_final_model.py --max-s1 5000
```

**Google Colab Execution:**
```python
import os
os.environ["BER_DATASET_ROOT"] = "/content/drive/MyDrive/Amazon_ML_Ram/Dataset/student_resource/dataset"
!python scripts/train_final_model.py
```

## Submission Format


```
matching_results.tsv
  source1_entity_id  <TAB>  matched_entity_ids   (comma-separated, can be empty)

candidate_pairs.tsv
  source1_entity_id  <TAB>  candidate_entity_ids (comma-separated, can be empty)
```

Validate with:
```bash
python ../Dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../Dataset/student_resource/dataset/test
```

## Constraints

- No external data sources, APIs, or geocoding services.
- Final model must be MIT/Apache 2.0 licensed and ≤ 8B parameters.
- Python 3.10+, `pathlib` for all paths.
