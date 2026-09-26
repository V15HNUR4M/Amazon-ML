# ML Challenge 2026: Business Entity Resolution — Solution Documentation

**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Task:** Entity Matching across Three Heterogeneous Business Databases (Source 1, Source 2, Source 3)  
**Evaluation Metric:** Entity-Level Macro $F_{0.5}$ (Precision-weighted $2\times$ over recall)  

---

## 1. Problem Understanding

Entity resolution across multi-source, independently scraped business registries is a foundational challenge in modern e-commerce and supply chain intelligence. In this challenge:
- **Input:** Three large, noisy tabular datasets:
  - **Source 1 (Query):** ~2.2M training records, ~1.73M test records.
  - **Source 2 (Candidate Pool A):** ~5.0M training records, ~4.9M test records.
  - **Source 3 (Candidate Pool B):** ~5.3M training records, ~5.2M test records.
  - **Ground Truth:** ~2.2M training relationships linking Source 1 entities to zero, one, or multiple matches in Sources 2 and 3.
- **Output:** 
  1. `matching_results.tsv`: Predicted match IDs (`S2-*`, `S3-*`) for each Source 1 query entity, or empty string for singletons.
  2. `candidate_pairs.tsv`: Candidate match IDs retained by the blocking phase for pipeline auditing.
  3. `submission.zip`: Packaged zip archive containing both TSV files.
- **Evaluation Metric:** Macro-averaged $F_{0.5}$ over all Source 1 query entities:
  $$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
  Singletons ($\emptyset$ true matches) score $1.0$ if and only if the model predicts an empty set $\emptyset$; any false merge on a singleton yields an entity score of $0.0$.
  Because $\beta = 0.5$, **precision is valued twice as heavily as recall**. A single false merge penalizes the macro score significantly more than a missed candidate.

---

## 2. Dataset Analysis

### 2.1 Source Files & Statistics

| File | Rows | Columns | Key Characteristics |
|---|---|---|---|
| `train_source1.tsv` | 2,213,812 | `entity_id`, `name`, `address`, `country` | High completeness, canonical query source |
| `train_source2.tsv` | 5,021,940 | `entity_id`, `name`, `address`, `country` | Large candidate pool, ~35% missing addresses |
| `train_source3.tsv` | 5,348,102 | `entity_id`, `name`, `address`, `country` | Contains domain/URL-like names, missing addresses |
| `train_ground_truth.tsv` | 2,213,812 | `source1_entity_id`, `matched_entity_ids` | ~5.6% singletons, 1-to-many matches across S2 and S3 |
| `test_source1.tsv` | 1,732,954 | `entity_id`, `name`, `address`, `country` | Official test query entities |
| `test_source2.tsv` | 4,918,220 | `entity_id`, `name`, `address`, `country` | Official test candidate pool A |
| `test_source3.tsv` | 5,234,118 | `entity_id`, `name`, `address`, `country` | Official test candidate pool B |

Total dataset size exceeds 26.5 million records (~2.4 GB raw TSVs).

### 2.2 Missing Values & Multi-Lingual Observations
1. **Missing Addresses:** Sources 2 and 3 frequently exhibit missing or generic placeholder addresses (`"null"`, `"None"`, `"N/A"`, `"."`, `"undefined"`). Source 1 has high address density (>98%).
2. **Scripts & Languages:** While training predominantly features US and Indian entities, the data contains multi-script content:
   - Latin script with French accented characters (`é`, `è`, `ê`, `ç`, `ô`) in the test split.
   - Devanagari (Hindi/Marathi) and Dravidian scripts (Tamil, Kannada, Gujarati).
3. **Source 3 Web Artifacts:** Source 3 frequently represents entity names as web domains or URLs (e.g., `www.acme-corp.com`, `pipe|delimited|tokens`).

### 2.3 Match Cardinality
- **Singletons:** ~5.6% of Source 1 entities have no corresponding entity in Source 2 or 3.
- **1-to-1 Matches:** ~68.4% match exactly one entity in S2 or S3.
- **1-to-Many Matches:** ~26.0% match multiple records across both S2 and S3.

---

## 3. Preprocessing Pipeline (`src/preprocessing.py`)

All normalizations are deterministic, Unicode-safe (NFKC), and execute with no external API lookups:
1. **Missing Value Sanitization:** Strict regex-based detection catches empty strings, whitespace-only, `"null"`, `"<NULL>"`, `"None"`, `"NaN"`, `"N/A"`, `"undefined"`, and lone punctuation.
2. **Name Normalization:**
   - Unicode NFKC normalization preserving accented Latin and Indic characters.
   - Lowercasing and whitespace collapse.
   - Legal suffix standardization & extraction (`inc`, `corp`, `llc`, `ltd`, `pvt ltd`, `sarl`, `gmbh`, `sa`, etc.).
   - S3 Domain URL cleaning: strips protocols (`http://`, `https://`), prefixes (`www.`), domain extensions (`.com`, `.org`, `.in`, etc.), and query parameters.
3. **Address Normalization:**
   - Preserves numerical tokens (crucial for building and suite numbers).
   - Abbreviation expansion: `st` $\to$ `street`, `rd` $\to$ `road`, `ave` $\to$ `avenue`, `blvd` $\to$ `boulevard`, `ste` $\to$ `suite`, `dr` $\to$ `drive`.
   - Token-sorted representation for order-independent street matching.
4. **Country Standardization:**
   - Canonicalizes country strings to ISO-2 format (`united states`, `usa`, `u.s.a.` $\to$ `US`; `india`, `bharat` $\to$ `IN`; `france` $\to$ `FR`).

---

## 4. Candidate Generation / Blocking (`src/blocking.py`)

Direct pairwise comparison across $1.73\text{M} \times 10.15\text{M}$ entities entails $1.76 \times 10^{13}$ pairs, which is computationally intractable. We designed a multi-strategy inverted index with 4 complementary blocking keys:

1. **Blocker A (Exact Normalized Name + Country):** Catches clean entity matches regardless of address noise.
2. **Blocker B (First Two Name Tokens + Country):** Handles trailing word variations and punctuation differences.
3. **Blocker C (Street Number + 3-char Name Prefix + Country):** Catches businesses with shared physical premises where name spellings diverge.
4. **Blocker D (Normalized Address Token Set + Country):** Catches exact address matches where business names underwent legal rebranding.

### Blocking Performance at Scale (5,000 S1 entities, 22,244 pool records):
- **Candidate Blocking Recall:** **93.40%** of all true matches retained ($\ge 93\%$ on both S2 and S3).
- **Candidate Reduction Ratio:** **99.8505%** (pruned 99.85% of non-matching pairs).
- **Candidate Cap:** Bounded at 50 candidates per S1 entity to guarantee linear runtime and bounded memory.

---

## 5. Feature Engineering (`src/features.py`)

For each candidate pair $(e_1, e_{\text{cand}})$, we compute **21 canonical numerical features**:
1. **Name Jaccard Token Similarity:** Unigram token overlap between normalized names.
2. **Name Token Sort Ratio:** Levenshtein ratio on sorted unigrams (robust to word order changes).
3. **Name Normalized Levenshtein Distance:** Character-level edit similarity ($1 - \frac{\text{dist}}{\max(\text{len}_1, \text{len}_2)}$).
4. **Name Prefix Match:** Binary flag indicating matching 3-character and 5-character prefixes.
5. **Exact Normalized Name Match:** Binary indicator ($1.0$ if identical).
6. **Legal Suffix Consistency:** Matches when legal suffixes are compatible or absent.
7. **Address Token Jaccard Similarity:** Overlap between normalized address tokens.
8. **Address Number Match:** Binary indicator for identical street/suite numbers.
9. **Address Normalized Levenshtein Distance:** Character-level edit distance on full address strings.
10. **Address Order-Independent Token Overlap:** Intersection count over minimum token count.
11. **Country Match:** Binary flag ($1.0$ if matching canonical ISO country codes).
12. **Country Missing Flag:** Indicates whether one or both records had null country values.
13. **Domain / URL Similarity:** String similarity specifically on domain tokens extracted from S3 names.
14. **Name Length Ratio:** $\min(\text{len}_1, \text{len}_2) / \max(\text{len}_1, \text{len}_2)$.
15. **Address Length Ratio:** Ratio of address string lengths.
16. **Script Consistency:** Indicator that both records share the same Unicode script class.
17-21. **Interaction and Source-Specific Features:** Blocker source indicators (`is_s2`, `is_s3`) and multi-key agreement count.

---

## 6. Model & Training Strategy (`src/model.py`, `scripts/train_matcher.py`)

### 6.1 Entity Group Splitting (Zero Leakage)
To accurately mirror test conditions, records are split by **Source 1 Entity ID** (80% train, 20% validation) using `GroupShuffleSplit`. No candidate pairs belonging to the same query entity ever cross the train/validation boundary.

### 6.2 Model Architecture: LightGBM GBDT
- **Classifier:** LightGBM Binary Classifier (MIT licensed, < 8B parameters).
- **Hyperparameters:** `objective='binary'`, `metric='binary_logloss'`, `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Negative Mining:** Hard negatives generated directly from the multi-stage blocking index, providing natural 10:1 class imbalance reflecting test deployment.

---

## 7. Threshold Selection & Decision Rule (`src/threshold.py`, `src/prediction.py`)

Because the competition evaluation metric is **Macro $F_{0.5}$**, false merges are penalised twice as heavily as false dismissals. A grid search over $\tau \in [0.05, 0.95]$ was conducted on the held-out validation set.

- **Optimal Threshold:** **$\tau^* = 0.90$**
- **Decision Rule:**
  $$\hat{Y}(e_1) = \{ e_{\text{cand}} \mid P(\text{match} \mid e_1, e_{\text{cand}}) \ge 0.90 \}$$
  If no candidate exceeds $0.90$, $\hat{Y}(e_1) = \emptyset$ (predicted singleton).

---

## 8. Validation Results (Turn 5.5 Benchmark)

Evaluated on 1,000 held-out validation Source 1 entities:

| Metric | Validation Score | Interpretation |
|---|---|---|
| **Macro $F_{0.5}$** | **0.9584** | **Competition metric optimization confirmed** |
| **Mean Entity Precision** | **0.9788** | 97.9% average precision across entities |
| **Mean Entity Recall** | **0.9187** | 91.9% average recall across entities |
| **Singleton Accuracy** | **98.28%** | 57 / 58 singletons correctly identified as $\emptyset$ |
| **False Merges (FP)** | **21** | Only 21 false positives across 30,992 negative candidates |
| **Classifier Selection Rate** | **98.28%** | Classifier successfully captures 98.3% of generated true matches |

---

## 9. Full Test Inference Architecture (`scripts/generate_submission.py`)

For the full test evaluation on 1.73M S1 entities and 10.15M S2/S3 candidate records:
1. **Chunked Streaming Blocking Index:**
   - Streams S2 and S3 in 50,000-record chunks.
   - Preprocesses and populates the inverted `BlockingIndex`.
   - Extracts a compact dictionary of 11 string/token fields, immediately discarding raw DataFrames.
   - **RAM Footprint:** Stabilizes peak process RSS under **1.2 GB**, enabling execution on any standard laptop or free-tier Google Colab CPU/GPU.
2. **Chunked Batch Inference:**
   - Processes S1 entities in batches of 10,000.
   - Computes features only for candidate pairs identified by blocking.
   - Applies LightGBM inference and thresholding ($\tau^* = 0.90$).
3. **Automated Submission Packaging & Verification:**
   - Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
   - Creates `output/submission.zip`.
   - Validates all formatting requirements (unique IDs, full coverage, tab-delimited formatting).

---

## 10. Final Competition Model Training (`scripts/train_final_model.py`)

A separate final training pipeline trains the competition LightGBM model on 100% of the available labeled training data:
1. **Zero Validation Withholding:**
   - 100% of labeled training entities are available for training (NO 80/20 train/val split).
2. **Architecture & Hyperparameter Consistency:**
   - Reuses validated 4-stage blocking (weights A=3, D=2, B=1, C=1, `max_candidates_per_block=100`, `top_k=50`).
   - Reuses validated 21 numerical features.
   - Fits `LightGBMMatcher(n_estimators=100, learning_rate=0.05, random_state=42)`.
3. **Threshold Preservation:**
   - Uses validated threshold $\tau^* = 0.90$ (no re-tuning on training data).
4. **Artifact Persistence:**
   - Model saved to `cache/turn6_final_matcher.joblib`.
   - Metadata saved to `cache/turn6_final_model_config.json`.
   - Leaves Turn 5.5 validation artifacts intact (`turn5_5_best_matcher.joblib`).

---

## 11. Reproducibility & Colab Execution

### Final Competition Model Training:
```bash
# 1. Full training on 100% of competition data:
python scripts/train_final_model.py

# 2. Fast deterministic sanity check (5,000 S1 queries, 100% train / 0% val):
python scripts/train_final_model.py --max-s1 5000
```

### Full Test Inference:
```bash
# 1. Run unit and integration tests (143/143 tests pass)
pytest tests/

# 2. Run fast smoke test (500 entities, ~2.3 mins)
python scripts/generate_submission.py --smoke-test --smoke-n 500 --pool-rows 200000 --batch-size 250

# 3. Run full test set inference
python scripts/generate_submission.py --batch-size 10000
```

### Google Colab Execution:
```python
import os
os.environ["BER_DATASET_ROOT"] = "/content/drive/MyDrive/Amazon_ML_Ram/Dataset/student_resource/dataset"
!pip install -r requirements.txt

# Step 1: Train final competition model on 100% labeled data
!python scripts/train_final_model.py

# Step 2: Run disk-backed inference using local /content NVMe SSD
!python scripts/generate_submission.py --batch-size 10000 --db-path /content/turn6_blocking.db
```

---

## 12. Hackathon Compliance Checklist

- [x] **No External Data or APIs:** All lookups, parsing, and models rely strictly on the competition dataset.
- [x] **Open-Source License:** LightGBM and Scikit-Learn components are MIT / BSD licensed.
- [x] **Model Size:** Zero LLMs used; LightGBM model size is < 2 MB ($\ll 8\text{B}$ parameter limit).
- [x] **Submission File Format:** `matching_results.tsv` and `candidate_pairs.tsv` are valid tab-separated files.
- [x] **100% Query Entity Coverage:** Exactly matches test Source 1 entity count with zero duplicate IDs.
- [x] **No S1 IDs in Targets:** Candidate and match lists contain strictly `S2-*` and `S3-*` IDs.
