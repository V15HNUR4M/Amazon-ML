# ML Challenge 2026: Business Entity Resolution — Solution Documentation

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Problem Understanding

*Describe the entity resolution task in your own words.*

- **Input:** Three independently scraped business databases (Source 1, 2, 3).
- **Output:** For each Source 1 entity, predict matching IDs from Source 2/3 (zero, one, or many).
- **Evaluation:** Macro-averaged F₀.₅ — precision-weighted. False merges penalised 2× over missed links.
- **Key challenge:** Blocking — reducing the comparison space without losing true matches.

*[Fill in: specific insights from EDA — noise patterns, missing fields, country distribution, match cardinality, etc.]*

---

## 2. Dataset Analysis

### 2.1 Source Files

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| train_source1.tsv | TODO | TODO | TODO |
| train_source2.tsv | TODO | TODO | TODO |
| train_source3.tsv | TODO | TODO | TODO |
| train_ground_truth.tsv | TODO | TODO | TODO |
| test_source1.tsv | TODO | TODO | TODO |
| test_source2.tsv | TODO | TODO | TODO |
| test_source3.tsv | TODO | TODO | TODO |

### 2.2 Missing Values

*[Fill in after running inspect_data.py]*

### 2.3 Match Cardinality

*[Fill in: % singletons, % with 1 match, % with >1 match, max matches]*

### 2.4 Observed Noise Patterns

*[Fill in: actual noise types observed in genuine matched pairs — spellings, abbreviations, address formats, etc.]*

### 2.5 Country Distribution

*[Fill in]*

---

## 3. Preprocessing

*[Fill in once preprocessing.py is implemented]*

- **Name normalisation:** TODO
- **Address normalisation:** TODO
- **Country normalisation:** TODO
- **Special cases:** TODO

---

## 4. Candidate Generation (Blocking)

*[Fill in once blocking.py is implemented]*

- **Blocking keys used:** TODO
- **Candidate pairs generated (train):** TODO
- **Blocking recall (% true pairs retained):** TODO
- **Reduction ratio:** TODO
- **How singletons are handled:** TODO

---

## 5. Feature Engineering

*[Fill in once features.py is implemented]*

**Name features:**
- TODO

**Address features:**
- TODO

**Other features:**
- TODO

---

## 6. Matching Model

*[Fill in once model.py is implemented]*

- **Model type:** TODO
- **Training set size:** TODO pairs
- **Positive/negative ratio:** TODO
- **Cross-validation strategy:** TODO

---

## 7. Training Strategy

*[Fill in once training is done]*

- **Train/validation split:** TODO
- **Negative sampling strategy:** TODO
- **Class imbalance handling:** TODO
- **Hyper-parameter tuning:** TODO

---

## 8. Validation Strategy

*[Fill in]*

- **Validation metric:** Macro F₀.₅
- **Held-out set:** TODO
- **Cross-validation folds:** TODO

---

## 9. Threshold Selection

*[Fill in once threshold.py is implemented]*

- **Method:** Grid search over predict_proba threshold on validation set.
- **Optimal threshold found:** TODO
- **Validation F₀.₅ at threshold:** TODO

---

## 10. Results

| Split | Precision | Recall | F₀.₅ |
|-------|-----------|--------|-------|
| Validation | TODO | TODO | TODO |
| Public Leaderboard | TODO | TODO | TODO |
| Private Leaderboard | TODO | TODO | TODO |

---

## 11. Error Analysis

### False Positives (Wrong merges)

*[Fill in: types of pairs that are incorrectly merged]*

### False Negatives (Missed matches)

*[Fill in: types of true matches that are missed]*

---

## 12. Computational Considerations

- **Hardware used:** TODO
- **Training time:** TODO
- **Inference time per entity:** TODO
- **Memory footprint:** TODO

---

## 13. Limitations

*[Fill in: known failure cases, edge cases, dataset assumptions]*

---

## 14. Reproducibility

**To reproduce results from scratch:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python -m src.main --mode full
```

**Random seeds fixed:** `Config.RANDOM_SEED = 42`

*[Fill in: any additional reproducibility notes]*

---

## 15. Hackathon Compliance

- [ ] No external databases, APIs, or geocoding services used.
- [ ] Model is MIT/Apache 2.0 licensed.
- [ ] Model has ≤ 8B parameters.
- [ ] Output files validated with `utils/validate_submission.py`.
- [ ] `matching_results.tsv` and `candidate_pairs.tsv` present in `output/`.
- [ ] Every Source 1 entity appears in submission (no missing rows).
- [ ] No duplicate `source1_entity_id` rows.
- [ ] No S1 IDs in `matched_entity_ids` (only S2-/S3-).

---

## Appendix

### A. Code Artefacts

All source code is in `src/` under `business_entity_resolution/`.  
Entry point: `python -m src.main --mode full`  
Dependencies: `requirements.txt`

### B. Additional Results

*[Attach plots, confusion matrices, PR curves, etc.]*
