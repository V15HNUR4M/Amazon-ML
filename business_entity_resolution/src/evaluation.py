"""
evaluation.py — Metrics for entity resolution evaluation.

The hackathon uses macro-averaged F_0.5 (precision-heavy).

Metric definitions
------------------
precision   = |predicted ∩ true| / |predicted|
recall      = |predicted ∩ true| / |true|
F_0.5       = (1.25 × precision × recall) / (0.25 × precision + recall)

For entities with no true matches (singletons):
  score = 1.0 if predicted is empty, else 0.0

Public interface
----------------
precision_recall_f05(y_true, y_pred) -> tuple[float, float, float]
entity_precision(true_ids, pred_ids) -> float
entity_recall(true_ids, pred_ids)    -> float
entity_f05(true_ids, pred_ids)       -> float
macro_f05(ground_truth, predictions) -> float
ground_truth_to_dict(gt_df)          -> dict[str, set[str]]
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

BETA: float = 0.5  # F_β with β=0.5
BETA_SQ: float = BETA * BETA  # 0.25


# ---------------------------------------------------------------------------
# Set-level metrics (one entity)
# ---------------------------------------------------------------------------


def entity_precision(true_ids: set[str], pred_ids: set[str]) -> float:
    """Precision for one Source 1 entity.

    Handles singletons:
    - true empty, pred empty: 1.0
    - pred empty, true non-empty: 0.0
    - true empty, pred non-empty: 0.0
    """
    if not true_ids and not pred_ids:
        return 1.0
    if not pred_ids or not true_ids:
        return 0.0
    common = len(true_ids & pred_ids)
    return common / len(pred_ids)


def entity_recall(true_ids: set[str], pred_ids: set[str]) -> float:
    """Recall for one Source 1 entity.

    Handles singletons:
    - true empty, pred empty: 1.0
    - pred empty, true non-empty: 0.0
    - true empty, pred non-empty: 0.0
    """
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids or not pred_ids:
        return 0.0
    common = len(true_ids & pred_ids)
    return common / len(true_ids)


def entity_f05(true_ids: set[str], pred_ids: set[str]) -> float:
    """F_0.5 score for one Source 1 entity.

    Handles singletons:
    - both empty sets → 1.0
    - non-empty pred on empty true → 0.0
    - empty pred on non-empty true → 0.0

    Parameters
    ----------
    true_ids:
        Ground-truth matched entity IDs (empty for singletons).
    pred_ids:
        Predicted matched entity IDs (empty for predicted singletons).

    Returns
    -------
    float in [0.0, 1.0]
    """
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids or not pred_ids:
        return 0.0

    common = len(true_ids & pred_ids)
    if common == 0:
        return 0.0

    p = common / len(pred_ids)
    r = common / len(true_ids)

    denom = BETA_SQ * p + r
    if denom == 0.0:
        return 0.0

    return (1.0 + BETA_SQ) * p * r / denom


# ---------------------------------------------------------------------------
# Corpus-level metric (macro-average)
# ---------------------------------------------------------------------------


def macro_f05(
    ground_truth: dict[str, set[str]],
    predictions: dict[str, set[str]],
) -> float:
    """Macro-averaged F_0.5 over all Source 1 entities.

    Parameters
    ----------
    ground_truth:
        Mapping: source1_entity_id → set of matched_entity_ids.
        Empty set for singletons.
    predictions:
        Mapping: source1_entity_id → set of predicted matched_entity_ids.
        Missing entities default to empty set (singleton prediction).

    Returns
    -------
    float — macro-average F_0.5 in [0.0, 1.0].
    """
    if not ground_truth:
        return 0.0

    total_f05 = 0.0
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_f05 += entity_f05(true_set, pred_set)

    return total_f05 / len(ground_truth)


# ---------------------------------------------------------------------------
# Convenience: build ground_truth dict from raw DataFrame
# ---------------------------------------------------------------------------


def ground_truth_to_dict(gt_df: pd.DataFrame) -> dict[str, set[str]]:
    """Convert raw ground-truth DataFrame to a dict of sets.

    Parameters
    ----------
    gt_df:
        DataFrame with columns: source1_entity_id, matched_entity_ids.

    Returns
    -------
    dict: source1_entity_id → set of matched IDs (empty set for singletons).
    """
    if gt_df is None:
        raise ValueError("gt_df DataFrame cannot be None")

    gt_dict: dict[str, set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        matched_raw = str(row.get("matched_entity_ids", "") or "")
        matches = {
            m.strip()
            for m in matched_raw.split(",")
            if m.strip() and m.strip().lower() not in ("nan", "none", "null")
        }
        gt_dict[s1_id] = matches

    return gt_dict


# ---------------------------------------------------------------------------
# Validation Error Analysis
# ---------------------------------------------------------------------------


def analyze_validation_errors(
    val_gt: dict[str, set[str]],
    val_predictions: dict[str, set[str]],
    val_pairs: pd.DataFrame,
    val_proba: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    """Perform detailed error analysis on validation set predictions.

    Explicitly separates:
    A. Candidate generation (blocking) errors: True matches that were never
       generated in the candidate pool.
    B. Classification errors:
       - False Negatives: True matches that were candidate pairs but rejected by classifier.
       - False Positives: Non-matches that were accepted by classifier.
       - Singleton errors: Singletons incorrectly given predictions.

    Parameters
    ----------
    val_gt:
        Validation ground truth mapping: s1_id -> set of true matched IDs.
    val_predictions:
        Validation predictions mapping: s1_id -> set of predicted IDs.
    val_pairs:
        Validation candidate pairs DataFrame with 's1_id' and 'candidate_id'.
    val_proba:
        Optional match probabilities.
    threshold:
        Optional decision threshold.

    Returns
    -------
    dict with detailed diagnostic counts and categorized breakdown.
    """
    total_val_entities = len(val_gt)
    total_true_matches = sum(len(matches) for matches in val_gt.values())

    # Build set of candidate IDs available per S1 entity
    cands_by_s1: dict[str, set[str]] = {}
    if len(val_pairs) > 0:
        for s1_id, grp in val_pairs.groupby("s1_id"):
            cands_by_s1[str(s1_id)] = set(grp["candidate_id"].astype(str))

    blocking_misses: list[tuple[str, str]] = []  # (s1_id, true_id)
    classifier_misses: list[tuple[str, str]] = []  # (s1_id, true_id)
    false_positives: list[tuple[str, str]] = []  # (s1_id, pred_id)
    true_positives: list[tuple[str, str]] = []  # (s1_id, matched_id)

    singleton_total = 0
    singleton_correct = 0
    singleton_false_positives = 0

    for s1_id, true_set in val_gt.items():
        pred_set = val_predictions.get(s1_id, set())
        avail_cands = cands_by_s1.get(s1_id, set())

        # Singleton evaluation
        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
            else:
                singleton_false_positives += 1
                for pid in pred_set:
                    false_positives.append((s1_id, pid))
            continue

        # True Positives
        for tid in (true_set & pred_set):
            true_positives.append((s1_id, tid))

        # False Positives
        for pid in (pred_set - true_set):
            false_positives.append((s1_id, pid))

        # False Negatives: distinguish blocking misses from classification misses
        for tid in (true_set - pred_set):
            if tid in avail_cands:
                classifier_misses.append((s1_id, tid))
            else:
                blocking_misses.append((s1_id, tid))

    # Error categorization by candidate source
    s2_classifier_misses = sum(1 for _, tid in classifier_misses if tid.startswith("S2"))
    s3_classifier_misses = sum(1 for _, tid in classifier_misses if tid.startswith("S3"))
    s2_blocking_misses = sum(1 for _, tid in blocking_misses if tid.startswith("S2"))
    s3_blocking_misses = sum(1 for _, tid in blocking_misses if tid.startswith("S3"))

    report: dict[str, Any] = {
        "total_val_entities": total_val_entities,
        "total_true_matches": total_true_matches,
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "total_false_negatives": len(blocking_misses) + len(classifier_misses),
        # Distinct error categories
        "blocking_misses_count": len(blocking_misses),
        "blocking_miss_rate_pct": (
            (len(blocking_misses) / total_true_matches * 100)
            if total_true_matches > 0
            else 0.0
        ),
        "classifier_misses_count": len(classifier_misses),
        "classifier_miss_rate_pct": (
            (len(classifier_misses) / total_true_matches * 100)
            if total_true_matches > 0
            else 0.0
        ),
        # Source breakdowns
        "s2_blocking_misses": s2_blocking_misses,
        "s3_blocking_misses": s3_blocking_misses,
        "s2_classifier_misses": s2_classifier_misses,
        "s3_classifier_misses": s3_classifier_misses,
        # Singleton breakdown
        "singleton_total": singleton_total,
        "singleton_correct": singleton_correct,
        "singleton_accuracy_pct": (
            (singleton_correct / singleton_total * 100) if singleton_total > 0 else 100.0
        ),
        "singleton_false_positives": singleton_false_positives,
    }

    return report
