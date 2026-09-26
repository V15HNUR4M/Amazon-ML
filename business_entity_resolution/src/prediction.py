"""
prediction.py — Candidate scoring, thresholding, and multi-match prediction.

Design intent
-------------
Translates pairwise match probabilities into final entity-level match sets:
1. Multi-match handling: An entity can match 0, 1, 2, or multiple candidates.
   Every candidate pair with P(match | features) >= threshold is selected.
2. Singleton handling: Entities with no candidates or where all candidates
   score below threshold are mapped to empty sets set().
3. Entity completeness: Guarantees that all requested Source 1 entities appear
   in the output mapping.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def predict_matches(
    pairs: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    s1_entities: Optional[Iterable[str]] = None,
) -> dict[str, set[str]]:
    """Convert candidate pair probabilities to entity-level match predictions.

    Parameters
    ----------
    pairs:
        DataFrame containing at least 's1_id' and 'candidate_id' columns.
    proba:
        1D array of match probabilities matching the rows of *pairs*.
    threshold:
        Decision threshold in [0.0, 1.0]. Candidates with proba >= threshold
        are predicted as matches.
    s1_entities:
        Optional full collection of Source 1 entity IDs. If supplied,
        entities with zero candidates or zero predictions above threshold
        will be explicitly present with value set() (singleton prediction).

    Returns
    -------
    dict: source1_entity_id -> set of predicted matched entity IDs.
    """
    predictions: dict[str, set[str]] = defaultdict(set)

    # Initialize all requested entities with empty sets
    if s1_entities is not None:
        for eid in s1_entities:
            predictions[str(eid)] = set()

    if pairs is None or len(pairs) == 0 or len(proba) == 0:
        return dict(predictions)

    if len(pairs) != len(proba):
        raise ValueError(
            f"Length mismatch: pairs has {len(pairs)} rows, but proba has {len(proba)} items"
        )

    # Filter pairs where probability meets or exceeds threshold
    p_vec = np.asarray(proba)
    mask = p_vec >= threshold

    matched_pairs = pairs[mask]
    for _, row in matched_pairs.iterrows():
        s1_id = str(row["s1_id"])
        cand_id = str(row["candidate_id"])
        predictions[s1_id].add(cand_id)

    # Also ensure any s1_id that appeared in pairs is present even if no candidate exceeded threshold
    for s1_id in pairs["s1_id"].unique():
        s1_str = str(s1_id)
        if s1_str not in predictions:
            predictions[s1_str] = set()

    return dict(predictions)
