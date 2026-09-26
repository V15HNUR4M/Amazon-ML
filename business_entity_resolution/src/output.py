"""
output.py — Write TSV submission files.

Submission format (from hackathon spec)
---------------------------------------
matching_results.tsv
    source1_entity_id \t matched_entity_ids
    One row per Source 1 entity.
    matched_entity_ids = comma-separated S2/S3 IDs, or empty string for singletons.
    No duplicate IDs within a list.

candidate_pairs.tsv
    source1_entity_id \t candidate_entity_ids
    One row per Source 1 entity.
    candidate_entity_ids = comma-separated S2/S3 IDs from blocking step.

Public interface
----------------
write_matching_results(predictions, path)
write_candidate_pairs(candidates_dict, path)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_matching_results(
    predictions: dict[str, set[str]],
    path: Path | str,
) -> None:
    """Write the final matching_results.tsv submission file.

    Parameters
    ----------
    predictions:
        Mapping: source1_entity_id → set of matched entity IDs.
        Use an empty set for singletons.
    path:
        Output file path (will be created/overwritten).

    Format
    ------
    source1_entity_id\\tmatched_entity_ids
    S1-00001\\tS2-00047,S2-00193,S3-00812
    S1-00002\\t

    Notes
    -----
    IDs within each row are sorted deterministically (lexicographic).
    All source1 entity IDs are guaranteed to appear exactly once.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in sorted(predictions.keys()):
        matched = predictions[s1_id]
        if matched:
            # Sort matched IDs deterministically for reproducibility
            matched_str = ",".join(sorted(matched))
        else:
            matched_str = ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})

    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(path, sep="\t", index=False)
    logger.info(
        "Wrote matching_results.tsv: %d rows (%d singletons) → %s",
        len(df),
        int((df["matched_entity_ids"] == "").sum()),
        path,
    )


def write_candidate_pairs(
    candidates_dict: dict[str, list[str]],
    path: Path | str,
) -> None:
    """Write the candidate_pairs.tsv blocking output file.

    Parameters
    ----------
    candidates_dict:
        Mapping: source1_entity_id → list of candidate entity IDs.
        Use an empty list when no candidates were found.
    path:
        Output file path (will be created/overwritten).

    Format
    ------
    source1_entity_id\\tcandidate_entity_ids
    S1-00001\\tS2-00047,S2-00193,S3-00812,S3-00999

    Notes
    -----
    Candidate IDs within each row are sorted deterministically (lexicographic).
    All source1 entity IDs are guaranteed to appear exactly once.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in sorted(candidates_dict.keys()):
        cands = candidates_dict[s1_id]
        if cands:
            if isinstance(cands, str):
                cands_str = cands
            else:
                cands_str = ",".join(sorted(set(str(c) for c in cands)))
        else:
            cands_str = ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cands_str})

    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    df.to_csv(path, sep="\t", index=False)
    logger.info(
        "Wrote candidate_pairs.tsv: %d rows (%d no-candidate) → %s",
        len(df),
        int((df["candidate_entity_ids"] == "").sum()),
        path,
    )
