"""
config.py — Central configuration for Business Entity Resolution.

All path and hyper-parameter constants live here.
Import this module everywhere instead of hard-coding values.

Usage
-----
from src.config import Config
cfg = Config()
df = pd.read_csv(cfg.TRAIN_SOURCE1, sep="\t")
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository / workspace roots
# ---------------------------------------------------------------------------

# The project root is ONE level above the `src/` directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

import os

# Dataset root — can be overridden via environment variable BER_DATASET_ROOT (e.g. in Google Colab).
# Defaults to relative path from workspace root.
_ENV_DATASET_ROOT = os.environ.get("BER_DATASET_ROOT")
DATASET_ROOT: Path = (
    Path(_ENV_DATASET_ROOT).resolve()
    if _ENV_DATASET_ROOT
    else (
        PROJECT_ROOT.parent.parent
        / "Dataset"
        / "student_resource"
        / "dataset"
    )
)


class Config:
    """Central configuration object.

    Instantiate once and pass to pipeline stages, or import the module-level
    constants directly for simple scripts.
    """

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    # Dataset — train
    TRAIN_DIR: Path = DATASET_ROOT / "train"
    TRAIN_SOURCE1: Path = TRAIN_DIR / "train_source1.tsv"
    TRAIN_SOURCE2: Path = TRAIN_DIR / "train_source2.tsv"
    TRAIN_SOURCE3: Path = TRAIN_DIR / "train_source3.tsv"
    TRAIN_GROUND_TRUTH: Path = TRAIN_DIR / "train_ground_truth.tsv"

    # Dataset — test
    TEST_DIR: Path = DATASET_ROOT / "test"
    TEST_SOURCE1: Path = TEST_DIR / "test_source1.tsv"
    TEST_SOURCE2: Path = TEST_DIR / "test_source2.tsv"
    TEST_SOURCE3: Path = TEST_DIR / "test_source3.tsv"

    # Project artefact directories (relative to project root)
    CACHE_DIR: Path = PROJECT_ROOT / "cache"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"

    # Output file names
    MATCHING_RESULTS_FILE: Path = OUTPUT_DIR / "matching_results.tsv"
    CANDIDATE_PAIRS_FILE: Path = OUTPUT_DIR / "candidate_pairs.tsv"

    # ------------------------------------------------------------------
    # TSV read settings
    # ------------------------------------------------------------------
    TSV_SEP: str = "\t"
    TSV_ENCODING: str = "utf-8"

    # ------------------------------------------------------------------
    # Hyper-parameters (placeholders — to be tuned after data inspection)
    # ------------------------------------------------------------------

    # Blocking
    BLOCKING_TOP_K: int = 50                 # max candidates per S1 entity
    MAX_CANDIDATES_PER_BLOCK: int = 100      # max posting list size per block to prevent explosion
    MIN_TOKEN_LEN: int = 3
    BLOCKER_A_ENABLED: bool = True           # Exact normalized, compact, no-suffix, domain stem
    BLOCKER_B_ENABLED: bool = True           # Selective token blocks
    BLOCKER_C_ENABLED: bool = True           # Address number + street/city tokens
    BLOCKER_D_ENABLED: bool = True           # Name + Address composite blocks
    BLOCKING_THRESHOLD: float | None = None  # optional similarity floor

    # Model
    # TODO: set after choosing model type
    MATCH_THRESHOLD: float | None = None     # probability cut-off

    # Evaluation
    FBETA: float = 0.5  # F_0.5 (precision-heavy)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    RANDOM_SEED: int = 42

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Config(DATASET_ROOT={self.DATASET_ROOT}, "
            f"PROJECT_ROOT={PROJECT_ROOT})"
        )
