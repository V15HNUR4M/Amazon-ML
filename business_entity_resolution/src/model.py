"""
model.py — Machine Learning matching models for entity resolution.

Design intent
-------------
Provides concrete matching models (Logistic Regression baseline and LightGBM tree-based model)
that operate on candidate pair features to output calibrated match probabilities.

Key characteristics:
1. Operates on the 21 numerical pairwise features from src.features.
2. Supports entity-level GroupKFold / GroupShuffleSplit to prevent data leakage.
3. Multi-match aware: outputs P(match | features) per candidate pair.
4. Compliant with competition constraints:
   - Tree model (LightGBM) under MIT license, well within the <= 8B parameter limit.
   - CPU-compatible and Colab-portable.
   - Reproducible via fixed random states.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import ground_truth_to_dict

logger = logging.getLogger(__name__)

# Canonical list of 21 pairwise numerical features
DEFAULT_FEATURE_NAMES: list[str] = [
    # Name features (10)
    "name_exact_match",
    "name_compact_exact_match",
    "name_no_suffix_match",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_levenshtein_sim",
    "name_compact_levenshtein_sim",
    "legal_suffix_match",
    "domain_stem_match",
    "script_match",
    # Address features (6)
    "address_exact_match",
    "address_token_jaccard",
    "address_token_overlap_count",
    "address_number_match",
    "address_levenshtein_sim",
    "address_missing_either",
    # Country features (2)
    "country_exact_match",
    "country_missing_either",
    # Meta features (3)
    "source_is_s2",
    "name_len_diff_ratio",
    "address_len_diff_ratio",
]


# ---------------------------------------------------------------------------
# Entity-Level Group Splitter
# ---------------------------------------------------------------------------


def entity_train_val_split(
    pairs: pd.DataFrame,
    ground_truth: dict[str, set[str]] | pd.DataFrame,
    val_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]], dict[str, set[str]]]:
    """Split candidate pairs and ground truth strictly by Source 1 entity group.

    Guarantees zero overlap between train and validation Source 1 entities:
        set(train_s1_ids) & set(val_s1_ids) == empty set

    Parameters
    ----------
    pairs:
        Candidate pairs DataFrame with at least column 's1_id'.
    ground_truth:
        Ground truth mapping (dict or DataFrame).
    val_size:
        Fraction of Source 1 entities reserved for validation (default 0.20).
    random_state:
        Seed for deterministic entity grouping.

    Returns
    -------
    train_pairs:
        Subset of pairs whose s1_id belongs to train entities.
    val_pairs:
        Subset of pairs whose s1_id belongs to validation entities.
    train_gt:
        Ground truth dict for train entities.
    val_gt:
        Ground truth dict for validation entities.
    """
    if isinstance(ground_truth, pd.DataFrame):
        gt_dict = ground_truth_to_dict(ground_truth)
    else:
        gt_dict = ground_truth

    # Collect all unique S1 entities across both pairs and ground truth
    pairs_s1 = set(pairs["s1_id"].unique()) if len(pairs) > 0 else set()
    all_s1_list = sorted(list(set(gt_dict.keys()) | pairs_s1))

    if not all_s1_list:
        return pairs.copy(), pairs.copy(), {}, {}

    rng = np.random.RandomState(random_state)
    shuffled_s1 = rng.permutation(all_s1_list)

    n_val = max(1, int(len(all_s1_list) * val_size))
    val_s1_set = set(shuffled_s1[:n_val])
    train_s1_set = set(shuffled_s1[n_val:])

    # Critical safety assertion: zero leakage
    assert len(train_s1_set & val_s1_set) == 0, "Data leakage detected: train and validation S1 sets overlap!"

    # Split pairs
    train_pairs = pairs[pairs["s1_id"].isin(train_s1_set)].copy().reset_index(drop=True)
    val_pairs = pairs[pairs["s1_id"].isin(val_s1_set)].copy().reset_index(drop=True)

    # Split ground truth
    train_gt = {eid: gt_dict[eid] for eid in train_s1_set if eid in gt_dict}
    val_gt = {eid: gt_dict[eid] for eid in val_s1_set if eid in gt_dict}

    logger.info(
        "Entity split: %d train S1 (%d pairs), %d val S1 (%d pairs)",
        len(train_s1_set),
        len(train_pairs),
        len(val_s1_set),
        len(val_pairs),
    )

    return train_pairs, val_pairs, train_gt, val_gt


# ---------------------------------------------------------------------------
# Base Matcher Class
# ---------------------------------------------------------------------------


class MatcherBase:
    """Abstract base class for candidate pair matching models."""

    def __init__(
        self,
        feature_names: Optional[list[str]] = None,
        random_state: int = 42,
    ) -> None:
        self.feature_names = feature_names or list(DEFAULT_FEATURE_NAMES)
        self.random_state = random_state
        self.is_fitted: bool = False

    def _extract_feature_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Extract and validate the numerical feature matrix."""
        if isinstance(X, pd.DataFrame):
            # Check if feature columns are present in DataFrame
            missing = [f for f in self.feature_names if f not in X.columns]
            if not missing:
                mat = X[self.feature_names].to_numpy(dtype=np.float32)
            elif X.shape[1] == len(self.feature_names):
                mat = X.to_numpy(dtype=np.float32)
            else:
                raise ValueError(
                    f"DataFrame is missing required feature columns: {missing}"
                )
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got {X.shape[1]}"
                )
            mat = X if X.dtype == np.float32 else X.astype(np.float32, copy=False)
        else:
            raise TypeError(f"Unsupported feature container: {type(X)}")

        # Handle any residual NaNs by imputing 0.0
        if not isinstance(mat, np.memmap):
            if np.isnan(mat).any():
                mat = np.nan_to_num(mat, nan=0.0, copy=False)

        return mat

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "MatcherBase":
        """Train the model on pairwise features X and binary labels y."""
        raise NotImplementedError("Subclasses must implement fit()")

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return 1D array of match probabilities in [0.0, 1.0]."""
        raise NotImplementedError("Subclasses must implement predict_proba()")

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binarize probabilities using decision threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importances(self) -> dict[str, float]:
        """Return mapping of feature names to importance scores."""
        raise NotImplementedError("Subclasses must implement get_feature_importances()")

    def save(self, path: Path | str) -> None:
        """Persist trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved %s model to %s", self.__class__.__name__, path)

    @classmethod
    def load(cls, path: Path | str) -> "MatcherBase":
        """Load trained model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        instance = joblib.load(path)
        logger.info("Loaded %s model from %s", instance.__class__.__name__, path)
        return instance


# ---------------------------------------------------------------------------
# Logistic Regression Baseline Matcher
# ---------------------------------------------------------------------------


class LogisticRegressionMatcher(MatcherBase):
    """Logistic Regression matching model with standardized features and balanced weighting."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict = "balanced",
        random_state: int = 42,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__(feature_names=feature_names, random_state=random_state)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.pipeline: Optional[Pipeline] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "LogisticRegressionMatcher":
        """Fit standardized Logistic Regression model."""
        X_mat = self._extract_feature_matrix(X)
        y_vec = np.asarray(y, dtype=int)

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                        random_state=self.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.pipeline.fit(X_mat, y_vec)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_mat = self._extract_feature_matrix(X)
        if len(X_mat) == 0:
            return np.empty((0,), dtype=np.float32)
        # Probabilities for class 1 (match)
        return self.pipeline.predict_proba(X_mat)[:, 1].astype(np.float32)

    def get_feature_importances(self) -> dict[str, float]:
        """Compute feature importance from absolute standardized coefficients."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        coefs = np.abs(clf.coef_[0])
        total = np.sum(coefs) or 1.0
        normalized = coefs / total
        return dict(zip(self.feature_names, normalized.tolist()))


# ---------------------------------------------------------------------------
# LightGBM Gradient Boosted Matcher
# ---------------------------------------------------------------------------


class LightGBMMatcher(MatcherBase):
    """Gradient boosted decision tree matcher using LightGBM.

    Complies with competition requirements:
    - MIT license (LightGBM)
    - Well below <= 8B parameters (< 100k parameters)
    - Fast CPU execution and low RAM footprint
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 10,
        class_weight: str | dict = "balanced",
        random_state: int = 42,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__(feature_names=feature_names, random_state=random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.class_weight = class_weight
        self.clf: Optional[Any] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "LightGBMMatcher":
        """Fit LightGBM binary classifier."""
        import lightgbm as lgb

        X_mat = self._extract_feature_matrix(X)
        y_vec = np.asarray(y, dtype=np.int32)

        self.clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            class_weight=self.class_weight,
            random_state=self.random_state,
            importance_type="gain",
            verbose=-1,
            n_jobs=-1,
        )
        self.clf.fit(X_mat, y_vec)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if not self.is_fitted or self.clf is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_mat = self._extract_feature_matrix(X)
        if len(X_mat) == 0:
            return np.empty((0,), dtype=np.float32)
        return self.clf.predict_proba(X_mat)[:, 1].astype(np.float32)

    def get_feature_importances(self, importance_type: str = "gain") -> dict[str, float]:
        """Return feature importance scores sorted by importance."""
        if not self.is_fitted or self.clf is None:
            raise RuntimeError("Model is not fitted yet.")
        importances = self.clf.booster_.feature_importance(importance_type=importance_type)
        total = np.sum(importances) or 1.0
        normalized = (importances / total).tolist()
        return dict(zip(self.feature_names, normalized))
