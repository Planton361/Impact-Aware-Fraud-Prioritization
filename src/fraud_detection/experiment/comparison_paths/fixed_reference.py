"""Deterministic fixed-reference comparison path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..prioritization.composition import compose_candidate_reranking
from ..prioritization.inputs import amount_gain_candidate_features


def fixed_reference_candidate_scores(
    candidate_pool: pd.DataFrame,
    amount: object,
) -> np.ndarray:
    features = amount_gain_candidate_features(candidate_pool, amount)
    return (
        features["p_fraud"].to_numpy(dtype=float)
        * features["log1p_amount"].to_numpy(dtype=float)
    )


def fixed_reference_ranking(
    candidate_pool: pd.DataFrame,
    amount: object,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return ``(raw_scores, complete_ranking)`` without fitting a model."""

    raw_scores = fixed_reference_candidate_scores(candidate_pool, amount)
    ranking = compose_candidate_reranking(candidate_pool, raw_scores)
    return raw_scores, ranking
