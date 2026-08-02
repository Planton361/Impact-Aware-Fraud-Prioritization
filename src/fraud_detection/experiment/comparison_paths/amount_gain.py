"""Learned Amount-Gain comparison path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EffectiveExperimentConfig
from ..prioritization.composition import compose_candidate_reranking
from ..prioritization.inputs import candidate_group
from ..prioritization.lambdarank import (
    CandidateAmountGainRanker,
    CandidateRankerConfig,
)


def fit_inner_amount_gain_path(
    *,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    train_candidate_relevance: np.ndarray,
    validation_candidate_relevance: np.ndarray,
    train_candidate_pool: pd.DataFrame,
    validation_candidate_pool: pd.DataFrame,
    target_budget: int,
    gain_profile: str,
    random_state: int,
    effective_config: EffectiveExperimentConfig,
) -> tuple[
    CandidateRankerConfig,
    CandidateAmountGainRanker,
    np.ndarray,
    pd.DataFrame,
]:
    """Return ``(config, ranker, raw_scores, complete_ranking)`` for one inner fit."""

    config = CandidateRankerConfig(
        target_budget=target_budget,
        gain_profile=gain_profile,
        effective_config=effective_config,
        random_state=random_state,
        n_estimators=effective_config.ranker_max_estimators,
        early_stopping_rounds=(
            effective_config.ranker_early_stopping_rounds
        ),
    )
    ranker = CandidateAmountGainRanker(config).fit(
        train_features,
        train_candidate_relevance,
        candidate_group(effective_config.candidate_pool_size),
        eval_X=validation_features,
        eval_relevance=validation_candidate_relevance,
        eval_group=candidate_group(effective_config.candidate_pool_size),
    )
    raw_scores = ranker.decision_function(validation_features)
    ranking = compose_candidate_reranking(
        validation_candidate_pool,
        raw_scores,
    )
    return config, ranker, raw_scores, ranking


def fit_final_amount_gain_path(
    *,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_candidate_relevance: np.ndarray,
    test_candidate_pool: pd.DataFrame,
    target_budget: int,
    selected_gain: str,
    random_state: int,
    final_n_estimators: int,
    effective_config: EffectiveExperimentConfig,
) -> tuple[
    CandidateRankerConfig,
    CandidateAmountGainRanker,
    np.ndarray,
    pd.DataFrame,
]:
    """Return ``(config, ranker, raw_scores, complete_ranking)`` for one final fit."""

    config = CandidateRankerConfig(
        target_budget=target_budget,
        gain_profile=selected_gain,
        effective_config=effective_config,
        random_state=random_state,
        n_estimators=final_n_estimators,
        early_stopping_rounds=(
            effective_config.ranker_early_stopping_rounds
        ),
    )
    ranker = CandidateAmountGainRanker(config).fit(
        train_features,
        train_candidate_relevance,
        candidate_group(effective_config.candidate_pool_size),
    )
    raw_scores = ranker.decision_function(test_features)
    ranking = compose_candidate_reranking(test_candidate_pool, raw_scores)
    return config, ranker, raw_scores, ranking
