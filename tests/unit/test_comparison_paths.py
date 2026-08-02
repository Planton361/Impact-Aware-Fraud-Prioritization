from inspect import signature

import numpy as np
import pandas as pd
import pytest

from fraud_detection.experiment.comparison_paths import (
    amount_gain,
    bce_baseline,
    fixed_reference,
    p_only,
)
from fraud_detection.experiment.config import resolve_experiment_profile
from fraud_detection.experiment.prioritization import composition
from fraud_detection.experiment.prioritization.inputs import (
    build_candidate_pool,
    candidate_rows_by_bce,
)
from fraud_detection.experiment.prioritization.lambdarank import (
    CandidateRankerConfig,
)

pytestmark = pytest.mark.unit

_CANONICAL = resolve_experiment_profile("canonical")


class _FakeRanker:
    instances: list["_FakeRanker"] = []

    def __init__(self, config: CandidateRankerConfig) -> None:
        self.config = config
        self.fit_args: tuple[object, ...] | None = None
        self.fit_kwargs: dict[str, object] | None = None
        self.score_features: object | None = None
        self.instances.append(self)

    def fit(self, *args: object, **kwargs: object) -> "_FakeRanker":
        self.fit_args = args
        self.fit_kwargs = kwargs
        return self

    def decision_function(self, features: object) -> np.ndarray:
        self.score_features = features
        return np.arange(len(features), dtype=float)


def _pool(rows: int = 1000, pool_size: int = 1000) -> pd.DataFrame:
    return build_candidate_pool(
        np.linspace(0.0, 1.0, rows),
        np.arange(rows),
        candidate_pool_size=pool_size,
    )


def _amount_features() -> pd.DataFrame:
    probability = np.linspace(0.0, 1.0, 1000)
    log_amount = np.log1p(np.arange(1000, dtype=float))
    return pd.DataFrame(
        {
            "p_fraud": probability,
            "log1p_amount": log_amount,
            "p_fraud_x_log1p_amount": probability * log_amount,
        }
    )


def _p_only_features() -> pd.DataFrame:
    return pd.DataFrame({"p_fraud": np.linspace(0.0, 1.0, 1000)})


def test_bce_baseline_ordering_and_ties_remain_exact() -> None:
    assert bce_baseline.bce_rank_positions is composition.bce_rank_positions
    assert bce_baseline.validate_full_ranking is composition.validate_full_ranking
    pool = build_candidate_pool(
        np.array([0.2, 0.9, 0.9, 0.1, 0.8]),
        np.arange(5),
        candidate_pool_size=3,
    )
    ranking = bce_baseline.baseline_bce_ranking(pool)
    assert (
        ranking.sort_values("final_rank_position")["original_position"].tolist()
        == [1, 2, 4, 0, 3]
    )
    np.testing.assert_allclose(ranking["raw_ranker_score"], pool["p_fraud"])
    assert ranking["candidate_rank_by_ranker"].isna().all()
    assert sorted(ranking["final_rank_position"]) == [1, 2, 3, 4, 5]
    assert not [
        name
        for name in vars(bce_baseline)
        if name.startswith("fit") or "Ranker" in name
    ]


def test_fixed_reference_formula_tuple_and_no_fit_boundary() -> None:
    assert (
        fixed_reference.compose_candidate_reranking
        is composition.compose_candidate_reranking
    )
    pool = build_candidate_pool(
        np.array([0.2, 0.9, 0.9, 0.1, 0.8]),
        np.arange(5),
        candidate_pool_size=3,
    )
    amount = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    candidates = candidate_rows_by_bce(pool)
    positions = candidates["original_position"].to_numpy(dtype=int)
    expected = candidates["p_fraud"].to_numpy() * np.log1p(amount[positions])
    raw_scores, ranking = fixed_reference.fixed_reference_ranking(pool, amount)
    np.testing.assert_allclose(raw_scores, expected)
    assert sorted(ranking["final_rank_position"]) == [1, 2, 3, 4, 5]
    assert not [
        name
        for name in vars(fixed_reference)
        if name.startswith("fit") or "Ranker" in name
    ]


def test_inner_amount_gain_path_uses_validation_and_fresh_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        amount_gain.compose_candidate_reranking
        is composition.compose_candidate_reranking
    )
    _FakeRanker.instances.clear()
    monkeypatch.setattr(amount_gain, "CandidateAmountGainRanker", _FakeRanker)
    composed: list[tuple[pd.DataFrame, np.ndarray]] = []
    ranking = pd.DataFrame({"result": ["inner"]})
    monkeypatch.setattr(
        amount_gain,
        "compose_candidate_reranking",
        lambda pool, scores: composed.append((pool, scores)) or ranking,
    )
    train_pool = _pool()
    validation_pool = _pool()
    train_features = _amount_features()
    validation_features = _amount_features()
    train_relevance = np.tile(np.arange(5), 200)
    validation_relevance = train_relevance.copy()

    result = amount_gain.fit_inner_amount_gain_path(
        train_features=train_features,
        validation_features=validation_features,
        train_candidate_relevance=train_relevance,
        validation_candidate_relevance=validation_relevance,
        train_candidate_pool=train_pool,
        validation_candidate_pool=validation_pool,
        target_budget=20,
        gain_profile="linear",
        random_state=42,
        effective_config=_CANONICAL,
    )
    config, ranker, raw_scores, returned_ranking = result
    assert isinstance(config, CandidateRankerConfig)
    assert ranker is _FakeRanker.instances[0]
    assert ranker.fit_args == (train_features, train_relevance, [1000])
    assert ranker.fit_kwargs == {
        "eval_X": validation_features,
        "eval_relevance": validation_relevance,
        "eval_group": [1000],
    }
    assert ranker.fit_args[2] is not ranker.fit_kwargs["eval_group"]
    assert ranker.score_features is validation_features
    assert len(composed) == 1
    assert composed[0][0] is validation_pool
    assert composed[0][1] is raw_scores
    assert returned_ranking is ranking


def test_final_learned_paths_use_distinct_models_and_same_relevance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert p_only.compose_candidate_reranking is composition.compose_candidate_reranking
    _FakeRanker.instances.clear()
    monkeypatch.setattr(amount_gain, "CandidateAmountGainRanker", _FakeRanker)
    monkeypatch.setattr(p_only, "CandidateAmountGainRanker", _FakeRanker)
    amount_ranking = pd.DataFrame({"result": ["amount"]})
    p_only_ranking = pd.DataFrame({"result": ["p_only"]})
    monkeypatch.setattr(
        amount_gain,
        "compose_candidate_reranking",
        lambda pool, scores: amount_ranking,
    )
    monkeypatch.setattr(
        p_only,
        "compose_candidate_reranking",
        lambda pool, scores: p_only_ranking,
    )
    pool = _pool()
    relevance = np.tile(np.arange(5), 200)
    amount_train = _amount_features()
    amount_test = _amount_features()
    probability_train = _p_only_features()
    probability_test = _p_only_features()

    amount_result = amount_gain.fit_final_amount_gain_path(
        train_features=amount_train,
        test_features=amount_test,
        train_candidate_relevance=relevance,
        test_candidate_pool=pool,
        target_budget=50,
        selected_gain="exponential",
        random_state=13,
        final_n_estimators=77,
        effective_config=_CANONICAL,
    )
    p_only_result = p_only.fit_final_p_only_path(
        train_features=probability_train,
        test_features=probability_test,
        train_candidate_relevance=relevance,
        test_candidate_pool=pool,
        target_budget=50,
        selected_gain="exponential",
        random_state=13,
        final_n_estimators=77,
        effective_config=_CANONICAL,
    )
    amount_config, amount_ranker, amount_raw, returned_amount = amount_result
    probability_config, probability_ranker, probability_raw, returned_p = (
        p_only_result
    )
    assert amount_ranker is not probability_ranker
    assert amount_config is not probability_config
    assert amount_config == probability_config
    assert amount_ranker.fit_args == (amount_train, relevance, [1000])
    assert probability_ranker.fit_args == (probability_train, relevance, [1000])
    assert amount_ranker.fit_args[1] is probability_ranker.fit_args[1] is relevance
    assert amount_ranker.fit_args[2] is not probability_ranker.fit_args[2]
    assert amount_ranker.fit_kwargs == probability_ranker.fit_kwargs == {}
    assert list(probability_train.columns) == ["p_fraud"]
    assert list(probability_test.columns) == ["p_fraud"]
    assert amount_ranker.score_features is amount_test
    assert probability_ranker.score_features is probability_test
    np.testing.assert_array_equal(amount_raw, np.arange(1000, dtype=float))
    np.testing.assert_array_equal(probability_raw, np.arange(1000, dtype=float))
    assert returned_amount is amount_ranking
    assert returned_p is p_only_ranking
    assert "amount" not in signature(p_only.fit_final_p_only_path).parameters
    with pytest.raises(ValueError, match="exactly p_fraud"):
        p_only.fit_final_p_only_path(
            train_features=amount_train,
            test_features=probability_test,
            train_candidate_relevance=relevance,
            test_candidate_pool=pool,
            target_budget=50,
            selected_gain="exponential",
            random_state=13,
            final_n_estimators=77,
            effective_config=_CANONICAL,
        )


def test_smoke_ranker_boundary_receives_reduced_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = resolve_experiment_profile("smoke-synthetic")
    _FakeRanker.instances.clear()
    monkeypatch.setattr(amount_gain, "CandidateAmountGainRanker", _FakeRanker)
    monkeypatch.setattr(
        amount_gain,
        "compose_candidate_reranking",
        lambda _pool, _scores: pd.DataFrame({"result": ["smoke"]}),
    )
    features = pd.DataFrame(
        np.arange(600, dtype=float).reshape(200, 3),
        columns=["p_fraud", "log1p_amount", "p_fraud_x_log1p_amount"],
    )
    pool = _pool(rows=200, pool_size=200)

    config, ranker, _scores, _ranking = amount_gain.fit_inner_amount_gain_path(
        train_features=features,
        validation_features=features.copy(),
        train_candidate_relevance=np.tile(np.arange(5), 40),
        validation_candidate_relevance=np.tile(np.arange(5), 40),
        train_candidate_pool=pool,
        validation_candidate_pool=pool.copy(),
        target_budget=20,
        gain_profile="linear",
        random_state=42,
        effective_config=smoke,
    )

    assert config.effective_config is smoke
    assert config.n_estimators == 30
    assert config.early_stopping_rounds == 5
    assert ranker.fit_args is not None
    assert ranker.fit_args[2] == [200]
    assert ranker.fit_kwargs is not None
    assert ranker.fit_kwargs["eval_group"] == [200]
