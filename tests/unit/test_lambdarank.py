import sys
from importlib import reload
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fraud_detection.experiment.config import resolve_experiment_profile
from fraud_detection.experiment.prioritization import lambdarank as _lambdarank

pytestmark = pytest.mark.unit

_CANONICAL = resolve_experiment_profile("canonical")


class _FakeModel:
    instances: list["_FakeModel"] = []

    def __init__(self, **parameters: object) -> None:
        self.parameters = parameters
        self.fit_args: tuple[object, ...] | None = None
        self.fit_kwargs: dict[str, object] | None = None
        self.best_iteration_ = 7
        self.evals_result_ = {
            "inner_validation": {
                "ndcg@20": [float(value) / 10.0 for value in range(1, 8)]
            }
        }
        self.prediction = np.linspace(0.0, 1.0, 1000)
        self.predict_iteration: int | None = None
        self.n_estimators_ = int(parameters["n_estimators"])
        self.instances.append(self)

    def fit(self, *args: object, **kwargs: object) -> "_FakeModel":
        self.fit_args = args
        self.fit_kwargs = kwargs
        return self

    def predict(self, X: object, *, num_iteration: int) -> np.ndarray:
        self.predict_iteration = num_iteration
        return self.prediction


def _install_fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    callbacks: list[dict[str, object]] = []

    def early_stopping(**kwargs: object) -> dict[str, object]:
        callbacks.append(kwargs)
        return kwargs

    _FakeModel.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(
            LGBMRanker=_FakeModel,
            early_stopping=early_stopping,
        ),
    )
    return callbacks


def _features(columns: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        np.arange(1000 * columns, dtype=float).reshape(1000, columns),
        columns=[f"feature_{index}" for index in range(columns)],
    )


def test_import_is_lightgbm_lazy_and_config_parameters_are_frozen() -> None:
    previous = sys.modules.pop("lightgbm", None)
    try:
        module = reload(_lambdarank)
        assert "lightgbm" not in sys.modules
    finally:
        if previous is not None:
            sys.modules["lightgbm"] = previous

    config = module.CandidateRankerConfig(
        target_budget=20,
        gain_profile="linear",
        effective_config=_CANONICAL,
        random_state=13,
    )
    assert config.label_gain == (0, 1, 2, 3, 4)
    assert config.truncation_level == 23
    assert config.eval_at == (20,)
    assert config.model_parameters() == {
        "objective": "lambdarank",
        "label_gain": [0, 1, 2, 3, 4],
        "lambdarank_truncation_level": 23,
        "random_state": 13,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_child_samples": 20,
        "min_child_weight": 0.001,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "verbosity": -1,
    }


def test_invalid_ranker_config_values_are_rejected() -> None:
    invalid = (
        {"target_budget": 6, "gain_profile": "linear"},
        {"target_budget": 20, "gain_profile": "unknown"},
        {"target_budget": 20, "gain_profile": "linear", "random_state": True},
        {"target_budget": 20, "gain_profile": "linear", "n_estimators": 0},
        {"target_budget": 20, "gain_profile": "linear", "n_estimators": 501},
        {
            "target_budget": 20,
            "gain_profile": "linear",
            "early_stopping_rounds": 49,
        },
    )
    for values in invalid:
        with pytest.raises(ValueError):
            _lambdarank.CandidateRankerConfig(
                effective_config=_CANONICAL,
                **values,
            )


def test_inner_fit_preserves_groups_early_stopping_and_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks = _install_fake_lightgbm(monkeypatch)
    config = _lambdarank.CandidateRankerConfig(
        target_budget=20,
        gain_profile="exponential",
        effective_config=_CANONICAL,
        random_state=42,
    )
    ranker = _lambdarank.CandidateAmountGainRanker(config).fit(
        _features(),
        np.tile(np.arange(5), 200),
        [1000],
        eval_X=_features(),
        eval_relevance=np.tile(np.arange(5), 200),
        eval_group=[1000],
    )
    model = _FakeModel.instances[0]
    assert model.fit_kwargs is not None
    assert model.fit_kwargs["group"] == [1000]
    assert model.fit_kwargs["eval_group"] == [[1000]]
    assert model.fit_kwargs["eval_names"] == ["inner_validation"]
    assert model.fit_kwargs["eval_metric"] == "ndcg"
    assert model.fit_kwargs["eval_at"] == (20,)
    assert callbacks == [
        {
            "stopping_rounds": 50,
            "first_metric_only": True,
            "verbose": False,
        }
    ]
    assert ranker.best_iteration_ == 7
    assert ranker.used_early_stopping_ is True
    assert ranker.evaluation_metric_ == "ndcg@20"
    assert ranker.evals_result_ == model.evals_result_


def test_final_fit_uses_fixed_tree_count_without_early_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks = _install_fake_lightgbm(monkeypatch)
    config = _lambdarank.CandidateRankerConfig(
        target_budget=50,
        gain_profile="linear",
        effective_config=_CANONICAL,
        random_state=7,
        n_estimators=123,
    )
    ranker = _lambdarank.CandidateAmountGainRanker(config).fit(
        _features(),
        np.tile(np.arange(5), 200),
        [1000],
    )
    model = _FakeModel.instances[0]
    assert model.fit_kwargs == {"group": [1000]}
    assert callbacks == []
    assert ranker.best_iteration_ == 123
    assert ranker.used_early_stopping_ is False
    assert ranker.evaluation_metric_ is None
    assert ranker.evals_result_ is None


def test_decision_function_uses_best_iteration_and_rejects_invalid_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lightgbm(monkeypatch)
    config = _lambdarank.CandidateRankerConfig(
        target_budget=20,
        gain_profile="linear",
        effective_config=_CANONICAL,
        n_estimators=11,
    )
    ranker = _lambdarank.CandidateAmountGainRanker(config).fit(
        _features(),
        np.tile(np.arange(5), 200),
        [1000],
    )
    model = _FakeModel.instances[0]
    scores = ranker.decision_function(_features())
    assert scores.shape == (1000,)
    assert model.predict_iteration == 11
    for invalid in (
        np.zeros((1000, 1)),
        np.full(1000, np.nan),
    ):
        model.prediction = invalid
        with pytest.raises(RuntimeError):
            ranker.decision_function(_features())
