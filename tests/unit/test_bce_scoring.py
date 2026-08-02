from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_detection.artifacts import _ordered_index_sha256, score_vector_sha256
from fraud_detection.experiment.config import (
    BCE_FEATURES,
    BCE_L2_ALPHA,
    BCE_LEARNING_RATE,
    BCE_MAX_ITER,
    BCE_TOL,
    EXPECTED_DEDUPLICATED_SHA256,
    resolve_experiment_profile,
)
from fraud_detection.experiment.preparation import bce as bce_scoring
from fraud_detection.experiment.preparation.bce import (
    BCEFitConfig,
    BCELogisticRegression,
    bce_gradient_logits,
    binary_cross_entropy_from_logits,
    fit_inner_bce_fold,
    fit_outer_bce_scores_after_freeze,
    generate_oof_bce_scores,
    sigmoid_stable,
)

pytestmark = pytest.mark.unit


_DIAGNOSTIC_COLUMNS = [
    "outer_seed",
    "inner_fold",
    "fit_type",
    "oof_fold",
    "fit_row_count",
    "fit_fraud_count",
    "score_row_count",
    "learning_rate",
    "tol",
    "max_iter",
    "l2_alpha",
    "n_iter_",
    "success_",
    "converged_by_tolerance",
    "message_",
    "first_loss",
    "penultimate_loss",
    "final_loss",
    "final_absolute_loss_change",
    "loss_history_finite",
    "coefficients_finite",
    "coefficient_l2_norm",
    "intercept",
    "scores_finite",
    "score_min",
    "score_max",
    "score_sha256",
]


def _config(random_state: int = 42, **changes: Any) -> BCEFitConfig:
    values = {
        "learning_rate": BCE_LEARNING_RATE,
        "tol": BCE_TOL,
        "max_iter": BCE_MAX_ITER,
        "l2_alpha": BCE_L2_ALPHA,
        "random_state": random_state,
    }
    values.update(changes)
    return BCEFitConfig(**values)


def _finite_difference(function, logits: np.ndarray) -> np.ndarray:
    epsilon = 1e-6
    gradient = np.zeros_like(logits, dtype=float)
    for index in range(len(logits)):
        step = np.zeros_like(logits, dtype=float)
        step[index] = epsilon
        gradient[index] = (
            function(logits + step) - function(logits - step)
        ) / (2 * epsilon)
    return gradient


def _same_array_bits(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _fit_reference_with_original_arithmetic(
    X: np.ndarray,
    y: np.ndarray,
    config: BCEFitConfig,
) -> dict[str, Any]:
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)
    coef = np.zeros(X_array.shape[1], dtype=float)
    intercept = 0.0
    loss_history: list[float] = []
    previous_loss = None

    for iteration in range(1, config.max_iter + 1):
        logits = X_array @ coef + intercept
        per_sample = (
            np.logaddexp(0.0, logits) - y_array * logits
        )
        base_loss = float(np.mean(per_sample))
        probabilities = sigmoid_stable(logits)
        gradient = probabilities - y_array
        gradient_logits = gradient / gradient.shape[0]
        l2_term = 0.5 * config.l2_alpha * np.sum(coef * coef)
        loss = float(base_loss + l2_term)
        loss_history.append(loss)
        gradient_coef = X_array.T @ gradient_logits + config.l2_alpha * coef
        gradient_intercept = float(np.sum(gradient_logits))
        coef -= config.learning_rate * gradient_coef
        intercept -= config.learning_rate * gradient_intercept

        if (
            previous_loss is not None
            and abs(previous_loss - loss) < config.tol
            and iteration < config.max_iter
        ):
            return {
                "coef": coef,
                "intercept": float(intercept),
                "n_iter": iteration,
                "loss_history": np.asarray(loss_history, dtype=float),
                "converged": True,
                "message": "Converged by loss difference tolerance.",
            }
        previous_loss = loss

    return {
        "coef": coef,
        "intercept": float(intercept),
        "n_iter": config.max_iter,
        "loss_history": np.asarray(loss_history, dtype=float),
        "converged": False,
        "message": "Reached maximum iterations without convergence.",
    }


def _assert_fit_matches_validated_reference(
    X: np.ndarray,
    y: np.ndarray,
    config: BCEFitConfig,
) -> BCELogisticRegression:
    reference = _fit_reference_with_original_arithmetic(X, y, config)
    model = BCELogisticRegression(config).fit(X, y)
    assert _same_array_bits(model.coef_, reference["coef"])
    assert np.float64(model.intercept_).tobytes() == np.float64(
        reference["intercept"]
    ).tobytes()
    assert model.n_iter_ == reference["n_iter"]
    assert _same_array_bits(
        np.asarray(model.loss_history_, dtype=float),
        reference["loss_history"],
    )
    assert model.classes_.tolist() == [0, 1]
    assert model.n_features_in_ == X.shape[1]
    assert model.success_ is True
    assert model.converged_by_tolerance_ is reference["converged"]
    assert model.message_ == reference["message"]

    expected_scores = X @ reference["coef"] + reference["intercept"]
    actual_scores = model.decision_function(X)
    assert _same_array_bits(actual_scores, expected_scores)
    expected_probability_1 = 1.0 / (1.0 + np.exp(-expected_scores))
    expected_probabilities = np.column_stack(
        (1.0 - expected_probability_1, expected_probability_1)
    )
    actual_probabilities = model.predict_proba(X)
    assert _same_array_bits(actual_probabilities, expected_probabilities)
    assert score_vector_sha256(
        actual_probabilities[:, 1],
        score_type="bce-validation-hoist-test",
    ) == score_vector_sha256(
        expected_probabilities[:, 1],
        score_type="bce-validation-hoist-test",
    )
    return model


def _assert_public_helper_validation_unchanged(function) -> None:
    invalid_cases = (
        ((np.array([0.0]), np.array([0.0])), {"reduction": "invalid"},
         "reduction must be one of {'mean', 'sum', 'none'}."),
        ((np.array([[0.0]]), np.array([0.0])), {},
         "logits must be one-dimensional."),
        ((np.array([]), np.array([])), {},
         "logits must contain at least one element."),
        ((np.array([np.nan]), np.array([0.0])), {},
         "logits must not contain NaN or infinite values."),
        ((np.array([0.0]), np.array([[0.0]])), {},
         "y must be one-dimensional."),
        ((np.array([0.0]), np.array([])), {},
         "y must contain at least one element."),
        ((np.array([0.0]), np.array([np.inf])), {},
         "y must not contain NaN or infinite values."),
        ((np.array([0.0, 1.0]), np.array([0.0])), {},
         "All input arrays must have the same length."),
        ((np.array([0.0]), np.array([2.0])), {},
         "y must be binary with values {0, 1}."),
    )
    for arguments, keywords, expected_message in invalid_cases:
        with pytest.raises(ValueError) as error:
            function(*arguments, **keywords)
        assert str(error.value) == expected_message


class _FakeBCEModel:
    instances: list[_FakeBCEModel] = []

    def __init__(self, config: BCEFitConfig) -> None:
        self.config = config
        self.coef_ = np.zeros(len(BCE_FEATURES), dtype=float)
        self.intercept_ = 0.0
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = len(BCE_FEATURES)
        self.n_iter_ = 2
        self.loss_history_ = [0.7, 0.6]
        self.success_ = True
        self.converged_by_tolerance_ = True
        self.message_ = "Converged by loss difference tolerance."
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        type(self).instances.append(self)

    def fit(self, X: object, y: object) -> _FakeBCEModel:
        self.fit_X = np.asarray(X, dtype=float).copy()
        self.fit_y = np.asarray(y, dtype=int).copy()
        self.coef_ = np.zeros(self.fit_X.shape[1], dtype=float)
        self.n_features_in_ = self.fit_X.shape[1]
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        row_count = np.asarray(X).shape[0]
        probability = np.linspace(0.2, 0.8, row_count, dtype=float)
        return np.column_stack((1.0 - probability, probability))


class _RecordingScaler:
    instances: list[_RecordingScaler] = []

    def __init__(self) -> None:
        self.fit_input: np.ndarray | None = None
        self.transform_inputs: list[np.ndarray] = []
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        type(self).instances.append(self)

    def fit_transform(self, X: object) -> np.ndarray:
        self.fit_input = np.asarray(X, dtype=float).copy()
        self.mean_ = self.fit_input.mean(axis=0)
        scale = self.fit_input.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return (self.fit_input - self.mean_) / self.scale_

    def transform(self, X: object) -> np.ndarray:
        values = np.asarray(X, dtype=float).copy()
        self.transform_inputs.append(values)
        assert self.mean_ is not None
        assert self.scale_ is not None
        return (values - self.mean_) / self.scale_


def _bce_frame(indices: np.ndarray) -> pd.DataFrame:
    row_count = len(indices)
    values = {
        feature: np.linspace(
            position,
            position + 1.0,
            row_count,
            dtype=float,
        )
        for position, feature in enumerate(BCE_FEATURES, start=1)
    }
    values["Class"] = np.resize(np.array([0, 1], dtype=int), row_count)
    return pd.DataFrame(values, index=indices)


def _fake_oof_scores(
    X: object,
    y: object,
    *,
    n_splits: int,
    config: BCEFitConfig,
    fit_diagnostics_callback=None,
) -> np.ndarray:
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=int)
    scores = np.empty(X_array.shape[0], dtype=float)
    positions = np.arange(X_array.shape[0])
    for fold_number, holdout in enumerate(
        np.array_split(positions, n_splits),
        start=1,
    ):
        train = np.setdiff1d(positions, holdout)
        model = _FakeBCEModel(config).fit(X_array[train], y_array[train])
        holdout_scores = model.predict_proba(X_array[holdout])[:, 1]
        scores[holdout] = holdout_scores
        if fit_diagnostics_callback is not None:
            fit_diagnostics_callback(
                fold_number,
                model,
                train.copy(),
                holdout.copy(),
                holdout_scores.copy(),
            )
    return scores


def _record_events() -> tuple[list[tuple[str, str, dict[str, Any]]], Any]:
    events: list[tuple[str, str, dict[str, Any]]] = []

    def emit(level: str, event: str, **fields: Any) -> None:
        events.append((level, event, fields))

    return events, emit


def _reset_fakes() -> None:
    _FakeBCEModel.instances = []
    _RecordingScaler.instances = []


def test_stable_sigmoid_and_bce_formula_remain_finite() -> None:
    extreme = np.array([-1000.0, -20.0, 0.0, 20.0, 1000.0])
    probabilities = sigmoid_stable(extreme)
    assert np.isfinite(probabilities).all()
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))

    logits = np.array([0.0, 1.0, -1.0])
    target = np.array([0.0, 1.0, 0.0])
    expected = np.logaddexp(0.0, logits) - target * logits
    actual = binary_cross_entropy_from_logits(
        logits,
        target,
        reduction="none",
    )
    np.testing.assert_allclose(actual, expected)
    assert binary_cross_entropy_from_logits(
        logits,
        target,
        reduction="sum",
    ) == float(np.sum(expected))
    assert binary_cross_entropy_from_logits(
        logits,
        target,
        reduction="mean",
    ) == float(np.mean(expected))
    _assert_public_helper_validation_unchanged(
        binary_cross_entropy_from_logits
    )


def test_bce_gradient_matches_definition_and_finite_difference() -> None:
    logits = np.array([0.3, -0.7, 1.2])
    target = np.array([0.0, 1.0, 1.0])
    np.testing.assert_allclose(
        bce_gradient_logits(logits, target, reduction="none"),
        sigmoid_stable(logits) - target,
    )
    def function(values: np.ndarray) -> float:
        return float(
            binary_cross_entropy_from_logits(values, target, reduction="mean")
        )
    np.testing.assert_allclose(
        bce_gradient_logits(logits, target, reduction="mean"),
        _finite_difference(function, logits),
        atol=1e-5,
    )
    expected = sigmoid_stable(logits) - target
    assert _same_array_bits(
        np.asarray(bce_gradient_logits(logits, target, reduction="sum")),
        expected,
    )
    _assert_public_helper_validation_unchanged(bce_gradient_logits)


def test_bce_fit_config_accepts_only_canonical_shape() -> None:
    config = _config(random_state=7)
    assert config == BCEFitConfig(
        learning_rate=0.1,
        tol=1e-6,
        max_iter=10_000,
        l2_alpha=0.0,
        random_state=7,
    )

    invalid = (
        {"learning_rate": 0.0},
        {"learning_rate": -0.1},
        {"tol": 0.0},
        {"tol": -1e-6},
        {"max_iter": 0},
        {"max_iter": -1},
        {"l2_alpha": -0.01},
        {"random_state": True},
        {"random_state": 1.5},
    )
    for changes in invalid:
        with pytest.raises(ValueError):
            _config(**changes)

    rng = np.random.default_rng(314159)
    X = rng.normal(0.0, 0.005, size=(24, len(BCE_FEATURES)))
    # An identical feature row with opposite labels makes separation impossible.
    X[1] = X[0]
    y = np.tile(np.array([0, 1], dtype=int), 12)
    first = BCELogisticRegression(_config(random_state=314159)).fit(X, y)
    second = BCELogisticRegression(_config(random_state=314159)).fit(X, y)

    for model in (first, second):
        scores = model.predict_proba(X)
        bce_scoring._validate_bce_fit(
            model,
            scores[:, 1],
            context="real-estimator-test",
        )
        assert model.n_features_in_ == len(BCE_FEATURES) == 28
        assert model.success_ is True
        assert model.converged_by_tolerance_ is True
        assert model.n_iter_ is not None
        assert 1 <= model.n_iter_ < BCE_MAX_ITER
        assert len(model.loss_history_) == model.n_iter_
        assert np.isfinite(model.loss_history_).all()
        assert np.isfinite(model.coef_).all()
        assert np.isfinite(model.intercept_)
        assert scores.shape == (24, 2)
        assert np.isfinite(scores).all()
        assert np.all((0.0 <= scores) & (scores <= 1.0))
        np.testing.assert_allclose(scores.sum(axis=1), 1.0)

    np.testing.assert_allclose(first.coef_, second.coef_)
    np.testing.assert_allclose(first.intercept_, second.intercept_)
    np.testing.assert_allclose(
        first.predict_proba(X),
        second.predict_proba(X),
    )

    balanced_model = _assert_fit_matches_validated_reference(
        X,
        y,
        _config(random_state=314159),
    )
    assert balanced_model.converged_by_tolerance_ is True

    imbalanced_X = np.zeros((40, len(BCE_FEATURES)), dtype=float)
    imbalanced_y = np.zeros(40, dtype=float)
    imbalanced_y[:2] = 1.0
    imbalanced_model = _assert_fit_matches_validated_reference(
        imbalanced_X,
        imbalanced_y,
        _config(random_state=271828),
    )
    assert imbalanced_model.converged_by_tolerance_ is True

    maximum_X = rng.normal(size=(48, len(BCE_FEATURES)))
    maximum_y = (maximum_X[:, 0] >= 0.0).astype(float)
    maximum_model = _assert_fit_matches_validated_reference(
        maximum_X,
        maximum_y,
        _config(max_iter=37, tol=1e-30),
    )
    assert maximum_model.n_iter_ == 37
    assert maximum_model.converged_by_tolerance_ is False

    invalid_fit_cases = (
        (np.array([0.0]), np.array([0.0]),
         "X must be two-dimensional."),
        (np.empty((0, len(BCE_FEATURES))), np.array([]),
         "X must contain at least one sample and one feature."),
        (np.full((1, len(BCE_FEATURES)), np.nan), np.array([0.0]),
         "X must not contain NaN or infinite values."),
        (np.zeros((1, len(BCE_FEATURES))), np.array([[0.0]]),
         "y must be one-dimensional."),
        (np.zeros((1, len(BCE_FEATURES))), np.array([]),
         "y must contain at least one element."),
        (np.zeros((2, len(BCE_FEATURES))), np.array([0.0]),
         "X and y must have the same number of samples."),
        (np.zeros((1, len(BCE_FEATURES))), np.array([np.inf]),
         "y must not contain NaN or infinite values."),
        (np.zeros((1, len(BCE_FEATURES))), np.array([2.0]),
         "y must be binary with values {0, 1}."),
    )
    for invalid_X, invalid_y, expected_message in invalid_fit_cases:
        with pytest.raises(ValueError) as error:
            BCELogisticRegression(_config()).fit(invalid_X, invalid_y)
        assert str(error.value) == expected_message


def test_bce_fit_validation_rejects_non_convergence() -> None:
    model = _FakeBCEModel(_config())
    model.converged_by_tolerance_ = False
    with pytest.raises(RuntimeError, match="did not converge"):
        bce_scoring._validate_bce_fit(
            model,
            np.array([0.2, 0.8]),
            context="test",
        )


def test_bce_fit_validation_rejects_invalid_state_and_scores() -> None:
    model = _FakeBCEModel(_config())
    model.n_iter_ = 3
    model.loss_history_ = [0.7, 0.6]
    with pytest.raises(RuntimeError, match="loss history"):
        bce_scoring._validate_bce_fit(
            model,
            np.array([0.2, 0.8]),
            context="test",
        )

    model = _FakeBCEModel(_config())
    model.coef_[0] = np.nan
    with pytest.raises(RuntimeError, match="coefficients"):
        bce_scoring._validate_bce_fit(
            model,
            np.array([0.2, 0.8]),
            context="test",
        )

    model = _FakeBCEModel(_config())
    model.intercept_ = np.inf
    with pytest.raises(RuntimeError, match="coefficients"):
        bce_scoring._validate_bce_fit(
            model,
            np.array([0.2, 0.8]),
            context="test",
        )

    model = _FakeBCEModel(_config())
    with pytest.raises(RuntimeError, match="non-finite BCE scores"):
        bce_scoring._validate_bce_fit(
            model,
            np.array([0.2, np.nan]),
            context="test",
        )
    with pytest.raises(RuntimeError, match=r"outside \[0,1\]"):
        bce_scoring._validate_bce_fit(
            model,
            np.array([0.2, 1.1]),
            context="test",
        )


def test_oof_scoring_is_deterministic_complete_and_fold_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = np.column_stack(
        (
            np.arange(20, dtype=float),
            np.linspace(-2.0, 3.0, 20),
        )
    )
    y = np.array([0, 1] * 10, dtype=int)
    monkeypatch.setattr(bce_scoring, "BCELogisticRegression", _FakeBCEModel)
    monkeypatch.setattr(bce_scoring, "StandardScaler", _RecordingScaler)

    def run_once():
        _reset_fakes()
        callback_rows = []

        def capture(fold, _model, train, holdout, scores):
            callback_rows.append(
                (fold, train.copy(), holdout.copy(), scores.copy())
            )
            train[:] = -1
            holdout[:] = -1
            scores[:] = -1.0

        result = generate_oof_bce_scores(
            X,
            y,
            n_splits=5,
            config=_config(random_state=13),
            fit_diagnostics_callback=capture,
        )
        return (
            result,
            callback_rows,
            list(_RecordingScaler.instances),
        )

    first_scores, first_callbacks, first_scalers = run_once()
    second_scores, second_callbacks, _second_scalers = run_once()

    np.testing.assert_allclose(first_scores, second_scores)
    assert np.isfinite(first_scores).all()
    assert np.all((0.0 <= first_scores) & (first_scores <= 1.0))
    assert [row[0] for row in first_callbacks] == [1, 2, 3, 4, 5]
    assert [row[0] for row in second_callbacks] == [1, 2, 3, 4, 5]
    for first, second in zip(first_callbacks, second_callbacks, strict=True):
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
    assigned = np.concatenate([row[2] for row in first_callbacks])
    np.testing.assert_array_equal(np.sort(assigned), np.arange(len(y)))
    for _fold, _train, holdout, _scores in first_callbacks:
        np.testing.assert_array_equal(np.bincount(y[holdout]), np.array([2, 2]))
    assert len(first_scalers) == 5
    for scaler, (_, train, holdout, _scores) in zip(
        first_scalers,
        first_callbacks,
        strict=True,
    ):
        np.testing.assert_array_equal(scaler.fit_input, X[train])
        np.testing.assert_array_equal(scaler.transform_inputs[0], X[holdout])


def test_inner_bce_contract_uses_separate_full_training_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    inner_train = _bce_frame(np.arange(100, 120))
    inner_validation = _bce_frame(np.arange(200, 208))
    events, emit = _record_events()
    monkeypatch.setattr(bce_scoring, "BCELogisticRegression", _FakeBCEModel)
    monkeypatch.setattr(bce_scoring, "StandardScaler", _RecordingScaler)
    monkeypatch.setattr(
        bce_scoring,
        "generate_oof_bce_scores",
        _fake_oof_scores,
    )
    monkeypatch.setattr(bce_scoring, "_emit_experiment_status", emit)

    result = fit_inner_bce_fold(
        outer_seed=42,
        inner_fold=2,
        inner_train=inner_train,
        inner_validation=inner_validation,
        effective_config=resolve_experiment_profile("canonical"),
    )

    assert list(result) == [
        "p_train_oof",
        "p_validation",
        "oof_fold_number",
        "diagnostics",
        "loss_history",
        "scaler_mean",
        "scaler_scale",
    ]
    assert len(_FakeBCEModel.instances) == 6
    assert len(_RecordingScaler.instances) == 1
    scaler = _RecordingScaler.instances[0]
    np.testing.assert_array_equal(
        scaler.fit_input,
        inner_train.loc[:, BCE_FEATURES].to_numpy(dtype=float),
    )
    np.testing.assert_array_equal(
        scaler.transform_inputs[0],
        inner_validation.loc[:, BCE_FEATURES].to_numpy(dtype=float),
    )
    assert {
        model.config.random_state for model in _FakeBCEModel.instances
    } == {100042}
    assert result["diagnostics"]["fit_type"].tolist() == [
        "OOF",
        "OOF",
        "OOF",
        "OOF",
        "OOF",
        "Full Inner Train",
    ]
    assert result["diagnostics"].columns.tolist() == _DIAGNOSTIC_COLUMNS
    assert result["loss_history"].columns.tolist() == [
        "outer_seed",
        "inner_fold",
        "fit_type",
        "oof_fold",
        "iteration",
        "loss",
    ]
    assert len(events) == 6
    assert {event for _level, event, _fields in events} == {
        "inner-bce-complete"
    }
    assert {fields["total"] for _level, _event, fields in events} == {90}


def test_outer_bce_contract_preserves_tuple_frames_metadata_and_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _reset_fakes()
    train_index = np.arange(100, 120)
    test_index = np.arange(200, 208)
    dataframe = pd.concat(
        (_bce_frame(train_index), _bce_frame(test_index)),
    )
    events, emit = _record_events()
    identity = pd.Series(
        {
            "train_index_sha256": _ordered_index_sha256(train_index),
            "test_index_sha256": _ordered_index_sha256(test_index),
        }
    )
    monkeypatch.setattr(bce_scoring, "BCELogisticRegression", _FakeBCEModel)
    monkeypatch.setattr(bce_scoring, "StandardScaler", _RecordingScaler)
    monkeypatch.setattr(
        bce_scoring,
        "generate_oof_bce_scores",
        _fake_oof_scores,
    )
    monkeypatch.setattr(
        bce_scoring,
        "preflight_split_identity",
        lambda _output_root, _seed: identity,
    )
    monkeypatch.setattr(bce_scoring, "_emit_experiment_status", emit)

    result = fit_outer_bce_scores_after_freeze(
        dataframe=dataframe,
        output_root=tmp_path,
        outer_seed=7,
        train_index=train_index,
        test_index=test_index,
        effective_config=resolve_experiment_profile("canonical"),
        data_sha256=EXPECTED_DEDUPLICATED_SHA256,
    )
    oof, test, metadata, diagnostics, loss_history = result

    assert oof.columns.tolist() == ["train_index", "y_true", "p_oof"]
    assert test.columns.tolist() == [
        "test_index",
        "y_true",
        "p_full_train_test",
    ]
    assert list(metadata) == [
        "deduplicated_dataframe_sha256",
        "bce_learning_rate",
        "bce_tol",
        "bce_max_iter",
        "bce_l2_alpha",
        "bce_oof_folds",
        "train_index_sha256",
        "test_index_sha256",
        "oof_score_sha256",
        "test_score_sha256",
        "fit_count",
        "converged_fit_count",
        "score_source",
    ]
    assert metadata["deduplicated_dataframe_sha256"] == (
        EXPECTED_DEDUPLICATED_SHA256
    )
    assert metadata["fit_count"] == 6
    assert metadata["converged_fit_count"] == 6
    assert metadata["score_source"] == "generated_from_current_configuration"
    assert diagnostics["fit_type"].tolist() == [
        "Outer OOF",
        "Outer OOF",
        "Outer OOF",
        "Outer OOF",
        "Outer OOF",
        "Full Outer Train",
    ]
    assert diagnostics.columns.tolist() == _DIAGNOSTIC_COLUMNS
    assert loss_history.columns.tolist() == [
        "outer_seed",
        "inner_fold",
        "fit_type",
        "oof_fold",
        "iteration",
        "loss",
    ]
    assert len(_FakeBCEModel.instances) == 6
    assert len(_RecordingScaler.instances) == 1
    scaler = _RecordingScaler.instances[0]
    np.testing.assert_array_equal(
        scaler.fit_input,
        dataframe.loc[train_index, BCE_FEATURES].to_numpy(dtype=float),
    )
    np.testing.assert_array_equal(
        scaler.transform_inputs[0],
        dataframe.loc[test_index, BCE_FEATURES].to_numpy(dtype=float),
    )
    assert {model.config.random_state for model in _FakeBCEModel.instances} == {
        7
    }
    assert len(events) == 6
    assert {event for _level, event, _fields in events} == {
        "final-bce-complete"
    }
    assert {fields["total"] for _level, _event, fields in events} == {30}


def test_smoke_bce_boundary_uses_two_oof_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    observed_splits: list[int] = []

    def record_oof(*args: object, n_splits: int, **kwargs: object) -> np.ndarray:
        observed_splits.append(n_splits)
        return _fake_oof_scores(*args, n_splits=n_splits, **kwargs)

    monkeypatch.setattr(bce_scoring, "BCELogisticRegression", _FakeBCEModel)
    monkeypatch.setattr(bce_scoring, "StandardScaler", _RecordingScaler)
    monkeypatch.setattr(bce_scoring, "generate_oof_bce_scores", record_oof)
    monkeypatch.setattr(
        bce_scoring,
        "_emit_experiment_status",
        lambda *_args, **_kwargs: None,
    )

    result = fit_inner_bce_fold(
        outer_seed=42,
        inner_fold=1,
        inner_train=_bce_frame(np.arange(100, 120)),
        inner_validation=_bce_frame(np.arange(200, 208)),
        effective_config=resolve_experiment_profile("smoke-synthetic"),
    )

    assert observed_splits == [2]
    assert len(result["diagnostics"]) == 3
