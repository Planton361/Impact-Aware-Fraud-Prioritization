"""Canonical BCE scoring boundary for the frozen experiment runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from fraud_detection.artifacts import (
    _ordered_index_sha256,
    score_vector_sha256,
)

from ..config import (
    BCE_FEATURES,
    BCE_L2_ALPHA,
    BCE_LEARNING_RATE,
    BCE_MAX_ITER,
    BCE_TOL,
    EffectiveExperimentConfig,
    _emit_experiment_status,
    _final_bce_completed,
    _inner_bce_completed,
)
from .data import preflight_split_identity

Reduction = Literal["mean", "sum", "none"]


def sigmoid_stable(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for logits."""
    values = np.asarray(logits, dtype=float)
    probabilities = np.empty_like(values, dtype=float)
    positive = values >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1.0 + exp_values)
    return probabilities


def _as_1d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one element.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or infinite values.")
    return array


def _validate_binary_labels(y: np.ndarray) -> np.ndarray:
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("y must be binary with values {0, 1}.")
    return y


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")


def _validate_lengths(*arrays: np.ndarray) -> int:
    sample_count = arrays[0].shape[0]
    for array in arrays[1:]:
        if array.shape[0] != sample_count:
            raise ValueError("All input arrays must have the same length.")
    return sample_count


def _apply_reduction(
    values: np.ndarray,
    reduction: Reduction,
) -> np.ndarray | float:
    if reduction == "none":
        return values
    if reduction == "sum":
        return float(np.sum(values))
    return float(np.mean(values))


def _reduce_gradient(
    values: np.ndarray,
    reduction: Reduction,
) -> np.ndarray | float:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values
    return values / values.shape[0]


def _binary_cross_entropy_from_validated_arrays(
    logits_array: np.ndarray,
    y_array: np.ndarray,
) -> float:
    per_sample = np.logaddexp(0.0, logits_array) - y_array * logits_array
    return float(np.mean(per_sample))


def _bce_gradient_from_validated_arrays(
    logits_array: np.ndarray,
    y_array: np.ndarray,
) -> np.ndarray:
    probabilities = sigmoid_stable(logits_array)
    gradient = probabilities - y_array
    return gradient / gradient.shape[0]


def binary_cross_entropy_from_logits(
    logits: Any,
    y: Any,
    reduction: Reduction = "mean",
) -> np.ndarray | float:
    _validate_reduction(reduction)
    logits_array = _as_1d_float_array(logits, "logits")
    y_array = _as_1d_float_array(y, "y")
    _validate_lengths(logits_array, y_array)
    y_array = _validate_binary_labels(y_array)

    if reduction == "mean":
        return _binary_cross_entropy_from_validated_arrays(
            logits_array,
            y_array,
        )
    per_sample = np.logaddexp(0.0, logits_array) - y_array * logits_array
    return _apply_reduction(per_sample, reduction)


def bce_gradient_logits(
    logits: Any,
    y: Any,
    reduction: Reduction = "mean",
) -> np.ndarray | float:
    _validate_reduction(reduction)
    logits_array = _as_1d_float_array(logits, "logits")
    y_array = _as_1d_float_array(y, "y")
    _validate_lengths(logits_array, y_array)
    y_array = _validate_binary_labels(y_array)

    if reduction == "mean":
        return _bce_gradient_from_validated_arrays(logits_array, y_array)
    probabilities = sigmoid_stable(logits_array)
    gradient = probabilities - y_array
    return _reduce_gradient(gradient, reduction)


@dataclass(frozen=True, slots=True)
class BCEFitConfig:
    learning_rate: float
    tol: float
    max_iter: int
    l2_alpha: float
    random_state: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be > 0.")
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, (int, np.integer))
            or self.max_iter <= 0
        ):
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(self.l2_alpha) or self.l2_alpha < 0:
            raise ValueError("l2_alpha must be >= 0.")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state,
            (int, np.integer),
        ):
            raise ValueError("random_state must be an int.")


class BCELogisticRegression:
    def __init__(self, config: BCEFitConfig) -> None:
        if not isinstance(config, BCEFitConfig):
            raise ValueError("config must be a BCEFitConfig.")
        self.config = config

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.n_iter_: int | None = None
        self.loss_history_: list[float] = []
        self.success_: bool = False
        self.converged_by_tolerance_: bool = False
        self.message_: str = "not fit"

        self._is_fitted_ = False

    def _validate_X(self, x: Any) -> np.ndarray:
        X = np.asarray(x, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if X.size == 0:
            raise ValueError("X must contain at least one sample and one feature.")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinite values.")
        return X

    def _validate_y(
        self,
        y: Any,
        n_samples: int | None = None,
    ) -> np.ndarray:
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if y_array.size == 0:
            raise ValueError("y must contain at least one element.")
        if n_samples is not None and y_array.size != n_samples:
            raise ValueError("X and y must have the same number of samples.")
        if not np.isfinite(y_array).all():
            raise ValueError("y must not contain NaN or infinite values.")
        unique = set(np.unique(y_array))
        if unique - {0.0, 1.0}:
            raise ValueError("y must be binary with values {0, 1}.")
        return y_array

    def _check_is_fitted(self) -> None:
        if not self._is_fitted_:
            raise RuntimeError(
                "Estimator must be fitted before this method is called."
            )

    def fit(self, X: Any, y: Any) -> BCELogisticRegression:
        X_array = self._validate_X(X)
        y_array = self._validate_y(y, n_samples=X_array.shape[0])

        _, n_features = X_array.shape
        self.n_features_in_ = n_features
        self.classes_ = np.array([0, 1], dtype=int)

        coef = np.zeros(n_features, dtype=float)
        intercept = 0.0
        self.loss_history_ = []
        self.success_ = False
        self.converged_by_tolerance_ = False
        self.message_ = "fit in progress"

        previous_loss = None
        for iteration in range(1, self.config.max_iter + 1):
            logits = X_array @ coef + intercept
            base_loss = _binary_cross_entropy_from_validated_arrays(
                logits,
                y_array,
            )
            gradient_logits = _bce_gradient_from_validated_arrays(
                logits,
                y_array,
            )

            l2_term = 0.5 * self.config.l2_alpha * np.sum(coef * coef)
            loss = float(base_loss + l2_term)
            self.loss_history_.append(loss)

            if not np.isscalar(gradient_logits):
                gradient_logits_array = np.asarray(gradient_logits)
            else:
                gradient_logits_array = np.full_like(
                    logits,
                    fill_value=gradient_logits,
                    dtype=float,
                )

            gradient_coef = (
                X_array.T @ gradient_logits_array
                + self.config.l2_alpha * coef
            )
            gradient_intercept = float(np.sum(gradient_logits_array))

            coef -= self.config.learning_rate * gradient_coef
            intercept -= self.config.learning_rate * gradient_intercept

            if (
                previous_loss is not None
                and abs(previous_loss - loss) < self.config.tol
                and iteration < self.config.max_iter
            ):
                self.coef_ = coef
                self.intercept_ = float(intercept)
                self.n_iter_ = iteration
                self.success_ = True
                self.converged_by_tolerance_ = True
                self.message_ = "Converged by loss difference tolerance."
                self._is_fitted_ = True
                return self
            previous_loss = loss

        self.coef_ = coef
        self.intercept_ = float(intercept)
        self.n_iter_ = self.config.max_iter
        self.success_ = True
        self.converged_by_tolerance_ = False
        self.message_ = "Reached maximum iterations without convergence."
        self._is_fitted_ = True
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must have the same number of features as during fit."
            )
        return X_array @ self.coef_ + self.intercept_

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        logits = self.decision_function(X)
        probability_1 = 1.0 / (1.0 + np.exp(-logits))
        probability_0 = 1.0 - probability_1
        return np.column_stack((probability_0, probability_1))

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        self._check_is_fitted()
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0.0, 1.0].")
        probabilities = self.predict_proba(X)
        return np.asarray(probabilities[:, 1] >= threshold, dtype=int)


def _validate_2d_finite_matrix(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"{name} must contain at least one sample and one feature."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or infinite values.")
    return array


def _validate_random_state(random_state: object) -> int:
    if isinstance(random_state, bool) or not isinstance(
        random_state,
        (int, np.integer),
    ):
        raise ValueError("random_state must be an int.")
    return int(random_state)


def _validate_n_splits(n_splits: object) -> int:
    if isinstance(n_splits, bool) or not isinstance(
        n_splits,
        (int, np.integer),
    ):
        raise ValueError("n_splits must be a positive integer greater than 1.")
    if n_splits <= 1:
        raise ValueError("n_splits must be a positive integer greater than 1.")
    return int(n_splits)


def _validate_oof_y(y: object, n_samples: int) -> np.ndarray:
    y_array = _as_1d_float_array(y, "y")
    if y_array.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of samples.")
    unique = set(np.unique(y_array))
    if unique - {0.0, 1.0}:
        raise ValueError("y must be binary with values {0, 1}.")
    if unique != {0.0, 1.0}:
        raise ValueError("y must contain both legit and fraud cases.")
    return y_array.astype(int)


def generate_oof_bce_scores(
    X: object,
    y: object,
    *,
    n_splits: int,
    config: BCEFitConfig,
    fit_diagnostics_callback: (
        Callable[
            [
                int,
                BCELogisticRegression,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
            None,
        ]
        | None
    ) = None,
) -> np.ndarray:
    """Generate fold-held-out BCE scores with fold-local scaling."""
    X_array = _validate_2d_finite_matrix(X, "X")
    y_array = _validate_oof_y(y, X_array.shape[0])
    n_splits_int = _validate_n_splits(n_splits)
    if not isinstance(config, BCEFitConfig):
        raise ValueError("config must be a BCEFitConfig.")
    random_state = _validate_random_state(config.random_state)
    if fit_diagnostics_callback is not None and not callable(
        fit_diagnostics_callback
    ):
        raise ValueError(
            "fit_diagnostics_callback must be callable or None."
        )

    class_counts = np.bincount(y_array, minlength=2)
    min_class_count = int(class_counts.min())
    if n_splits_int > min_class_count:
        raise ValueError(
            "n_splits must be less than or equal to the number of samples "
            "in each class."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits_int,
        shuffle=True,
        random_state=random_state,
    )
    scores = np.full(X_array.shape[0], np.nan, dtype=float)
    assigned = np.zeros(X_array.shape[0], dtype=bool)

    for fold_number, (train_index, holdout_index) in enumerate(
        splitter.split(X_array, y_array),
        start=1,
    ):
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_array[train_index])
        X_fold_holdout = scaler.transform(X_array[holdout_index])

        model = BCELogisticRegression(config)
        model.fit(X_fold_train, y_array[train_index])
        fold_scores = model.predict_proba(X_fold_holdout)[:, 1]
        if not np.isfinite(fold_scores).all():
            raise ValueError(
                "OOF fraud scores must not contain NaN or infinite values."
            )
        if (fold_scores < 0).any() or (fold_scores > 1).any():
            raise ValueError("OOF fraud scores must be in [0, 1].")
        scores[holdout_index] = fold_scores
        assigned[holdout_index] = True
        if fit_diagnostics_callback is not None:
            fit_diagnostics_callback(
                fold_number,
                model,
                train_index.copy(),
                holdout_index.copy(),
                fold_scores.copy(),
            )

    if not assigned.all():
        raise ValueError(
            "Each training sample must receive exactly one OOF fraud score."
        )
    if not np.isfinite(scores).all():
        raise ValueError(
            "OOF fraud scores must not contain NaN or infinite values."
        )
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("OOF fraud scores must be in [0, 1].")
    return scores


def _bce_config(random_state: int) -> BCEFitConfig:
    return BCEFitConfig(
        learning_rate=BCE_LEARNING_RATE,
        tol=BCE_TOL,
        max_iter=BCE_MAX_ITER,
        l2_alpha=BCE_L2_ALPHA,
        random_state=int(random_state),
    )


def _validate_bce_fit(
    model: BCELogisticRegression,
    produced_scores: object,
    *,
    context: str,
) -> None:
    scores = np.asarray(produced_scores, dtype=float)
    history = np.asarray(model.loss_history_, dtype=float)
    coefficients = np.asarray(model.coef_, dtype=float)
    if not bool(model.converged_by_tolerance_):
        raise RuntimeError(f"{context}: BCE fit did not converge by tolerance.")
    if model.n_iter_ is None or not 1 <= int(model.n_iter_) < BCE_MAX_ITER:
        raise RuntimeError(f"{context}: invalid BCE n_iter_.")
    if history.shape != (int(model.n_iter_),) or not np.isfinite(history).all():
        raise RuntimeError(f"{context}: invalid BCE loss history.")
    if not np.isfinite(coefficients).all() or not np.isfinite(model.intercept_):
        raise RuntimeError(f"{context}: non-finite BCE coefficients.")
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise RuntimeError(f"{context}: non-finite BCE scores.")
    if (scores < 0.0).any() or (scores > 1.0).any():
        raise RuntimeError(f"{context}: BCE scores outside [0,1].")


def _bce_diagnostic_row(
    *,
    outer_seed: int,
    inner_fold: int,
    fit_type: str,
    oof_fold: int | None,
    model: BCELogisticRegression,
    fit_y: np.ndarray,
    produced_scores: np.ndarray,
) -> dict[str, Any]:
    history = np.asarray(model.loss_history_, dtype=float)
    coefficients = np.asarray(model.coef_, dtype=float)
    return {
        "outer_seed": int(outer_seed),
        "inner_fold": int(inner_fold),
        "fit_type": fit_type,
        "oof_fold": oof_fold,
        "fit_row_count": int(fit_y.shape[0]),
        "fit_fraud_count": int(fit_y.sum()),
        "score_row_count": int(produced_scores.shape[0]),
        "learning_rate": BCE_LEARNING_RATE,
        "tol": BCE_TOL,
        "max_iter": BCE_MAX_ITER,
        "l2_alpha": BCE_L2_ALPHA,
        "n_iter_": int(model.n_iter_),
        "success_": bool(model.success_),
        "converged_by_tolerance": bool(model.converged_by_tolerance_),
        "message_": str(model.message_),
        "first_loss": float(history[0]),
        "penultimate_loss": float(history[-2]),
        "final_loss": float(history[-1]),
        "final_absolute_loss_change": float(abs(history[-1] - history[-2])),
        "loss_history_finite": bool(np.isfinite(history).all()),
        "coefficients_finite": bool(np.isfinite(coefficients).all()),
        "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
        "intercept": float(model.intercept_),
        "scores_finite": bool(np.isfinite(produced_scores).all()),
        "score_min": float(np.min(produced_scores)),
        "score_max": float(np.max(produced_scores)),
        "score_sha256": score_vector_sha256(
            produced_scores,
            score_type=(
                f"inner_bce.seed_{outer_seed}.fold_{inner_fold}."
                f"{fit_type}.oof_{oof_fold}"
            ),
        ),
    }


def _loss_history_frame(
    *,
    outer_seed: int,
    inner_fold: int,
    fit_type: str,
    oof_fold: int | None,
    model: BCELogisticRegression,
) -> pd.DataFrame:
    history = np.asarray(model.loss_history_, dtype=float)
    return pd.DataFrame(
        {
            "outer_seed": outer_seed,
            "inner_fold": inner_fold,
            "fit_type": fit_type,
            "oof_fold": oof_fold,
            "iteration": np.arange(1, history.shape[0] + 1),
            "loss": history,
        }
    )


def fit_inner_bce_fold(
    *,
    outer_seed: int,
    inner_fold: int,
    inner_train: pd.DataFrame,
    inner_validation: pd.DataFrame,
    effective_config: EffectiveExperimentConfig,
) -> dict[str, Any]:
    X_train = inner_train.loc[:, BCE_FEATURES]
    y_train = inner_train["Class"].to_numpy(dtype=int)
    X_validation = inner_validation.loc[:, BCE_FEATURES]
    bce_random_state = 100000 + int(outer_seed)
    config = _bce_config(bce_random_state)
    diagnostics: list[dict[str, Any]] = []
    loss_frames: list[pd.DataFrame] = []
    oof_fold_number = np.full(len(inner_train), -1, dtype=int)

    def capture(
        fold_number: int,
        model: BCELogisticRegression,
        train_position: np.ndarray,
        holdout_position: np.ndarray,
        holdout_scores: np.ndarray,
    ) -> None:
        _validate_bce_fit(
            model,
            holdout_scores,
            context=(
                f"seed={outer_seed},inner_fold={inner_fold},"
                f"oof_fold={fold_number}"
            ),
        )
        if (oof_fold_number[holdout_position] != -1).any():
            raise RuntimeError("An inner OOF position was assigned twice.")
        oof_fold_number[holdout_position] = int(fold_number)
        diagnostics.append(
            _bce_diagnostic_row(
                outer_seed=outer_seed,
                inner_fold=inner_fold,
                fit_type="OOF",
                oof_fold=fold_number,
                model=model,
                fit_y=y_train[train_position],
                produced_scores=holdout_scores,
            )
        )
        loss_frames.append(
            _loss_history_frame(
                outer_seed=outer_seed,
                inner_fold=inner_fold,
                fit_type="OOF",
                oof_fold=fold_number,
                model=model,
            )
        )
        _emit_experiment_status(
            "INFO",
            "inner-bce-complete",
            seed=outer_seed,
            inner_fold=inner_fold,
            fit_kind="oof",
            oof_fold=fold_number,
            completed=_inner_bce_completed(
                outer_seed,
                inner_fold,
                fold_number,
                effective_config,
            ),
            total=(
                len(effective_config.seeds)
                * effective_config.inner_folds
                * (effective_config.bce_oof_folds + 1)
            ),
        )

    p_train_oof = generate_oof_bce_scores(
        X_train,
        y_train,
        n_splits=effective_config.bce_oof_folds,
        config=config,
        fit_diagnostics_callback=capture,
    )
    if (oof_fold_number == -1).any():
        raise RuntimeError("Inner OOF score assignment is incomplete.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    full_model = BCELogisticRegression(config)
    full_model.fit(X_train_scaled, y_train)
    p_validation = full_model.predict_proba(X_validation_scaled)[:, 1]
    _validate_bce_fit(
        full_model,
        p_validation,
        context=f"seed={outer_seed},inner_fold={inner_fold},full_inner_train",
    )
    diagnostics.append(
        _bce_diagnostic_row(
            outer_seed=outer_seed,
            inner_fold=inner_fold,
            fit_type="Full Inner Train",
            oof_fold=None,
            model=full_model,
            fit_y=y_train,
            produced_scores=p_validation,
        )
    )
    loss_frames.append(
        _loss_history_frame(
            outer_seed=outer_seed,
            inner_fold=inner_fold,
            fit_type="Full Inner Train",
            oof_fold=None,
            model=full_model,
        )
    )
    _emit_experiment_status(
        "INFO",
        "inner-bce-complete",
        seed=outer_seed,
        inner_fold=inner_fold,
        fit_kind="full",
        oof_fold=None,
        completed=_inner_bce_completed(
            outer_seed,
            inner_fold,
            effective_config.bce_oof_folds + 1,
            effective_config,
        ),
        total=(
            len(effective_config.seeds)
            * effective_config.inner_folds
            * (effective_config.bce_oof_folds + 1)
        ),
    )
    return {
        "p_train_oof": p_train_oof,
        "p_validation": p_validation,
        "oof_fold_number": oof_fold_number,
        "diagnostics": pd.DataFrame(diagnostics),
        "loss_history": pd.concat(loss_frames, ignore_index=True),
        "scaler_mean": scaler.mean_.copy(),
        "scaler_scale": scaler.scale_.copy(),
    }


def fit_outer_bce_scores_after_freeze(
    *,
    dataframe: pd.DataFrame,
    output_root: Path,
    outer_seed: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    effective_config: EffectiveExperimentConfig,
    data_sha256: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Generate the converged outer BCE vectors after selection.

    The configuration, split, fold-local scaling, OOF splitter and full-train
    scaling follow the empirical method. No persisted score vector is required.
    """

    identity = preflight_split_identity(output_root, outer_seed)
    if (
        _ordered_index_sha256(train_index) != identity["train_index_sha256"]
        or _ordered_index_sha256(test_index) != identity["test_index_sha256"]
    ):
        raise RuntimeError(f"Outer split hash mismatch for seed {outer_seed}.")
    train_labels = dataframe.loc[train_index, "Class"].to_numpy(dtype=int)
    test_labels = dataframe.loc[test_index, "Class"].to_numpy(dtype=int)
    X_train = dataframe.loc[train_index, BCE_FEATURES]
    X_test = dataframe.loc[test_index, BCE_FEATURES]
    config = _bce_config(outer_seed)
    diagnostics: list[dict[str, Any]] = []
    loss_frames: list[pd.DataFrame] = []

    def capture(
        fold_number: int,
        model: BCELogisticRegression,
        train_position: np.ndarray,
        _holdout_position: np.ndarray,
        holdout_scores: np.ndarray,
    ) -> None:
        _validate_bce_fit(
            model,
            holdout_scores,
            context=f"seed={outer_seed},outer_oof_fold={fold_number}",
        )
        diagnostics.append(
            _bce_diagnostic_row(
                outer_seed=outer_seed,
                inner_fold=0,
                fit_type="Outer OOF",
                oof_fold=fold_number,
                model=model,
                fit_y=train_labels[train_position],
                produced_scores=holdout_scores,
            )
        )
        loss_frames.append(
            _loss_history_frame(
                outer_seed=outer_seed,
                inner_fold=0,
                fit_type="Outer OOF",
                oof_fold=fold_number,
                model=model,
            )
        )
        _emit_experiment_status(
            "INFO",
            "final-bce-complete",
            seed=outer_seed,
            fit_kind="oof",
            oof_fold=fold_number,
            completed=_final_bce_completed(
                outer_seed,
                fold_number,
                effective_config,
            ),
            total=(
                len(effective_config.seeds)
                * (effective_config.bce_oof_folds + 1)
            ),
        )

    p_oof = generate_oof_bce_scores(
        X_train,
        train_labels,
        n_splits=effective_config.bce_oof_folds,
        config=config,
        fit_diagnostics_callback=capture,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    full_model = BCELogisticRegression(config)
    full_model.fit(X_train_scaled, train_labels)
    p_test = full_model.predict_proba(X_test_scaled)[:, 1]
    _validate_bce_fit(
        full_model,
        p_test,
        context=f"seed={outer_seed},full_outer_train",
    )
    diagnostics.append(
        _bce_diagnostic_row(
            outer_seed=outer_seed,
            inner_fold=0,
            fit_type="Full Outer Train",
            oof_fold=None,
            model=full_model,
            fit_y=train_labels,
            produced_scores=p_test,
        )
    )
    loss_frames.append(
        _loss_history_frame(
            outer_seed=outer_seed,
            inner_fold=0,
            fit_type="Full Outer Train",
            oof_fold=None,
            model=full_model,
        )
    )
    _emit_experiment_status(
        "INFO",
        "final-bce-complete",
        seed=outer_seed,
        fit_kind="full",
        oof_fold=None,
        completed=_final_bce_completed(
            outer_seed,
            effective_config.bce_oof_folds + 1,
            effective_config,
        ),
        total=(
            len(effective_config.seeds)
            * (effective_config.bce_oof_folds + 1)
        ),
    )
    oof = pd.DataFrame(
        {
            "train_index": train_index,
            "y_true": train_labels,
            "p_oof": p_oof,
        }
    )
    test = pd.DataFrame(
        {
            "test_index": test_index,
            "y_true": test_labels,
            "p_full_train_test": p_test,
        }
    )
    metadata = {
        "deduplicated_dataframe_sha256": data_sha256,
        "bce_learning_rate": BCE_LEARNING_RATE,
        "bce_tol": BCE_TOL,
        "bce_max_iter": BCE_MAX_ITER,
        "bce_l2_alpha": BCE_L2_ALPHA,
        "bce_oof_folds": effective_config.bce_oof_folds,
        "train_index_sha256": _ordered_index_sha256(train_index),
        "test_index_sha256": _ordered_index_sha256(test_index),
        "oof_score_sha256": score_vector_sha256(
            p_oof,
            score_type=f"outer_oof_bce.seed_{outer_seed}",
        ),
        "test_score_sha256": score_vector_sha256(
            p_test,
            score_type=f"outer_test_bce.seed_{outer_seed}",
        ),
        "fit_count": len(diagnostics),
        "converged_fit_count": int(
            sum(
                bool(row["converged_by_tolerance"])
                for row in diagnostics
            )
        ),
        "score_source": "generated_from_current_configuration",
    }
    return (
        oof,
        test,
        metadata,
        pd.DataFrame(diagnostics),
        pd.concat(loss_frames, ignore_index=True),
    )
