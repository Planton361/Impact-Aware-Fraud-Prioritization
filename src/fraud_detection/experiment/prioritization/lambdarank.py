"""Private shared LambdaRank implementation for learned comparison paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .. import config as _contracts
from ..config import EffectiveExperimentConfig
from .inputs import label_gain_for_profile as _label_gain_for_profile
from .selection import (
    _positive_int,
    eval_at_for_budget,
    truncation_for_budget,
)


def _one_dimensional_finite(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one element.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _matrix(values: object, name: str) -> tuple[np.ndarray, list[str] | None]:
    columns = None
    if isinstance(values, pd.DataFrame):
        columns = [str(column) for column in values.columns]
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values.")
    return arr, columns


def _lgbm_input(values: np.ndarray, columns: list[str] | None):
    if columns is None:
        return values
    return pd.DataFrame(values, columns=columns)


def _relevance(values: object, n_samples: int, name: str) -> np.ndarray:
    arr = _one_dimensional_finite(values, name)
    if arr.shape[0] != n_samples:
        raise ValueError(f"{name} and the feature matrix must have the same length.")
    if not np.equal(arr, np.floor(arr)).all():
        raise ValueError(f"{name} must contain integer labels in 0..4.")
    if (arr < 0).any() or (arr > 4).any():
        raise ValueError(f"{name} must contain integer labels in 0..4.")
    return arr.astype(int)


def _single_group(group: object, n_samples: int, name: str) -> list[int]:
    arr = np.asarray(group)
    if arr.ndim != 1 or arr.size != 1:
        raise ValueError(f"{name} must contain exactly one ranking group.")
    value = _positive_int(arr[0], f"{name}[0]")
    if value != n_samples:
        raise ValueError(f"{name} length must equal the candidate-pool size.")
    return [value]


@dataclass(frozen=True)
class CandidateRankerConfig:
    target_budget: int
    gain_profile: str
    effective_config: EffectiveExperimentConfig
    random_state: int = 42
    n_estimators: int | None = None
    early_stopping_rounds: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.effective_config, EffectiveExperimentConfig):
            raise ValueError(
                "effective_config must be an EffectiveExperimentConfig."
            )
        truncation_for_budget(self.target_budget, self.effective_config)
        _label_gain_for_profile(self.gain_profile)
        if self.gain_profile not in self.effective_config.enabled_gain_profiles:
            raise ValueError("gain_profile is not enabled by the profile.")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state,
            (int, np.integer),
        ):
            raise ValueError("random_state must be an integer.")
        estimators = _positive_int(
            self.effective_config.ranker_max_estimators
            if self.n_estimators is None
            else self.n_estimators,
            "n_estimators",
        )
        object.__setattr__(self, "n_estimators", estimators)
        if estimators > self.effective_config.ranker_max_estimators:
            raise ValueError(
                "n_estimators cannot exceed the configured tree ceiling."
            )
        stopping = _positive_int(
            self.effective_config.ranker_early_stopping_rounds
            if self.early_stopping_rounds is None
            else self.early_stopping_rounds,
            "early_stopping_rounds",
        )
        object.__setattr__(self, "early_stopping_rounds", stopping)
        if stopping != self.effective_config.ranker_early_stopping_rounds:
            raise ValueError(
                "early_stopping_rounds does not match the profile."
            )

    @property
    def label_gain(self) -> tuple[int, int, int, int, int]:
        return _label_gain_for_profile(self.gain_profile)

    @property
    def truncation_level(self) -> int:
        return truncation_for_budget(self.target_budget, self.effective_config)

    @property
    def eval_at(self) -> tuple[int]:
        return eval_at_for_budget(self.target_budget, self.effective_config)

    def model_parameters(self) -> dict[str, object]:
        return {
            "objective": "lambdarank",
            "label_gain": list(self.label_gain),
            "lambdarank_truncation_level": self.truncation_level,
            "random_state": int(self.random_state),
            "n_estimators": int(self.n_estimators),
            "learning_rate": _contracts.RANKER_LEARNING_RATE,
            "num_leaves": _contracts.RANKER_NUM_LEAVES,
            "min_child_samples": _contracts.RANKER_MIN_CHILD_SAMPLES,
            "min_child_weight": _contracts.RANKER_MIN_CHILD_WEIGHT,
            "reg_lambda": _contracts.RANKER_REG_LAMBDA,
            "n_jobs": _contracts.RANKER_N_JOBS,
            "verbosity": _contracts.RANKER_VERBOSITY,
        }


class CandidateAmountGainRanker:
    """Fixed-capacity LambdaRank model for one candidate pool and budget."""

    def __init__(self, config: CandidateRankerConfig) -> None:
        if not isinstance(config, CandidateRankerConfig):
            raise ValueError("config must be a CandidateRankerConfig.")
        self.config = config
        self.model_: Any | None = None
        self.best_iteration_: int | None = None
        self.n_features_in_: int | None = None
        self.feature_names_in_: list[str] | None = None
        self.evals_result_: dict[str, dict[str, list[float]]] | None = None
        self.used_early_stopping_: bool = False
        self.evaluation_metric_: str | None = None

    def _build_model(self):
        try:
            from lightgbm import LGBMRanker
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for CandidateAmountGainRanker."
            ) from exc
        return LGBMRanker(**self.config.model_parameters())

    def fit(
        self,
        X: object,
        relevance: object,
        group: object,
        *,
        eval_X: object | None = None,
        eval_relevance: object | None = None,
        eval_group: object | None = None,
    ) -> "CandidateAmountGainRanker":
        X_arr, columns = _matrix(X, "X")
        rel = _relevance(relevance, X_arr.shape[0], "relevance")
        train_group = _single_group(group, X_arr.shape[0], "group")
        eval_arguments = (eval_X, eval_relevance, eval_group)
        has_any_eval = any(value is not None for value in eval_arguments)
        has_all_eval = all(value is not None for value in eval_arguments)
        if has_any_eval and not has_all_eval:
            raise ValueError(
                "eval_X, eval_relevance, and eval_group must be supplied together."
            )
        if (
            has_all_eval
            and self.config.n_estimators
            != self.config.effective_config.ranker_max_estimators
        ):
            raise ValueError(
                "Inner early stopping must use the configured tree ceiling."
            )

        model = self._build_model()
        fit_kwargs: dict[str, object] = {}
        if has_all_eval:
            eval_arr, eval_columns = _matrix(eval_X, "eval_X")
            if eval_arr.shape[1] != X_arr.shape[1]:
                raise ValueError(
                    "eval_X must have the same number of features as X."
                )
            if columns is not None and eval_columns is not None:
                if columns != eval_columns:
                    raise ValueError("eval_X feature columns must match X.")
            eval_rel = _relevance(
                eval_relevance,
                eval_arr.shape[0],
                "eval_relevance",
            )
            valid_group = _single_group(
                eval_group,
                eval_arr.shape[0],
                "eval_group",
            )
            try:
                from lightgbm import early_stopping
            except ImportError as exc:
                raise ImportError(
                    "lightgbm is required for CandidateAmountGainRanker."
                ) from exc
            fit_kwargs = {
                "eval_X": _lgbm_input(eval_arr, columns),
                "eval_y": eval_rel,
                "eval_names": ["inner_validation"],
                "eval_group": [valid_group],
                "eval_metric": "ndcg",
                "eval_at": self.config.eval_at,
                "callbacks": [
                    early_stopping(
                        stopping_rounds=self.config.early_stopping_rounds,
                        first_metric_only=True,
                        verbose=False,
                    )
                ],
            }

        model.fit(
            _lgbm_input(X_arr, columns),
            rel,
            group=train_group,
            **fit_kwargs,
        )
        if has_all_eval:
            best_iteration = int(model.best_iteration_)
            if not 1 <= best_iteration <= (
                self.config.effective_config.ranker_max_estimators
            ):
                raise RuntimeError("LightGBM returned an invalid best_iteration_.")
            self.used_early_stopping_ = True
            self.evaluation_metric_ = f"ndcg@{self.config.target_budget}"
        else:
            best_iteration = int(self.config.n_estimators)
            self.used_early_stopping_ = False
            self.evaluation_metric_ = None

        self.model_ = model
        self.best_iteration_ = best_iteration
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = columns
        self.evals_result_ = (
            {
                str(dataset): {
                    str(metric): [float(value) for value in values]
                    for metric, values in metrics.items()
                }
                for dataset, metrics in model.evals_result_.items()
            }
            if has_all_eval
            else None
        )
        return self

    def decision_function(self, X: object) -> np.ndarray:
        if self.model_ is None or self.n_features_in_ is None:
            raise RuntimeError(
                "CandidateAmountGainRanker must be fitted before scoring."
            )
        X_arr, columns = _matrix(X, "X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X must have the fitted number of features.")
        if self.feature_names_in_ is not None and columns is not None:
            if self.feature_names_in_ != columns:
                raise ValueError("X feature columns must match the fitted columns.")
        scores = np.asarray(
            self.model_.predict(
                _lgbm_input(X_arr, self.feature_names_in_),
                num_iteration=self.best_iteration_,
            ),
            dtype=float,
        )
        if scores.ndim != 1 or scores.shape[0] != X_arr.shape[0]:
            raise RuntimeError("LightGBM returned an invalid score-vector shape.")
        if not np.isfinite(scores).all():
            raise RuntimeError("LightGBM returned non-finite raw ranker scores.")
        return scores
