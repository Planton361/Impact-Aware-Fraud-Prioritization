"""Public experiment result and final result-record construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from fraud_detection.artifacts import score_vector_sha256

from .config import (
    METHOD_BASELINE,
    METHOD_FIXED,
    RANKER_LEARNING_RATE,
    RANKER_MIN_CHILD_SAMPLES,
    RANKER_MIN_CHILD_WEIGHT,
    RANKER_N_JOBS,
    RANKER_NUM_LEAVES,
    RANKER_REG_LAMBDA,
    EffectiveExperimentConfig,
    ExperimentPhase,
)
from .evaluation.metrics import matched_budget_metrics
from .prioritization.lambdarank import CandidateAmountGainRanker


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Structured outcome of a completed requested experiment phase."""

    output_root: Path
    requested_phase: ExperimentPhase
    status: Literal["COMPLETE", "PHASE_COMPLETE"]
    completed_phases: tuple[str, ...]


def ranking_with_context(
    ranking: pd.DataFrame,
    *,
    row_labels: np.ndarray,
    amounts: np.ndarray,
    outer_seed: int,
    target_budget: int,
    score_path: str,
    method_family: str,
    selected_gain: str,
    selection_status: str,
    truncation_level: int,
    final_n_estimators: int,
    effective_config: EffectiveExperimentConfig,
) -> pd.DataFrame:
    output = ranking.copy()
    output.insert(0, "seed", int(outer_seed))
    output.insert(1, "target_budget", int(target_budget))
    output.insert(
        2,
        "primary_budget",
        target_budget in effective_config.primary_budgets,
    )
    output.insert(3, "score_path", score_path)
    output.insert(4, "method_family", method_family)
    output["Class"] = np.asarray(row_labels, dtype=int)
    output["Amount"] = np.asarray(amounts, dtype=float)
    output["selected_gain"] = selected_gain
    output["selection_status"] = selection_status
    output["truncation_level"] = int(truncation_level)
    output["final_n_estimators"] = int(final_n_estimators)
    output["score_type"] = (
        "fraud_probability"
        if method_family == METHOD_BASELINE
        else (
            "ordinal_candidate_postprocessing"
            if method_family == METHOD_FIXED
            else "raw_lambdarank_score_with_ordinal_full_order"
        )
    )
    return output


def score_path(method_family: str, budget: int) -> str:
    if method_family == METHOD_BASELINE:
        return METHOD_BASELINE
    return f"{method_family}_k{int(budget)}"


def matched_metric_row(
    *,
    outer_seed: int,
    target_budget: int,
    method_family: str,
    score_path: str,
    ranking: pd.DataFrame,
    y_true: np.ndarray,
    amount: np.ndarray,
    selected_gain: str,
    selection_status: str,
    truncation_level: int,
    final_n_estimators: int,
    effective_config: EffectiveExperimentConfig,
) -> dict[str, Any]:
    metrics = matched_budget_metrics(
        y_true,
        amount,
        ranking,
        target_budget,
    )
    return {
        "seed": int(outer_seed),
        "target_budget": int(target_budget),
        "primary_budget": target_budget in effective_config.primary_budgets,
        "method_family": method_family,
        "score_path": score_path,
        "selected_gain": selected_gain,
        "selection_status": selection_status,
        "truncation_level": int(truncation_level),
        "eval_at": int(target_budget),
        "final_n_estimators": int(final_n_estimators),
        **metrics,
    }


def model_row(
    *,
    seed: int,
    budget: int,
    model_type: str,
    config: dict[str, Any],
    ranker: CandidateAmountGainRanker,
    train_pool_hash: str,
    test_pool_hash: str,
    raw_scores: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    if ranker.model_ is None or ranker.best_iteration_ is None:
        raise RuntimeError("Final ranker is not fitted.")
    trained_estimators = int(ranker.model_.n_estimators_)
    expected_estimators = int(config["final_n_estimators"])
    if trained_estimators != expected_estimators:
        raise RuntimeError(
            f"Final {model_type} tree count mismatch: "
            f"{trained_estimators} != {expected_estimators}"
        )
    return {
        "seed": seed,
        "target_budget": budget,
        "model_type": model_type,
        "selected_gain": config["selected_gain"],
        "selection_status": config["selection_status"],
        "label_gain": json.dumps(config["label_gain"]),
        "truncation_level": config["truncation"],
        "eval_at": budget,
        "configured_n_estimators": expected_estimators,
        "trained_n_estimators": trained_estimators,
        "learning_rate": RANKER_LEARNING_RATE,
        "num_leaves": RANKER_NUM_LEAVES,
        "min_child_samples": RANKER_MIN_CHILD_SAMPLES,
        "min_child_weight": RANKER_MIN_CHILD_WEIGHT,
        "reg_lambda": RANKER_REG_LAMBDA,
        "n_jobs": RANKER_N_JOBS,
        "feature_names": json.dumps(feature_names),
        "train_candidate_pool_sha256": train_pool_hash,
        "test_candidate_pool_sha256": test_pool_hash,
        "raw_ranker_score_sha256": score_vector_sha256(
            raw_scores,
            score_type=f"final.{model_type}.seed_{seed}.k_{budget}",
        ),
        "config_hash": config["config_hash"],
        "scores_finite": bool(np.isfinite(raw_scores).all()),
    }


def selected_gain_row(
    *,
    seed: int,
    target_budget: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "target_budget": target_budget,
        "selected_gain": config["selected_gain"],
        "selection_status": config["selection_status"],
        "final_n_estimators": config["final_n_estimators"],
        "truncation_level": config["truncation"],
        "eval_at": target_budget,
        "config_hash": config["config_hash"],
    }


def fixed_reference_path_row(
    *,
    seed: int,
    target_budget: int,
    train_pool_hash: str,
    test_pool_hash: str,
    raw_scores: np.ndarray,
    effective_config: EffectiveExperimentConfig,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "target_budget": target_budget,
        "formula": "p_fraud * log1p(Amount)",
        "train_candidate_pool_sha256": train_pool_hash,
        "test_candidate_pool_sha256": test_pool_hash,
        "candidate_pool_size": effective_config.candidate_pool_size,
        "raw_candidate_score_sha256": score_vector_sha256(
            raw_scores,
            score_type=f"fixed_reference.seed_{seed}.k_{target_budget}",
        ),
    }
