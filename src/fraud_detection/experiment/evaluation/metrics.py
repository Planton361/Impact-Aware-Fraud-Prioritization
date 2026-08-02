"""Metric calculation for completed rankings and matched Top-k selections."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..prioritization.composition import validate_full_ranking


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _one_dimensional_finite(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one element.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _binary_labels(values: object, name: str = "y_true") -> np.ndarray:
    arr = _one_dimensional_finite(values, name)
    if set(np.unique(arr)) - {0.0, 1.0}:
        raise ValueError(f"{name} must contain only binary values 0 and 1.")
    return arr.astype(int)


def _amounts(values: object, name: str = "amount") -> np.ndarray:
    arr = _one_dimensional_finite(values, name)
    if (arr < 0.0).any():
        raise ValueError(f"{name} must be non-negative.")
    return arr


def _dcg(gains: np.ndarray, k: int) -> float:
    effective = min(int(k), gains.shape[0])
    if effective <= 0:
        return 0.0
    positions = np.arange(1, effective + 1, dtype=float)
    return float(
        np.sum(gains[:effective] / np.log2(positions + 1.0))
    )


def amount_ndcg_at_k(
    y_true: object,
    amount: object,
    final_rank_position: object,
    k: object,
    *,
    minimum_fraud_amount: float | None = None,
) -> float:
    y = _binary_labels(y_true)
    amounts = _amounts(amount)
    ranks = _one_dimensional_finite(
        final_rank_position,
        "final_rank_position",
    )
    budget = _positive_int(k, "k")
    if not (len(y) == len(amounts) == len(ranks)):
        raise ValueError("Metric inputs must have the same length.")
    if budget > len(y):
        raise ValueError("k cannot exceed the number of rows.")
    if not np.array_equal(
        np.sort(ranks.astype(int)),
        np.arange(1, len(y) + 1),
    ):
        raise ValueError("final_rank_position must be a complete permutation.")

    eligible = y == 1
    if minimum_fraud_amount is not None:
        if not math.isfinite(float(minimum_fraud_amount)):
            raise ValueError("minimum_fraud_amount must be finite.")
        eligible &= amounts >= float(minimum_fraud_amount)
    gains_all = np.where(eligible, amounts, 0.0)
    ideal = np.sort(gains_all)[::-1]
    denominator = _dcg(ideal, budget)
    if denominator <= 0.0:
        return 0.0
    order = np.argsort(ranks, kind="mergesort")
    return _dcg(gains_all[order], budget) / denominator


def cutoff_tie_diagnostics(
    ranking: pd.DataFrame,
    target_budget: object,
) -> dict[str, int | float]:
    budget = _positive_int(target_budget, "target_budget")
    if budget > len(ranking):
        raise ValueError("target_budget cannot exceed ranking length.")
    validate_full_ranking(ranking)
    raw = ranking["raw_ranker_score"].to_numpy(dtype=float)
    ranks = ranking["final_rank_position"].to_numpy(dtype=int)
    cutoff_index = int(np.flatnonzero(ranks == budget)[0])
    cutoff_score = float(raw[cutoff_index])
    if not np.isfinite(cutoff_score):
        raise ValueError("The raw score at the matched cutoff must be finite.")
    tied = np.isfinite(raw) & (raw == cutoff_score)
    tied_ranks = ranks[tied]
    return {
        "unique_raw_ranker_scores": int(np.unique(raw[np.isfinite(raw)]).size),
        "cutoff_raw_ranker_score": cutoff_score,
        "cutoff_tie_size": int(tied.sum()),
        "cutoff_tie_rank_min": int(tied_ranks.min()),
        "cutoff_tie_rank_max": int(tied_ranks.max()),
    }


def matched_budget_metrics(
    y_true: object,
    amount: object,
    ranking: pd.DataFrame,
    target_budget: object,
) -> dict[str, int | float]:
    y = _binary_labels(y_true)
    amounts = _amounts(amount)
    budget = _positive_int(target_budget, "target_budget")
    if len(y) != len(ranking) or len(amounts) != len(ranking):
        raise ValueError("y_true, amount, and ranking must have the same length.")
    if budget > len(ranking):
        raise ValueError("target_budget cannot exceed ranking length.")
    validate_full_ranking(ranking)

    ranks = ranking["final_rank_position"].to_numpy(dtype=int)
    selected = ranks <= budget
    selected_fraud = selected & (y == 1)
    total_fraud = int((y == 1).sum())
    if total_fraud <= 0:
        raise ValueError("At least one fraud case is required.")
    total_fraud_amount = float(amounts[y == 1].sum())
    if total_fraud_amount <= 0.0:
        raise ValueError("Total fraud Amount must be positive.")

    fraud_count = int(selected_fraud.sum())
    fraud_amount_sum = float(amounts[selected_fraud].sum())
    q90_threshold = float(np.quantile(amounts[y == 1], 0.90))
    q90_mask = (y == 1) & (amounts >= q90_threshold)
    q90_count = int(q90_mask.sum())
    q90_captured = int((q90_mask & selected).sum())
    legit_selected = selected & (y == 0)
    legit_amounts = amounts[y == 0]
    high_legit_threshold = (
        float(np.quantile(legit_amounts, 0.90))
        if legit_amounts.size
        else 0.0
    )
    high_legit_selected = legit_selected & (amounts >= high_legit_threshold)
    ties = cutoff_tie_diagnostics(ranking, budget)

    return {
        "target_budget": budget,
        "prevented_loss_ratio_at_k": fraud_amount_sum / total_fraud_amount,
        "frauds_at_k": fraud_count,
        "precision_at_k": fraud_count / float(budget),
        "recall_at_k": fraud_count / float(total_fraud),
        "fraud_amount_sum_at_k": fraud_amount_sum,
        "legit_count_at_k": int(legit_selected.sum()),
        "amount_ndcg_at_k": amount_ndcg_at_k(y, amounts, ranks, budget),
        "q90_threshold": q90_threshold,
        "q90_total_fraud_count": q90_count,
        "q90_captured_fraud_count": q90_captured,
        "q90_captured_ratio_at_k": (
            q90_captured / float(q90_count) if q90_count else 0.0
        ),
        "q90_amount_ndcg_at_k": amount_ndcg_at_k(
            y,
            amounts,
            ranks,
            budget,
            minimum_fraud_amount=q90_threshold,
        ),
        "high_amount_legit_threshold_q90": high_legit_threshold,
        "high_amount_legit_count_at_k": int(high_legit_selected.sum()),
        "mean_legit_amount_at_k": (
            float(amounts[legit_selected].mean())
            if legit_selected.any()
            else 0.0
        ),
        **ties,
    }
