"""Diagnostic analysis-row construction for final experiment results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.artifacts import _integer_vector_sha256

from ..config import METHOD_BASELINE


def replacement_rows(
    *,
    seed: int,
    budget: int,
    method_family: str,
    baseline_ranking: pd.DataFrame,
    comparison_ranking: pd.DataFrame,
    y_true: np.ndarray,
    amount: np.ndarray,
) -> list[dict[str, Any]]:
    baseline_selected = set(
        baseline_ranking.loc[
            baseline_ranking["final_rank_position"].astype(int) <= budget,
            "original_position",
        ].astype(int)
    )
    comparison_selected = set(
        comparison_ranking.loc[
            comparison_ranking["final_rank_position"].astype(int) <= budget,
            "original_position",
        ].astype(int)
    )
    rows = []
    for subset_name, positions in (
        ("added_vs_bce", sorted(comparison_selected - baseline_selected)),
        ("removed_from_bce", sorted(baseline_selected - comparison_selected)),
    ):
        position_arr = np.asarray(positions, dtype=int)
        fraud_mask = (
            y_true[position_arr] == 1
            if position_arr.size
            else np.zeros(0, dtype=bool)
        )
        rows.append(
            {
                "seed": seed,
                "target_budget": budget,
                "method_family": method_family,
                "subset": subset_name,
                "case_count": int(position_arr.size),
                "fraud_count": int(fraud_mask.sum()),
                "fraud_amount_sum": (
                    float(amount[position_arr][fraud_mask].sum())
                    if position_arr.size
                    else 0.0
                ),
                "legit_count": int(position_arr.size - fraud_mask.sum()),
                "row_index_sha256": _integer_vector_sha256(
                    comparison_ranking.loc[
                        position_arr,
                        "row_index",
                    ].to_numpy(dtype=int)
                    if position_arr.size
                    else np.asarray([], dtype=int),
                    vector_type=(
                        f"replacement.seed_{seed}.k_{budget}."
                        f"{method_family}.{subset_name}"
                    ),
                ),
            }
        )
    return rows


def boundary_rows(
    *,
    seed: int,
    budget: int,
    method_family: str,
    baseline_ranking: pd.DataFrame,
    comparison_ranking: pd.DataFrame,
    y_true: np.ndarray,
    amount: np.ndarray,
) -> list[dict[str, Any]]:
    baseline_rank = baseline_ranking["final_rank_position"].to_numpy(dtype=int)
    comparison_rank = comparison_ranking[
        "final_rank_position"
    ].to_numpy(dtype=int)
    lower = max(1, budget - 3)
    upper = min(len(baseline_rank), budget + 3)
    relevant = (
        ((baseline_rank >= lower) & (baseline_rank <= upper))
        | ((comparison_rank >= lower) & (comparison_rank <= upper))
        | ((baseline_rank <= budget) != (comparison_rank <= budget))
    )
    positions = np.flatnonzero(relevant)
    rows = []
    for position in positions:
        rows.append(
            {
                "seed": seed,
                "target_budget": budget,
                "method_family": method_family,
                "row_index": int(
                    comparison_ranking.iloc[position]["row_index"]
                ),
                "Class": int(y_true[position]),
                "Amount": float(amount[position]),
                "p_fraud": float(
                    comparison_ranking.iloc[position]["p_fraud"]
                ),
                "bce_rank_position": int(baseline_rank[position]),
                "comparison_rank_position": int(comparison_rank[position]),
                "rank_shift_vs_bce": int(
                    comparison_rank[position] - baseline_rank[position]
                ),
                "crossed_into_topk": bool(
                    baseline_rank[position] > budget
                    and comparison_rank[position] <= budget
                ),
                "dropped_from_topk": bool(
                    baseline_rank[position] <= budget
                    and comparison_rank[position] > budget
                ),
            }
        )
    return rows


def global_metric_rows(
    *,
    seed: int,
    budget: int,
    method_family: str,
    score_path: str,
    y_true: np.ndarray,
    ranking: pd.DataFrame,
) -> list[dict[str, Any]]:
    priority = ranking["priority_order_score"].to_numpy(dtype=float)
    rows = [
        {
            "seed": seed,
            "target_budget": budget,
            "method_family": method_family,
            "score_path": score_path,
            "metric": "roc_auc_of_final_order",
            "value": float(roc_auc_score(y_true, priority)),
        },
        {
            "seed": seed,
            "target_budget": budget,
            "method_family": method_family,
            "score_path": score_path,
            "metric": "average_precision_of_final_order",
            "value": float(average_precision_score(y_true, priority)),
        },
    ]
    if method_family == METHOD_BASELINE:
        probability = ranking["p_fraud"].to_numpy(dtype=float)
        rows.append(
            {
                "seed": seed,
                "target_budget": budget,
                "method_family": method_family,
                "score_path": score_path,
                "metric": "brier_score_probability",
                "value": float(np.mean((probability - y_true) ** 2)),
            }
        )
    return rows


def tie_diagnostic_row(
    *,
    outer_seed: int,
    target_budget: int,
    method_family: str,
    score_path: str,
    metric_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": outer_seed,
        "target_budget": target_budget,
        "method_family": method_family,
        "score_path": score_path,
        **{
            key: metric_row[key]
            for key in (
                "unique_raw_ranker_scores",
                "cutoff_raw_ranker_score",
                "cutoff_tie_size",
                "cutoff_tie_rank_min",
                "cutoff_tie_rank_max",
            )
        },
    }


def high_amount_legit_row(
    *,
    outer_seed: int,
    target_budget: int,
    method_family: str,
    metric_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": outer_seed,
        "target_budget": target_budget,
        "method_family": method_family,
        "high_amount_legit_threshold_q90": metric_row[
            "high_amount_legit_threshold_q90"
        ],
        "legit_count_at_k": metric_row["legit_count_at_k"],
        "high_amount_legit_count_at_k": metric_row[
            "high_amount_legit_count_at_k"
        ],
        "mean_legit_amount_at_k": metric_row[
            "mean_legit_amount_at_k"
        ],
    }


def hard_impact_row(
    *,
    outer_seed: int,
    target_budget: int,
    method_family: str,
    metric_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": outer_seed,
        "target_budget": target_budget,
        "method_family": method_family,
        "prevented_loss_ratio_at_k": metric_row[
            "prevented_loss_ratio_at_k"
        ],
        "fraud_amount_sum_at_k": metric_row[
            "fraud_amount_sum_at_k"
        ],
        "amount_ndcg_at_k": metric_row["amount_ndcg_at_k"],
        "q90_threshold": metric_row["q90_threshold"],
        "q90_captured_ratio_at_k": metric_row[
            "q90_captured_ratio_at_k"
        ],
        "q90_amount_ndcg_at_k": metric_row[
            "q90_amount_ndcg_at_k"
        ],
    }
