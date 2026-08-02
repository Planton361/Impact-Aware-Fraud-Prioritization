import numpy as np
import pandas as pd
import pytest

from fraud_detection.experiment.config import (
    METHOD_BASELINE,
    METHOD_FIXED,
    METHOD_P_ONLY,
    resolve_experiment_profile,
)
from fraud_detection.experiment.evaluation.diagnostics import (
    boundary_rows,
    global_metric_rows,
    hard_impact_row,
    high_amount_legit_row,
    replacement_rows,
    tie_diagnostic_row,
)
from fraud_detection.experiment.records import (
    fixed_reference_path_row,
    ranking_with_context,
    score_path,
    selected_gain_row,
)

pytestmark = pytest.mark.unit


def test_result_and_analysis_rows_preserve_schemas_and_ordering() -> None:
    effective_config = resolve_experiment_profile("canonical")
    baseline = pd.DataFrame(
        {
            "row_index": [10, 11, 12, 13],
            "original_position": [0, 1, 2, 3],
            "final_rank_position": [1, 2, 3, 4],
            "p_fraud": [0.9, 0.8, 0.7, 0.6],
            "priority_order_score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    comparison = baseline.copy()
    comparison["final_rank_position"] = [3, 1, 2, 4]
    comparison["priority_order_score"] = [2.0, 4.0, 3.0, 1.0]
    y_true = np.array([1, 0, 1, 0])
    amount = np.array([10.0, 20.0, 30.0, 40.0])

    assert score_path(METHOD_BASELINE, 20) == METHOD_BASELINE
    assert score_path(METHOD_P_ONLY, 20) == f"{METHOD_P_ONLY}_k20"
    contextual = ranking_with_context(
        comparison,
        row_labels=y_true,
        amounts=amount,
        outer_seed=42,
        target_budget=20,
        score_path=score_path(METHOD_P_ONLY, 20),
        method_family=METHOD_P_ONLY,
        selected_gain="linear",
        selection_status="SELECTED",
        truncation_level=23,
        final_n_estimators=17,
        effective_config=effective_config,
    )
    assert list(contextual.columns[:5]) == [
        "seed",
        "target_budget",
        "primary_budget",
        "score_path",
        "method_family",
    ]
    assert contextual["score_type"].unique().tolist() == [
        "raw_lambdarank_score_with_ordinal_full_order"
    ]

    config = {
        "selected_gain": "linear",
        "selection_status": "SELECTED",
        "final_n_estimators": 17,
        "truncation": 23,
        "config_hash": "c" * 64,
    }
    assert list(selected_gain_row(seed=42, target_budget=20, config=config)) == [
        "seed",
        "target_budget",
        "selected_gain",
        "selection_status",
        "final_n_estimators",
        "truncation_level",
        "eval_at",
        "config_hash",
    ]
    fixed = fixed_reference_path_row(
        seed=42,
        target_budget=20,
        train_pool_hash="a" * 64,
        test_pool_hash="b" * 64,
        raw_scores=np.array([0.1, 0.2]),
        effective_config=effective_config,
    )
    assert list(fixed) == [
        "seed",
        "target_budget",
        "formula",
        "train_candidate_pool_sha256",
        "test_candidate_pool_sha256",
        "candidate_pool_size",
        "raw_candidate_score_sha256",
    ]
    assert fixed["formula"] == "p_fraud * log1p(Amount)"

    replacements = replacement_rows(
        seed=42,
        budget=2,
        method_family=METHOD_P_ONLY,
        baseline_ranking=baseline,
        comparison_ranking=comparison,
        y_true=y_true,
        amount=amount,
    )
    assert [row["subset"] for row in replacements] == [
        "added_vs_bce",
        "removed_from_bce",
    ]
    assert [row["fraud_amount_sum"] for row in replacements] == [30.0, 10.0]
    boundaries = boundary_rows(
        seed=42,
        budget=2,
        method_family=METHOD_P_ONLY,
        baseline_ranking=baseline,
        comparison_ranking=comparison,
        y_true=y_true,
        amount=amount,
    )
    assert [row["row_index"] for row in boundaries] == [10, 11, 12, 13]
    assert boundaries[0]["dropped_from_topk"] is True
    assert boundaries[2]["crossed_into_topk"] is True
    global_rows = global_metric_rows(
        seed=42,
        budget=2,
        method_family=METHOD_BASELINE,
        score_path=METHOD_BASELINE,
        y_true=y_true,
        ranking=baseline,
    )
    assert [row["metric"] for row in global_rows] == [
        "roc_auc_of_final_order",
        "average_precision_of_final_order",
        "brier_score_probability",
    ]

    metric_ranking = pd.DataFrame(
        {
            "priority_order_score": [0.9, 0.8, 0.7, 0.1],
            "p_fraud": [0.9, 0.8, 0.7, 0.1],
        }
    )
    metric_labels = np.array([1, 0, 1, 0])
    numeric_rows = global_metric_rows(
        seed=42,
        budget=20,
        method_family=METHOD_BASELINE,
        score_path=METHOD_BASELINE,
        y_true=metric_labels,
        ranking=metric_ranking,
    )
    numeric_values = {row["metric"]: row["value"] for row in numeric_rows}
    assert list(numeric_values) == [
        "roc_auc_of_final_order",
        "average_precision_of_final_order",
        "brier_score_probability",
    ]
    np.testing.assert_allclose(
        numeric_values["roc_auc_of_final_order"],
        0.75,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        numeric_values["average_precision_of_final_order"],
        5 / 6,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        numeric_values["brier_score_probability"],
        0.1875,
        rtol=1e-12,
        atol=1e-12,
    )

    non_bce_rows = global_metric_rows(
        seed=42,
        budget=20,
        method_family=METHOD_P_ONLY,
        score_path=score_path(METHOD_P_ONLY, 20),
        y_true=metric_labels,
        ranking=metric_ranking,
    )
    assert [row["metric"] for row in non_bce_rows] == [
        "roc_auc_of_final_order",
        "average_precision_of_final_order",
    ]

    metric_row = {
        "unique_raw_ranker_scores": 2,
        "cutoff_raw_ranker_score": 0.5,
        "cutoff_tie_size": 2,
        "cutoff_tie_rank_min": 2,
        "cutoff_tie_rank_max": 3,
        "high_amount_legit_threshold_q90": 40.0,
        "legit_count_at_k": 1,
        "high_amount_legit_count_at_k": 1,
        "mean_legit_amount_at_k": 40.0,
        "prevented_loss_ratio_at_k": 0.5,
        "fraud_amount_sum_at_k": 30.0,
        "amount_ndcg_at_k": 0.75,
        "q90_threshold": 25.0,
        "q90_captured_ratio_at_k": 1.0,
        "q90_amount_ndcg_at_k": 1.0,
    }
    assert list(
        tie_diagnostic_row(
            outer_seed=42,
            target_budget=20,
            method_family=METHOD_FIXED,
            score_path=score_path(METHOD_FIXED, 20),
            metric_row=metric_row,
        )
    ) == [
        "seed",
        "target_budget",
        "method_family",
        "score_path",
        "unique_raw_ranker_scores",
        "cutoff_raw_ranker_score",
        "cutoff_tie_size",
        "cutoff_tie_rank_min",
        "cutoff_tie_rank_max",
    ]
    assert list(
        high_amount_legit_row(
            outer_seed=42,
            target_budget=20,
            method_family=METHOD_FIXED,
            metric_row=metric_row,
        )
    ) == [
        "seed",
        "target_budget",
        "method_family",
        "high_amount_legit_threshold_q90",
        "legit_count_at_k",
        "high_amount_legit_count_at_k",
        "mean_legit_amount_at_k",
    ]
    assert list(
        hard_impact_row(
            outer_seed=42,
            target_budget=20,
            method_family=METHOD_FIXED,
            metric_row=metric_row,
        )
    ) == [
        "seed",
        "target_budget",
        "method_family",
        "prevented_loss_ratio_at_k",
        "fraud_amount_sum_at_k",
        "amount_ndcg_at_k",
        "q90_threshold",
        "q90_captured_ratio_at_k",
        "q90_amount_ndcg_at_k",
    ]
