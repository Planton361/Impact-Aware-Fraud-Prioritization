import numpy as np
import pytest

from fraud_detection.experiment.evaluation.metrics import (
    amount_ndcg_at_k,
    cutoff_tie_diagnostics,
    matched_budget_metrics,
)
from fraud_detection.experiment.prioritization.composition import (
    compose_candidate_reranking,
)
from fraud_detection.experiment.prioritization.inputs import build_candidate_pool

pytestmark = pytest.mark.unit


def _ranking(raw_scores: object = (5.0, 4.0, 3.0, 2.0, 1.0)):
    pool = build_candidate_pool(
        np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
        np.arange(5),
        candidate_pool_size=5,
    )
    return compose_candidate_reranking(pool, raw_scores)


def _labels_and_amounts() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([1, 0, 1, 0, 1]),
        np.array([10.0, 100.0, 30.0, 40.0, 50.0]),
    )


def test_amount_ndcg_matches_the_mathematical_definition() -> None:
    observed = amount_ndcg_at_k(
        [1, 1, 0],
        [10.0, 30.0, 100.0],
        [1, 3, 2],
        2,
    )
    expected = 10.0 / (30.0 + 10.0 / np.log2(3.0))

    assert observed == pytest.approx(expected)


def test_thresholded_amount_ndcg_regression_remains_one_half() -> None:
    result = amount_ndcg_at_k(
        [1, 0, 1, 0],
        [10.0, 100.0, 30.0, 40.0],
        [2, 1, 3, 4],
        3,
        minimum_fraud_amount=20.0,
    )

    assert result == pytest.approx(0.5)


def test_non_finite_minimum_fraud_amount_is_rejected() -> None:
    with pytest.raises(ValueError, match="minimum_fraud_amount must be finite"):
        amount_ndcg_at_k(
            [1, 0],
            [10.0, 20.0],
            [1, 2],
            1,
            minimum_fraud_amount=np.inf,
        )


def test_cutoff_tie_diagnostics_preserve_exact_schema_and_ranks() -> None:
    diagnostics = cutoff_tie_diagnostics(
        _ranking([0.9, 0.8, 0.8, 0.1, 0.0]),
        2,
    )

    assert diagnostics == {
        "unique_raw_ranker_scores": 4,
        "cutoff_raw_ranker_score": 0.8,
        "cutoff_tie_size": 2,
        "cutoff_tie_rank_min": 2,
        "cutoff_tie_rank_max": 3,
    }


def test_matched_metrics_preserve_exact_key_ordering() -> None:
    y_true, amount = _labels_and_amounts()
    result = matched_budget_metrics(y_true, amount, _ranking(), 2)

    assert list(result) == [
        "target_budget",
        "prevented_loss_ratio_at_k",
        "frauds_at_k",
        "precision_at_k",
        "recall_at_k",
        "fraud_amount_sum_at_k",
        "legit_count_at_k",
        "amount_ndcg_at_k",
        "q90_threshold",
        "q90_total_fraud_count",
        "q90_captured_fraud_count",
        "q90_captured_ratio_at_k",
        "q90_amount_ndcg_at_k",
        "high_amount_legit_threshold_q90",
        "high_amount_legit_count_at_k",
        "mean_legit_amount_at_k",
        "unique_raw_ranker_scores",
        "cutoff_raw_ranker_score",
        "cutoff_tie_size",
        "cutoff_tie_rank_min",
        "cutoff_tie_rank_max",
    ]


def test_fraud_amount_proxy_coverage_counts_only_selected_fraud_rows() -> None:
    y_true, amount = _labels_and_amounts()
    result = matched_budget_metrics(y_true, amount, _ranking(), 2)

    assert result["frauds_at_k"] == 1
    assert result["fraud_amount_sum_at_k"] == 10.0
    assert result["prevented_loss_ratio_at_k"] == pytest.approx(10.0 / 90.0)


def test_q90_fraud_diagnostics_preserve_current_quantile_behavior() -> None:
    y_true, amount = _labels_and_amounts()
    result = matched_budget_metrics(
        y_true,
        amount,
        _ranking([4.0, 3.0, 2.0, 1.0, 5.0]),
        2,
    )

    assert result["q90_threshold"] == pytest.approx(46.0)
    assert result["q90_total_fraud_count"] == 1
    assert result["q90_captured_fraud_count"] == 1
    assert result["q90_captured_ratio_at_k"] == 1.0
    assert result["q90_amount_ndcg_at_k"] == 1.0


def test_high_amount_legitimate_diagnostics_preserve_current_behavior() -> None:
    y_true, amount = _labels_and_amounts()
    result = matched_budget_metrics(y_true, amount, _ranking(), 2)

    assert result["high_amount_legit_threshold_q90"] == pytest.approx(94.0)
    assert result["high_amount_legit_count_at_k"] == 1
    assert result["mean_legit_amount_at_k"] == 100.0


def test_invalid_permutations_and_fraud_zero_cases_are_rejected() -> None:
    y_true, amount = _labels_and_amounts()
    malformed = _ranking()
    malformed.loc[0, "final_rank_position"] = malformed.loc[
        1, "final_rank_position"
    ]
    with pytest.raises(ValueError, match="complete rank permutation"):
        matched_budget_metrics(y_true, amount, malformed, 2)
    with pytest.raises(ValueError, match="At least one fraud case is required"):
        matched_budget_metrics(np.zeros(5), amount, _ranking(), 2)
    with pytest.raises(ValueError, match="Total fraud Amount must be positive"):
        matched_budget_metrics(
            [1, 0, 0, 0, 0],
            [0.0, 100.0, 30.0, 40.0, 50.0],
            _ranking(),
            2,
        )
