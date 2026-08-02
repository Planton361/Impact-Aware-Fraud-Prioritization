import numpy as np
import pandas as pd
import pytest

from fraud_detection.experiment.config import (
    CANDIDATE_POOL_SIZE,
    GAIN_PROFILES,
)
from fraud_detection.experiment.prioritization.inputs import (
    amount_gain_candidate_features,
    build_amount_gain_relevance_labels,
    build_candidate_pool,
    candidate_group,
    candidate_pool_hash,
    candidate_rows_by_bce,
    label_gain_for_profile,
    p_only_candidate_features,
    relevance_distribution,
    validate_candidate_pool,
)

pytestmark = pytest.mark.unit


def _pool(rows: int = 30, pool_size: int = 20) -> pd.DataFrame:
    return build_candidate_pool(
        np.linspace(0.0, 1.0, rows),
        np.arange(10_000, 10_000 + rows),
        candidate_pool_size=pool_size,
    )


def test_frozen_pool_size_gain_vectors_and_group() -> None:
    assert CANDIDATE_POOL_SIZE == 1000
    assert GAIN_PROFILES == {
        "exponential": (0, 1, 3, 7, 15),
        "linear": (0, 1, 2, 3, 4),
    }
    assert label_gain_for_profile("linear") == (0, 1, 2, 3, 4)
    assert label_gain_for_profile("exponential") == (0, 1, 3, 7, 15)
    first = candidate_group(CANDIDATE_POOL_SIZE)
    second = candidate_group(CANDIDATE_POOL_SIZE)
    assert first == second == [1000]
    assert first is not second


def test_candidate_membership_uses_only_probability_and_position() -> None:
    scores = np.array([0.5, 0.9, 0.9, 0.2, 0.8])
    indices = np.arange(20_000, 20_005)
    first = build_candidate_pool(scores, indices, candidate_pool_size=3)
    second = build_candidate_pool(scores.copy(), indices, candidate_pool_size=3)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    candidates = candidate_rows_by_bce(first)
    assert candidates["original_position"].tolist() == [1, 2, 4]
    assert candidates["row_index"].tolist() == [20_001, 20_002, 20_004]


def test_candidate_pool_schema_nullable_dtypes_and_ranks_are_stable() -> None:
    pool = _pool(rows=8, pool_size=3)
    assert list(pool.columns) == [
        "row_index",
        "original_position",
        "candidate_flag",
        "candidate_rank_by_bce",
        "candidate_pool_size",
        "candidate_index",
        "candidate_pool_sha256",
        "p_fraud",
    ]
    assert str(pool["candidate_rank_by_bce"].dtype) == "Int64"
    assert str(pool["candidate_index"].dtype) == "Int64"
    assert candidate_rows_by_bce(pool)["candidate_rank_by_bce"].tolist() == [1, 2, 3]
    validate_candidate_pool(pool, expected_pool_size=3)


def test_candidate_pool_rejects_invalid_identity_and_probability() -> None:
    scores = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="unique"):
        build_candidate_pool(scores, [1, 1, 2, 3, 4], candidate_pool_size=3)
    with pytest.raises(ValueError, match="missing"):
        build_candidate_pool(scores, [1, 2, np.nan, 4, 5], candidate_pool_size=3)
    for invalid in (np.nan, np.inf, -0.1, 1.1):
        changed = scores.copy()
        changed[0] = invalid
        with pytest.raises(ValueError):
            build_candidate_pool(changed, np.arange(5), candidate_pool_size=3)


def test_candidate_pool_hash_is_deterministic_and_alignment_sensitive() -> None:
    pool = _pool()
    same = _pool()
    changed = build_candidate_pool(
        np.linspace(0.0, 1.0, 30),
        np.arange(20_000, 20_030),
        candidate_pool_size=20,
    )
    assert candidate_pool_hash(pool) == candidate_pool_hash(same)
    assert candidate_pool_hash(pool) != candidate_pool_hash(changed)
    inconsistent = pool.copy()
    inconsistent.loc[0, "candidate_pool_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="inconsistent"):
        candidate_pool_hash(inconsistent)


def test_relevance_uses_only_fraud_amount_and_labels_zero_through_four() -> None:
    y_true = [0, 1, 1, 1, 1, 0]
    labels, thresholds = build_amount_gain_relevance_labels(
        y_true,
        [100_000.0, 10.0, 20.0, 30.0, 40.0, 200_000.0],
    )
    _, changed_legitimate = build_amount_gain_relevance_labels(
        y_true,
        [0.0, 10.0, 20.0, 30.0, 40.0, 1.0],
    )
    np.testing.assert_allclose(thresholds, [17.5, 25.0, 32.5])
    np.testing.assert_array_equal(thresholds, changed_legitimate)
    np.testing.assert_array_equal(labels, [0, 1, 2, 3, 4, 0])
    assert relevance_distribution(labels) == {
        "0": 2,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
    }


def test_training_thresholds_are_reused_without_recalculation() -> None:
    labels, thresholds = build_amount_gain_relevance_labels(
        [1, 1, 1, 1, 0],
        [11.0, 12.0, 13.0, 14.0, 1_000.0],
        fraud_amount_thresholds=[10.0, 20.0, 30.0],
    )
    np.testing.assert_array_equal(thresholds, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(labels, [2, 2, 2, 2, 0])


def test_invalid_relevance_inputs_are_rejected() -> None:
    invalid_inputs = (
        ([0, 2], [10.0, 20.0]),
        ([0, 1], [10.0, np.nan]),
        ([0, 1], [10.0, -1.0]),
        ([0, 0], [10.0, 20.0]),
    )
    for y_true, amounts in invalid_inputs:
        with pytest.raises(ValueError):
            build_amount_gain_relevance_labels(y_true, amounts)


def test_amount_gain_features_preserve_candidate_order_and_values() -> None:
    pool = _pool()
    amount = np.arange(30, dtype=float)
    candidates = candidate_rows_by_bce(pool)
    positions = candidates["original_position"].to_numpy(dtype=int)
    expected_probability = candidates["p_fraud"].to_numpy(dtype=float)
    expected_log_amount = np.log1p(amount[positions])
    features = amount_gain_candidate_features(pool, amount)
    assert list(features.columns) == [
        "p_fraud",
        "log1p_amount",
        "p_fraud_x_log1p_amount",
    ]
    np.testing.assert_allclose(features["p_fraud"], expected_probability)
    np.testing.assert_allclose(features["log1p_amount"], expected_log_amount)
    np.testing.assert_allclose(
        features["p_fraud_x_log1p_amount"],
        expected_probability * expected_log_amount,
    )


def test_p_only_features_contain_exactly_candidate_probability() -> None:
    pool = _pool()
    features = p_only_candidate_features(pool)
    assert list(features.columns) == ["p_fraud"]
    np.testing.assert_allclose(
        features["p_fraud"],
        candidate_rows_by_bce(pool)["p_fraud"],
    )
