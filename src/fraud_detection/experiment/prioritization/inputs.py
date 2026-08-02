"""Canonical ranking inputs for the frozen comparison paths."""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pandas as pd

from ..config import GAIN_PROFILES


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


def _probabilities(values: object, name: str = "p_fraud") -> np.ndarray:
    arr = _one_dimensional_finite(values, name)
    if (arr < 0.0).any() or (arr > 1.0).any():
        raise ValueError(f"{name} must be in [0, 1].")
    return arr


def _amounts(values: object, name: str = "amount") -> np.ndarray:
    arr = _one_dimensional_finite(values, name)
    if (arr < 0.0).any():
        raise ValueError(f"{name} must be non-negative.")
    return arr


def _validate_1d_finite(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one element.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must not contain NaN or infinite values.")
    return arr


def _validate_binary_y_true(y_true: object) -> np.ndarray:
    y = _validate_1d_finite(y_true, "y_true")
    unique = set(np.unique(y))
    if unique - {0.0, 1.0}:
        raise ValueError("y_true must be binary with values {0, 1}.")
    if not (y == 1.0).any():
        raise ValueError("y_true must contain at least one fraud case.")
    return y.astype(int)


def _validate_amount(amount: object) -> np.ndarray:
    amount_arr = _validate_1d_finite(amount, "amount")
    if (amount_arr < 0).any():
        raise ValueError("amount must be non-negative.")
    return amount_arr


def _validate_quantiles(quantiles: object) -> np.ndarray:
    quantile_arr = _validate_1d_finite(quantiles, "quantiles")
    if quantile_arr.shape[0] != 3:
        raise ValueError("quantiles must contain exactly three values.")
    if (quantile_arr < 0).any() or (quantile_arr > 1).any():
        raise ValueError("quantiles must be in [0, 1].")
    if np.any(np.diff(quantile_arr) < 0):
        raise ValueError("quantiles must be sorted in non-decreasing order.")
    return quantile_arr


def _validate_thresholds(fraud_amount_thresholds: object) -> np.ndarray:
    thresholds = _validate_1d_finite(
        fraud_amount_thresholds,
        "fraud_amount_thresholds",
    )
    if thresholds.shape[0] != 3:
        raise ValueError(
            "fraud_amount_thresholds must contain exactly three values."
        )
    if (thresholds < 0).any():
        raise ValueError("fraud_amount_thresholds must be non-negative.")
    if np.any(np.diff(thresholds) < 0):
        raise ValueError(
            "fraud_amount_thresholds must be sorted in non-decreasing order."
        )
    return thresholds


def label_gain_for_profile(
    gain_profile: object,
) -> tuple[int, int, int, int, int]:
    if not isinstance(gain_profile, str) or gain_profile not in GAIN_PROFILES:
        raise ValueError(
            "gain_profile must be exactly one of {'exponential', 'linear'}."
        )
    return GAIN_PROFILES[gain_profile]


def _candidate_pool_sha256(
    candidate_positions: np.ndarray,
    row_index: pd.Index,
    scores: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"fraud_detection.candidate_pool_sha256.v1\0")
    for position in candidate_positions:
        position_int = int(position)
        row_value = row_index[position_int]
        row_encoded = json.dumps(
            row_value.item() if isinstance(row_value, np.generic) else row_value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(struct.pack("<q", position_int))
        digest.update(struct.pack("<Q", len(row_encoded)))
        digest.update(row_encoded)
        digest.update(struct.pack("<d", float(scores[position_int])))
    return digest.hexdigest()


def build_candidate_pool(
    p_fraud: object,
    row_index: object,
    *,
    candidate_pool_size: int,
) -> pd.DataFrame:
    """Select a deterministic BCE-only candidate pool.

    The function intentionally has no label, Amount, Time, relevance, ranker-score,
    or metric argument. Ties are resolved by original vector position.
    """

    scores = _probabilities(p_fraud)
    pool_size = _positive_int(candidate_pool_size, "candidate_pool_size")
    if pool_size > scores.shape[0]:
        raise ValueError("candidate_pool_size cannot exceed the number of rows.")

    index = pd.Index(row_index)
    if len(index) != scores.shape[0]:
        raise ValueError("row_index and p_fraud must have the same length.")
    if not index.is_unique:
        raise ValueError("row_index must contain unique values.")
    if index.hasnans:
        raise ValueError("row_index must not contain missing values.")

    original_position = np.arange(scores.shape[0], dtype=np.int64)
    candidate_positions = np.lexsort((original_position, -scores))[:pool_size]
    candidate_flag = np.zeros(scores.shape[0], dtype=bool)
    candidate_flag[candidate_positions] = True
    candidate_rank = np.zeros(scores.shape[0], dtype=np.int64)
    candidate_rank[candidate_positions] = np.arange(1, pool_size + 1, dtype=np.int64)
    pool_hash = _candidate_pool_sha256(candidate_positions, index, scores)

    frame = pd.DataFrame(
        {
            "row_index": index.to_numpy(copy=True),
            "original_position": original_position,
            "candidate_flag": candidate_flag,
            "candidate_rank_by_bce": pd.Series(candidate_rank).where(
                candidate_flag,
                pd.NA,
            ).astype("Int64"),
            "candidate_pool_size": np.full(
                scores.shape[0],
                pool_size,
                dtype=np.int64,
            ),
            "candidate_index": pd.Series(original_position).where(
                candidate_flag,
                pd.NA,
            ).astype("Int64"),
            "candidate_pool_sha256": np.full(
                scores.shape[0],
                pool_hash,
                dtype=object,
            ),
            "p_fraud": scores,
        }
    )
    validate_candidate_pool(frame, expected_pool_size=pool_size)
    return frame


def validate_candidate_pool(
    candidate_pool: pd.DataFrame,
    *,
    expected_pool_size: int,
) -> None:
    required = {
        "row_index",
        "original_position",
        "candidate_flag",
        "candidate_rank_by_bce",
        "candidate_pool_size",
        "candidate_index",
        "candidate_pool_sha256",
        "p_fraud",
    }
    missing = sorted(required - set(candidate_pool.columns))
    if missing:
        raise ValueError(f"candidate_pool is missing columns: {missing}")
    if candidate_pool.empty:
        raise ValueError("candidate_pool must contain at least one row.")

    pool_size = _positive_int(expected_pool_size, "expected_pool_size")
    if len(candidate_pool) < pool_size:
        raise ValueError("candidate_pool has fewer rows than expected_pool_size.")
    if not candidate_pool["row_index"].is_unique:
        raise ValueError("candidate_pool row_index values must be unique.")
    positions = candidate_pool["original_position"].to_numpy(dtype=int)
    if not np.array_equal(positions, np.arange(len(candidate_pool))):
        raise ValueError("candidate_pool original_position must be a full 0-based range.")

    flags = candidate_pool["candidate_flag"].to_numpy(dtype=bool)
    if int(flags.sum()) != pool_size:
        raise ValueError("candidate_pool must contain exactly expected_pool_size candidates.")
    declared_sizes = candidate_pool["candidate_pool_size"].to_numpy(dtype=int)
    if not np.all(declared_sizes == pool_size):
        raise ValueError("candidate_pool_size metadata is inconsistent.")
    hashes = candidate_pool["candidate_pool_sha256"].astype(str).unique()
    if hashes.size != 1 or len(hashes[0]) != 64:
        raise ValueError("candidate_pool_sha256 metadata is inconsistent.")

    ranks = candidate_pool.loc[flags, "candidate_rank_by_bce"].to_numpy(dtype=int)
    if not np.array_equal(np.sort(ranks), np.arange(1, pool_size + 1)):
        raise ValueError("candidate_rank_by_bce must be a complete 1-based range.")
    if candidate_pool.loc[~flags, "candidate_rank_by_bce"].notna().any():
        raise ValueError("Non-candidates must not have candidate_rank_by_bce.")

    scores = _probabilities(candidate_pool["p_fraud"], "candidate_pool.p_fraud")
    expected = np.lexsort((positions, -scores))[:pool_size]
    actual = (
        candidate_pool.loc[flags]
        .sort_values("candidate_rank_by_bce", kind="mergesort")[
            "original_position"
        ]
        .to_numpy(dtype=int)
    )
    if not np.array_equal(actual, expected):
        raise ValueError("candidate_pool is not the deterministic BCE top pool.")


def candidate_rows_by_bce(candidate_pool: pd.DataFrame) -> pd.DataFrame:
    expected_pool_size = int(candidate_pool["candidate_pool_size"].iloc[0])
    validate_candidate_pool(
        candidate_pool,
        expected_pool_size=expected_pool_size,
    )
    return (
        candidate_pool.loc[candidate_pool["candidate_flag"].astype(bool)]
        .sort_values("candidate_rank_by_bce", kind="mergesort")
        .reset_index(drop=True)
    )


def build_amount_gain_relevance_labels(
    y_true: object,
    amount: object,
    *,
    fraud_amount_thresholds: object | None = None,
    quantiles: object = (0.25, 0.5, 0.75),
) -> tuple[np.ndarray, np.ndarray]:
    """Build train-scoped Amount-proxy relevance labels."""

    y = _validate_binary_y_true(y_true)
    amount_arr = _validate_amount(amount)
    if amount_arr.shape[0] != y.shape[0]:
        raise ValueError("amount and y_true must have the same length.")

    if fraud_amount_thresholds is None:
        threshold_arr = np.quantile(
            amount_arr[y == 1],
            _validate_quantiles(quantiles),
        )
    else:
        threshold_arr = _validate_thresholds(fraud_amount_thresholds)

    labels = np.zeros(y.shape[0], dtype=int)
    fraud_mask = y == 1
    labels[fraud_mask] = (
        np.searchsorted(
            threshold_arr,
            amount_arr[fraud_mask],
            side="right",
        )
        + 1
    )
    return labels, threshold_arr.astype(float, copy=True)


def amount_gain_candidate_features(
    candidate_pool: pd.DataFrame,
    amount: object,
) -> pd.DataFrame:
    amounts = _amounts(amount)
    if amounts.shape[0] != len(candidate_pool):
        raise ValueError("amount and candidate_pool must have the same length.")
    candidates = candidate_rows_by_bce(candidate_pool)
    positions = candidates["original_position"].to_numpy(dtype=int)
    p_fraud = _probabilities(candidates["p_fraud"].to_numpy(dtype=float))
    log_amount = np.log1p(amounts[positions])
    return pd.DataFrame(
        {
            "p_fraud": p_fraud,
            "log1p_amount": log_amount,
            "p_fraud_x_log1p_amount": p_fraud * log_amount,
        }
    )


def p_only_candidate_features(candidate_pool: pd.DataFrame) -> pd.DataFrame:
    candidates = candidate_rows_by_bce(candidate_pool)
    return pd.DataFrame(
        {
            "p_fraud": _probabilities(
                candidates["p_fraud"].to_numpy(dtype=float),
            )
        }
    )


def candidate_group(candidate_pool_size: int) -> list[int]:
    return [_positive_int(candidate_pool_size, "candidate_pool_size")]


def candidate_pool_hash(candidate_pool: pd.DataFrame) -> str:
    hashes = candidate_pool["candidate_pool_sha256"].astype(str).unique()
    if hashes.size != 1:
        raise RuntimeError("Candidate pool contains inconsistent hashes.")
    return str(hashes[0])


def relevance_distribution(labels: object) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=5)
    return {str(label): int(counts[label]) for label in range(5)}
