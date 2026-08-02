"""Construct and validate complete rankings from candidates and the BCE remainder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inputs import (
    candidate_rows_by_bce as _candidate_rows_by_bce,
)
from .inputs import (
    validate_candidate_pool as _validate_candidate_pool,
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


def _finite_score_vector(scores: object) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1:
        raise ValueError("scores must be one-dimensional.")
    if arr.size == 0:
        raise ValueError("scores must contain at least one element.")
    if not np.isfinite(arr).all():
        raise ValueError("scores must not contain NaN or infinite values.")
    return arr


def _descending_score_order(scores: object) -> np.ndarray:
    scores_arr = _finite_score_vector(scores)
    return np.lexsort((np.arange(scores_arr.shape[0]), -scores_arr))


def bce_rank_positions(scores: np.ndarray) -> np.ndarray:
    order = _descending_score_order(scores)
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.int64)
    return ranks


def compose_candidate_reranking(
    candidate_pool: pd.DataFrame,
    raw_candidate_scores: object,
) -> pd.DataFrame:
    """Compose candidates-first full ordering from candidate raw scores.

    ``raw_candidate_scores`` must be aligned to ``candidate_rows_by_bce``.
    Candidate ties use ``candidate_rank_by_bce``; non-candidates retain their
    relative BCE ordering. ``priority_order_score`` is purely ordinal.
    """

    pool_size = int(candidate_pool["candidate_pool_size"].iloc[0])
    _validate_candidate_pool(candidate_pool, expected_pool_size=pool_size)
    raw_scores = _one_dimensional_finite(
        raw_candidate_scores,
        "raw_candidate_scores",
    )
    if raw_scores.shape[0] != pool_size:
        raise ValueError(
            "raw_candidate_scores length must equal candidate_pool_size."
        )

    candidates = _candidate_rows_by_bce(candidate_pool)
    candidate_positions = candidates["original_position"].to_numpy(dtype=int)
    candidate_bce_ranks = candidates["candidate_rank_by_bce"].to_numpy(dtype=int)
    candidate_local_order = np.lexsort((candidate_bce_ranks, -raw_scores))
    ordered_candidate_positions = candidate_positions[candidate_local_order]

    all_scores = candidate_pool["p_fraud"].to_numpy(dtype=float)
    bce_order = _descending_score_order(all_scores)
    flags = candidate_pool["candidate_flag"].to_numpy(dtype=bool)
    ordered_non_candidate_positions = bce_order[~flags[bce_order]]
    final_order = np.concatenate(
        (ordered_candidate_positions, ordered_non_candidate_positions)
    )
    if not np.array_equal(np.sort(final_order), np.arange(len(candidate_pool))):
        raise RuntimeError("Full candidate reranking did not produce a permutation.")

    final_rank = np.empty(len(candidate_pool), dtype=np.int64)
    final_rank[final_order] = np.arange(1, len(candidate_pool) + 1, dtype=np.int64)
    candidate_ranker_rank = np.zeros(len(candidate_pool), dtype=np.int64)
    candidate_ranker_rank[ordered_candidate_positions] = np.arange(
        1,
        pool_size + 1,
        dtype=np.int64,
    )
    raw_full = np.full(len(candidate_pool), np.nan, dtype=float)
    raw_full[candidate_positions] = raw_scores

    output = candidate_pool.copy()
    output["raw_ranker_score"] = raw_full
    output["candidate_rank_by_ranker"] = pd.Series(
        candidate_ranker_rank
    ).where(flags, pd.NA).astype("Int64")
    output["final_rank_position"] = final_rank
    output["bce_rank_position"] = bce_rank_positions(all_scores)
    output["priority_order_score"] = (
        len(output) - final_rank + 1
    ).astype(float)
    validate_full_ranking(output)
    return output


def validate_full_ranking(ranking: pd.DataFrame) -> None:
    required = {
        "row_index",
        "original_position",
        "candidate_flag",
        "candidate_rank_by_bce",
        "p_fraud",
        "raw_ranker_score",
        "candidate_rank_by_ranker",
        "final_rank_position",
        "bce_rank_position",
        "priority_order_score",
    }
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise ValueError(f"ranking is missing columns: {missing}")
    n_rows = len(ranking)
    if n_rows == 0 or not ranking["row_index"].is_unique:
        raise ValueError("ranking must contain a non-empty unique row_index set.")
    expected_ranks = np.arange(1, n_rows + 1)
    final_ranks = ranking["final_rank_position"].to_numpy(dtype=int)
    bce_ranks = ranking["bce_rank_position"].to_numpy(dtype=int)
    if not np.array_equal(np.sort(final_ranks), expected_ranks):
        raise ValueError("final_rank_position must be a complete rank permutation.")
    if not np.array_equal(np.sort(bce_ranks), expected_ranks):
        raise ValueError("bce_rank_position must be a complete rank permutation.")
    if not np.isfinite(ranking["priority_order_score"].to_numpy(dtype=float)).all():
        raise ValueError("priority_order_score must contain only finite values.")

    flags = ranking["candidate_flag"].to_numpy(dtype=bool)
    if flags.any():
        candidate_final = final_ranks[flags]
        non_candidate_final = final_ranks[~flags]
        is_bce_baseline = np.array_equal(final_ranks, bce_ranks)
        if not is_bce_baseline and non_candidate_final.size:
            if int(candidate_final.max()) >= int(non_candidate_final.min()):
                raise ValueError("All candidates must precede all non-candidates.")
            non_candidate_order = np.flatnonzero(~flags)[
                np.argsort(non_candidate_final, kind="mergesort")
            ]
            expected_non_candidate_order = np.flatnonzero(~flags)[
                np.argsort(bce_ranks[~flags], kind="mergesort")
            ]
            if not np.array_equal(
                non_candidate_order,
                expected_non_candidate_order,
            ):
                raise ValueError(
                    "Non-candidates must retain relative BCE ordering."
                )
