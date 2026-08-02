import numpy as np
import pandas as pd
import pytest

from fraud_detection.experiment.prioritization.composition import (
    bce_rank_positions,
    compose_candidate_reranking,
    validate_full_ranking,
)
from fraud_detection.experiment.prioritization.inputs import build_candidate_pool

pytestmark = pytest.mark.unit


def _pool(rows: int = 6, pool_size: int = 3) -> pd.DataFrame:
    return build_candidate_pool(
        np.linspace(1.0, 0.0, rows),
        np.arange(10_000, 10_000 + rows),
        candidate_pool_size=pool_size,
    )


def test_bce_ranks_descend_with_original_position_tie_breaking() -> None:
    ranks = bce_rank_positions(np.array([0.4, 0.8, 0.8, 0.3]))

    np.testing.assert_array_equal(ranks, np.array([3, 1, 2, 4]))
    assert ranks.dtype == np.dtype("int64")


def test_candidate_reranking_rejects_non_finite_or_misaligned_scores() -> None:
    pool = _pool()
    with pytest.raises(
        ValueError,
        match="raw_candidate_scores must contain only finite values",
    ):
        compose_candidate_reranking(pool, [np.inf, 0.2, 0.1])
    with pytest.raises(
        ValueError,
        match="raw_candidate_scores length must equal candidate_pool_size",
    ):
        compose_candidate_reranking(pool, [0.2, 0.1])


def test_all_candidates_precede_non_candidates() -> None:
    ranking = compose_candidate_reranking(_pool(), [1.0, 3.0, 2.0])
    candidates = ranking.loc[ranking["candidate_flag"]]
    non_candidates = ranking.loc[~ranking["candidate_flag"]]

    assert candidates["final_rank_position"].max() == 3
    assert non_candidates["final_rank_position"].min() == 4


def test_non_candidates_retain_relative_bce_order() -> None:
    ranking = compose_candidate_reranking(_pool(), [1.0, 3.0, 2.0])
    non_candidates = ranking.loc[~ranking["candidate_flag"]]

    assert (
        non_candidates.sort_values("final_rank_position")["row_index"].tolist()
        == non_candidates.sort_values("bce_rank_position")["row_index"].tolist()
    )


def test_candidate_score_ties_retain_candidate_bce_order() -> None:
    ranking = compose_candidate_reranking(_pool(), np.ones(3))
    candidates = ranking.loc[ranking["candidate_flag"]].sort_values(
        "final_rank_position"
    )

    assert candidates["candidate_rank_by_bce"].tolist() == [1, 2, 3]


def test_ranking_schema_nullable_dtypes_and_permutations_remain_exact() -> None:
    ranking = compose_candidate_reranking(_pool(), [1.0, 3.0, 2.0])

    assert list(ranking.columns) == [
        "row_index",
        "original_position",
        "candidate_flag",
        "candidate_rank_by_bce",
        "candidate_pool_size",
        "candidate_index",
        "candidate_pool_sha256",
        "p_fraud",
        "raw_ranker_score",
        "candidate_rank_by_ranker",
        "final_rank_position",
        "bce_rank_position",
        "priority_order_score",
    ]
    assert str(ranking["candidate_rank_by_bce"].dtype) == "Int64"
    assert str(ranking["candidate_rank_by_ranker"].dtype) == "Int64"
    assert ranking["final_rank_position"].dtype == np.dtype("int64")
    assert ranking["bce_rank_position"].dtype == np.dtype("int64")
    assert ranking["priority_order_score"].dtype == np.dtype("float64")
    assert ranking.loc[~ranking["candidate_flag"], "raw_ranker_score"].isna().all()
    assert sorted(ranking["final_rank_position"]) == list(range(1, 7))
    assert sorted(ranking["bce_rank_position"]) == list(range(1, 7))


def test_validation_rejects_malformed_permutations_or_candidate_ordering() -> None:
    ranking = compose_candidate_reranking(_pool(), [1.0, 3.0, 2.0])
    malformed = ranking.copy()
    malformed.loc[0, "final_rank_position"] = malformed.loc[
        1, "final_rank_position"
    ]
    with pytest.raises(
        ValueError,
        match="final_rank_position must be a complete rank permutation",
    ):
        validate_full_ranking(malformed)

    misordered = ranking.copy()
    candidate_index = misordered.loc[
        misordered["candidate_flag"], "final_rank_position"
    ].idxmax()
    non_candidate_index = misordered.loc[
        ~misordered["candidate_flag"], "final_rank_position"
    ].idxmin()
    candidate_rank = misordered.loc[candidate_index, "final_rank_position"]
    non_candidate_rank = misordered.loc[
        non_candidate_index, "final_rank_position"
    ]
    misordered.loc[candidate_index, "final_rank_position"] = non_candidate_rank
    misordered.loc[non_candidate_index, "final_rank_position"] = candidate_rank
    with pytest.raises(
        ValueError,
        match="All candidates must precede all non-candidates",
    ):
        validate_full_ranking(misordered)
