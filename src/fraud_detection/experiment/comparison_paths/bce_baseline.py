"""Complete descending-BCE comparison path."""

from __future__ import annotations

import pandas as pd

from ..prioritization.composition import (
    bce_rank_positions,
    validate_full_ranking,
)
from ..prioritization.inputs import validate_candidate_pool


def baseline_bce_ranking(candidate_pool: pd.DataFrame) -> pd.DataFrame:
    pool_size = int(candidate_pool["candidate_pool_size"].iloc[0])
    validate_candidate_pool(candidate_pool, expected_pool_size=pool_size)
    scores = candidate_pool["p_fraud"].to_numpy(dtype=float)
    ranks = bce_rank_positions(scores)
    output = candidate_pool.copy()
    output["raw_ranker_score"] = scores
    output["candidate_rank_by_ranker"] = pd.array(
        [pd.NA] * len(output),
        dtype="Int64",
    )
    output["final_rank_position"] = ranks
    output["bce_rank_position"] = ranks.copy()
    output["priority_order_score"] = (
        len(output) - ranks + 1
    ).astype(float)
    validate_full_ranking(output)
    return output
