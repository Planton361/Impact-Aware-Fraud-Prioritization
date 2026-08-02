import numpy as np
import pytest

from fraud_detection.artifacts import (
    canonical_json_sha256,
    score_vector_sha256,
)

pytestmark = pytest.mark.unit


def test_fixed_artifact_hash_baselines_are_byte_identical() -> None:
    assert canonical_json_sha256({"b": 2, "a": [1, 3]}) == (
        "41206cfbbd2c91b0c47347e15c006841"
        "398a76c34ce20c0ae03c5062a8febaef"
    )
    assert score_vector_sha256(
        np.array([0.0, 0.25, 1.0]),
        score_type="unit.score",
    ) == (
        "4ef83d7a6199276031906f0c933b172a"
        "737622d082e50b7856a9b87c68902418"
    )


def test_score_type_changes_the_score_vector_digest() -> None:
    values = np.array([0.0, 0.25, 1.0])

    assert score_vector_sha256(
        values,
        score_type="unit.score",
    ) != score_vector_sha256(values, score_type="other.score")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([[0.1, 0.2]]), "one-dimensional"),
        (np.array([]), "at least one element"),
        (np.array([0.1, np.nan]), "only finite values"),
    ],
)
def test_invalid_score_vectors_are_rejected(values: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        score_vector_sha256(values, score_type="unit.score")
