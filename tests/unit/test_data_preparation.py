from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.artifacts import _ordered_index_sha256
from fraud_detection.experiment.config import (
    BCE_FEATURES,
    OUTER_SEEDS,
    resolve_experiment_profile,
)
from fraud_detection.experiment.preparation import data as data_preparation
from fraud_detection.experiment.preparation.data import (
    apply_full_row_deduplication,
    dataframe_content_sha256,
    expected_v_columns,
    outer_split_indices,
    required_columns,
    validate_creditcard_schema,
    verify_outer_split_without_test_labels,
)

pytestmark = pytest.mark.unit


def _creditcard_frame(
    rows: int = 100,
    *,
    fraud_rows: int = 20,
    seed: int = 123,
) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    dataframe = pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            **{
                column: random.normal(size=rows)
                for column in expected_v_columns()
            },
            "Amount": random.uniform(0.0, 1000.0, size=rows),
            "Class": np.concatenate(
                [
                    np.ones(fraud_rows, dtype=int),
                    np.zeros(rows - fraud_rows, dtype=int),
                ]
            ),
        }
    )
    return dataframe


def _row(
    marker: float,
    *,
    time: float | None = None,
    amount: float | None = None,
    target: int = 0,
) -> dict[str, float | int]:
    return {
        "Time": marker if time is None else time,
        **{
            column: marker + position / 100.0
            for position, column in enumerate(expected_v_columns(), start=1)
        },
        "Amount": marker + 10.0 if amount is None else amount,
        "Class": target,
    }


def _frame(rows: list[dict[str, float | int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=required_columns())


def _write_split_identity(
    output_root: Path,
    *,
    seed: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> None:
    preflight_dir = output_root / "preflight"
    preflight_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "seed": seed,
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "train_fraud": 378,
                "test_fraud": 95,
                "train_index_sha256": _ordered_index_sha256(train_index),
                "test_index_sha256": _ordered_index_sha256(test_index),
                "passed": True,
            }
        ]
    ).to_csv(preflight_dir / "outer_split_identity.csv", index=False)


def test_canonical_schema_and_bce_feature_roles() -> None:
    expected_features = [f"V{index}" for index in range(1, 29)]
    assert expected_v_columns() == expected_features
    assert tuple(expected_v_columns()) == BCE_FEATURES
    assert required_columns() == [
        "Time",
        *expected_features,
        "Amount",
        "Class",
    ]
    assert "Time" not in BCE_FEATURES
    assert "Amount" not in BCE_FEATURES


def test_schema_requires_every_column_to_be_numeric() -> None:
    missing = _creditcard_frame().drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_creditcard_schema(missing)

    non_numeric = _creditcard_frame()
    non_numeric["V4"] = "not-numeric"
    with pytest.raises(ValueError, match="Non-numeric columns"):
        validate_creditcard_schema(non_numeric)


def test_schema_rejects_invalid_required_values() -> None:
    nan_frame = _creditcard_frame()
    nan_frame.loc[0, "V1"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_creditcard_schema(nan_frame)

    infinite_frame = _creditcard_frame()
    infinite_frame.loc[0, "V2"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_creditcard_schema(infinite_frame)

    non_binary_frame = _creditcard_frame()
    non_binary_frame.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_creditcard_schema(non_binary_frame)

    negative_amount_frame = _creditcard_frame()
    negative_amount_frame.loc[0, "Amount"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        validate_creditcard_schema(negative_amount_frame)


def test_stable_deduplication_preserves_source_alignment_and_audit() -> None:
    duplicate = _row(3.0, target=1)
    frame = _frame(
        [
            _row(1.0),
            duplicate,
            _row(2.0),
            duplicate,
            _row(4.0),
            duplicate,
        ]
    ).set_axis([90, 12, 8, 77, 3, 44], axis="index")

    processed, audit = apply_full_row_deduplication(
        frame,
        source_data_sha256="source-sha256",
    )

    assert processed.index.tolist() == [90, 12, 8, 3]
    pd.testing.assert_frame_equal(processed, frame.loc[[90, 12, 8, 3]])
    pd.testing.assert_frame_equal(
        processed[["Amount", "Class"]],
        frame.loc[processed.index, ["Amount", "Class"]],
    )
    metadata = audit.to_metadata()
    assert audit.deduplication_enabled is True
    assert audit.rows_removed_as_duplicate_followups == 2
    assert (audit.class_0_before, audit.class_0_after) == (3, 3)
    assert (audit.class_1_before, audit.class_1_after) == (3, 1)
    assert metadata["duplicate_definition"] == "exact full-row equality"
    assert metadata["duplicate_keep"] == "first"
    assert metadata["rows_before_deduplication"] == 6
    assert metadata["rows_after_deduplication"] == 4
    assert metadata["source_data_sha256"] == "source-sha256"
    assert metadata["deduplicated_dataframe_sha256"] == (
        dataframe_content_sha256(processed)
    )
    assert metadata["columns_used_for_deduplication"] == frame.columns.tolist()


def test_deduplication_uses_exact_full_row_equality() -> None:
    base = _row(5.0, time=10.0, amount=25.0)
    different_feature = dict(base)
    different_feature["V17"] = float(different_feature["V17"]) + 0.001
    different_amount = dict(base)
    different_amount["Amount"] = 26.0
    different_time = dict(base)
    different_time["Time"] = 11.0
    different_class = dict(base)
    different_class["Class"] = 1
    frame = _frame(
        [
            base,
            different_feature,
            different_amount,
            different_time,
            different_class,
        ]
    )

    processed, audit = apply_full_row_deduplication(
        frame,
        source_data_sha256="source-sha256",
    )

    pd.testing.assert_frame_equal(processed, frame)
    assert audit.rows_removed_as_duplicate_followups == 0


def test_dataframe_hash_is_deterministic_and_index_independent() -> None:
    frame = _creditcard_frame(rows=20, fraud_rows=4)
    changed_index = frame.set_axis(np.arange(100, 120), axis="index")

    assert dataframe_content_sha256(frame) == dataframe_content_sha256(frame)
    assert dataframe_content_sha256(frame) == dataframe_content_sha256(
        changed_index
    )
    assert dataframe_content_sha256(frame) != dataframe_content_sha256(
        frame.iloc[::-1]
    )


def test_outer_split_is_deterministic_and_seed_specific() -> None:
    frame = _creditcard_frame()
    first_train, first_test = outer_split_indices(frame, OUTER_SEEDS[0])
    second_train, second_test = outer_split_indices(frame, OUTER_SEEDS[0])
    different_train, different_test = outer_split_indices(frame, OUTER_SEEDS[1])

    np.testing.assert_array_equal(first_train, second_train)
    np.testing.assert_array_equal(first_test, second_test)
    assert not np.array_equal(first_train, different_train)
    assert not np.array_equal(first_test, different_test)


def test_outer_split_is_stratified_complete_and_source_aligned() -> None:
    frame = _creditcard_frame().set_axis(np.arange(1000, 1100), axis="index")
    train_index, test_index = outer_split_indices(frame, OUTER_SEEDS[0])

    assert train_index.dtype.kind == "i"
    assert test_index.dtype.kind == "i"
    assert len(train_index) == 80
    assert len(test_index) == 20
    assert set(train_index).isdisjoint(test_index)
    assert set(train_index) | set(test_index) == set(frame.index)
    assert int(frame.loc[train_index, "Class"].sum()) == 16
    assert int(frame.loc[test_index, "Class"].sum()) == 4
    pd.testing.assert_frame_equal(
        frame.loc[train_index, ["Amount", "Class"]],
        frame[["Amount", "Class"]].loc[train_index],
    )
    pd.testing.assert_frame_equal(
        frame.loc[test_index, ["Amount", "Class"]],
        frame[["Amount", "Class"]].loc[test_index],
    )


def test_split_identity_verification_accepts_matching_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_index = np.arange(226980, dtype=int)
    test_index = np.arange(226980, 283726, dtype=int)
    _write_split_identity(
        tmp_path,
        seed=OUTER_SEEDS[0],
        train_index=train_index,
        test_index=test_index,
    )
    monkeypatch.setattr(
        data_preparation,
        "outer_split_indices",
        lambda _dataframe, _seed: (train_index, test_index),
    )

    actual_train, actual_test = verify_outer_split_without_test_labels(
        _creditcard_frame(rows=10, fraud_rows=2),
        tmp_path,
        OUTER_SEEDS[0],
        resolve_experiment_profile("canonical"),
    )

    np.testing.assert_array_equal(actual_train, train_index)
    np.testing.assert_array_equal(actual_test, test_index)


def test_split_identity_verification_rejects_changed_indices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_index = np.arange(226980, dtype=int)
    test_index = np.arange(226980, 283726, dtype=int)
    _write_split_identity(
        tmp_path,
        seed=OUTER_SEEDS[0],
        train_index=train_index,
        test_index=test_index,
    )
    changed_train = train_index.copy()
    changed_train[:2] = changed_train[1::-1]
    monkeypatch.setattr(
        data_preparation,
        "outer_split_indices",
        lambda _dataframe, _seed: (changed_train, test_index),
    )
    with pytest.raises(RuntimeError, match="train split mismatch"):
        verify_outer_split_without_test_labels(
            _creditcard_frame(rows=10, fraud_rows=2),
            tmp_path,
            OUTER_SEEDS[0],
            resolve_experiment_profile("canonical"),
        )

    changed_test = test_index.copy()
    changed_test[:2] = changed_test[1::-1]
    monkeypatch.setattr(
        data_preparation,
        "outer_split_indices",
        lambda _dataframe, _seed: (train_index, changed_test),
    )
    with pytest.raises(RuntimeError, match="test split mismatch"):
        verify_outer_split_without_test_labels(
            _creditcard_frame(rows=10, fraud_rows=2),
            tmp_path,
            OUTER_SEEDS[0],
            resolve_experiment_profile("canonical"),
        )
