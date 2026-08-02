from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from fraud_detection.errors import ProductError
from fraud_detection.experiment.config import resolve_experiment_profile
from fraud_detection.experiment.preparation import data as data_preparation
from fraud_detection.experiment.preparation.data import (
    dataframe_content_sha256,
    load_experiment_data,
    outer_split_indices,
    required_columns,
)
from fraud_detection.experiment.preparation.synthetic import (
    _validate_synthetic_frame,
    generate_synthetic_data,
)
from fraud_detection.experiment.prioritization.inputs import (
    build_amount_gain_relevance_labels,
    build_candidate_pool,
    validate_candidate_pool,
)

pytestmark = pytest.mark.unit


def _smoke_data() -> tuple[pd.DataFrame, dict[str, object]]:
    return generate_synthetic_data(
        resolve_experiment_profile("smoke-synthetic")
    )


def test_synthetic_frame_preserves_exact_canonical_input_contract() -> None:
    smoke = resolve_experiment_profile("smoke-synthetic")
    dataframe, metadata = _smoke_data()

    assert dataframe.columns.tolist() == required_columns()
    assert len(dataframe) == smoke.synthetic_row_target == 5_000
    assert dataframe.isna().any().any() is np.False_
    assert np.isfinite(dataframe.to_numpy(dtype=float)).all()
    assert pd.api.types.is_integer_dtype(dataframe["Class"])
    assert set(dataframe["Class"].unique()) == {0, 1}
    assert int(dataframe["Class"].sum()) < len(dataframe) / 2
    assert (dataframe[["Time", "Amount"]] >= 0.0).all().all()
    assert not dataframe.duplicated().any()
    assert dataframe["Amount"].quantile(0.95) > (
        3.0 * dataframe["Amount"].median()
    )
    fraud_v1 = dataframe.loc[dataframe["Class"] == 1, "V1"]
    legitimate_v1 = dataframe.loc[dataframe["Class"] == 0, "V1"]
    assert fraud_v1.mean() > legitimate_v1.mean() + 0.5
    assert fraud_v1.min() < legitimate_v1.max()
    assert metadata["synthetic_generation_seed"] == (
        smoke.synthetic_generation_seed
    )


def test_synthetic_generation_and_identity_are_deterministic() -> None:
    first, _first_metadata = _smoke_data()
    second, _second_metadata = _smoke_data()

    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert dataframe_content_sha256(first) == dataframe_content_sha256(second)


def test_different_synthetic_generation_seeds_change_identity() -> None:
    smoke = resolve_experiment_profile("smoke-synthetic")
    first, _metadata = generate_synthetic_data(smoke)
    changed, _changed_metadata = generate_synthetic_data(
        replace(
            smoke,
            synthetic_generation_seed=smoke.synthetic_generation_seed + 1,
        )
    )

    assert dataframe_content_sha256(first) != dataframe_content_sha256(changed)


def test_synthetic_classes_support_all_configured_stratified_splits() -> None:
    smoke = resolve_experiment_profile("smoke-synthetic")
    dataframe, _metadata = _smoke_data()
    train_index, test_index = outer_split_indices(dataframe, smoke.seeds[0])

    for index in (train_index, test_index):
        assert set(dataframe.loc[index, "Class"].unique()) == {0, 1}
    outer_train = dataframe.loc[train_index]
    splitter = StratifiedKFold(
        n_splits=smoke.inner_folds,
        shuffle=True,
        random_state=100_000 + smoke.seeds[0],
    )
    for inner_train_position, validation_position in splitter.split(
        outer_train,
        outer_train["Class"],
    ):
        inner_train = outer_train.iloc[inner_train_position]
        validation = outer_train.iloc[validation_position]
        assert set(inner_train["Class"].unique()) == {0, 1}
        assert set(validation["Class"].unique()) == {0, 1}
        assert int(inner_train["Class"].value_counts().min()) >= (
            smoke.bce_oof_folds
        )


def test_synthetic_amounts_construct_all_relevance_levels() -> None:
    dataframe, _metadata = _smoke_data()
    labels, thresholds = build_amount_gain_relevance_labels(
        dataframe["Class"],
        dataframe["Amount"],
    )

    assert set(labels[dataframe["Class"].to_numpy(dtype=int) == 1]) == {
        1,
        2,
        3,
        4,
    }
    assert np.all(np.diff(thresholds) > 0.0)


def test_synthetic_frame_supports_the_configured_candidate_pool() -> None:
    smoke = resolve_experiment_profile("smoke-synthetic")
    dataframe, _metadata = _smoke_data()
    _train_index, test_index = outer_split_indices(dataframe, smoke.seeds[0])
    outer_test = dataframe.loc[test_index]
    probability_proxy = 1.0 / (1.0 + np.exp(-outer_test["V1"].to_numpy()))

    pool = build_candidate_pool(
        probability_proxy,
        outer_test.index,
        candidate_pool_size=smoke.candidate_pool_size,
    )

    validate_candidate_pool(
        pool,
        expected_pool_size=smoke.candidate_pool_size,
    )
    assert int(pool["candidate_flag"].sum()) == 200


def test_synthetic_loader_ignores_data_path_and_records_evidence_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = resolve_experiment_profile("smoke-synthetic")

    def unexpected_path_access(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Synthetic preparation accessed the data path.")

    monkeypatch.setattr(data_preparation, "file_sha256", unexpected_path_access)
    monkeypatch.setattr(
        data_preparation,
        "load_creditcard_csv",
        unexpected_path_access,
    )
    missing_path = tmp_path / "does-not-exist.csv"

    dataframe, metadata = load_experiment_data(missing_path, smoke)

    assert not missing_path.exists()
    assert len(dataframe) == smoke.synthetic_row_target
    assert metadata["data_source_kind"] == "synthetic"
    assert metadata["synthetic_requested_row_count"] == 5_000
    assert metadata["synthetic_generated_row_count"] == 5_000
    assert metadata["synthetic_generated_fraud_count"] == 100
    assert metadata["synthetic_generated_legitimate_count"] == 4_900
    assert metadata["synthetic_data_identity"] == dataframe_content_sha256(
        dataframe
    )
    assert metadata["evidence_classification"] == "non-evidentiary"
    evidence_boundary = str(metadata["evidence_boundary"])
    assert "synthetic engineering data" in evidence_boundary.lower()
    assert "not thesis evidence" in evidence_boundary.lower()
    assert "not comparable" in evidence_boundary.lower()


def test_smoke_preflight_uses_in_memory_data_without_a_real_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = resolve_experiment_profile("smoke-synthetic")
    missing_path = tmp_path / "missing" / "creditcard.csv"
    output_root = tmp_path / "outputs" / "synthetic-preflight"

    def unexpected_path_access(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Synthetic preflight accessed the configured data path.")

    monkeypatch.setattr(data_preparation, "file_sha256", unexpected_path_access)
    monkeypatch.setattr(
        data_preparation,
        "load_creditcard_csv",
        unexpected_path_access,
    )

    dataframe, metadata = data_preparation._initialize_preflight(
        repository_root=tmp_path,
        output_root=output_root,
        data_path=missing_path,
        effective_config=smoke,
    )

    assert not missing_path.exists()
    assert len(dataframe) == smoke.synthetic_row_target
    assert metadata["synthetic_data_identity"] == dataframe_content_sha256(
        dataframe
    )
    preflight = json.loads(
        (output_root / "preflight" / "preflight_validation.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert preflight["status"] == "PASS"
    assert preflight["data"]["data_source_kind"] == "synthetic"
    assert preflight["data"]["synthetic_generated_row_count"] == 5_000
    assert "raw_path" not in preflight["data"]
    assert all(preflight["stop_conditions_checked"].values())


def test_real_profiles_keep_the_resolved_csv_loading_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[Path] = []
    expected = pd.DataFrame({"sentinel": [1]})

    def fake_real_loader(path: Path):
        received.append(path)
        return expected, {"source": "real"}

    def unexpected_generator(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("A real profile entered synthetic generation.")

    monkeypatch.setattr(
        data_preparation,
        "load_deduplicated_data",
        fake_real_loader,
    )
    monkeypatch.setattr(
        data_preparation,
        "generate_synthetic_data",
        unexpected_generator,
    )

    for profile in ("canonical", "mini-real"):
        data_path = tmp_path / profile / "creditcard.csv"
        dataframe, metadata = load_experiment_data(
            data_path,
            resolve_experiment_profile(profile),
        )

        assert dataframe is expected
        assert metadata == {"source": "real"}
        assert received[-1] == data_path.resolve()


def test_invalid_synthetic_generation_configuration_fails_precisely(
) -> None:
    cases = (
        ({"synthetic_row_target": None}, "FD-SYNTHETIC-ROW-TARGET"),
        ({"synthetic_row_target": 500}, "FD-SYNTHETIC-ROW-TARGET"),
        ({"synthetic_generation_seed": None}, "FD-SYNTHETIC-SEED"),
    )
    for changes, expected_code in cases:
        effective = replace(
            resolve_experiment_profile("smoke-synthetic"),
            **changes,
        )

        with pytest.raises(ProductError) as captured:
            generate_synthetic_data(effective)

        assert captured.value.code == expected_code


def test_synthetic_validation_rejects_invalid_generated_contracts() -> None:
    cases = (
        ("schema", "FD-SYNTHETIC-SCHEMA"),
        ("non_finite", "FD-SYNTHETIC-VALUES"),
        ("negative", "FD-SYNTHETIC-VALUES"),
        ("class", "FD-SYNTHETIC-CLASS"),
        ("class_support", "FD-SYNTHETIC-CLASS-SUPPORT"),
        ("duplicate", "FD-SYNTHETIC-DUPLICATES"),
    )
    smoke = resolve_experiment_profile("smoke-synthetic")
    for mutation, expected_code in cases:
        dataframe, _metadata = _smoke_data()
        if mutation == "schema":
            dataframe = dataframe.drop(columns="V28")
        elif mutation == "non_finite":
            dataframe.loc[0, "V1"] = np.inf
        elif mutation == "negative":
            dataframe.loc[0, "Amount"] = -1.0
        elif mutation == "class":
            dataframe.loc[0, "Class"] = 2
        elif mutation == "class_support":
            dataframe["Class"] = 0
            dataframe.loc[:4, "Class"] = 1
        else:
            dataframe.iloc[-1] = dataframe.iloc[0]

        with pytest.raises(ProductError) as captured:
            _validate_synthetic_frame(dataframe, smoke)

        assert captured.value.code == expected_code
