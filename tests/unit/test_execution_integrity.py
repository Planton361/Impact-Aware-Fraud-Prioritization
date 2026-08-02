"""Focused contracts for completed-run integrity validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pytest

from fraud_detection.experiment.config import resolve_experiment_profile
from fraud_detection.experiment.execution.integrity import (
    _validate_selected_gain_numeric_contract,
)
from fraud_detection.experiment.execution.manifest import (
    _build_run_manifest,
    _produced_artifacts,
    _validate_run_manifest,
    write_completed_run_manifest,
)

pytestmark = pytest.mark.unit


def _selected_gain_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outer_seed": [42, 42, 42],
            "target_budget": [20, 50, 100],
            "best_iteration_fold_1": [8.0, 10.0, 12.0],
            "best_iteration_fold_2": [9.0, 11.0, 13.0],
            "best_iteration_fold_3": [7.0, 9.0, 11.0],
            "final_n_estimators": [9.0, 10.0, 12.0],
        }
    )


def _synthetic_metadata() -> dict[str, object]:
    return {
        "rows_before_deduplication": 5000,
        "rows_removed_as_duplicate_followups": 0,
        "rows_after_deduplication": 5000,
        "class_0_before": 4900,
        "class_0_after": 4900,
        "class_1_before": 100,
        "class_1_after": 100,
        "deduplicated_dataframe_sha256": "a" * 64,
        "synthetic_generator_schema": "fraud_detection.synthetic_engineering.v1",
        "synthetic_generation_seed": 314159,
        "synthetic_requested_row_count": 5000,
    }


def _real_metadata() -> dict[str, object]:
    return {
        "rows_before_deduplication": 284807,
        "rows_removed_as_duplicate_followups": 1081,
        "rows_after_deduplication": 283726,
        "class_0_before": 284315,
        "class_0_after": 283253,
        "class_1_before": 492,
        "class_1_after": 473,
        "deduplicated_dataframe_sha256": "b" * 64,
    }


def test_smoke_selected_gain_and_manifest_contracts(tmp_path: Path) -> None:
    frame = _selected_gain_frame()
    frame["best_iteration_fold_3"] = np.nan
    smoke = resolve_experiment_profile("smoke-synthetic")

    _validate_selected_gain_numeric_contract(
        frame,
        smoke,
    )

    output_root = tmp_path / "run"
    (output_root / "comparison").mkdir(parents=True)
    (output_root / "preflight").mkdir()
    (output_root / "comparison" / "checksums.sha256").write_text(
        "inventory\n", encoding="ascii"
    )
    (output_root / "preflight" / "preflight_validation.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (output_root / ".cache").write_text("ignored", encoding="utf-8")
    (output_root / "comparison" / "write.lock").write_text(
        "ignored", encoding="utf-8"
    )
    artifacts = _produced_artifacts(output_root)

    manifest_path = output_root / "RUN_MANIFEST.json"
    temporary_manifest_path = output_root / ".RUN_MANIFEST.json.tmp"
    assert not manifest_path.exists()
    written = write_completed_run_manifest(
        output_root=output_root,
        effective_config=smoke,
        data_metadata=_synthetic_metadata(),
    )
    first_bytes = manifest_path.read_bytes()
    loaded = json.loads(first_bytes.decode("utf-8"))
    assert loaded == written
    assert loaded["status"] == "COMPLETE"
    assert loaded["profile"] == "smoke-synthetic"
    assert loaded["evidence_classification"] == "non-evidentiary"
    assert loaded["effective_config"] == smoke.as_dict()
    assert loaded["produced_artifacts"] == artifacts
    assert not temporary_manifest_path.exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_completed_run_manifest(
            output_root=output_root,
            effective_config=smoke,
            data_metadata=_synthetic_metadata(),
        )
    assert manifest_path.read_bytes() == first_bytes

    manifest = _build_run_manifest(
        effective_config=smoke,
        data_metadata=_synthetic_metadata(),
        produced_artifacts=artifacts,
    )
    repeated = _build_run_manifest(
        effective_config=smoke,
        data_metadata=_synthetic_metadata(),
        produced_artifacts=copy.deepcopy(artifacts),
    )
    artifact_paths = [entry["path"] for entry in artifacts]
    _validate_run_manifest(
        manifest,
        effective_config=smoke,
        available_artifact_paths=frozenset(artifact_paths),
    )

    assert manifest == repeated
    assert json.dumps(manifest, allow_nan=False) == json.dumps(
        repeated, allow_nan=False
    )
    assert list(manifest) == [
        "schema",
        "status",
        "profile",
        "evidence_classification",
        "completed_phases",
        "effective_config",
        "data_summary",
        "produced_artifacts",
    ]
    assert manifest["schema"] == "fraud_detection.run_manifest.v1"
    assert manifest["status"] == "COMPLETE"
    assert manifest["profile"] == "smoke-synthetic"
    assert manifest["evidence_classification"] == "non-evidentiary"
    assert manifest["effective_config"] == smoke.as_dict()
    assert manifest["effective_config"] == {
        "profile_name": "smoke-synthetic",
        "evidence_classification": "non-evidentiary",
        "data_source_kind": "synthetic",
        "synthetic_row_target": 5000,
        "synthetic_generation_seed": 314159,
        "seeds": [42],
        "target_budgets": [20, 50, 100],
        "primary_budgets": [20, 50, 100],
        "supplementary_budgets": [],
        "bce_oof_folds": 2,
        "inner_folds": 2,
        "candidate_pool_size": 200,
        "enabled_gain_profiles": ["exponential", "linear"],
        "ranker_max_estimators": 30,
        "ranker_early_stopping_rounds": 5,
    }
    data_summary = manifest["data_summary"]
    assert isinstance(data_summary, dict)
    assert data_summary["source_counts"] == {
        "kind": "generated",
        "rows": 5000,
        "fraud": 100,
        "legitimate": 4900,
    }
    assert data_summary["deduplicated_counts"] == {
        "rows": 5000,
        "fraud": 100,
        "legitimate": 4900,
    }
    assert data_summary["removed_duplicate_count"] == 0
    assert data_summary["synthetic"] == {
        "generator_schema": "fraud_detection.synthetic_engineering.v1",
        "generation_seed": 314159,
        "requested_row_count": 5000,
    }
    assert artifact_paths == sorted(set(artifact_paths))
    assert artifact_paths == [
        "comparison/checksums.sha256",
        "preflight/preflight_validation.json",
    ]
    assert all(not PurePosixPath(path).is_absolute() for path in artifact_paths)
    assert all(".." not in PurePosixPath(path).parts for path in artifact_paths)
    assert all(set(entry) == {"path", "group", "format"} for entry in artifacts)
    serialized_keys = {
        str(key)
        for mapping in (manifest, manifest["effective_config"], data_summary)
        if isinstance(mapping, dict)
        for key in mapping
    }
    assert serialized_keys.isdisjoint(
        {
            "git_sha",
            "branch",
            "tag",
            "dirty_status",
            "source_file_hash",
            "module_hash",
            "source_tree_inventory",
            "historical_commit",
            "previous_artifact_hash",
            "image_byte_identity",
        }
    )


def test_smoke_selected_gain_rejects_invalid_fold_contracts() -> None:
    smoke = resolve_experiment_profile("smoke-synthetic")

    active_missing = _selected_gain_frame()
    active_missing["best_iteration_fold_2"] = np.nan
    active_missing["best_iteration_fold_3"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite active fold values.*fold_2"):
        _validate_selected_gain_numeric_contract(active_missing, smoke)

    inactive_populated = _selected_gain_frame()
    with pytest.raises(
        RuntimeError, match="unexpectedly populated inactive fold column.*fold_3"
    ):
        _validate_selected_gain_numeric_contract(inactive_populated, smoke)

    missing_column = _selected_gain_frame().drop(columns="best_iteration_fold_2")
    with pytest.raises(RuntimeError, match="missing active fold column.*fold_2"):
        _validate_selected_gain_numeric_contract(missing_column, smoke)

    invalid_iteration = _selected_gain_frame()
    invalid_iteration["best_iteration_fold_1"] = 0.0
    invalid_iteration["best_iteration_fold_3"] = np.nan
    with pytest.raises(RuntimeError, match="invalid active fold iterations.*fold_1"):
        _validate_selected_gain_numeric_contract(invalid_iteration, smoke)

    malformed_numbering = _selected_gain_frame().rename(
        columns={"best_iteration_fold_3": "best_iteration_fold_4"}
    )
    with pytest.raises(RuntimeError, match="malformed fold-column numbering"):
        _validate_selected_gain_numeric_contract(malformed_numbering, smoke)

    invalid_other_numeric = _selected_gain_frame()
    invalid_other_numeric["best_iteration_fold_3"] = np.nan
    invalid_other_numeric["final_n_estimators"] = np.nan
    with pytest.raises(RuntimeError, match="contains non-finite numeric values"):
        _validate_selected_gain_numeric_contract(invalid_other_numeric, smoke)

    artifacts = [
        {"path": "comparison/checksums.sha256", "group": "integrity", "format": "sha256"},
        {"path": "preflight/preflight_validation.json", "group": "preflight", "format": "json"},
    ]
    expected_paths = frozenset(entry["path"] for entry in artifacts)
    base_manifest = _build_run_manifest(
        effective_config=smoke,
        data_metadata=_synthetic_metadata(),
        produced_artifacts=copy.deepcopy(artifacts),
    )

    absolute = copy.deepcopy(base_manifest)
    absolute["produced_artifacts"][0]["path"] = "C:/absolute/result.csv"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="unsafe artifact path"):
        _validate_run_manifest(
            absolute,
            effective_config=smoke,
            available_artifact_paths=expected_paths,
        )

    traversal = copy.deepcopy(base_manifest)
    traversal["produced_artifacts"][0]["path"] = "comparison/../escape.csv"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="unsafe artifact path"):
        _validate_run_manifest(
            traversal,
            effective_config=smoke,
            available_artifact_paths=expected_paths,
        )

    duplicate = copy.deepcopy(base_manifest)
    duplicate["produced_artifacts"].append(  # type: ignore[union-attr]
        copy.deepcopy(duplicate["produced_artifacts"][0])  # type: ignore[index]
    )
    with pytest.raises(RuntimeError, match="sorted and unique"):
        _validate_run_manifest(
            duplicate,
            effective_config=smoke,
            available_artifact_paths=expected_paths,
        )

    missing = copy.deepcopy(base_manifest)
    missing["produced_artifacts"] = missing["produced_artifacts"][:-1]  # type: ignore[index]
    with pytest.raises(RuntimeError, match="does not match produced files"):
        _validate_run_manifest(
            missing,
            effective_config=smoke,
            available_artifact_paths=expected_paths,
        )

    for non_finite in (float("nan"), float("inf")):
        invalid_number = copy.deepcopy(base_manifest)
        invalid_number["data_summary"]["removed_duplicate_count"] = non_finite  # type: ignore[index]
        with pytest.raises(RuntimeError, match="non-finite numbers"):
            _validate_run_manifest(
                invalid_number,
                effective_config=smoke,
                available_artifact_paths=expected_paths,
            )


def test_canonical_selected_gain_requires_all_retained_folds() -> None:
    canonical = resolve_experiment_profile("canonical")
    frame = _selected_gain_frame()

    _validate_selected_gain_numeric_contract(frame, canonical)

    frame["best_iteration_fold_3"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite active fold values.*fold_3"):
        _validate_selected_gain_numeric_contract(frame, canonical)

    artifacts = [
        {"path": "comparison/checksums.sha256", "group": "integrity", "format": "sha256"},
    ]
    artifact_paths = frozenset({"comparison/checksums.sha256"})
    canonical_manifest = _build_run_manifest(
        effective_config=canonical,
        data_metadata=_real_metadata(),
        produced_artifacts=copy.deepcopy(artifacts),
    )
    _validate_run_manifest(
        canonical_manifest,
        effective_config=canonical,
        available_artifact_paths=artifact_paths,
    )
    assert canonical_manifest["effective_config"] == canonical.as_dict()
    assert canonical_manifest["effective_config"] == {
        "profile_name": "canonical",
        "evidence_classification": "thesis-evidentiary",
        "data_source_kind": "real",
        "synthetic_row_target": None,
        "synthetic_generation_seed": None,
        "seeds": [42, 7, 13, 123, 202],
        "target_budgets": [5, 10, 20, 50, 100, 200, 500],
        "primary_budgets": [20, 50, 100],
        "supplementary_budgets": [5, 10, 200, 500],
        "bce_oof_folds": 5,
        "inner_folds": 3,
        "candidate_pool_size": 1000,
        "enabled_gain_profiles": ["exponential", "linear"],
        "ranker_max_estimators": 500,
        "ranker_early_stopping_rounds": 50,
    }
    assert "synthetic" not in canonical_manifest["data_summary"]

    partial = copy.deepcopy(canonical_manifest)
    partial["completed_phases"] = ["inner_selection"]
    with pytest.raises(RuntimeError, match="completed phases are incomplete"):
        _validate_run_manifest(
            partial,
            effective_config=canonical,
            available_artifact_paths=artifact_paths,
        )

    mini = resolve_experiment_profile("mini-real")
    mini_manifest = _build_run_manifest(
        effective_config=mini,
        data_metadata=_real_metadata(),
        produced_artifacts=copy.deepcopy(artifacts),
    )
    _validate_run_manifest(
        mini_manifest,
        effective_config=mini,
        available_artifact_paths=artifact_paths,
    )
    assert (
        mini_manifest["evidence_classification"]
        == mini.evidence_classification
    )
    assert "not thesis evidence" in mini.evidence_classification
    assert "synthetic" not in mini_manifest["data_summary"]
