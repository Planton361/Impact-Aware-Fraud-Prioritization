from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from fraud_detection.experiment import ExperimentConfig, build_experiment_plan
from fraud_detection.experiment.config import (
    EXPECTED_RAW_SHA256,
    resolve_experiment_profile,
)
from fraud_detection.experiment.execution import planning

pytestmark = pytest.mark.contract


def _repository(tmp_path: Path, *, with_data: bool = False) -> Path:
    root = tmp_path / "repository"
    package_root = root / "src" / "fraud_detection"
    package_root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    (package_root / "__init__.py").write_text("\n", encoding="utf-8")
    environment = root / "environment"
    environment.mkdir(parents=True)
    (environment / "final_experiment_requirements.txt").write_text(
        "",
        encoding="utf-8",
    )
    (environment / "bootstrap_requirements.txt").write_text(
        "",
        encoding="utf-8",
    )
    (root / "outputs").mkdir()
    if with_data:
        data_root = root / "data"
        data_root.mkdir()
        (data_root / "creditcard.csv").write_text(
            "synthetic fixture\n",
            encoding="utf-8",
        )
    return root


def _error_codes(plan: planning.ExperimentPlan) -> set[str]:
    return {error.code for error in plan.prerequisite_errors}


def test_missing_repository_root_returns_prerequisite_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(planning, "find_repository_root", lambda: None)

    plan = build_experiment_plan(
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("outputs/run"),
    )

    assert _error_codes(plan) == {"FD-ROOT-NOT-FOUND"}
    assert plan.data_status == "unavailable"
    assert plan.config.phase == "all"


def test_missing_dataset_is_reported_without_loading_it(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)

    plan = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("outputs/run"),
    )

    assert "FD-DATA-MISSING" in _error_codes(plan)
    assert plan.data_status == "missing"

    invalid_root_plan = build_experiment_plan(
        repository_root=root / "nested",
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("outputs/run-invalid-root"),
    )
    assert _error_codes(invalid_root_plan) == {"FD-ROOT-NOT-FOUND"}


def test_valid_synthetic_prerequisites_build_complete_run_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path, with_data=True)
    monkeypatch.setattr(
        planning,
        "_sha256_file",
        lambda _path: EXPECTED_RAW_SHA256,
    )

    plan = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("outputs/run"),
    )

    assert plan.prerequisite_errors == ()
    assert plan.data_status == "verified"
    assert plan.actual_data_sha256 == EXPECTED_RAW_SHA256
    assert plan.repository_root == root.resolve()
    assert plan.config.repository_root == root.resolve()
    assert plan.config == ExperimentConfig(
        data_path=plan.config.data_path,
        output_root=plan.config.output_root,
        phase=plan.config.phase,
        repository_root=root / "elsewhere",
    )
    assert "repository_root" not in repr(plan.config)
    assert "repository_root" not in plan.as_dict()

    def unexpected_data_access(_path: Path) -> str:
        raise AssertionError("synthetic planning must not access project data")

    monkeypatch.setattr(planning, "_sha256_file", unexpected_data_access)
    smoke_plan = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("generated/runs/smoke-synthetic"),
        profile="smoke-synthetic",
    )

    assert smoke_plan.prerequisite_errors == ()
    assert smoke_plan.data_status == "synthetic"
    assert smoke_plan.actual_data_sha256 is None
    assert smoke_plan.as_dict()["data"]["path"] is None
    assert smoke_plan.as_dict()["evidence_notice"] == [
        "Deterministic synthetic engineering data.",
        "Not thesis evidence.",
        "Not comparable with canonical empirical results.",
    ]


def test_unsafe_output_path_is_rejected(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)

    plan = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("generated/run"),
    )

    assert "FD-OUTPUT-UNSAFE" in _error_codes(plan)


def test_existing_output_path_is_rejected(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    (root / "outputs" / "existing").mkdir()

    plan = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("outputs/existing"),
    )

    assert "FD-OUTPUT-EXISTS" in _error_codes(plan)


def test_planning_surface_rejects_removed_forwarded_arguments(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)

    assert "forwarded_arguments" not in inspect.signature(
        build_experiment_plan
    ).parameters
    assert "forwarded_arguments" not in {
        field.name for field in fields(planning.ExperimentPlan)
    }

    canonical = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("outputs/run"),
        inspect_data=False,
    )
    assert canonical.profile == "canonical"
    assert canonical.data_status == "not_inspected"
    assert "forwarded_arguments" not in canonical.as_dict()

    smoke = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("generated/runs/smoke"),
        profile="smoke-synthetic",
        inspect_data=False,
    )
    assert smoke.profile == "smoke-synthetic"
    assert smoke.data_status == "synthetic"
    assert "forwarded_arguments" not in smoke.as_dict()

    with pytest.raises(TypeError):
        build_experiment_plan(
            repository_root=root,
            data_path=Path("data/creditcard.csv"),
            output_dir=Path("outputs/forwarded"),
            forwarded_arguments=("--candidate-pool-size", "999"),
        )


@pytest.mark.parametrize(
    "requirements",
    (
        "numpy>=2.5.1\n",
        "numpy==2.5.1\nNumPy==2.5.1\n",
    ),
)
def test_invalid_pinned_requirements_are_rejected(
    tmp_path: Path,
    requirements: str,
) -> None:
    root = _repository(tmp_path)
    (
        root / "environment" / "final_experiment_requirements.txt"
    ).write_text(requirements, encoding="utf-8")

    plan = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("outputs/run"),
    )

    assert "FD-ENV-REQUIREMENTS" in _error_codes(plan)


def test_successful_plan_contains_all_phase_experiment_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path, with_data=True)
    monkeypatch.setattr(
        planning,
        "_sha256_file",
        lambda _path: EXPECTED_RAW_SHA256,
    )

    plan = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("outputs/run"),
    )

    assert plan.config == ExperimentConfig(
        data_path=(root / "data" / "creditcard.csv").resolve(),
        output_root=(root / "outputs" / "run").resolve(),
        phase="all",
    )
    assert plan.repository_root == root.resolve()
    assert plan.config.repository_root == root.resolve()
    assert plan.as_dict()["execution"] == (
        "fraud_detection.experiment.run_experiment"
    )
    assert plan.profile == "canonical"
    assert plan.effective_config == resolve_experiment_profile("canonical")
    assert plan.as_dict()["profile"] == "canonical"
    assert plan.as_dict()["effective_config"] == (
        plan.effective_config.as_dict()
    )
    assert plan.as_dict()["task_counts"] == {
        "inner_bce_fits": 90,
        "inner_ranker_fits": 210,
        "selection_freeze_configurations": 35,
        "final_bce_fits": 30,
        "final_ranker_fits": 70,
    }
    assert plan.as_dict()["known_task_total"] == 435


def test_planning_does_not_create_the_output_or_process_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path, with_data=True)
    hashed: list[Path] = []

    def synthetic_hash(path: Path) -> str:
        hashed.append(path)
        return EXPECTED_RAW_SHA256

    monkeypatch.setattr(planning, "_sha256_file", synthetic_hash)
    dry_output_path = root / "generated" / "runs" / "dry-run"
    dry_plan = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("generated/runs/dry-run"),
        profile="mini-real",
        inspect_data=False,
    )

    assert dry_plan.prerequisite_errors == ()
    assert dry_plan.data_status == "not_inspected"
    assert dry_plan.actual_data_sha256 is None
    assert hashed == []
    assert not dry_output_path.exists()
    dry_payload = dry_plan.as_dict()
    assert dry_payload["effective_config"]["bce_oof_folds"] == 5
    assert dry_payload["effective_config"]["inner_folds"] == 3
    assert dry_payload["effective_config"]["candidate_pool_size"] == 1000
    assert dry_payload["execution_notes"] == [
        "Runtime depends on the local environment; no duration is guaranteed.",
        "Individual model fits are not resumable.",
    ]

    output_path = root / "generated" / "runs" / "new-run"

    plan = build_experiment_plan(
        repository_root=root,
        data_path=Path("data/creditcard.csv"),
        output_dir=Path("generated/runs/new-run"),
        profile="mini-real",
    )

    assert plan.prerequisite_errors == ()
    assert plan.profile == "mini-real"
    assert plan.effective_config == resolve_experiment_profile("mini-real")
    assert hashed == [(root / "data" / "creditcard.csv").resolve()]
    assert not output_path.exists()
