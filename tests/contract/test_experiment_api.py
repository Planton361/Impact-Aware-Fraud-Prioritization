from __future__ import annotations

import argparse
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path

import pytest

import fraud_detection.experiment as experiment
from fraud_detection.experiment import (
    ExperimentConfig,
    ExperimentPhase,
    ExperimentPlan,
    ExperimentResult,
    build_experiment_plan,
    config,
    records,
    run_experiment,
)
from fraud_detection.experiment.execution import integrity, pipeline, planning

pytestmark = pytest.mark.contract


def test_evaluation_modules_replace_superseded_root_modules() -> None:
    for module_name in (
        "candidate_" "rerank",
        "ranking",
        "metrics",
    ):
        assert find_spec(f"fraud_detection.{module_name}") is None
    assert (
        find_spec("fraud_detection.experiment.prioritization.composition")
        is not None
    )
    assert find_spec("fraud_detection.experiment.evaluation.metrics") is not None


def test_public_exports_preserve_real_object_identity() -> None:
    assert experiment.__all__ == [
        "ExperimentConfig",
        "ExperimentPhase",
        "ExperimentPlan",
        "ExperimentResult",
        "build_experiment_plan",
        "run_experiment",
    ]
    assert ExperimentConfig is config.ExperimentConfig
    assert ExperimentPhase is config.ExperimentPhase
    assert ExperimentPlan is planning.ExperimentPlan
    assert ExperimentResult is records.ExperimentResult
    assert build_experiment_plan is planning.build_experiment_plan
    assert pipeline.ExperimentResult is records.ExperimentResult
    assert pipeline.run_validation_phase is integrity.run_validation_phase
    assert pipeline.write_failure_record is integrity.write_failure_record
    assert run_experiment is pipeline.run_experiment


def _recording_phases(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
    def record(name: str):
        def phase(
            _args: argparse.Namespace,
            _effective_config: config.EffectiveExperimentConfig,
            *_extra: object,
        ) -> None:
            calls.append(name)

        return phase

    monkeypatch.setattr(pipeline, "_prepare_output_root", record("preflight"))
    monkeypatch.setattr(pipeline, "run_inner_phase", record("inner"))
    monkeypatch.setattr(pipeline, "run_final_phase", record("final"))
    monkeypatch.setattr(pipeline, "run_validation_phase", record("qa"))

    def record_manifest(**_kwargs: object) -> None:
        calls.append("manifest")

    monkeypatch.setattr(
        pipeline,
        "write_completed_run_manifest",
        record_manifest,
    )


def _config(tmp_path: Path, phase: str = "all") -> ExperimentConfig:
    package_root = tmp_path / "src" / "fraud_detection"
    package_root.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("fixture\n", encoding="utf-8")
    (package_root / "__init__.py").write_text("\n", encoding="utf-8")
    return ExperimentConfig(
        data_path=tmp_path / "creditcard.csv",
        output_root=tmp_path / "outputs" / "run",
        phase=phase,  # type: ignore[arg-type]
        repository_root=tmp_path,
    )


def test_all_runs_preflight_then_every_phase_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _recording_phases(monkeypatch, calls)

    result = run_experiment(_config(tmp_path))

    assert calls == ["preflight", "inner", "final", "qa", "manifest"]
    assert result == ExperimentResult(
        output_root=(tmp_path / "outputs" / "run").resolve(),
        requested_phase="all",
        status="COMPLETE",
        completed_phases=("inner", "final", "qa"),
    )


@pytest.mark.parametrize("phase", ["inner", "final", "qa"])
def test_single_phase_runs_only_after_preflight(
    phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _recording_phases(monkeypatch, calls)

    result = run_experiment(_config(tmp_path, phase))

    assert calls == ["preflight", phase]
    assert result.status == "PHASE_COMPLETE"
    assert result.completed_phases == (phase,)


def test_run_experiment_is_silent_without_event_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    _recording_phases(monkeypatch, calls)

    run_experiment(_config(tmp_path, "inner"))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_supplied_sink_receives_structured_status_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    events: list[tuple[str, Mapping[str, object]]] = []
    _recording_phases(monkeypatch, calls)
    base = _config(tmp_path, "qa")
    config = ExperimentConfig(
        data_path=base.data_path,
        output_root=base.output_root,
        phase=base.phase,
        repository_root=base.repository_root,
        event_sink=lambda kind, payload: events.append((kind, payload)),
    )

    run_experiment(config)

    status_events = [
        payload["event"]
        for kind, payload in events
        if kind == "status"
    ]
    assert status_events[0] == "experiment-start"
    assert "preflight-complete" in status_events
    assert "phase-start" in status_events
    assert status_events[-1] == "experiment-exit"
    assert all(isinstance(payload, Mapping) for _, payload in events)


def test_exception_propagates_and_writes_failure_record_after_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ControlledFailure(RuntimeError):
        pass

    _config(tmp_path, "inner")
    output_root = tmp_path / "outputs" / "run"
    recorded: list[tuple[Path, str, BaseException]] = []

    def preflight(
        args: argparse.Namespace,
        _effective_config: config.EffectiveExperimentConfig,
        _repository_root: Path,
    ) -> None:
        Path(args.output_dir, "logs").mkdir(parents=True)

    def fail_inner(
        _args: argparse.Namespace,
        _effective_config: config.EffectiveExperimentConfig,
    ) -> None:
        raise ControlledFailure("controlled phase failure")

    def record_failure(
        *,
        output_root: Path,
        phase: str,
        error: BaseException,
    ) -> None:
        recorded.append((output_root, phase, error))

    monkeypatch.setattr(pipeline, "_prepare_output_root", preflight)
    monkeypatch.setattr(pipeline, "run_inner_phase", fail_inner)
    monkeypatch.setattr(pipeline, "write_failure_record", record_failure)

    with pytest.raises(ControlledFailure) as captured:
        run_experiment(
            ExperimentConfig(
                data_path=tmp_path / "creditcard.csv",
                output_root=output_root,
                phase="inner",
                repository_root=tmp_path,
            )
        )

    assert recorded == [(output_root.resolve(), "inner", captured.value)]

    recorded.clear()
    manifest_output = tmp_path / "outputs" / "manifest-failure"

    def complete_phase(
        _args: argparse.Namespace,
        _effective_config: config.EffectiveExperimentConfig,
    ) -> dict[str, object]:
        return {}

    def fail_manifest(**_kwargs: object) -> None:
        raise ControlledFailure("controlled manifest failure")

    monkeypatch.setattr(pipeline, "run_inner_phase", complete_phase)
    monkeypatch.setattr(pipeline, "run_final_phase", complete_phase)
    monkeypatch.setattr(pipeline, "run_validation_phase", complete_phase)
    monkeypatch.setattr(
        pipeline,
        "write_completed_run_manifest",
        fail_manifest,
    )

    with pytest.raises(ControlledFailure, match="manifest failure") as manifest_error:
        run_experiment(
            ExperimentConfig(
                data_path=tmp_path / "creditcard.csv",
                output_root=manifest_output,
                phase="all",
                repository_root=tmp_path,
            )
        )

    assert recorded == [
        (manifest_output.resolve(), "all", manifest_error.value)
    ]


def test_invalid_phase_is_rejected_before_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def preflight(
        _args: argparse.Namespace,
        _effective_config: config.EffectiveExperimentConfig,
        _repository_root: Path,
    ) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(pipeline, "_prepare_output_root", preflight)

    with pytest.raises(ValueError, match="Unsupported experiment phase"):
        run_experiment(_config(tmp_path, "invalid"))

    assert called is False


def test_run_experiment_requires_experiment_config() -> None:
    with pytest.raises(TypeError, match="ExperimentConfig"):
        run_experiment(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("profile", ("mini-real", "smoke-synthetic"))
def test_run_experiment_resolves_once_and_threads_same_effective_config(
    profile: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective = config.resolve_experiment_profile(profile)
    _config(tmp_path)
    resolved: list[str] = []
    received: list[config.EffectiveExperimentConfig] = []
    roots: list[Path] = []

    def resolve(profile: str) -> config.EffectiveExperimentConfig:
        resolved.append(profile)
        return effective

    def record(
        _args: argparse.Namespace,
        effective_config: config.EffectiveExperimentConfig,
        *_extra: object,
    ) -> None:
        received.append(effective_config)
        if _extra:
            roots.append(_extra[0])  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "resolve_experiment_profile", resolve)
    monkeypatch.setattr(pipeline, "_prepare_output_root", record)
    monkeypatch.setattr(pipeline, "run_inner_phase", record)
    monkeypatch.setattr(pipeline, "run_final_phase", record)
    monkeypatch.setattr(pipeline, "run_validation_phase", record)

    def record_manifest(
        *,
        effective_config: config.EffectiveExperimentConfig,
        **_kwargs: object,
    ) -> None:
        received.append(effective_config)

    monkeypatch.setattr(
        pipeline,
        "write_completed_run_manifest",
        record_manifest,
    )

    run_experiment(
        ExperimentConfig(
            data_path=tmp_path / "creditcard.csv",
            output_root=tmp_path / "outputs" / "run",
            profile=profile,  # type: ignore[arg-type]
        )
    )

    assert resolved == [profile]
    assert received == [effective, effective, effective, effective, effective]
    assert all(value is effective for value in received)
    assert roots == [tmp_path.resolve()]


@pytest.mark.parametrize(
    ("profile", "seed_count", "budget_count"),
    (
        ("canonical", 5, 7),
        ("mini-real", 3, 3),
        ("smoke-synthetic", 1, 3),
    ),
)
def test_execution_namespace_uses_profile_dimensions(
    profile: str,
    seed_count: int,
    budget_count: int,
    tmp_path: Path,
) -> None:
    experiment_config = ExperimentConfig(
        data_path=tmp_path / "creditcard.csv",
        output_root=tmp_path / "outputs" / "run",
        profile=profile,  # type: ignore[arg-type]
    )
    effective = config.resolve_experiment_profile(profile)

    namespace = pipeline._namespace_from_config(
        experiment_config,
        effective,
    )

    assert len(namespace.seeds) == seed_count
    assert len(namespace.target_budgets) == budget_count
    assert namespace.candidate_pool_size == effective.candidate_pool_size
