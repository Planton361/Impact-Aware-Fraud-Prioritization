import hashlib
import json
import os
import subprocess
import sys
from importlib import import_module, util
from io import StringIO
from pathlib import Path

import pytest
import typer
from rich.console import Console
from typer._click._compat import strip_ansi

from fraud_detection.cli import main
from fraud_detection.cli.output import (
    CommandReport,
    LongCommandProjection,
    ShellState,
    render_error,
    render_report,
)
from fraud_detection.errors import ProductError
from fraud_detection.experiment import (
    ExperimentConfig,
    ExperimentPlan,
    ExperimentResult,
    run_experiment,
)
from fraud_detection.presentation import (
    METHOD_ORDER,
    PresentationConfig,
    PresentationError,
    PresentationResult,
    PresentationStepResult,
)
from fraud_detection.setup import (
    DiagnosticFinding,
    DiagnosticReport,
    SetupFailure,
    SetupResult,
)
from fraud_detection.setup import run_check as public_run_check
from fraud_detection.setup import run_doctor as public_run_doctor
from fraud_detection.setup import run_setup as public_run_setup

pytestmark = pytest.mark.contract
cli_app = import_module("fraud_detection.cli.app")
cli_experiment = import_module("fraud_detection.cli.experiment")
cli_presentation = import_module("fraud_detection.cli.presentation")


class _TTYBuffer(StringIO):
    def isatty(self) -> bool:
        return True


def _inspection_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspection_inventory(root: Path, paths: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _inspection_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]


def _write_inspection_run(root: Path, profile: str) -> Path:
    config = import_module("fraud_detection.experiment.config")
    effective = config.resolve_experiment_profile(profile)
    relative_paths = {
        "comparison/checksums.sha256",
        "comparison/final_qa.json",
        "diagnostics/global_metrics_seedwise.csv",
        "figure_data/all_budget_matched_results.csv",
        "final_outer_run/final_outer_manifest.json",
        "inner_validation/inner_validation_manifest.json",
        "preflight/preflight_validation.json",
        "selection_freeze/selection_manifest.json",
        *(
            f"final_outer_run/seed_{seed}/ranking_dump.csv"
            for seed in effective.seeds
        ),
    }
    for relative in sorted(relative_paths - {"comparison/checksums.sha256"}):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")
    phase_manifests = {
        "preflight/preflight_validation.json": {
            "schema": "ranker_gain_validation.preflight.v2",
            "status": "PASS",
        },
        "inner_validation/inner_validation_manifest.json": {"status": "PASS"},
        "selection_freeze/selection_manifest.json": {
            "schema": "ranker_gain_validation.selection_manifest.v1",
            "outer_test_selection_locked": True,
            "outer_test_labels_used_for_selection": False,
        },
        "final_outer_run/final_outer_manifest.json": {
            "schema": "ranker_gain_validation.final_outer.v1",
            "status": "PASS",
        },
        "comparison/final_qa.json": {"status": "PASS"},
    }
    for relative, value in phase_manifests.items():
        (root / relative).write_text(json.dumps(value), encoding="utf-8")
    checksum_path = root / "comparison" / "checksums.sha256"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text(
        "".join(
            f"{_inspection_sha256(root / relative)}  {relative}\n"
            for relative in sorted(
                relative_paths - {"comparison/checksums.sha256"}
            )
        ),
        encoding="utf-8",
    )
    if effective.data_source_kind == "synthetic":
        source_rows, source_fraud, source_legitimate = 5_000, 100, 4_900
        rows, fraud, legitimate, removed = 5_000, 100, 4_900, 0
        identity = "a" * 64
    else:
        source_rows, source_fraud, source_legitimate = 284_807, 492, 284_315
        rows, fraud, legitimate, removed = 283_726, 473, 283_253, 1_081
        identity = config.EXPECTED_DEDUPLICATED_SHA256
    data_summary: dict[str, object] = {
        "source_kind": effective.data_source_kind,
        "data_identity": identity,
        "source_counts": {
            "kind": (
                "generated" if effective.data_source_kind == "synthetic" else "raw"
            ),
            "rows": source_rows,
            "fraud": source_fraud,
            "legitimate": source_legitimate,
        },
        "deduplicated_counts": {
            "rows": rows,
            "fraud": fraud,
            "legitimate": legitimate,
        },
        "removed_duplicate_count": removed,
    }
    if effective.data_source_kind == "synthetic":
        data_summary["synthetic"] = {
            "generator_schema": "fraud_detection.synthetic_engineering.v1",
            "generation_seed": effective.synthetic_generation_seed,
            "requested_row_count": effective.synthetic_row_target,
        }
    groups = {
        "comparison": "integrity",
        "diagnostics": "aggregation",
        "figure_data": "aggregation",
        "final_outer_run": "final_outer",
        "inner_validation": "inner_selection",
        "preflight": "preflight",
        "selection_freeze": "selection_freeze",
    }
    manifest = {
        "schema": "fraud_detection.run_manifest.v1",
        "status": "COMPLETE",
        "profile": profile,
        "evidence_classification": effective.evidence_classification,
        "completed_phases": [
            "preflight",
            "inner_selection",
            "selection_freeze",
            "final_outer",
            "aggregation",
            "qa",
        ],
        "effective_config": effective.as_dict(),
        "data_summary": data_summary,
        "produced_artifacts": [
            {
                "path": relative,
                "group": (
                    "qa"
                    if relative == "comparison/final_qa.json"
                    else groups[relative.split("/", maxsplit=1)[0]]
                ),
                "format": Path(relative).suffix.removeprefix("."),
            }
            for relative in sorted(relative_paths)
        ],
    }
    (root / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return root


def _write_inspection_partial(root: Path) -> Path:
    path = root / "preflight" / "preflight_validation.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "ranker_gain_validation.preflight.v2",
                "status": "PASS",
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_inspection_presentation(root: Path, profile: str) -> Path:
    config = import_module("fraud_detection.experiment.config")
    catalog = import_module("fraud_detection.presentation.catalog")
    derivations = import_module(
        "fraud_detection.presentation.preparation.derivations"
    )
    effective = config.resolve_experiment_profile(profile)
    role = "canonical" if profile == "canonical" else "engineering"
    selected = (
        catalog.CANONICAL_ARTIFACT_IDS
        if role == "canonical"
        else catalog.ENGINEERING_ARTIFACT_IDS
    )
    selection = catalog.build_profile_selection_registry(
        presentation_role=role,
        profile=profile,
        evidence_classification=effective.evidence_classification,
        data_source_kind=effective.data_source_kind,
    )
    root.mkdir(parents=True)
    (root / "PRESENTATION_SELECTION.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    prepared_paths = (
        derivations.CANONICAL_OUTPUT_PATHS
        if role == "canonical"
        else derivations.ENGINEERING_OUTPUT_PATHS
    )
    data_root = root / "data"
    prepared_files: list[Path] = []
    for relative in prepared_paths:
        target = data_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")
        prepared_files.append(target)
    data_manifest: dict[str, object] = {
        "schema": "fraud_detection.chapter5_presentation_data.r6.v1",
        "status": "PASS",
        "profile": profile,
        "presentation_role": role,
        "evidence_classification": effective.evidence_classification,
        "data_source_kind": effective.data_source_kind,
        "seeds": list(effective.seeds),
        "budgets": list(effective.target_budgets),
        "primary_budgets": list(effective.primary_budgets),
        "candidate_pool_size": effective.candidate_pool_size,
        "selected_catalog_artifact_ids": list(selected),
        "outputs": _inspection_inventory(data_root, prepared_files),
    }
    if role == "engineering":
        data_manifest.update(
            {
                "evidence_statement": selection["evidence_statement"],
                "comparability_boundary": selection["comparability_boundary"],
            }
        )
    (data_root / "PRESENTATION_DATA_MANIFEST.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )

    logical_count = 9 if role == "canonical" else 1
    figure_root = root / "figures"
    figure_files: list[Path] = []
    for index in range(logical_count):
        for suffix in ("pdf", "png", "svg"):
            target = figure_root / f"figure-{index}.{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture\n", encoding="utf-8")
            figure_files.append(target)
    figure_manifest: dict[str, object] = {
        "schema": (
            "fraud_detection.chapter5_figure_render.r7b.v1"
            if role == "canonical"
            else "fraud_detection.engineering_figure_render.v1"
        ),
        "status": "PASS",
        "rendered_stems": [f"figure-{index}" for index in range(logical_count)],
        "outputs": _inspection_inventory(figure_root, figure_files),
    }
    if role == "engineering":
        figure_manifest.update(
            {
                "profile": profile,
                "presentation_role": role,
                "evidence_classification": effective.evidence_classification,
                "logical_figure_count": logical_count,
                "rendered_file_count": len(figure_files),
            }
        )
    (figure_root / "FIGURE_RENDER_MANIFEST.json").write_text(
        json.dumps(figure_manifest), encoding="utf-8"
    )

    table_root = root / "tables"
    table_files: list[Path] = []
    for index in range(logical_count):
        for suffix in ("csv", "tex"):
            target = table_root / f"table-{index}.{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture\n", encoding="utf-8")
            table_files.append(target)
    table_manifest: dict[str, object] = {
        "schema": (
            "fraud_detection.chapter5_table_render.r6.v1"
            if role == "canonical"
            else "fraud_detection.engineering_table_render.v1"
        ),
        "status": "PASS",
        "tables": [{"stem": f"table-{index}"} for index in range(logical_count)],
        "outputs": _inspection_inventory(table_root, table_files),
        "latex_preview": {"status": "SKIPPED_NO_ENGINE", "engine": None},
    }
    if role == "engineering":
        table_manifest.update(
            {
                "profile": profile,
                "presentation_role": role,
                "evidence_classification": effective.evidence_classification,
                "logical_table_count": logical_count,
                "rendered_file_count": len(table_files),
            }
        )
    (table_root / "TABLE_RENDER_MANIFEST.json").write_text(
        json.dumps(table_manifest), encoding="utf-8"
    )
    return root


def _experiment_plan(
    tmp_path: Path,
    *,
    profile: str = "canonical",
    output: Path | None = None,
    prerequisite_errors: tuple[ProductError, ...] = (),
) -> ExperimentPlan:
    root = tmp_path.resolve()
    return ExperimentPlan(
        repository_root=root,
        config=ExperimentConfig(
            data_path=root / "data" / "creditcard.csv",
            output_root=output or root / "outputs" / "run",
            phase="all",
            profile=profile,
        ),
        data_status="synthetic" if profile == "smoke-synthetic" else "verified",
        actual_data_sha256=None if profile == "smoke-synthetic" else "synthetic",
        prerequisite_errors=prerequisite_errors,
    )


def _presentation_plan(
    experiment_root: Path,
    profile: str,
) -> object:
    role = "canonical" if profile == "canonical" else "engineering"
    evidence = {
        "canonical": "thesis-evidentiary",
        "mini-real": "engineering mini profile — not thesis evidence",
        "smoke-synthetic": "non-evidentiary",
    }[profile]
    figure_count = 9 if role == "canonical" else 1
    table_count = 9 if role == "canonical" else 1
    return cli_presentation.PresentationBuildPlan(
        experiment_root=experiment_root,
        profile=profile,
        presentation_role=role,
        evidence_classification=evidence,
        data_source_kind="synthetic" if profile == "smoke-synthetic" else "real",
        expected_figure_count=figure_count,
        expected_table_count=table_count,
        expected_scope=(
            "full canonical thesis catalog"
            if role == "canonical"
            else "1 engineering figure and 1 engineering table"
        ),
    )


def _write_cli_run_manifest(
    experiment_root: Path,
    profile: str,
    *,
    status: str = "COMPLETE",
    completed_phases: tuple[str, ...] = (
        "preflight",
        "inner_selection",
        "selection_freeze",
        "final_outer",
        "aggregation",
        "qa",
    ),
    schema: str = "fraud_detection.run_manifest.v1",
) -> Path:
    experiment_root.mkdir(parents=True, exist_ok=True)
    evidence = {
        "canonical": "thesis-evidentiary",
        "mini-real": "engineering mini profile — not thesis evidence",
        "smoke-synthetic": "non-evidentiary",
    }[profile]
    manifest = {
        "schema": schema,
        "status": status,
        "profile": profile,
        "evidence_classification": evidence,
        "completed_phases": list(completed_phases),
        "effective_config": {},
        "data_summary": {},
        "produced_artifacts": [],
    }
    path = experiment_root / "RUN_MANIFEST.json"
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return experiment_root


def test_subprocess_imports_use_temporary_caches(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    mpl_config = tmp_path / "mplconfig"
    mpl_config.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(repository_root / "src"), environment.get("PYTHONPATH", ""))
        if value
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(mpl_config)
    program = """
import os
from pathlib import Path

import matplotlib
import fraud_detection.cli
import fraud_detection.presentation

root = Path.cwd().resolve()
config = Path(os.environ["MPLCONFIGDIR"]).resolve()
assert config.is_relative_to(root)
assert Path(matplotlib.get_configdir()).resolve() == config
assert Path(matplotlib.get_cachedir()).resolve().is_relative_to(config)
assert not (root / "generated").exists()
assert not (root / "outputs").exists()
assert not (root / "thesis_build").exists()
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    diagnostic = "\n".join(
        value.strip()
        for value in (completed.stdout, completed.stderr)
        if value.strip()
    )
    assert completed.returncode == 0, diagnostic
    assert mpl_config.resolve().is_relative_to(tmp_path.resolve())
    assert not list(tmp_path.rglob("__pycache__"))


def test_removed_direct_runner_has_no_module_replacement() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    assert not (repository_root / "scripts" / "run_amount_gain_ranker.py").exists()
    assert util.find_spec("fraud_detection.experiment." + "cli") is None
    experiment_execution = import_module(
        "fraud_detection.experiment.execution.pipeline"
    )
    assert run_experiment is experiment_execution.run_experiment


def test_top_level_help_exposes_main_command_groups(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def captured_help(arguments: list[str]) -> tuple[str, str]:
        returncode = main(arguments)
        captured = capsys.readouterr()
        output = strip_ansi(captured.out + captured.err)

        assert returncode == 0
        assert "\x1b" not in output
        return output, " ".join(output.split())

    output, _normalized_output = captured_help(["--help"])

    root_command = typer.main.get_command(cli_app.app)
    commands = list(root_command.commands)
    assert commands == ["setup", "check", "run", "build", "inspect"]
    for command in commands:
        assert command in output
    assert "Run fast, read-only diagnostics without downloads" not in output
    assert "Inspect known artifact and manifest paths" not in output
    assert "Plan or run the complete frozen experiment" not in output
    assert "Build all presentation-only artifacts in sequence" not in output

    _setup_output, setup_help = captured_help(["setup", "--help"])

    for phrase in (
        "Prepare or repair the local environment",
        "install pinned packages",
        "acquire, reuse, or validate the canonical dataset",
        "never runs an experiment",
        "fraud-detection setup",
        "fraud-detection --json setup",
    ):
        assert phrase in setup_help

    _check_output, check_help = captured_help(["check", "--help"])

    for phrase in (
        "fast read-only environment diagnostics by default",
        "--full",
        "--require-data",
        "no repair or model fit",
        "fraud-detection check",
        "fraud-detection --json check --full",
    ):
        assert phrase in check_help
    for unsupported in ("--yes", "--fix", "--install"):
        assert unsupported not in check_help

    output, normalized_output = captured_help(["run", "--help"])

    assert "Plan and run one complete profile-aware experiment safely" in output
    for phrase in (
        "live progress",
        "--plain emits stable text",
        "--json emits one final document",
        "no linear ETA",
    ):
        assert phrase in normalized_output
    for option in ("--profile", "--output", "--data", "--dry-run", "--yes"):
        assert option in output
    for removed in ("--output-dir", "--data-path", "--phase"):
        assert removed not in output
    for example in (
        "run --profile smoke-synthetic --dry-run",
        "run --profile smoke-synthetic",
        "run --profile mini-real --yes",
        "--output outputs/canonical-final-2 --yes",
    ):
        assert example in normalized_output

    output, normalized_output = captured_help(["inspect", "--help"])

    for phrase in (
        "Read-only semantic inspection",
        "repository, experiment, partial-run, or presentation root",
        "without searching parent directories",
        "manifest and checksum validation",
    ):
        assert phrase in normalized_output
    assert "PATH" in output
    assert "--root" not in output
    for example in (
        "inspect .",
        "inspect outputs/canonical-final",
        "inspect generated/runs/interrupted",
        "inspect generated/presentations/canonical-final",
        "--json inspect outputs/canonical-final",
    ):
        assert example in normalized_output

    output, normalized_output = captured_help(["build", "--help"])

    assert "Build canonical or engineering presentation artifacts" in (
        normalized_output
    )
    assert "completed run manifest without fitting models" in normalized_output
    for phrase in (
        "live progress",
        "--plain emits stable text",
        "--json emits one final document",
        "no linear ETA",
    ):
        assert phrase in normalized_output
    assert "EXPERIMENT_PATH" in output
    for option in ("--output", "--force"):
        assert option in output
    for removed in ("--output-dir", "--output-root", "--root", "--yes"):
        assert removed not in output
    for example in (
        "build generated/runs/smoke-synthetic",
        "build generated/runs/mini-real",
        "--output generated/presentations/canonical-copy",
        "--force",
    ):
        assert example in normalized_output


def test_removed_legacy_entry_points_return_usage_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("removed commands must fail before dispatch")

    monkeypatch.setattr(cli_app, "build_experiment_plan", unexpected)
    monkeypatch.setattr(cli_app, "execute_experiment", unexpected)
    monkeypatch.setattr(cli_app, "inspect_presentation_input", unexpected)
    monkeypatch.setattr(cli_app, "inspect_path", unexpected)
    monkeypatch.setattr(cli_presentation, "build_presentation", unexpected)
    monkeypatch.setattr(cli_app, "run_doctor", unexpected)

    for arguments in (
        ["doctor", "--help"],
        ["experiment", "plan", "--help"],
        ["experiment", "run", "--help"],
        ["presentation", "all", "--help"],
        ["artifacts", "list", "--help"],
    ):
        returncode = main(arguments)
        captured = capsys.readouterr()

        assert returncode == 2
        assert captured.out == ""
        assert "No such command" in captured.err
        assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ("plan", "run"))
def test_removed_experiment_subcommands_are_rejected(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    returncode = main(["experiment", command, "--help"])
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert returncode == 2
    assert "No such command" in output
    assert "Traceback" not in output


def test_unknown_command_returns_usage_code_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("usage failures must precede planning and execution")

    monkeypatch.setattr(cli_app, "build_experiment_plan", unexpected)
    monkeypatch.setattr(cli_app, "execute_experiment", unexpected)
    monkeypatch.setattr(cli_app, "inspect_presentation_input", unexpected)
    monkeypatch.setattr(cli_app, "inspect_path", unexpected)
    monkeypatch.setattr(cli_presentation, "build_presentation", unexpected)
    monkeypatch.setattr(cli_app, "run_doctor", unexpected)
    monkeypatch.setattr(cli_app, "run_check", unexpected)
    monkeypatch.setattr(cli_app, "run_setup", unexpected)

    cases = (
        (["--plain", "unknown-command"], "No such command"),
        (
            ["--plain", "run", "--profile", "smoke-synthetic", "--unknown"],
            "No such option",
        ),
        (
            ["--plain", "run", "--profile", "unknown-profile"],
            "Unknown profile",
        ),
        (["--plain", "run", "--dry-run"], "--profile is required"),
        (
            [
                "--plain",
                "run",
                "--profile",
                "smoke-synthetic",
                "--data",
                "data/creditcard.csv",
            ],
            "--data is not valid",
        ),
        (["--plain", "build"], "Missing argument"),
        (
            ["--plain", "build", "outputs/complete", "--unknown"],
            "No such option",
        ),
        (
            ["--plain", "build", "outputs/complete", "trailing"],
            "unexpected extra argument",
        ),
        (["--plain", "inspect"], "Missing argument"),
        (["--plain", "inspect", ".", "--unknown"], "No such option"),
        (
            ["--plain", "inspect", ".", "trailing"],
            "unexpected extra argument",
        ),
        (["--plain", "check", "--unknown"], "No such option"),
        (["--plain", "check", "trailing"], "unexpected extra argument"),
        (["--plain", "setup", "--unknown"], "No such option"),
        (["--plain", "setup", "trailing"], "unexpected extra argument"),
    )
    for arguments, message in cases:
        returncode = main(arguments)
        captured = capsys.readouterr()

        assert returncode == 2
        assert "FD-USAGE" in captured.err
        assert message in captured.err
        assert "Traceback" not in captured.err


def test_expected_error_has_no_python_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unsupported = tmp_path / "unsupported"
    unsupported.mkdir()
    inspected_file = tmp_path / "file.txt"
    inspected_file.write_text("fixture\n", encoding="utf-8")
    failure_cases = (
        (tmp_path / "missing", "FD-INSPECT-NOT-FOUND"),
        (inspected_file, "FD-INSPECT-NOT-DIRECTORY"),
        (unsupported, "FD-INSPECT-UNSUPPORTED"),
    )
    for path, code in failure_cases:
        returncode = main(["--plain", "inspect", str(path)])
        captured = capsys.readouterr()

        assert returncode == 1
        assert code in captured.err
        assert "Traceback" not in captured.err
        if code == "FD-INSPECT-UNSUPPORTED":
            for sentinel in (
                "pyproject.toml",
                "RUN_MANIFEST.json",
                "phase manifest",
                "selection and stage manifests",
            ):
                assert sentinel in captured.err
            assert "count=0" not in captured.err

    repository = tmp_path / "repository"
    (repository / "src" / "fraud_detection").mkdir(parents=True)
    (repository / "src" / "fraud_detection" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (repository / "pyproject.toml").write_text(
        "[project]\n"
        'name = "fraud-detection"\n'
        "[project.scripts]\n"
        'fraud-detection = "fraud_detection.cli:main"\n',
        encoding="utf-8",
    )
    reference_root = repository / "reference_results"
    reference_root.mkdir()
    for name in (
        "central_topk_results.csv",
        "selected_configuration_summary.csv",
        "data_identity.json",
    ):
        (reference_root / name).write_text("fixture\n", encoding="utf-8")
    assert not (repository / ".git").exists()
    returncode = main(["--plain", "inspect", str(repository)])
    captured = capsys.readouterr()

    assert returncode == 0
    assert captured.err == ""
    assert captured.out.startswith("TYPE repository\nSTATUS VALID\n")
    assert "package_name=fraud_detection" in captured.out
    assert "console_entry_point=fraud-detection" in captured.out
    assert "smoke-synthetic" in captured.out
    assert "fraud-detection run --profile smoke-synthetic --dry-run" in captured.out
    artifacts_module = import_module("fraud_detection.artifacts")
    assert callable(artifacts_module.inspect_path)
    assert artifacts_module.inspect_path(repository).path_type == "repository"
    for removed_name in (
        "inventory_artifacts",
        "_known_artifact_category",
    ):
        assert not hasattr(artifacts_module, removed_name)

    completed_runs = {
        profile: _write_inspection_run(
            tmp_path / "runs" / profile,
            profile,
        )
        for profile in ("smoke-synthetic", "mini-real", "canonical")
    }
    snapshots = {
        path: {
            item.relative_to(path).as_posix(): item.read_bytes()
            for item in path.rglob("*")
            if item.is_file()
        }
        for path in completed_runs.values()
    }
    data_stage = import_module(
        "fraud_detection.presentation.preparation.data"
    )
    real_group_loader = data_stage._load_ranking_groups
    pandas = import_module("pandas")
    ranking_rows = []
    for method in METHOD_ORDER:
        for row_index in range(2):
            ranking_rows.append(
                {
                    "seed": 42,
                    "target_budget": 20,
                    "method_family": method,
                    "score_path": (
                        method if method == METHOD_ORDER[0] else f"{method}_k20"
                    ),
                    "row_index": row_index,
                    "original_position": row_index,
                    "candidate_flag": row_index == 0,
                    "candidate_pool_size": 1,
                    "candidate_pool_sha256": "b" * 64,
                    "p_fraud": 1.0 - row_index / 10.0,
                    "raw_ranker_score": (
                        1.0 - row_index / 10.0
                        if method == METHOD_ORDER[0] or row_index == 0
                        else None
                    ),
                    "final_rank_position": row_index + 1,
                    "bce_rank_position": row_index + 1,
                    "priority_order_score": -row_index,
                    "Class": int(row_index == 0),
                    "Amount": float(row_index + 1),
                    "selected_gain": "linear",
                    "selection_status": "selected",
                    "truncation_level": 1,
                    "final_n_estimators": 1,
                    "score_type": "ranking_score",
                }
            )
    ranking_frame = pandas.DataFrame(ranking_rows)

    class TinyStore:
        def __init__(self, frame: object) -> None:
            self.frame = frame

        def ranking(self, _seed: int) -> object:
            return self.frame.copy()

    assert len(real_group_loader(TinyStore(ranking_frame), (42,), (20,), 1)) == 4
    with pytest.raises(RuntimeError, match="Full-ranking coverage mismatch"):
        real_group_loader(
            TinyStore(
                ranking_frame.loc[
                    ranking_frame["method_family"] != METHOD_ORDER[-1]
                ]
            ),
            (42,),
            (20,),
            1,
        )
    duplicated = pandas.concat(
        [
            ranking_frame,
            ranking_frame.loc[
                ranking_frame["method_family"] == METHOD_ORDER[1]
            ],
        ],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="Ranking candidate count differs"):
        real_group_loader(TinyStore(duplicated), (42,), (20,), 1)

    def validated_group_identities(
        store: object,
        seeds: tuple[int, ...],
        budgets: tuple[int, ...],
        _pool_size: int,
    ) -> dict[tuple[int, int, str], object]:
        if store.root.name == "missing-group":
            raise RuntimeError("Full-ranking coverage mismatch: missing group.")
        if store.root.name == "duplicate-group":
            raise RuntimeError("Ranking candidate count differs for duplicate group.")
        return {
            (seed, budget, method): object()
            for seed in seeds
            for budget in budgets
            for method in METHOD_ORDER
        }

    monkeypatch.setattr(
        data_stage,
        "_load_ranking_groups",
        validated_group_identities,
    )
    expected_group_counts = {
        "smoke-synthetic": 12,
        "mini-real": 36,
        "canonical": 140,
    }
    for profile, path in completed_runs.items():
        returncode = main(["--plain", "inspect", str(path)])
        captured = capsys.readouterr()

        assert returncode == 0
        assert captured.err == ""
        assert captured.out.startswith("TYPE experiment\nSTATUS COMPLETE\n")
        assert f"PROFILE {profile}" in captured.out
        assert "CHECKSUMS VERIFIED" in captured.out
        assert "PRESENTATION_COMPATIBLE true" in captured.out
        assert "DETAIL seeds=" in captured.out
        assert "DETAIL budgets=" in captured.out
        assert "DETAIL result_grid_dimensions=" in captured.out
        assert (
            f"DETAIL ranking_group_count={expected_group_counts[profile]}"
            in captured.out
        )
        assert f"NEXT fraud-detection build {path}" in captured.out
        assert {
            item.relative_to(path).as_posix(): item.read_bytes()
            for item in path.rglob("*")
            if item.is_file()
        } == snapshots[path]

    smoke_run = completed_runs["smoke-synthetic"]
    returncode = main(["--json", "inspect", str(smoke_run)])
    captured = capsys.readouterr()
    smoke_payload = json.loads(captured.out)

    assert returncode == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert smoke_payload["details"]["ranking_group_count"] == 12
    assert {
        item.relative_to(smoke_run).as_posix(): item.read_bytes()
        for item in smoke_run.rglob("*")
        if item.is_file()
    } == snapshots[smoke_run]

    missing_group = _write_inspection_run(
        tmp_path / "invalid-groups" / "missing-group",
        "smoke-synthetic",
    )
    duplicated_group = _write_inspection_run(
        tmp_path / "invalid-groups" / "duplicate-group",
        "smoke-synthetic",
    )
    invalid_group_snapshots = {
        path: {
            item.relative_to(path).as_posix(): item.read_bytes()
            for item in path.rglob("*")
            if item.is_file()
        }
        for path in (missing_group, duplicated_group)
    }
    for path, message in (
        (missing_group, "Full-ranking coverage mismatch"),
        (duplicated_group, "Ranking candidate count differs"),
    ):
        returncode = main(["--plain", "inspect", str(path)])
        captured = capsys.readouterr()

        assert returncode == 1
        assert message in captured.err
        assert "Traceback" not in captured.err
        assert {
            item.relative_to(path).as_posix(): item.read_bytes()
            for item in path.rglob("*")
            if item.is_file()
        } == invalid_group_snapshots[path]

    partial = _write_inspection_partial(tmp_path / "partial")
    partial_snapshot = {
        item.relative_to(partial).as_posix(): item.read_bytes()
        for item in partial.rglob("*")
        if item.is_file()
    }
    returncode = main(["--plain", "inspect", str(partial)])
    captured = capsys.readouterr()

    assert returncode == 0
    assert captured.err == ""
    assert captured.out.startswith(
        "TYPE experiment-partial\nSTATUS INCOMPLETE\n"
    )
    assert "COMPLETED_PHASES preflight" in captured.out
    assert "PRESENTATION_COMPATIBLE false" in captured.out
    assert "No COMPLETE RUN_MANIFEST.json" in captured.out
    assert "Fit-level resume is unsupported" in captured.out
    assert "new output directory" in captured.out
    assert "ranking_group_count" not in captured.out

    returncode = main(["--json", "inspect", str(partial)])
    captured = capsys.readouterr()
    partial_payload = json.loads(captured.out)

    assert returncode == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert partial_payload["command"] == "inspect"
    assert partial_payload["result"] == "success"
    assert partial_payload["exit_code"] == 0
    assert partial_payload["path_type"] == "experiment-partial"
    assert partial_payload["status"] == "INCOMPLETE"
    assert partial_payload["presentation_compatible"] is False
    assert partial_payload["completed_phases"] == ["preflight"]
    assert partial_payload["missing_phases"]
    assert partial_payload["warnings"]
    assert "ranking_group_count" not in partial_payload["details"]
    assert {
        item.relative_to(partial).as_posix(): item.read_bytes()
        for item in partial.rglob("*")
        if item.is_file()
    } == partial_snapshot

    presentations = {
        profile: _write_inspection_presentation(
            tmp_path / "presentations" / profile,
            profile,
        )
        for profile in ("smoke-synthetic", "canonical")
    }
    presentation_snapshots = {
        path: {
            item.relative_to(path).as_posix(): item.read_bytes()
            for item in path.rglob("*")
            if item.is_file()
        }
        for path in presentations.values()
    }
    for profile, path in presentations.items():
        returncode = main(["--plain", "inspect", str(path)])
        captured = capsys.readouterr()

        assert returncode == 0
        assert captured.err == ""
        assert captured.out.startswith("TYPE presentation\nSTATUS COMPLETE\n")
        assert f"PROFILE {profile}" in captured.out
        assert "CHECKSUMS VERIFIED" in captured.out
        assert "DETAIL latex_preview_status=SKIPPED_NO_ENGINE" in captured.out
        assert "LOGICAL_FIGURES 9" in captured.out if profile == "canonical" else (
            "LOGICAL_FIGURES 1" in captured.out
        )
        assert "LOGICAL_TABLES 9" in captured.out if profile == "canonical" else (
            "LOGICAL_TABLES 1" in captured.out
        )
        if profile == "smoke-synthetic":
            assert "Deterministic synthetic engineering data" in captured.out
            assert "Not thesis evidence" in captured.out
            assert "Not comparable with canonical empirical results" in captured.out
        assert f"figures in {path / 'figures'}" in captured.out
        assert f"tables in {path / 'tables'}" in captured.out
        assert {
            item.relative_to(path).as_posix(): item.read_bytes()
            for item in path.rglob("*")
            if item.is_file()
        } == presentation_snapshots[path]

    checksum_run = _write_inspection_run(
        tmp_path / "invalid" / "checksum",
        "smoke-synthetic",
    )
    (checksum_run / "diagnostics" / "global_metrics_seedwise.csv").write_text(
        "tampered\n", encoding="utf-8"
    )
    missing_run = _write_inspection_run(
        tmp_path / "invalid" / "missing",
        "smoke-synthetic",
    )
    (missing_run / "diagnostics" / "global_metrics_seedwise.csv").unlink()
    for path, code in (
        (checksum_run, "FD-INSPECT-CHECKSUM"),
        (missing_run, "FD-INSPECT-ARTIFACT"),
    ):
        returncode = main(["--plain", "inspect", str(path)])
        captured = capsys.readouterr()

        assert returncode == 1
        assert code in captured.err
        assert "Traceback" not in captured.err

    conflict = tmp_path / "conflict"
    (conflict / "src" / "fraud_detection").mkdir(parents=True)
    (conflict / "src" / "fraud_detection" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (conflict / "pyproject.toml").write_text("not valid TOML", encoding="utf-8")
    (conflict / "RUN_MANIFEST.json").write_text("not JSON", encoding="utf-8")
    returncode = main(["--plain", "inspect", str(conflict)])
    captured = capsys.readouterr()

    assert returncode == 1
    assert "FD-INSPECT-CONFLICT" in captured.err
    assert "repository, experiment" in captured.err
    assert "malformed" not in captured.err

    returncode = main(["--json", "artifacts", "list", "--root", str(repository)])
    captured = capsys.readouterr()

    assert returncode == 2
    payload = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert payload["command"] == "artifacts list"
    assert payload["exit_code"] == 2
    assert payload["error"]["code"] == "FD-USAGE"
    assert "No such command" in payload["error"]["summary"]
    assert "Traceback" not in captured.out

    experiment_execution = import_module(
        "fraud_detection.experiment.execution.pipeline"
    )
    presentation_build = import_module("fraud_detection.presentation.build")
    public_presentation = import_module("fraud_detection.presentation")
    assert run_experiment is experiment_execution.run_experiment
    assert public_presentation.build_presentation is presentation_build.build_presentation

    monkeypatch.setattr(cli_app, "find_repository_root", lambda: None)

    returncode = main(["--plain", "setup"])
    captured = capsys.readouterr()

    assert returncode == 1
    assert "ERROR FD-ROOT-NOT-FOUND" in captured.err
    assert "Traceback" not in captured.err


def test_debug_prints_traceback_for_unexpected_internal_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class BrokenCommand:
        def main(self, **_kwargs: object) -> int:
            raise RuntimeError("controlled internal failure")

    monkeypatch.setattr(
        typer.main,
        "get_command",
        lambda _application: BrokenCommand(),
    )

    returncode = main(["--plain", "--debug", "inspect", "."])
    captured = capsys.readouterr()

    assert returncode == 1
    assert "ERROR FD-UNEXPECTED" in captured.err
    assert "Traceback (most recent call last)" in captured.err
    assert "RuntimeError: controlled internal failure" in captured.err


def test_setup_report_is_stable_plain_text_without_ansi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stdout = StringIO()
    state = ShellState(stdout=stdout, stderr=StringIO())
    report = CommandReport()
    report.passed("Environment ready.")
    report.info("No action required.")

    render_report(state, "setup", report, exit_code=0)

    assert stdout.getvalue() == (
        "PASS Environment ready.\n"
        "INFO No action required.\n"
        "STATUS command=setup result=pass exit_code=0\n"
    )
    assert "\x1b[" not in stdout.getvalue()

    diagnostics_module = import_module("fraud_detection.setup.diagnostics")
    environment_module = import_module("fraud_detection.setup.environment")
    assert public_run_doctor is diagnostics_module.run_doctor
    assert public_run_check is diagnostics_module.run_check
    assert public_run_setup is environment_module.run_setup

    root = tmp_path.resolve()
    sentinel = root / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    monkeypatch.setattr(cli_app, "find_repository_root", lambda: root)
    quick_calls: list[dict[str, object]] = []
    full_calls: list[dict[str, object]] = []
    quick_report = DiagnosticReport(
        command="doctor",
        repository_root=root,
        findings=(
            DiagnosticFinding(
                code="FD-PYTHON",
                status="PASS",
                summary="Python 3.12 is available.",
            ),
            DiagnosticFinding(
                code="",
                status="WARN",
                summary=(
                    "data/creditcard.csv is absent. The dataset is optional "
                    "for this check."
                ),
            ),
            DiagnosticFinding(
                code="",
                status="INFO",
                summary="Quick diagnostic scope completed.",
            ),
        ),
        elapsed_seconds=0.01,
    )
    data_report = DiagnosticReport(
        command="doctor",
        repository_root=root,
        findings=(
            DiagnosticFinding(
                code="FD-DATA-IDENTITY",
                status="PASS",
                summary="Canonical data identity is validated.",
                details={
                    "raw_rows": 284_807,
                    "raw_fraud": 492,
                    "deduplicated_rows": 283_726,
                    "deduplicated_fraud": 473,
                    "deduplicated_legitimate": 283_253,
                    "removed_duplicates": 1_081,
                },
            ),
        ),
        elapsed_seconds=0.02,
    )
    full_report = DiagnosticReport(
        command="check",
        repository_root=root,
        findings=data_report.findings,
        elapsed_seconds=0.03,
    )

    def fake_quick(**kwargs: object) -> DiagnosticReport:
        quick_calls.append(kwargs)
        return data_report if kwargs["require_data"] else quick_report

    def fake_full(**kwargs: object) -> DiagnosticReport:
        full_calls.append(kwargs)
        return full_report

    monkeypatch.setattr(cli_app, "run_doctor", fake_quick)
    monkeypatch.setattr(cli_app, "run_check", fake_full)

    returncode = main(["--plain", "check"])
    captured = capsys.readouterr()

    assert returncode == 0
    assert quick_calls == [{"repository_root": root, "require_data": False}]
    assert full_calls == []
    assert captured.out.startswith(
        "CHECK mode=quick require_data=false read_only=true\n"
    )
    assert "PASS Python 3" in captured.out
    assert "result=pass exit_code=0 mode=quick" in captured.out
    assert "passed=2 warnings=1 failed=0 skipped=0" in captured.out
    assert "NEXT fraud-detection run --profile smoke-synthetic --dry-run" in (
        captured.out
    )
    assert "WARNING data/creditcard" in captured.err
    assert "\x1b[" not in captured.out + captured.err
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert sorted(path.name for path in root.iterdir()) == ["sentinel.txt"]

    quick_call_count = len(quick_calls)
    returncode = main(["--plain", "doctor"])
    captured = capsys.readouterr()

    assert returncode == 2
    assert len(quick_calls) == quick_call_count
    assert captured.out == ""
    assert "No such command" in captured.err
    assert "Traceback" not in captured.err

    returncode = main(["--plain", "check", "--require-data"])
    captured = capsys.readouterr()

    assert returncode == 0
    assert quick_calls[-1] == {
        "repository_root": root,
        "require_data": True,
    }
    assert "require_data=true" in captured.out
    assert "dataset=validated" in captured.out
    assert "NEXT fraud-detection run --profile smoke-synthetic" in captured.out
    assert "--dry-run" not in captured.out

    returncode = main(["--plain", "check", "--full", "--require-data"])
    captured = capsys.readouterr()

    assert returncode == 0
    assert full_calls == [{"repository_root": root, "require_data": True}]
    assert "mode=full" in captured.out
    assert "require_data=true" in captured.out
    assert "dataset=validated" in captured.out
    assert "NEXT fraud-detection run --profile smoke-synthetic" in captured.out

    returncode = main(["--json", "check"])
    captured = capsys.readouterr()
    check_payload = json.loads(captured.out)

    assert returncode == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert check_payload["command"] == "check"
    assert check_payload["result"] == "pass"
    assert check_payload["exit_code"] == 0
    assert check_payload["mode"] == "quick"
    assert check_payload["require_data"] is False
    assert check_payload["repository_root"] == root.as_posix()
    assert check_payload["passed_count"] == 2
    assert check_payload["warning_count"] == 1
    assert check_payload["failed_count"] == 0
    assert check_payload["skipped_count"] == 0
    assert isinstance(check_payload["findings"], list)
    assert check_payload["suggested_command"].endswith("--dry-run")
    assert "\x1b[" not in captured.out

    failure_report = DiagnosticReport(
        command="doctor",
        repository_root=root,
        findings=(
            DiagnosticFinding(
                code="FD-DATA-MISSING",
                status="FAIL",
                summary="Canonical dataset is unavailable.",
                recovery=(
                    "Run fraud-detection setup.",
                    "Follow data/README.md for the supported manual path.",
                ),
            ),
        ),
        elapsed_seconds=0.01,
    )
    monkeypatch.setattr(cli_app, "run_doctor", lambda **_kwargs: failure_report)
    returncode = main(["--plain", "check", "--require-data"])
    captured = capsys.readouterr()

    assert returncode == 1
    assert "FAIL Canonical dataset is unavailable" in captured.err
    assert "WHY Canonical real-data validation cannot proceed" in captured.err
    assert "MODIFICATION none; diagnostics are read-only" in captured.err
    assert "RECOVERY 1 Run fraud-detection setup" in captured.err
    assert "NEXT fraud-detection setup" in captured.out
    assert "Traceback" not in captured.out + captured.err

    def interrupt(**_kwargs: object) -> DiagnosticReport:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_app, "run_doctor", interrupt)
    returncode = main(["--plain", "check"])
    captured = capsys.readouterr()

    assert returncode == 130
    assert "FD-INTERRUPTED" in captured.err
    assert "Ctrl-C" in captured.err
    assert "Traceback" not in captured.err

    setup_calls: list[dict[str, object]] = []
    setup_result = SetupResult(
        dataset_path=root / "data" / "creditcard.csv",
        dataset_status="reused",
        interpreter=root / ".venv" / "Scripts" / "python.exe",
        expected_sha256="a" * 64,
    )

    def fake_setup(**kwargs: object) -> SetupResult:
        setup_calls.append(kwargs)
        progress = kwargs["progress"]
        assert callable(progress)
        for message in (
            "START Virtual environment — creating or reusing",
            "PASS Virtual environment — ready",
            "START pip — installing pinned version 26.2",
            "PASS pip — installed",
            "START Full diagnostic — validating installed repository",
            "PASS Full diagnostic — completed",
        ):
            progress(message)
        return setup_result

    monkeypatch.setattr(cli_app, "run_setup", fake_setup)
    before = len(setup_calls)
    returncode = main(["--plain", "setup"])
    captured = capsys.readouterr()

    assert returncode == 0
    assert len(setup_calls) == before + 1
    assert setup_calls[-1]["repository_root"] == root
    assert setup_calls[-1]["capture_install_output"] is True
    assert captured.err == ""
    assert captured.out.startswith("SETUP_PLAN mutation=true\n")
    assert "PLAN Python validation" in captured.out
    assert "PLAN Virtual environment target=" in captured.out
    assert "PLAN Dependency installation pinned=true" in captured.out
    assert "PLAN Package installation local=true" in captured.out
    assert "PLAN Dataset action=acquire and validate canonical dataset" in (
        captured.out
    )
    assert "START Virtual environment — creating or reusing" in captured.out
    assert "PASS Virtual environment — ready" in captured.out
    assert "START pip — installing pinned version 26.2" in captured.out
    assert "PASS pip — installed" in captured.out
    assert "START Full diagnostic — validating installed repository" in captured.out
    assert "PASS Full diagnostic — completed" in captured.out
    assert "RESULT command=setup result=pass exit_code=0" in captured.out
    assert "NEXT fraud-detection check --full" in captured.out
    assert "THEN fraud-detection run --profile smoke-synthetic" in captured.out
    assert "experiment execution" not in captured.out.lower()
    assert "\x1b[" not in captured.out

    before = len(setup_calls)
    returncode = main(["--json", "setup"])
    captured = capsys.readouterr()
    setup_payload = json.loads(captured.out)

    assert returncode == 0
    assert len(setup_calls) == before + 1
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert setup_payload["command"] == "setup"
    assert setup_payload["result"] == "pass"
    assert setup_payload["exit_code"] == 0
    assert setup_payload["repository_root"] == root.as_posix()
    assert setup_payload["python_executable"] == (
        root / ".venv" / "Scripts" / "python.exe"
    ).as_posix()
    assert setup_payload["environment_status"] == "prepared"
    assert setup_payload["package_status"] == "installed"
    assert setup_payload["dataset_status"] == "reused"
    assert setup_payload["data_identity_status"] == "validated"
    assert isinstance(setup_payload["findings"], list)
    assert setup_payload["suggested_command"] == "fraud-detection check --full"

    setup_failure = SetupFailure(
        "FD-PYTHON-VERSION",
        "Python 3.12 is required.",
        ("Install Python 3.12 and rerun setup.",),
        {"side_effects_started": False},
    )

    def fail_setup(**_kwargs: object) -> SetupResult:
        raise setup_failure

    monkeypatch.setattr(cli_app, "run_setup", fail_setup)
    returncode = main(["--plain", "setup"])
    captured = capsys.readouterr()

    assert returncode == 1
    assert "FAIL Python validation — Python 3.12 is required" in captured.err
    assert "MODIFICATION none" in captured.err
    assert "NEXT fraud-detection setup" in captured.err
    assert "Traceback" not in captured.out + captured.err


def test_product_error_rendering_is_stable_plain_text() -> None:
    stderr = StringIO()
    state = ShellState(plain=True, stdout=StringIO(), stderr=stderr)

    render_error(
        state,
        "sample",
        ProductError(
            "FD-SAMPLE",
            "A controlled error occurred.",
            ("Correct the sample input.",),
            {"field": "value"},
            7,
        ),
    )

    assert stderr.getvalue() == (
        "ERROR FD-SAMPLE A controlled error occurred.\n"
        "DETAIL field=value\n"
        "RECOVERY 1 Correct the sample input.\n"
    )


def test_presentation_command_dispatches_direct_api_without_subprocesses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path.resolve()
    experiments = {
        profile: _write_cli_run_manifest(
            root / "outputs" / f"{profile}-run",
            profile,
        )
        for profile in ("smoke-synthetic", "mini-real", "canonical")
    }
    plans = {
        path: _presentation_plan(path, profile)
        for profile, path in experiments.items()
    }
    inspected: list[Path] = []
    observed: list[PresentationConfig] = []

    def fake_inspect(experiment_root: Path) -> object:
        inspected.append(experiment_root)
        return plans[experiment_root]

    def fake_build_presentation(
        config: PresentationConfig,
    ) -> PresentationResult:
        observed.append(config)
        plan = plans[config.experiment_root]
        assert config.event_sink is not None
        prepared_count = 29 if plan.profile == "canonical" else 2
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "build-data-start",
                "fields": {"phase": "\x1b[31mdata\x1b[0m"},
            },
        )
        for event, fields in (
            (
                "build-data-frame",
                {"completed": prepared_count, "total": prepared_count},
            ),
            ("build-data-complete", {"frames": prepared_count}),
            ("render-figures-start", {}),
            (
                "render-figure",
                {
                    "completed": plan.expected_figure_count,
                    "total": plan.expected_figure_count,
                },
            ),
            (
                "render-figures-complete",
                {"figures": plan.expected_figure_count},
            ),
            ("render-tables-start", {}),
            (
                "render-table",
                {
                    "completed": plan.expected_table_count,
                    "total": plan.expected_table_count,
                },
            ),
            (
                "table-preview",
                {"status": "SKIPPED_NO_ENGINE"},
            ),
            (
                "render-tables-complete",
                {"tables": plan.expected_table_count},
            ),
        ):
            config.event_sink(
                "status",
                {
                    "level": "WARN" if event == "table-preview" else "PASS",
                    "event": event,
                    "fields": fields,
                },
            )
        return PresentationResult(
            output_root=config.output_root,
            data_dir=config.output_root / "data",
            figures_dir=config.output_root / "figures",
            tables_dir=config.output_root / "tables",
            preview_dir=config.output_root / "preview",
            steps=(
                PresentationStepResult(
                    step="data",
                    manifest_path=(
                        config.output_root
                        / "data"
                        / "PRESENTATION_DATA_MANIFEST.json"
                    ),
                    manifest={
                        "status": "PASS",
                        "outputs": [
                            {"path": f"prepared-{index}.csv"}
                            for index in range(prepared_count)
                        ],
                    },
                    elapsed_seconds=0.1,
                ),
                PresentationStepResult(
                    step="figures",
                    manifest_path=(
                        config.output_root
                        / "figures"
                        / "FIGURE_RENDER_MANIFEST.json"
                    ),
                    manifest={
                        "status": "PASS",
                        "logical_figure_count": plan.expected_figure_count,
                    },
                    elapsed_seconds=0.2,
                ),
                PresentationStepResult(
                    step="tables",
                    manifest_path=(
                        config.output_root
                        / "tables"
                        / "TABLE_RENDER_MANIFEST.json"
                    ),
                    manifest={
                        "status": "PASS",
                        "logical_table_count": plan.expected_table_count,
                        "latex_preview": {
                            "status": "SKIPPED_NO_ENGINE",
                            "engine": None,
                        },
                    },
                    elapsed_seconds=0.3,
                ),
            ),
            status="COMPLETE",
        )

    def fail_subprocess(*_args: object, **_kwargs: object) -> None:
        pytest.fail("presentation execution must not start a subprocess")

    monkeypatch.setattr(cli_app, "find_repository_root", lambda: root)
    monkeypatch.setattr(cli_app, "inspect_presentation_input", fake_inspect)
    monkeypatch.setattr(
        cli_presentation,
        "build_presentation",
        fake_build_presentation,
    )
    monkeypatch.setattr(subprocess, "run", fail_subprocess)
    monkeypatch.setattr(subprocess, "Popen", fail_subprocess)

    for profile, experiment in experiments.items():
        before = len(observed)
        returncode = main(["--plain", "build", str(experiment)])
        captured = capsys.readouterr()

        assert returncode == 0
        assert captured.err == ""
        assert captured.out.count(
            "Optional LaTeX preview — skipped (engine not available)"
        ) == 1
        assert len(observed) == before + 1
        assert inspected[-1] == experiment
        config = observed[-1]
        assert config.repository_root == root
        assert config.experiment_root == experiment
        assert config.output_root == (
            root / "generated" / "presentations" / experiment.name
        )
        assert config.width_mm == 160.0
        assert config.preview_dir == config.output_root / "preview"
        assert config.force is False
        assert config.event_sink is not None
        assert f"profile={profile}" in captured.out
        assert "experiment_status=COMPLETE" in captured.out
        assert f"presentation_role={plans[experiment].presentation_role}" in (
            captured.out
        )
        assert "RESULT command=build result=pass" in captured.out
        assert "PROGRESS command=build" in captured.out
        assert 'operation="figures"' in captured.out
        assert 'operation="tables"' in captured.out
        assert (
            f'operation="figures" '
            f"completed={plans[experiment].expected_figure_count} "
            f"total={plans[experiment].expected_figure_count}"
        ) in captured.out
        assert "NEXT fraud-detection inspect" in captured.out
        assert "\x1b[" not in captured.out
        assert "ETA" not in captured.out
        if profile == "canonical":
            assert "not thesis evidence" not in captured.out.lower()
        else:
            assert captured.out.lower().count("not thesis evidence") >= 2

    rich_plan_output = _TTYBuffer()
    smoke_plan = plans[experiments["smoke-synthetic"]]
    cli_presentation.render_presentation_plan(
        ShellState(stdout=rich_plan_output, stderr=_TTYBuffer()),
        root,
        smoke_plan,
        root / "generated" / "presentations" / "smoke-synthetic-run",
        force=False,
    )
    plan_text = strip_ansi(rich_plan_output.getvalue())
    assert "Build plan" in plan_text
    assert "Presentation build" not in plan_text
    logical_live = LongCommandProjection(
        state=ShellState(plain=True, stdout=StringIO(), stderr=StringIO()),
        command="build",
        command_kind="build",
        profile="smoke-synthetic",
        evidence_classification="non-evidentiary",
        output_path="generated/presentations/smoke-synthetic-run",
        evidence_notices=(),
    )
    live_text = StringIO()
    Console(file=live_text, force_terminal=False).print(
        logical_live.live_table()
    )
    assert live_text.getvalue().count("Presentation build") == 1

    json_output = Path("generated/presentations/json-smoke")
    returncode = main(
        [
            "--json",
            "build",
            str(experiments["smoke-synthetic"]),
            "--output",
            str(json_output),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert returncode == 0
    assert captured.err == ""
    assert len(captured.out.strip().splitlines()) == 1
    assert payload["schema"] == "fraud_detection.cli.v1"
    assert payload["command"] == "build"
    assert payload["status"] == "pass"
    assert payload["result"] == "pass"
    assert payload["presentation_status"] == "COMPLETE"
    assert payload["exit_code"] == 0
    assert payload["experiment_path"] == "outputs/smoke-synthetic-run"
    assert payload["presentation_path"] == json_output.as_posix()
    assert payload["profile"] == "smoke-synthetic"
    assert payload["presentation_role"] == "engineering"
    assert payload["evidence_classification"] == "non-evidentiary"
    assert payload["prepared_data_count"] == 2
    assert payload["figure_count"] == 1
    assert payload["table_count"] == 1
    assert payload["phase_counts"]["figures"] == {
        "completed": 1,
        "total": 1,
    }
    assert payload["phase_counts"]["tables"] == {
        "completed": 1,
        "total": 1,
    }
    assert payload["event_summary"]["table-preview"] == 1
    assert payload["warnings"] == [
        "Deterministic synthetic engineering data; not thesis evidence; "
        "not comparable with canonical empirical results."
    ]
    assert payload["suggested_command"].startswith("fraud-detection inspect ")
    assert set(payload["manifest_paths"]) == {
        "selection",
        "data",
        "figures",
        "tables",
    }
    assert payload["events"][0]["payload"]["fields"]["phase"] == "data"
    assert "\x1b[" not in captured.out

    existing_output = root / "generated" / "presentations" / "existing"
    existing_output.mkdir(parents=True)
    sentinel = existing_output / "keep.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    before = len(observed)
    rejected = main(
        [
            "--plain",
            "build",
            str(experiments["mini-real"]),
            "--output",
            "generated/presentations/existing",
        ]
    )
    captured = capsys.readouterr()
    assert rejected == 1
    assert len(observed) == before
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert "FD-PRESENTATION-OUTPUT-EXISTS" in captured.err

    forced = main(
        [
            "--plain",
            "build",
            str(experiments["mini-real"]),
            "--output",
            "generated/presentations/existing",
            "--force",
        ]
    )
    captured = capsys.readouterr()
    assert forced == 0
    assert len(observed) == before + 1
    assert observed[-1].force is True
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"

    before = len(observed)
    unsafe = main(
        [
            "--plain",
            "build",
            str(experiments["canonical"]),
            "--output",
            "src/unsafe",
            "--force",
        ]
    )
    captured = capsys.readouterr()
    assert unsafe == 1
    assert len(observed) == before
    assert "FD-OUTPUT-UNSAFE" in captured.err

    legacy = main(
        [
            "--plain",
            "presentation",
            "all",
            "--experiment-root",
            str(experiments["smoke-synthetic"]),
            "--output-dir",
            "generated/presentations/legacy",
        ]
    )
    captured = capsys.readouterr()
    assert legacy == 2
    assert len(observed) == before
    assert captured.out == ""
    assert "No such command" in captured.err

    rich_plan = plans[experiments["smoke-synthetic"]]
    rich_config = PresentationConfig(
        repository_root=root,
        experiment_root=experiments["smoke-synthetic"],
        output_root=root / "generated" / "presentations" / "rich",
    )
    for no_color in (False, True):
        rich_stdout = _TTYBuffer()
        rich_stderr = _TTYBuffer()
        rich_state = ShellState(
            stdout=rich_stdout,
            stderr=rich_stderr,
            no_color=no_color,
        )
        result_code, result, details = cli_presentation.execute_presentation(
            rich_config,
            rich_state,
            rich_plan,
        )
        cli_presentation.render_presentation_result(
            rich_state,
            result_code,
            result,
            details,
            rich_plan,
        )
        rich_text = rich_stdout.getvalue()
        assert result_code == 0
        assert "Presentation build" in rich_text
        assert "Figures" in rich_text
        assert "1/1" in rich_text
        assert "ETA" not in rich_text
        if no_color:
            assert "\x1b[" not in rich_text
            assert "\u2713" in rich_text
        else:
            assert "\x1b[" in rich_text


def test_presentation_stage_error_uses_product_error_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path.resolve()
    build_calls: list[PresentationConfig] = []

    def unexpected_build(config: PresentationConfig) -> PresentationResult:
        build_calls.append(config)
        pytest.fail("invalid input must fail before presentation dispatch")

    monkeypatch.setattr(cli_app, "find_repository_root", lambda: root)
    monkeypatch.setattr(
        cli_presentation,
        "build_presentation",
        unexpected_build,
    )

    missing = root / "outputs" / "missing"
    file_path = root / "outputs" / "run.txt"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("not a run\n", encoding="utf-8")
    arbitrary = root / "arbitrary"
    arbitrary.mkdir()
    presentation_root = root / "generated" / "existing-presentation"
    presentation_root.mkdir(parents=True)
    input_cases = (
        (missing, "FD-EXPERIMENT-PATH", "does not exist"),
        (file_path, "FD-EXPERIMENT-PATH", "provided path is a file"),
        (arbitrary, "FD-EXPERIMENT-MANIFEST", "Expected RUN_MANIFEST.json"),
        (
            presentation_root,
            "FD-EXPERIMENT-MANIFEST",
            "Expected RUN_MANIFEST.json",
        ),
        (root, "FD-EXPERIMENT-MANIFEST", "Expected RUN_MANIFEST.json"),
    )
    for experiment_path, code, message in input_cases:
        returncode = main(["--plain", "build", str(experiment_path)])
        captured = capsys.readouterr()

        assert returncode == 1
        assert build_calls == []
        assert code in captured.err
        assert message in captured.err
        assert "Expected a completed experiment-run root" in captured.err
        assert "fraud-detection build outputs/canonical-final" in captured.err
        assert "Traceback" not in captured.err

    incomplete_root = _write_cli_run_manifest(
        root / "outputs" / "incomplete",
        "canonical",
        status="FAILED",
        completed_phases=("preflight", "inner_selection"),
    )
    incomplete_output = (
        root / "generated" / "presentations" / incomplete_root.name
    )
    incomplete = main(["--plain", "build", str(incomplete_root)])
    captured = capsys.readouterr()

    assert incomplete == 1
    assert build_calls == []
    assert not incomplete_output.exists()
    assert "ERROR FD-RUN-INCOMPLETE" in captured.err
    assert "run is incomplete" in captured.err
    assert "presentation was not started" in captured.err
    assert "no presentation output was created or cleared" in captured.err
    assert "COMPLETE RUN_MANIFEST.json" in captured.err
    assert "Fit-level resume is not supported" in captured.err
    assert "new experiment output directory is required" in captured.err
    assert "completed_phases=['preflight', 'inner_selection']" in captured.err
    assert "missing_phases=" in captured.err
    assert "Traceback" not in captured.err

    incomplete_json = main(["--json", "build", str(incomplete_root)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert incomplete_json == 1
    assert captured.err == ""
    assert len(captured.out.strip().splitlines()) == 1
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["error"]["code"] == "FD-RUN-INCOMPLETE"
    assert payload["error"]["details"]["experiment_status"] == "FAILED"
    assert payload["error"]["details"]["completed_phases"] == [
        "preflight",
        "inner_selection",
    ]
    assert not incomplete_output.exists()

    invalid_root = _write_cli_run_manifest(
        root / "outputs" / "invalid",
        "canonical",
        schema="invalid",
    )
    invalid = main(["--plain", "build", str(invalid_root)])
    captured = capsys.readouterr()
    assert invalid == 1
    assert build_calls == []
    assert "ERROR FD-PRESENTATION-INPUT" in captured.err
    assert "invalid or mismatched" in captured.err
    assert "schema is invalid" not in captured.err
    assert "Traceback" not in captured.err

    complete_root = _write_cli_run_manifest(
        root / "outputs" / "complete",
        "canonical",
    )
    plan = _presentation_plan(complete_root, "canonical")
    monkeypatch.setattr(
        cli_app,
        "inspect_presentation_input",
        lambda _path: plan,
    )
    output_root = root / "generated" / "presentations" / "failure"
    completed = PresentationStepResult(
        step="data",
        manifest_path=output_root / "data" / "PRESENTATION_DATA_MANIFEST.json",
        manifest={"status": "PASS"},
        elapsed_seconds=0.1,
    )

    def fail_build(config: PresentationConfig) -> PresentationResult:
        build_calls.append(config)
        assert config.event_sink is not None
        config.event_sink(
            "status",
            {
                "level": "PASS",
                "event": "build-data-complete",
                "fields": {"frames": 29},
            },
        )
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "render-figures-start",
                "fields": {},
            },
        )
        cause = RuntimeError("synthetic figure failure")
        raise PresentationError(
            failed_step="figures",
            completed_steps=(completed,),
            original_exception_type="RuntimeError",
            original_message=str(cause),
        ) from cause

    def fail_subprocess(*_args: object, **_kwargs: object) -> None:
        pytest.fail("presentation execution must not start a subprocess")

    monkeypatch.setattr(cli_presentation, "build_presentation", fail_build)
    monkeypatch.setattr(subprocess, "run", fail_subprocess)
    monkeypatch.setattr(subprocess, "Popen", fail_subprocess)

    returncode = main(
        [
            "--plain",
            "build",
            str(complete_root),
            "--output",
            "generated/presentations/failure",
        ]
    )
    captured = capsys.readouterr()

    assert returncode == 1
    assert "BUILD_PLAN" in captured.out
    assert "ERROR FD-PRESENTATION-BUILD" in captured.err
    assert "DETAIL failed_step=figures" in captured.err
    assert "DETAIL completed_steps=['data']" in captured.err
    assert "exception_type" not in captured.err
    assert "synthetic figure failure" not in captured.err
    assert "tables" not in captured.err
    assert "Traceback" not in captured.err

    failed_json = main(
        [
            "--json",
            "build",
            str(complete_root),
            "--output",
            "generated/presentations/failure-json",
        ]
    )
    captured = capsys.readouterr()
    failure_payload = json.loads(captured.out)
    assert failed_json == 1
    assert captured.err == ""
    assert len(captured.out.strip().splitlines()) == 1
    assert failure_payload["result"] == "fail"
    assert failure_payload["error_code"] == "FD-PRESENTATION-BUILD"
    assert failure_payload["failed_phase"] == "Figures"
    assert failure_payload["completed_steps"] == ["data"]
    assert failure_payload["event_summary"] == {
        "build-data-complete": 1,
        "render-figures-start": 1,
    }
    assert len(failure_payload["events"]) == 2
    assert failure_payload["source_experiment_unchanged"] is True

    debug_failure = main(
        [
            "--debug",
            "--plain",
            "build",
            str(complete_root),
            "--output",
            "generated/presentations/failure-debug",
        ]
    )
    captured = capsys.readouterr()
    assert debug_failure == 1
    assert "Traceback" in captured.err
    assert "PresentationError" in captured.err

    def interrupt_build(config: PresentationConfig) -> PresentationResult:
        assert config.event_sink is not None
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "render-tables-start",
                "fields": {},
            },
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(
        cli_presentation,
        "build_presentation",
        interrupt_build,
    )
    interrupted = main(
        [
            "--plain",
            "build",
            str(complete_root),
            "--output",
            "generated/presentations/interrupted",
        ]
    )
    captured = capsys.readouterr()
    assert interrupted == 130
    assert "ERROR FD-INTERRUPTED" in captured.err
    assert "Presentation build interrupted" in captured.err
    assert "source_experiment_unchanged=True" in captured.err
    assert "partial_output_possible=True" in captured.err
    assert "child-process" not in captured.err
    assert "Traceback" not in captured.err

    interrupted_json = main(
        [
            "--json",
            "build",
            str(complete_root),
            "--output",
            "generated/presentations/interrupted-json",
        ]
    )
    captured = capsys.readouterr()
    interruption_payload = json.loads(captured.out)
    assert interrupted_json == 130
    assert captured.err == ""
    assert interruption_payload["result"] == "interrupted"
    assert interruption_payload["exit_code"] == 130
    assert interruption_payload["event_summary"] == {
        "render-tables-start": 1
    }


def test_experiment_run_dispatches_direct_api_with_cli_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path.resolve()
    observed: list[ExperimentConfig] = []
    planning_calls: list[dict[str, object]] = []

    def fake_plan(**kwargs: object) -> ExperimentPlan:
        planning_calls.append(kwargs)
        profile = str(kwargs["profile"])
        output = Path(str(kwargs["output_dir"]))
        if not output.is_absolute():
            output = root / output
        data = Path(str(kwargs["data_path"]))
        if not data.is_absolute():
            data = root / data
        return ExperimentPlan(
            repository_root=root,
            config=ExperimentConfig(
                data_path=data,
                output_root=output,
                phase="all",
                profile=profile,
            ),
            data_status=(
                "synthetic" if profile == "smoke-synthetic" else "verified"
            ),
            actual_data_sha256=(
                None if profile == "smoke-synthetic" else "synthetic"
            ),
            prerequisite_errors=(),
        )

    def fake_run_experiment(config: ExperimentConfig) -> ExperimentResult:
        observed.append(config)
        assert config.event_sink is not None
        task_total = len(config.effective_config.seeds) * (
            config.effective_config.inner_folds
            * (config.effective_config.bce_oof_folds + 1)
        )
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "phase-start",
                "fields": {"phase": "inner"},
            },
        )
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "inner-bce-complete",
                "fields": {"completed": 1, "total": task_total},
            },
        )
        config.event_sink(
            "status",
            {
                "level": "PASS",
                "event": "phase-complete",
                "fields": {"phase": "inner"},
            },
        )
        config.event_sink("log", {"message": "synthetic progress"})
        return ExperimentResult(
            output_root=config.output_root,
            requested_phase="all",
            status="COMPLETE",
            completed_phases=("inner", "final", "qa"),
        )

    def fail_subprocess(*_args: object, **_kwargs: object) -> None:
        pytest.fail("experiment execution must not start a subprocess")

    monkeypatch.setattr(
        cli_app,
        "build_experiment_plan",
        fake_plan,
    )
    monkeypatch.setattr(
        cli_experiment,
        "run_experiment",
        fake_run_experiment,
    )
    monkeypatch.setattr(subprocess, "run", fail_subprocess)
    monkeypatch.setattr(subprocess, "Popen", fail_subprocess)

    dry_output = root / "generated" / "runs" / "smoke-synthetic"
    returncode = main(
        ["--plain", "run", "--profile", "smoke-synthetic", "--dry-run"]
    )
    captured = capsys.readouterr()
    assert returncode == 0
    assert observed == []
    assert planning_calls[-1]["inspect_data"] is False
    assert planning_calls[-1]["output_dir"] == Path(
        "generated/runs/smoke-synthetic"
    )
    assert not dry_output.exists()
    assert "PLAN data_path=not_applicable" in captured.out
    assert "Not thesis evidence" in captured.out

    expected_outputs = {
        "smoke-synthetic": root / "generated" / "runs" / "smoke-synthetic",
        "mini-real": root / "generated" / "runs" / "mini-real",
        "canonical": root / "outputs" / "canonical-final",
    }
    for profile, expected_output in expected_outputs.items():
        arguments = ["--plain", "run", "--profile", profile]
        if profile != "smoke-synthetic":
            arguments.append("--yes")
        returncode = main(arguments)
        captured = capsys.readouterr()

        assert returncode == 0
        assert observed[-1].profile == profile
        assert observed[-1].phase == "all"
        assert observed[-1].output_root == expected_output
        assert observed[-1].event_sink is not None
        assert "PHASE command=run status=START name=\"Inner validation\"" in (
            captured.out
        )
        assert "PROGRESS command=run" in captured.out
        assert "operation=\"BCE fits\"" in captured.out
        expected_bce_total = (
            len(observed[-1].effective_config.seeds)
            * observed[-1].effective_config.inner_folds
            * (observed[-1].effective_config.bce_oof_folds + 1)
        )
        assert f"total={expected_bce_total}" in captured.out
        assert "synthetic progress" not in captured.out
        assert "RESULT command=run result=pass exit_code=0" in captured.out
        assert f"NEXT fraud-detection build {expected_output.relative_to(root).as_posix()}" in captured.out
        assert "ETA" not in captured.out
        assert "\x1b[" not in captured.out
        if profile == "canonical":
            assert "Not thesis evidence" not in captured.out
        else:
            assert captured.out.count("Not thesis evidence") >= 2

    before_legacy = len(observed)
    legacy_returncode = main(
        [
            "--plain",
            "experiment",
            "run",
            "--output-dir",
            "outputs/legacy",
            "--yes",
        ]
    )
    legacy = capsys.readouterr()
    assert legacy_returncode == 2
    assert len(observed) == before_legacy
    assert legacy.out == ""
    assert "No such command" in legacy.err

    def interrupt(config: ExperimentConfig) -> ExperimentResult:
        assert config.event_sink is not None
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "phase-start",
                "fields": {"phase": "final"},
            },
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_experiment, "run_experiment", interrupt)
    interrupted = main(
        [
            "--plain",
            "run",
            "--profile",
            "smoke-synthetic",
            "--output",
            "generated/runs/interrupted",
        ]
    )
    captured = capsys.readouterr()
    assert interrupted == 130
    assert "ERROR FD-INTERRUPTED" in captured.err
    assert "Experiment interrupted" in captured.err
    assert "partial_output_possible=True" in captured.err
    assert "fit_level_resume=unsupported" in captured.err
    assert "FD-RUNNER-CHILD" not in captured.err
    assert "child" not in captured.err.lower()

    monkeypatch.setattr(cli_experiment, "run_experiment", fake_run_experiment)
    automatic_plain = main(
        [
            "run",
            "--profile",
            "smoke-synthetic",
            "--output",
            "generated/runs/automatic-plain",
        ]
    )
    captured = capsys.readouterr()
    assert automatic_plain == 0
    assert "START command=run" in captured.out
    assert "RESULT command=run result=pass" in captured.out
    assert "\x1b[" not in captured.out + captured.err


def test_experiment_run_still_requires_confirmation_without_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path.resolve()
    called = False
    planning_calls: list[dict[str, object]] = []

    def fake_plan(**kwargs: object) -> ExperimentPlan:
        planning_calls.append(kwargs)
        output = Path(str(kwargs["output_dir"]))
        if not output.is_absolute():
            output = root / output
        errors: tuple[ProductError, ...] = ()
        if output.name == "existing":
            errors = (
                ProductError(
                    "FD-OUTPUT-EXISTS",
                    "A complete run requires a new output directory.",
                    ("Choose --output with a new path.",),
                ),
            )
        return _experiment_plan(
            tmp_path,
            profile=str(kwargs["profile"]),
            output=output,
            prerequisite_errors=errors,
        )

    def unexpected_run(
        _plan: ExperimentPlan,
        _state: ShellState,
    ) -> tuple[int, dict[str, object]]:
        nonlocal called
        called = True
        return 0, {}

    monkeypatch.setattr(
        cli_app,
        "build_experiment_plan",
        fake_plan,
    )
    monkeypatch.setattr(cli_app, "execute_experiment", unexpected_run)

    for profile in ("mini-real", "canonical"):
        returncode = main(["--plain", "run", "--profile", profile])
        captured = capsys.readouterr()

        assert returncode == 1
        assert not called
        assert planning_calls[-1]["inspect_data"] is False
        assert "ERROR FD-CONFIRMATION-REQUIRED" in captured.err
        assert "Traceback" not in captured.err

    existing = main(
        [
            "--plain",
            "run",
            "--profile",
            "canonical",
            "--output",
            "outputs/existing",
            "--yes",
        ]
    )
    captured = capsys.readouterr()
    assert existing == 1
    assert not called
    assert "ERROR FD-OUTPUT-EXISTS" in captured.err

    prompts: list[str] = []

    def accept_default(text: str, **kwargs: object) -> object:
        prompts.append(text)
        return kwargs["default"]

    monkeypatch.setattr(
        cli_app.ShellState,
        "interactive_input",
        property(lambda _state: True),
    )
    monkeypatch.setattr(cli_app.typer, "prompt", accept_default)
    interactive = main(["--plain", "run", "--dry-run"])
    captured = capsys.readouterr()

    assert interactive == 0
    assert prompts == ["Select experiment profile"]
    assert planning_calls[-1]["profile"] == "smoke-synthetic"
    assert "recommended=true" in captured.out
    assert not called


def test_experiment_json_collects_structured_ansi_free_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _experiment_plan(tmp_path)
    assert cli_experiment.run_experiment is run_experiment
    stdout = StringIO()
    state = ShellState(
        json_output=True,
        stdout=stdout,
        stderr=StringIO(),
    )

    def fake_run_experiment(config: ExperimentConfig) -> ExperimentResult:
        assert config.event_sink is not None
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "phase-start",
                "fields": {"phase": "\x1b[31minner\x1b[0m"},
            },
        )
        config.event_sink(
            "log",
            {"message": "[2026-08-02T12:00:00Z] inner BCE seed=13 fold=2"},
        )
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "inner-bce-complete",
                "fields": {"completed": 2, "total": 7},
            },
        )
        config.event_sink(
            "status",
            {
                "level": "PASS",
                "event": "phase-complete",
                "fields": {"phase": "inner"},
            },
        )
        print(
            "[2026-08-02T12:00:01Z] final models seed=42 "
            "k=50 gain=linear trees=3"
        )
        sys.stdout.write("repository-owned diagnostic ")
        sys.stdout.write("context")
        sys.stdout.flush()
        return ExperimentResult(
            output_root=config.output_root,
            requested_phase="all",
            status="COMPLETE",
            completed_phases=("inner", "final", "qa"),
        )

    monkeypatch.setattr(
        cli_experiment,
        "run_experiment",
        fake_run_experiment,
    )

    returncode, details = cli_experiment.execute_experiment(plan, state)
    cli_experiment.render_run_result(
        state,
        plan,
        returncode,
        details,
    )
    payload = json.loads(stdout.getvalue())

    assert payload["result"] == "pass"
    assert payload["events"][0] == {
        "kind": "status",
        "payload": {
            "level": "INFO",
            "event": "phase-start",
            "fields": {"phase": "inner"},
        },
    }
    assert payload["phase_counts"]["inner_bce_fits"] == {
        "completed": 7,
        "total": 7,
    }
    assert payload["event_summary"] == {
        "inner-bce-complete": 1,
        "phase-complete": 1,
        "phase-start": 1,
    }
    assert payload["completed_phases"] == ["inner", "final", "qa"]
    assert payload["qa_status"] == "PASS"
    assert payload["suggested_command"].startswith("fraud-detection build ")
    assert payload["events"][-2]["payload"]["message"].endswith(
        "final models seed=42 k=50 gain=linear trees=3"
    )
    assert payload["events"][-1] == {
        "kind": "log",
        "payload": {"message": "repository-owned diagnostic context"},
    }
    assert len(stdout.getvalue().strip().splitlines()) == 1
    assert "ETA" not in stdout.getvalue()
    assert "435" not in stdout.getvalue()
    assert "\x1b[" not in stdout.getvalue()

    rich_result = _TTYBuffer()
    cli_experiment.render_run_result(
        ShellState(stdout=rich_result, stderr=_TTYBuffer()),
        plan,
        returncode,
        details,
    )
    assert "Manifest" in rich_result.getvalue()
    assert "outputs/run/RUN_MANIFEST.json" in rich_result.getvalue()
    assert "{'run':" not in rich_result.getvalue()

    deferred_plan = ExperimentPlan(
        repository_root=plan.repository_root,
        config=plan.config,
        data_status="not_inspected",
        actual_data_sha256=None,
        prerequisite_errors=(),
    )
    rich_plan = _TTYBuffer()
    cli_experiment.render_experiment_plan(
        ShellState(stdout=rich_plan, stderr=_TTYBuffer()),
        deferred_plan,
    )
    assert (
        "validation deferred to preflight · data/creditcard.csv"
        in strip_ansi(rich_plan.getvalue())
    )

    failure_stdout = StringIO()
    failure_stderr = StringIO()
    failure_state = ShellState(
        json_output=True,
        stdout=failure_stdout,
        stderr=failure_stderr,
    )

    def fail_after_events(config: ExperimentConfig) -> ExperimentResult:
        assert config.event_sink is not None
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "phase-start",
                "fields": {"phase": "final"},
            },
        )
        config.event_sink(
            "status",
            {
                "level": "INFO",
                "event": "final-ranker-complete",
                "fields": {"completed": 3, "total": 8},
            },
        )
        raise RuntimeError("controlled scientific boundary failure")

    monkeypatch.setattr(cli_experiment, "run_experiment", fail_after_events)
    returncode, details = cli_experiment.execute_experiment(plan, failure_state)
    cli_experiment.render_run_result(
        failure_state,
        plan,
        returncode,
        details,
    )
    failure_payload = json.loads(failure_stdout.getvalue())

    assert returncode == 1
    assert failure_stderr.getvalue() == ""
    assert failure_payload["result"] == "fail"
    assert failure_payload["error_code"] == "FD-EXPERIMENT-EXECUTION"
    assert failure_payload["failed_phase"] == "Final execution"
    assert failure_payload["failed_operation"] == "ranker fits"
    assert failure_payload["event_summary"] == {
        "final-ranker-complete": 1,
        "phase-start": 1,
    }
    assert len(failure_payload["events"]) == 2
    assert failure_payload["fit_level_resume"] == "unsupported"
    assert len(failure_stdout.getvalue().strip().splitlines()) == 1
    assert "controlled scientific boundary failure" not in (
        failure_stdout.getvalue()
    )

    human_stdout = StringIO()
    human_stderr = StringIO()
    cli_experiment.render_run_result(
        ShellState(plain=True, stdout=human_stdout, stderr=human_stderr),
        plan,
        returncode,
        details,
    )
    assert "ERROR FD-EXPERIMENT-EXECUTION" in human_stderr.getvalue()
    assert "failed_phase=Final execution" in human_stderr.getvalue()
    assert "Fit-level resume is unsupported" in human_stderr.getvalue()
    assert "Traceback" not in human_stderr.getvalue()

    debug_stdout = StringIO()
    debug_stderr = StringIO()
    cli_experiment.render_run_result(
        ShellState(
            plain=True,
            debug=True,
            stdout=debug_stdout,
            stderr=debug_stderr,
        ),
        plan,
        returncode,
        details,
    )
    assert "Traceback" in debug_stderr.getvalue()
    assert "RuntimeError" in debug_stderr.getvalue()

    clock_value = [10.0]
    rich_stdout = _TTYBuffer()
    rich_stderr = _TTYBuffer()
    projection = LongCommandProjection(
        state=ShellState(stdout=rich_stdout, stderr=rich_stderr),
        command="run",
        command_kind="run",
        profile="smoke-synthetic",
        evidence_classification="non-evidentiary",
        output_path="generated/runs/smoke-synthetic",
        evidence_notices=("Not thesis evidence.",),
        expected_counts={
            "inner_bce_fits": 90,
            "inner_ranker_fits": 210,
            "selection_freeze_configurations": 35,
            "final_bce_fits": 30,
            "final_ranker_fits": 70,
        },
        seeds=(42, 7, 13, 123, 202),
        budgets=(5, 10, 20, 50, 100, 200, 500),
        inner_fold_total=3,
        clock=lambda: clock_value[0],
    )
    projection.start()
    projection(
        "log",
        {"message": "[2026-08-02T12:00:00Z] inner BCE seed=13 fold=2"},
    )
    assert projection.current_phase == "inner"
    assert projection.current_operation == "BCE"
    assert projection.current_seed == 13
    assert projection.current_inner_fold == 2
    clock_value[0] = 10.5
    projection(
        "log",
        {
            "message": (
                "[2026-08-02T12:00:01Z] inner ranker seed=13 "
                "fold=2 k=100 gain=linear"
            )
        },
    )
    assert projection.current_operation == "Amount-Gain ranker"
    assert projection.current_budget == 100
    assert projection.current_gain == "linear"
    projection(
        "status",
        {
            "level": "INFO",
            "event": "inner-ranker-complete",
            "fields": {
                "seed": 13,
                "inner_fold": 2,
                "budget": 100,
                "gain": "linear",
                "completed": 103,
                "total": 210,
            },
        },
    )
    assert projection.latest_completed_operation == (
        "Amount-Gain ranker complete · seed 13 · fold 2 · "
        "k=100 · gain=linear · 103/210"
    )
    projection(
        "status",
        {
            "level": "INFO",
            "event": "selection-freeze-config-complete",
            "fields": {
                "seed": 13,
                "budget": 100,
                "completed": 5,
                "total": 35,
            },
        },
    )
    assert projection.current_phase == "selection"
    assert projection.current_operation is None
    assert projection.current_inner_fold is None
    assert projection.current_gain is None
    projection(
        "status",
        {
            "level": "PASS",
            "event": "selection-freeze-complete",
            "fields": {"completed": 35, "total": 35},
        },
    )
    assert "Selection Freeze" in projection.completed_phases
    assert projection.latest_completed_operation == "Selection frozen"
    clock_value[0] = 11.0
    projection(
        "log",
        {
            "message": (
                "[2026-08-02T12:00:02Z] final models seed=42 "
                "k=50 gain=linear trees=3"
            )
        },
    )
    assert projection.current_phase == "final"
    assert projection.current_operation == "final rankers"
    assert projection.current_seed == 42
    assert projection.current_budget == 50
    assert projection.current_tree_count == 3
    projection(
        "status",
        {
            "level": "INFO",
            "event": "final-ranker-complete",
            "fields": {
                "seed": 42,
                "budget": 50,
                "path": "selected_candidate_p_only",
                "completed": 8,
                "total": 70,
            },
        },
    )
    assert projection.latest_completed_operation == (
        "p-only final ranker complete · seed 42 · k=50 · 8/70"
    )
    assert projection.current_path is None
    projection.state.verbose = True
    projection("log", {"message": "verbose context above live"})
    projection.state.verbose = False
    active_render = StringIO()
    clock_value[0] = 11.4
    Console(file=active_render, force_terminal=False, width=100).print(
        projection.live_table()
    )
    active_text = active_render.getvalue()
    assert "Final execution" in active_text
    assert "final rankers" in active_text
    assert "42 (1/5)" in active_text
    assert "50 (4/7)" in active_text
    assert "linear" in active_text
    assert "Trees" in active_text and "3" in active_text
    assert "Running" in active_text and "Last event" in active_text
    projection(
        "status",
        {
            "level": "INFO",
            "event": "aggregation-start",
            "fields": {"phase": "final", "seeds": 5},
        },
    )
    assert projection.current_phase == "aggregation"
    assert projection.current_seed is None
    assert projection.current_budget is None
    projection(
        "status",
        {
            "level": "PASS",
            "event": "aggregation-complete",
            "fields": {"phase": "final"},
        },
    )
    assert "Aggregation" in projection.completed_phases
    assert projection.latest_completed_operation == "Aggregation complete"
    clock_value[0] = 12.5
    projection.mark_success(
        completed=(
            "preflight",
            "inner",
            "selection",
            "final",
            "aggregation",
            "qa",
        )
    )
    projection.close()
    rich_text = rich_stdout.getvalue()
    assert "Experiment run" in rich_text
    assert "Inner validation" in rich_text
    assert "Inner rankers" in rich_text
    assert "verbose context above live" in rich_text
    assert "final models seed=42" not in rich_text
    assert "2.5s" in rich_text
    assert "ETA" not in rich_text

    rendered = StringIO()
    console = Console(file=rendered, force_terminal=False, width=100)
    clock_value[0] = 13.0
    console.print(projection.live_table())
    final_render = rendered.getvalue()
    assert "2.5s" in final_render
    assert "Current" in final_render and "complete" in final_render
    assert "experiment complete" in final_render
    for exact_total in ("90/90", "210/210", "35/35", "30/30", "70/70"):
        assert exact_total in final_render
    assert "waiting for event" not in final_render
    assert "ETA" not in final_render
    assert "%" not in final_render

    plain_stdout = StringIO()
    plain_projection = LongCommandProjection(
        state=ShellState(plain=True, stdout=plain_stdout, stderr=StringIO()),
        command="run",
        command_kind="run",
        profile="smoke-synthetic",
        evidence_classification="non-evidentiary",
        output_path="generated/runs/smoke-synthetic",
        evidence_notices=("Not thesis evidence.",),
        clock=lambda: clock_value[0],
    )
    plain_projection.start()
    lines_before_heartbeat = plain_stdout.getvalue().splitlines()
    clock_value[0] = 20.0
    assert plain_projection.snapshot()["elapsed_seconds"] == 7.0
    assert plain_stdout.getvalue().splitlines() == lines_before_heartbeat

    verbose_stdout = StringIO()
    verbose_projection = LongCommandProjection(
        state=ShellState(
            plain=True,
            verbose=True,
            stdout=verbose_stdout,
            stderr=StringIO(),
        ),
        command="run",
        command_kind="run",
        profile="smoke-synthetic",
        evidence_classification="non-evidentiary",
        output_path="generated/runs/smoke-synthetic",
        evidence_notices=(),
        clock=lambda: clock_value[0],
    )
    recognized_final = (
        "[2026-08-02T12:00:03Z] final models seed=42 "
        "k=50 gain=linear trees=3"
    )
    verbose_projection("log", {"message": recognized_final})
    assert recognized_final not in verbose_stdout.getvalue()
    assert verbose_projection.current_operation == "final rankers"
    verbose_projection("log", {"message": "unrecognized repository log"})
    assert verbose_projection.events[-1]["payload"]["message"] == (
        "unrecognized repository log"
    )
    assert "unrecognized repository log" in verbose_stdout.getvalue()
