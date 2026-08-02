"""Direct presentation API output adapter."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path

from rich.table import Table

from fraud_detection.artifacts import display_path
from fraud_detection.cli.output import (
    JSON_SCHEMA,
    LongCommandProjection,
    ShellState,
    _emit_json,
    _format_status_value,
    render_error,
)
from fraud_detection.errors import ProductError
from fraud_detection.presentation import (
    PresentationConfig,
    PresentationError,
    PresentationResult,
    build_presentation,
)


@dataclass(frozen=True, slots=True)
class PresentationBuildPlan:
    """Validated display contract for one fit-free presentation build."""

    experiment_root: Path
    profile: str
    presentation_role: str
    evidence_classification: str
    data_source_kind: str
    expected_figure_count: int
    expected_table_count: int
    expected_scope: str


class PresentationInputError(RuntimeError):
    """Structured CLI translation of a failed presentation input boundary."""

    def __init__(
        self,
        *,
        original_exception_type: str,
        original_message: str,
        incomplete: bool = False,
        status: str | None = None,
        completed_phases: tuple[str, ...] = (),
        missing_phases: tuple[str, ...] = (),
    ) -> None:
        self.original_exception_type = original_exception_type
        self.original_message = original_message
        self.incomplete = incomplete
        self.status = status
        self.completed_phases = completed_phases
        self.missing_phases = missing_phases
        super().__init__(original_message)


def _completion_details(
    experiment_root: Path,
    expected_phases: tuple[str, ...],
) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    manifest_path = experiment_root / "RUN_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, Mapping):
        return None
    status = manifest.get("status")
    completed = manifest.get("completed_phases")
    if not isinstance(status, str) or not isinstance(completed, list):
        return None
    if not all(isinstance(phase, str) for phase in completed):
        return None
    completed_phases = tuple(completed)
    if status == "COMPLETE" and completed_phases == expected_phases:
        return None
    completed_set = set(completed_phases)
    missing = tuple(
        phase for phase in expected_phases if phase not in completed_set
    )
    return status, completed_phases, missing


def _catalog_counts(registry: object) -> tuple[int, int]:
    if not isinstance(registry, Mapping):
        raise RuntimeError("Presentation catalog is not a mapping.")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Presentation catalog artifact inventory is invalid.")
    figures = 0
    tables = 0
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise RuntimeError("Presentation catalog artifact entry is invalid.")
        input_path = artifact.get("input_data_file")
        if not isinstance(input_path, str):
            raise RuntimeError("Presentation catalog input path is invalid.")
        if "/figures/" in input_path:
            figures += 1
        elif "/tables/" in input_path:
            tables += 1
        else:
            raise RuntimeError("Presentation catalog artifact kind is invalid.")
    return figures, tables


def inspect_presentation_input(
    experiment_root: Path,
) -> PresentationBuildPlan:
    """Use the existing fit-free boundary to validate and summarize one run."""

    data_stage = import_module(
        "fraud_detection.presentation.preparation.data"
    )
    expected_phases = tuple(str(item) for item in data_stage.COMPLETED_PHASES)
    try:
        context = data_stage.load_presentation_input_context(experiment_root)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        completion = _completion_details(experiment_root, expected_phases)
        details = (
            {
                "incomplete": True,
                "status": completion[0],
                "completed_phases": completion[1],
                "missing_phases": completion[2],
            }
            if completion is not None
            else {}
        )
        raise PresentationInputError(
            original_exception_type=type(exc).__name__,
            original_message=str(exc),
            **details,
        ) from exc

    catalog_stage = import_module("fraud_detection.presentation.catalog")
    registry = catalog_stage.build_profile_selection_registry(
        presentation_role=context.presentation_role,
        profile=context.profile,
        evidence_classification=context.evidence_classification,
        data_source_kind=context.data_source_kind,
    )
    figure_count, table_count = _catalog_counts(registry)
    expected_scope = (
        "full canonical thesis catalog"
        if context.presentation_role == "canonical"
        else (
            f"{figure_count} engineering figure and "
            f"{table_count} engineering table"
        )
    )
    return PresentationBuildPlan(
        experiment_root=context.experiment_root,
        profile=str(context.profile),
        presentation_role=str(context.presentation_role),
        evidence_classification=str(context.evidence_classification),
        data_source_kind=str(context.data_source_kind),
        expected_figure_count=figure_count,
        expected_table_count=table_count,
        expected_scope=expected_scope,
    )


def evidence_statement(plan: PresentationBuildPlan) -> str:
    """Return the approved profile-specific presentation boundary."""

    if plan.profile == "smoke-synthetic":
        return (
            "Deterministic synthetic engineering data; not thesis evidence; "
            "not comparable with canonical empirical results."
        )
    if plan.profile == "mini-real":
        return (
            "Engineering mini profile; real canonical dataset; not thesis "
            "evidence; not comparable with canonical empirical results."
        )
    return "Canonical thesis-evidentiary presentation."


def render_presentation_plan(
    state: ShellState,
    repository_root: Path,
    plan: PresentationBuildPlan,
    output_root: Path,
    *,
    force: bool,
) -> None:
    """Display the validated, fit-free build handoff without JSON noise."""

    if state.json_output:
        return
    experiment_path = display_path(repository_root, plan.experiment_root)
    output_path = display_path(repository_root, output_root)
    if state.rich_enabled:
        table = Table(title="Build plan", show_header=False)
        table.add_column("Field", style="bold cyan")
        table.add_column("Value")
        table.add_row("Experiment", experiment_path)
        table.add_row("Profile", plan.profile)
        table.add_row("Experiment status", "COMPLETE")
        table.add_row("Presentation role", plan.presentation_role)
        table.add_row("Evidence", plan.evidence_classification)
        table.add_row("Expected scope", plan.expected_scope)
        table.add_row("Output", output_path)
        table.add_row("Force replacement", str(force).lower())
        state.console().print(table)
        state.console().print(evidence_statement(plan))
        return
    print(
        "BUILD_PLAN "
        f"experiment_path={json.dumps(experiment_path)} "
        f"profile={plan.profile} experiment_status=COMPLETE "
        f"presentation_role={plan.presentation_role} "
        f"evidence_classification={json.dumps(plan.evidence_classification)} "
        f"expected_scope={json.dumps(plan.expected_scope)} "
        f"output_path={json.dumps(output_path)} "
        f"force={str(force).lower()}",
        file=state.stdout,
    )
    print(f"EVIDENCE {evidence_statement(plan)}", file=state.stdout)


def _result_details(
    config: PresentationConfig,
    result: PresentationResult,
    projection: LongCommandProjection,
) -> dict[str, object]:
    step_manifests = {step.step: step.manifest for step in result.steps}
    data_manifest = step_manifests.get("data", {})
    figure_manifest = step_manifests.get("figures", {})
    table_manifest = step_manifests.get("tables", {})
    prepared_outputs = data_manifest.get("outputs", [])
    rendered_stems = figure_manifest.get("rendered_stems", [])
    rendered_tables = table_manifest.get("tables", [])
    prepared_data_count = (
        len(prepared_outputs) if isinstance(prepared_outputs, list) else 0
    )
    figure_count = figure_manifest.get("logical_figure_count")
    if not isinstance(figure_count, int) or isinstance(figure_count, bool):
        figure_count = (
            len(rendered_stems) if isinstance(rendered_stems, list) else 0
        )
    table_count = table_manifest.get("logical_table_count")
    if not isinstance(table_count, int) or isinstance(table_count, bool):
        table_count = (
            len(rendered_tables) if isinstance(rendered_tables, list) else 0
        )
    manifest_paths = {
        "selection": display_path(
            config.repository_root,
            result.output_root / "PRESENTATION_SELECTION.json",
        ),
        **{
            step.step: display_path(
                config.repository_root,
                step.manifest_path,
            )
            for step in result.steps
        },
    }
    details = projection.snapshot()
    details["event_completed_phases"] = details.pop("completed_phases")
    details.update({
        "steps": [
            {
                "step": step.step,
                "status": "PASS",
                "manifest_path": display_path(
                    config.repository_root,
                    step.manifest_path,
                ),
                "elapsed_seconds": step.elapsed_seconds,
            }
            for step in result.steps
        ],
        "output_path": display_path(
            config.repository_root,
            result.output_root,
        ),
        "experiment_path": display_path(
            config.repository_root,
            config.experiment_root,
        ),
        "prepared_data_count": prepared_data_count,
        "figure_count": figure_count,
        "table_count": table_count,
        "manifest_paths": manifest_paths,
        "completed_phases": [step.step for step in result.steps],
        "presentation_status": result.status,
        "result": "pass",
        "suggested_command": (
            "fraud-detection inspect "
            + display_path(config.repository_root, result.output_root)
        ),
    })
    return details


def _optional_preview_summary(result: PresentationResult) -> str | None:
    """Return a human-only summary for an unavailable optional preview."""

    table_step = next(
        (step for step in result.steps if step.step == "tables"),
        None,
    )
    if table_step is None:
        return None
    preview = table_step.manifest.get("latex_preview")
    if not isinstance(preview, Mapping):
        return None
    if preview.get("status") == "SKIPPED_NO_ENGINE":
        return "Optional LaTeX preview — skipped (engine not available)"
    return None


def execute_presentation(
    config: PresentationConfig,
    state: ShellState,
    plan: PresentationBuildPlan,
    *,
    command: str = "build",
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[int, PresentationResult | None, dict[str, object]]:
    """Run the direct presentation API with structured shell events."""

    if not callable(clock):
        raise TypeError("clock must be callable.")
    output_path = display_path(config.repository_root, config.output_root)
    output = LongCommandProjection(
        state=state,
        command=command,
        command_kind="build",
        profile=plan.profile,
        evidence_classification=plan.evidence_classification,
        output_path=output_path,
        evidence_notices=(evidence_statement(plan),),
        expected_counts={
            "figures": plan.expected_figure_count,
            "tables": plan.expected_table_count,
        },
        clock=clock,
    )
    configured = replace(config, event_sink=output)
    output.start()
    try:
        result = build_presentation(configured)
    except KeyboardInterrupt as exc:
        output.mark_failure(interrupted=True)
        details = output.snapshot()
        details.update(
            {
                "result": "interrupted",
                "error_code": "FD-INTERRUPTED",
                "error_message": "Presentation build interrupted.",
                "error_recovery": [
                    "The source experiment remains unchanged.",
                    "Presentation output may be partial; it is not COMPLETE.",
                    "Retry with a new --output path when ready.",
                ],
                "experiment_path": display_path(
                    config.repository_root,
                    config.experiment_root,
                ),
                "output_path": output_path,
                "partial_output_possible": True,
                "source_experiment_unchanged": True,
                "suggested_command": (
                    "fraud-detection build "
                    f"{display_path(config.repository_root, config.experiment_root)} "
                    "--output <NEW_PATH>"
                ),
                "_exception": exc,
            }
        )
        return 130, None, details
    except Exception as exc:
        failed_step = exc.failed_step if isinstance(exc, PresentationError) else None
        output.mark_failure(
            phase=failed_step,
            operation=(f"{failed_step} stage" if failed_step is not None else None),
        )
        details = output.snapshot()
        incomplete = (
            isinstance(exc, PresentationError)
            and exc.failed_step == "data"
            and exc.original_message
            in {
                "RUN_MANIFEST.json status is not COMPLETE.",
                "RUN_MANIFEST.json completed phases are incomplete.",
            }
        )
        if isinstance(exc, ProductError):
            code = exc.code
            message = exc.summary
            recovery = list(exc.recovery)
            exit_code = exc.exit_code
        elif incomplete:
            code = "FD-RUN-INCOMPLETE"
            message = (
                "The experiment run is incomplete; presentation was not started."
            )
            recovery = [
                "A COMPLETE RUN_MANIFEST.json is required.",
                "Fit-level resume is not supported.",
                "Start a new experiment in a new output directory.",
            ]
            exit_code = 1
        else:
            code = "FD-PRESENTATION-BUILD"
            stage = details.get("failed_phase") or "the active stage"
            message = f"Presentation build failed during {stage}."
            recovery = [
                "Correct the reported fit-free presentation failure.",
                "Retry with a new --output path, or use --force consciously.",
            ]
            exit_code = 1
        completed_steps = (
            [step.step for step in exc.completed_steps]
            if isinstance(exc, PresentationError)
            else []
        )
        details.update(
            {
                "result": "fail",
                "error_code": code,
                "error_message": message,
                "error_recovery": recovery,
                "experiment_path": display_path(
                    config.repository_root,
                    config.experiment_root,
                ),
                "output_path": output_path,
                "failed_step": failed_step,
                "completed_steps": completed_steps,
                "partial_output_possible": bool(completed_steps),
                "source_experiment_unchanged": True,
                "suggested_command": (
                    "fraud-detection build "
                    f"{display_path(config.repository_root, config.experiment_root)} "
                    "--output <NEW_PATH>"
                ),
                "_exception": exc,
            }
        )
        return exit_code, None, details
    else:
        if result.status != "COMPLETE":
            raise RuntimeError(
                f"Unexpected presentation status: {result.status!r}."
            )
        output.mark_success(completed=tuple(step.step for step in result.steps))
        return 0, result, _result_details(configured, result, output)
    finally:
        output.close()


def render_presentation_result(
    state: ShellState,
    returncode: int,
    result: PresentationResult | None,
    details: dict[str, object],
    plan: PresentationBuildPlan,
    *,
    command: str = "build",
) -> None:
    """Render one final fit-free presentation result or structured failure."""

    public_details = {
        key: value for key, value in details.items() if not key.startswith("_")
    }
    if state.json_output:
        _emit_json(
            state,
            {
                "schema": JSON_SCHEMA,
                "command": command,
                "status": "pass" if returncode == 0 else "error",
                "result": details["result"],
                "exit_code": returncode,
                "experiment_path": details.get("experiment_path"),
                "presentation_path": details.get("output_path"),
                "profile": plan.profile,
                "presentation_role": plan.presentation_role,
                "evidence_classification": plan.evidence_classification,
                **public_details,
            },
        )
    elif returncode:
        error = ProductError(
            str(details["error_code"]),
            str(details["error_message"]),
            tuple(str(item) for item in details["error_recovery"]),
            {
                "failed_phase": details.get("failed_phase"),
                "failed_step": details.get("failed_step"),
                "failed_operation": details.get("failed_operation"),
                "completed_steps": details.get("completed_steps", []),
                "partial_output_possible": details["partial_output_possible"],
                "source_experiment_unchanged": True,
            },
            returncode,
        )
        render_error(state, command, error)
        print(f"NEXT {details['suggested_command']}", file=state.stderr)
    elif state.rich_enabled:
        if result is None:
            raise TypeError("A successful build requires a presentation result.")
        table = Table(title="Presentation artifacts")
        table.add_column("Step", style="bold cyan")
        table.add_column("Status")
        table.add_column("Manifest")
        for row in details["steps"]:
            assert isinstance(row, dict)
            table.add_row(
                str(row["step"]),
                "[green]PASS[/]",
                str(row["manifest_path"]),
            )
        state.console().print(table)
        state.console().print(
            "[green]Build completed successfully.[/green]"
        )
        state.console().print(
            f"Profile: {plan.profile}; role: {plan.presentation_role}; "
            f"evidence: {plan.evidence_classification}"
        )
        state.console().print(
            f"Prepared data: {details['prepared_data_count']}; "
            f"figures: {details['figure_count']}; "
            f"tables: {details['table_count']}"
        )
        preview_summary = _optional_preview_summary(result)
        if preview_summary is not None:
            state.console().print(preview_summary)
        state.console().print(evidence_statement(plan))
        state.console().print(
            f"Elapsed: {float(details['elapsed_seconds']):.1f}s"
        )
        state.console().print(f"Next: {details['suggested_command']}")
    else:
        print(
            f"RESULT command={command.replace(' ', '_')} result=pass "
            "exit_code=0 presentation_status=COMPLETE "
            f"elapsed_seconds={float(details['elapsed_seconds']):.3f}",
            file=state.stdout,
        )
        print(
            "BUILD_RESULT "
            f"profile={plan.profile} "
            f"presentation_role={plan.presentation_role} "
            f"evidence_classification="
            f"{_format_status_value(plan.evidence_classification)} "
            f"prepared_data_count={details['prepared_data_count']} "
            f"figure_count={details['figure_count']} "
            f"table_count={details['table_count']} "
            f"output_path={json.dumps(str(details['output_path']))}",
            file=state.stdout,
        )
        manifest_paths = details["manifest_paths"]
        assert isinstance(manifest_paths, dict)
        for kind, path in manifest_paths.items():
            print(
                f"MANIFEST kind={kind} path={json.dumps(str(path))}",
                file=state.stdout,
            )
        if result is None:
            raise TypeError("A successful build requires a presentation result.")
        preview_summary = _optional_preview_summary(result)
        if preview_summary is not None:
            print(preview_summary, file=state.stdout)
        print(f"EVIDENCE {evidence_statement(plan)}", file=state.stdout)
        print(
            f"NEXT {details['suggested_command']}",
            file=state.stdout,
        )
    if returncode and state.debug:
        exception = details.get("_exception")
        if isinstance(exception, BaseException):
            traceback.print_exception(exception, file=state.stderr)
