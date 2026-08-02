"""Typer command tree, commands, confirmation, and exception translation."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import typer
from rich.table import Table
from typer._click import exceptions as click_exceptions

from fraud_detection.artifacts import (
    find_repository_root,
    inspect_path,
    safe_output_path,
)
from fraud_detection.cli.artifacts import render_inspection
from fraud_detection.cli.experiment import (
    execute_experiment,
    render_experiment_plan,
    render_run_result,
)
from fraud_detection.cli.output import (
    ShellState,
    project_setup_progress,
    render_check_start,
    render_diagnostic_report,
    render_error,
    render_setup_failure,
    render_setup_plan,
    render_setup_progress,
    render_setup_success,
)
from fraud_detection.cli.presentation import (
    PresentationInputError,
    execute_presentation,
    inspect_presentation_input,
    render_presentation_plan,
    render_presentation_result,
)
from fraud_detection.errors import ProductError
from fraud_detection.experiment import build_experiment_plan
from fraud_detection.experiment.config import (
    EXPERIMENT_PROFILE_NAMES,
    EffectiveExperimentConfig,
    resolve_experiment_profile,
)
from fraud_detection.presentation import PresentationConfig
from fraud_detection.setup import SetupFailure, run_check, run_doctor, run_setup

app = typer.Typer(
    name="fraud-detection",
    help="Diagnose, plan, and safely orchestrate the frozen fraud-detection project.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)

_ACTIVE_STATE: ShellState | None = None


@app.callback()
def root_options(
    ctx: typer.Context,
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Force stable line-oriented text.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit stable ANSI-free JSON.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Include commands and diagnostic details.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show Python tracebacks for failures.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable ANSI color while retaining interactive layout.",
    ),
) -> None:
    """Select global rendering and diagnostics behavior."""

    if plain and json_output:
        raise click_exceptions.UsageError(
            "--plain and --json are mutually exclusive."
        )
    state = ShellState(
        plain=plain,
        json_output=json_output,
        verbose=verbose,
        debug=debug,
        no_color=no_color,
    )
    ctx.ensure_object(dict)
    ctx.obj["state"] = state
    global _ACTIVE_STATE
    _ACTIVE_STATE = state


def _shell_state(ctx: typer.Context) -> ShellState:
    return ctx.find_root().obj["state"]


_RUN_HELP = (
    "Plan and run one complete profile-aware experiment safely. Interactive "
    "terminals show live progress; --plain emits stable text and --json emits "
    "one final document. Heterogeneous work has no linear ETA."
)
_RUN_EXAMPLES = """\
Examples:

  Synthetic dry run: fraud-detection run --profile smoke-synthetic --dry-run
  Synthetic execution: fraud-detection run --profile smoke-synthetic
  Mini-real automation: fraud-detection run --profile mini-real --yes
  Custom output: fraud-detection run --profile canonical --output outputs/canonical-final-2 --yes
"""
_CANONICAL_DATA_PATH = Path("data/creditcard.csv")
_BUILD_HELP = (
    "Build canonical or engineering presentation artifacts from one completed "
    "run manifest without fitting models. Interactive terminals show live "
    "progress; --plain emits stable text and --json emits one final document. "
    "Heterogeneous work has no linear ETA."
)
_BUILD_EXAMPLES = """\
Examples:

  Smoke run: fraud-detection build generated/runs/smoke-synthetic
  Mini-real run: fraud-detection build generated/runs/mini-real
  Custom output: fraud-detection build outputs/canonical-final --output generated/presentations/canonical-copy
  Safe replacement: fraud-detection build outputs/canonical-final --output generated/presentations/canonical-copy --force
"""
_BUILD_EXAMPLE = "fraud-detection build outputs/canonical-final"
_SETUP_HELP = (
    "Prepare or repair the local environment. Setup may install pinned packages "
    "and acquire, reuse, or validate the canonical dataset; it never runs an "
    "experiment."
)
_SETUP_EXAMPLES = """\
Examples:

  Prepare the environment: fraud-detection setup
  Stable text: fraud-detection --plain setup
  Structured output: fraud-detection --json setup
"""
_CHECK_HELP = (
    "Run fast read-only environment diagnostics by default. Use --full for the "
    "bounded integrity/import/help scope and --require-data for canonical dataset "
    "validation. Check performs no repair or model fit."
)
_CHECK_EXAMPLES = """\
Examples:

  Quick check: fraud-detection check
  Full check: fraud-detection check --full
  Require canonical data: fraud-detection check --require-data
  Plain output: fraud-detection --plain check
  JSON output: fraud-detection --json check --full
"""
_INSPECT_HELP = (
    "Read-only semantic inspection of an exact repository, experiment, "
    "partial-run, or presentation root, without searching parent directories. "
    "Completed outputs receive manifest and checksum validation."
)
_INSPECT_EXAMPLES = """\
Examples:

  Repository: fraud-detection inspect .
  Completed experiment: fraud-detection inspect outputs/canonical-final
  Partial experiment: fraud-detection inspect generated/runs/interrupted
  Presentation: fraud-detection inspect generated/presentations/canonical-final
  JSON: fraud-detection --json inspect outputs/canonical-final
"""


def _resolve_cli_profile(profile_name: str) -> EffectiveExperimentConfig:
    try:
        return resolve_experiment_profile(profile_name)
    except ValueError as exc:
        valid = ", ".join(EXPERIMENT_PROFILE_NAMES)
        raise click_exceptions.UsageError(
            f"Unknown profile {profile_name!r}. Valid profiles: {valid}."
        ) from exc


def _ordered_profiles() -> tuple[EffectiveExperimentConfig, ...]:
    profiles = tuple(
        resolve_experiment_profile(name) for name in EXPERIMENT_PROFILE_NAMES
    )
    return tuple(
        sorted(
            profiles,
            key=lambda item: (
                0
                if item.data_source_kind == "synthetic"
                else (
                    2
                    if item.evidence_classification == "thesis-evidentiary"
                    else 1
                )
            ),
        )
    )


def _profile_scope(profile: EffectiveExperimentConfig) -> str:
    return (
        f"{len(profile.seeds)} seed(s), {len(profile.target_budgets)} budget(s), "
        f"OOF folds={profile.bce_oof_folds}, inner folds={profile.inner_folds}, "
        f"candidate pool={profile.candidate_pool_size}"
    )


def _render_profile_choices(
    state: ShellState,
    profiles: tuple[EffectiveExperimentConfig, ...],
) -> None:
    recommended = profiles[0].profile_name
    if state.rich_enabled:
        table = Table(title="Experiment profiles")
        table.add_column("Profile", style="bold cyan")
        table.add_column("Source")
        table.add_column("Evidence role")
        table.add_column("Approximate scope")
        for profile in profiles:
            label = profile.profile_name
            if label == recommended:
                label += " (recommended)"
            table.add_row(
                label,
                profile.data_source_kind,
                profile.evidence_classification,
                _profile_scope(profile),
            )
        state.console().print(table)
        return
    for profile in profiles:
        print(
            f"PROFILE name={profile.profile_name} "
            f"recommended={str(profile.profile_name == recommended).lower()} "
            f"source={profile.data_source_kind} "
            f"evidence={profile.evidence_classification!r} "
            f"scope={_profile_scope(profile)!r}",
            file=state.stdout,
        )


def _select_profile(
    state: ShellState,
    profile_name: str | None,
) -> EffectiveExperimentConfig:
    if profile_name is not None:
        return _resolve_cli_profile(profile_name)
    profiles = _ordered_profiles()
    recommended = profiles[0].profile_name
    if not state.interactive_input:
        raise click_exceptions.UsageError(
            "--profile is required when stdin is not interactive. "
            f"Example: fraud-detection run --profile {recommended} --dry-run"
        )
    _render_profile_choices(state, profiles)
    selected = typer.prompt(
        "Select experiment profile",
        default=recommended,
        show_default=True,
    )
    return _resolve_cli_profile(str(selected))


def _default_output(profile: EffectiveExperimentConfig) -> Path:
    if profile.evidence_classification == "thesis-evidentiary":
        return Path("outputs/canonical-final")
    return Path("generated") / "runs" / profile.profile_name


def _run_handler(
    *,
    state: ShellState,
    profile_name: str | None,
    output: Path | None,
    data: Path | None,
    dry_run: bool,
    yes: bool,
    command: str,
) -> None:
    profile = _select_profile(state, profile_name)
    if profile.data_source_kind == "synthetic" and data is not None:
        raise click_exceptions.UsageError(
            "--data is not valid with the smoke-synthetic profile; synthetic "
            "engineering data is generated by the shared experiment pipeline."
        )
    selected_output = output or _default_output(profile)
    selected_data = data or _CANONICAL_DATA_PATH
    preview = build_experiment_plan(
        data_path=selected_data,
        output_dir=selected_output,
        profile=profile.profile_name,
        inspect_data=False,
    )
    render_experiment_plan(
        state,
        preview,
        emit_json=dry_run or bool(preview.prerequisite_errors),
        command=command,
    )
    if preview.prerequisite_errors:
        raise typer.Exit(1)
    if dry_run:
        return

    if profile.data_source_kind == "real" and not yes:
        if not state.interactive_input:
            raise ProductError(
                "FD-CONFIRMATION-REQUIRED",
                "Non-interactive real-data runs require --yes.",
                (
                    "Review the displayed profile, evidence role, data path, and output.",
                    f"Re-run with --profile {profile.profile_name} --yes.",
                ),
                {"plan": preview.as_dict()} if state.json_output else {},
            )
        if not typer.confirm("Start this real-data experiment?"):
            raise ProductError(
                "FD-RUN-CANCELLED",
                "The real-data experiment was cancelled before data access.",
                ("No recovery is required; experiment execution did not start.",),
            )

    plan = preview
    if profile.data_source_kind == "real":
        plan = build_experiment_plan(
            data_path=selected_data,
            output_dir=selected_output,
            profile=profile.profile_name,
        )
        if plan.prerequisite_errors:
            render_experiment_plan(
                state,
                plan,
                command=command,
                show_plan=False,
            )
            raise typer.Exit(1)

    returncode, details = execute_experiment(plan, state, command=command)
    render_run_result(
        state,
        plan,
        returncode,
        details,
        command=command,
    )
    if returncode:
        raise typer.Exit(returncode)


@app.command("setup", help=_SETUP_HELP, epilog=_SETUP_EXAMPLES)
def setup_command(ctx: typer.Context) -> None:
    """Prepare or repair the local environment without running experiments."""

    state = _shell_state(ctx)
    root = find_repository_root()
    if root is None:
        raise ProductError(
            "FD-ROOT-NOT-FOUND",
            "Repository root was not found.",
            ("Run setup from the repository checkout.",),
        )
    environment_path = root / ".venv"
    dataset_action = (
        "validate existing and reuse if canonical"
        if (root / "data" / "creditcard.csv").is_file()
        else "acquire and validate canonical dataset"
    )
    render_setup_plan(
        state,
        repository_root=root,
        environment_path=environment_path,
        dataset_action=dataset_action,
    )

    def progress(message: str) -> None:
        render_setup_progress(state, project_setup_progress(message))

    try:
        result = run_setup(
            repository_root=root,
            capture_install_output=True,
            progress=progress,
        )
    except SetupFailure as exc:
        render_setup_failure(
            state,
            repository_root=root,
            environment_path=environment_path,
            error=exc,
        )
        raise typer.Exit(exc.exit_code) from exc
    render_setup_success(
        state,
        repository_root=root,
        environment_path=environment_path,
        result=result,
    )


def _check_handler(
    *,
    state: ShellState,
    full: bool,
    require_data: bool,
    command: str,
) -> None:
    root = find_repository_root()
    mode = "full" if full else "quick"
    render_check_start(
        state,
        mode=mode,
        require_data=require_data,
        repository_root=root,
    )
    diagnostic = run_check if full else run_doctor
    try:
        report = diagnostic(
            repository_root=root,
            require_data=require_data,
        )
    except KeyboardInterrupt as exc:
        raise ProductError(
            "FD-INTERRUPTED",
            "Environment check interrupted by Ctrl-C.",
            ("Re-run the read-only check when ready.",),
            exit_code=130,
        ) from exc
    render_diagnostic_report(
        state,
        command=command,
        mode=mode,
        require_data=require_data,
        report=report,
    )
    if report.exit_code:
        raise typer.Exit(report.exit_code)


@app.command("check", help=_CHECK_HELP, epilog=_CHECK_EXAMPLES)
def check_command(
    ctx: typer.Context,
    full: bool = typer.Option(
        False,
        "--full",
        help="Run bounded syntax, import, integrity, and public-help checks.",
    ),
    require_data: bool = typer.Option(
        False,
        "--require-data",
        help="Require and validate the canonical local dataset identity.",
    ),
) -> None:
    """Run quick or full read-only diagnostics without repair or fitting."""

    _check_handler(
        state=_shell_state(ctx),
        full=full,
        require_data=require_data,
        command="check",
    )


@app.command("run", help=_RUN_HELP, epilog=_RUN_EXAMPLES)
def run_command(
    ctx: typer.Context,
    profile: str | None = typer.Option(
        None,
        "--profile",
        help=(
            "Experiment profile: "
            + ", ".join(
                item.profile_name for item in _ordered_profiles()
            )
            + "; required when stdin is not interactive."
        ),
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="New experiment output root; never overwritten or reused.",
    ),
    data: Path | None = typer.Option(
        None,
        "--data",
        help="Real-data CSV; invalid for smoke-synthetic.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the pure plan without dataset access or execution.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm a real-data run non-interactively.",
    ),
) -> None:
    """Plan and run one complete profile-aware experiment safely."""

    _run_handler(
        state=_shell_state(ctx),
        profile_name=profile,
        output=output,
        data=data,
        dry_run=dry_run,
        yes=yes,
        command="run",
    )


def _incomplete_run_error(
    error: PresentationInputError | None = None,
) -> ProductError:
    details: dict[str, object] = {}
    if error is not None:
        if error.status is not None:
            details["experiment_status"] = error.status
        if error.completed_phases:
            details["completed_phases"] = list(error.completed_phases)
        if error.missing_phases:
            details["missing_phases"] = list(error.missing_phases)
    return ProductError(
        "FD-RUN-INCOMPLETE",
        (
            "The experiment run is incomplete; presentation was not started "
            "and no presentation output was created or cleared."
        ),
        (
            "A COMPLETE RUN_MANIFEST.json with all semantic phases is required.",
            "Fit-level resume is not supported.",
            (
                "A new experiment output directory is required for a new run; "
                "build only after that run completes."
            ),
        ),
        details,
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _build_handler(
    *,
    state: ShellState,
    experiment_path: Path,
    output: Path | None,
    force: bool,
    command: str,
) -> None:
    root = find_repository_root()
    if root is None:
        raise ProductError(
            "FD-ROOT-NOT-FOUND",
            "Repository root was not found.",
            ("Run the command from the repository checkout.",),
        )
    root = root.resolve()
    resolved_experiment = (
        experiment_path
        if experiment_path.is_absolute()
        else root / experiment_path
    ).resolve()
    input_recovery = (
        "Expected a completed experiment-run root.",
        f"Example: {_BUILD_EXAMPLE}",
    )
    if not resolved_experiment.exists():
        raise ProductError(
            "FD-EXPERIMENT-PATH",
            (
                "Expected a completed experiment-run root, but the provided "
                "path does not exist."
            ),
            input_recovery,
            {"experiment_path": str(resolved_experiment)},
        )
    if not resolved_experiment.is_dir():
        raise ProductError(
            "FD-EXPERIMENT-PATH",
            (
                "Expected a completed experiment-run root directory; the "
                "provided path is a file."
            ),
            input_recovery,
            {"experiment_path": str(resolved_experiment)},
        )
    if not (resolved_experiment / "RUN_MANIFEST.json").is_file():
        raise ProductError(
            "FD-EXPERIMENT-MANIFEST",
            (
                "Expected RUN_MANIFEST.json at the provided completed "
                "experiment-run root."
            ),
            input_recovery,
            {"experiment_path": str(resolved_experiment)},
        )

    try:
        plan = inspect_presentation_input(resolved_experiment)
    except PresentationInputError as exc:
        if exc.incomplete:
            raise _incomplete_run_error(exc) from exc
        details = (
            {
                "exception_type": exc.original_exception_type,
                "reason": exc.original_message,
            }
            if state.debug
            else {}
        )
        raise ProductError(
            "FD-PRESENTATION-INPUT",
            (
                "The experiment run manifest or registered artifacts are "
                "invalid or mismatched; presentation was not started and no "
                "presentation output was created or cleared."
            ),
            (
                "Provide the root of one valid completed experiment run.",
                f"Example: {_BUILD_EXAMPLE}",
            ),
            details,
        ) from exc

    selected_output = output or (
        Path("generated") / "presentations" / resolved_experiment.name
    )
    resolved_output = safe_output_path(root, selected_output, "generated")
    if _paths_overlap(resolved_experiment, resolved_output):
        raise ProductError(
            "FD-OUTPUT-UNSAFE",
            "Presentation output must be disjoint from the experiment input root.",
            (
                "Choose a new child path below generated/presentations/.",
                "Never use the experiment run or one of its parents as output.",
            ),
            {"path": str(resolved_output)},
        )
    if resolved_output.exists() and (
        not force or not resolved_output.is_dir()
    ):
        recovery = (
            "Choose --output with a new path.",
            "Use --force only for conscious replacement of this safe directory.",
        )
        summary = (
            "Presentation output already exists and was not cleared or reused."
            if not force
            else "The existing presentation output path is not a directory."
        )
        raise ProductError(
            "FD-PRESENTATION-OUTPUT-EXISTS",
            summary,
            recovery,
            {"presentation_path": str(resolved_output)},
        )

    render_presentation_plan(
        state,
        root,
        plan,
        resolved_output,
        force=force,
    )
    config = PresentationConfig(
        repository_root=root,
        experiment_root=resolved_experiment,
        output_root=resolved_output,
        width_mm=160.0,
        preview_dir=resolved_output / "preview",
        force=force,
    )
    returncode, result, details = execute_presentation(
        config,
        state,
        plan,
        command=command,
    )
    render_presentation_result(
        state,
        returncode,
        result,
        details,
        plan,
        command=command,
    )
    if returncode:
        raise typer.Exit(returncode)


@app.command("build", help=_BUILD_HELP, epilog=_BUILD_EXAMPLES)
def build_command(
    ctx: typer.Context,
    experiment_path: Path = typer.Argument(
        ...,
        metavar="EXPERIMENT_PATH",
        help="Existing completed experiment-run root.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="New presentation output root below generated/.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace only the selected safe presentation output directory.",
    ),
) -> None:
    """Build fit-free artifacts from one completed experiment run manifest."""

    _build_handler(
        state=_shell_state(ctx),
        experiment_path=experiment_path,
        output=output,
        force=force,
        command="build",
    )


@app.command("inspect", help=_INSPECT_HELP, epilog=_INSPECT_EXAMPLES)
def inspect_command(
    ctx: typer.Context,
    path: Path = typer.Argument(
        ...,
        metavar="PATH",
        help="Exact repository, experiment, partial-run, or presentation root.",
    ),
) -> None:
    """Inspect one supported root without writing, searching, or executing."""

    render_inspection(
        _shell_state(ctx),
        inspect_path(path),
        command="inspect",
    )


def _provisional_state(arguments: Sequence[str]) -> ShellState:
    return ShellState(
        plain="--plain" in arguments,
        json_output="--json" in arguments,
        verbose="--verbose" in arguments,
        debug="--debug" in arguments,
        no_color="--no-color" in arguments,
    )


def _command_label(arguments: Sequence[str]) -> str:
    words = [item for item in arguments if not item.startswith("-")]
    return " ".join(words[:2]) if words else "fraud-detection"


def main(argv: Iterable[str] | None = None) -> int:
    """Dispatch the public console script."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    provisional = _provisional_state(arguments)
    global _ACTIVE_STATE
    _ACTIVE_STATE = None
    previous_no_color = os.environ.get("NO_COLOR")
    if provisional.no_color:
        os.environ["NO_COLOR"] = "1"
    command = typer.main.get_command(app)
    try:
        result = command.main(
            args=arguments,
            prog_name="fraud-detection",
            standalone_mode=False,
        )
        return int(result or 0)
    except click_exceptions.Exit as exc:
        return int(exc.exit_code)
    except ProductError as exc:
        state = _ACTIVE_STATE or provisional
        render_error(state, _command_label(arguments), exc)
        if state.debug:
            traceback.print_exc(file=state.stderr)
        return exc.exit_code
    except click_exceptions.ClickException as exc:
        state = _ACTIVE_STATE or provisional
        error = ProductError(
            "FD-USAGE",
            exc.format_message(),
            ("Run fraud-detection --help or the selected command with --help.",),
            exit_code=exc.exit_code,
        )
        render_error(state, _command_label(arguments), error)
        if state.debug:
            traceback.print_exc(file=state.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        state = _ACTIVE_STATE or provisional
        error = ProductError(
            "FD-INTERRUPTED",
            "Operation interrupted by Ctrl-C.",
            ("Re-run the command when ready.",),
            exit_code=130,
        )
        render_error(state, _command_label(arguments), error)
        if state.debug:
            traceback.print_exc(file=state.stderr)
        return 130
    except Exception as exc:
        state = _ACTIVE_STATE or provisional
        error = ProductError(
            "FD-UNEXPECTED",
            f"{type(exc).__name__}: {exc}",
            (
                "Re-run with --debug to inspect the Python traceback.",
                "Check the command inputs before retrying.",
            ),
        )
        render_error(state, _command_label(arguments), error)
        if state.debug:
            traceback.print_exc(file=state.stderr)
        return 1
    finally:
        _ACTIVE_STATE = None
        if provisional.no_color:
            if previous_no_color is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = previous_no_color
