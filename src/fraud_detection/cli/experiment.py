"""Experiment planning and public-run output adapters."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from dataclasses import replace

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
from fraud_detection.experiment import (
    ExperimentConfig,
    ExperimentPlan,
    run_experiment,
)


class _EventLogWriter:
    """Route repository-owned stdout lines into the structured event sink."""

    encoding = "utf-8"

    def __init__(self, sink: LongCommandProjection) -> None:
        self._sink = sink
        self._pending = ""

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("Captured experiment output must be text.")
        self._pending += value
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._emit(line.rstrip("\r"))
        return len(value)

    def flush(self) -> None:
        if self._pending:
            line, self._pending = self._pending, ""
            self._emit(line.rstrip("\r"))

    def isatty(self) -> bool:
        return False

    def _emit(self, line: str) -> None:
        if line:
            self._sink("log", {"message": line})


def render_experiment_plan(
    state: ShellState,
    plan: ExperimentPlan,
    *,
    emit_json: bool = True,
    command: str = "run",
    show_plan: bool = True,
) -> None:
    payload = plan.as_dict()
    if state.json_output:
        if emit_json:
            _emit_json(
                state,
                {
                    "schema": JSON_SCHEMA,
                    "command": command,
                    "status": (
                        "fail" if plan.prerequisite_errors else "pass"
                    ),
                    "result": (
                        "fail" if plan.prerequisite_errors else "pass"
                    ),
                    "exit_code": 1 if plan.prerequisite_errors else 0,
                    "plan": payload,
                },
            )
        return
    effective = payload["effective_config"]
    if not isinstance(effective, Mapping):
        raise TypeError("Effective experiment configuration must be a mapping.")
    data = payload["data"]
    if not isinstance(data, Mapping):
        raise TypeError("Experiment plan data must be a mapping.")
    evidence_notice = payload["evidence_notice"]
    execution_notes = payload["execution_notes"]
    if show_plan and state.rich_enabled:
        if data["path"] is None:
            data_label = "synthetic profile; no real-data path"
        elif data["status"] == "not_inspected":
            data_label = (
                f"validation deferred to preflight · {data['path']}"
            )
        else:
            data_label = f"{data['status']}: {data['path']}"
        table = Table(title="Frozen experiment plan", show_header=False)
        table.add_column("Field", style="bold cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("Profile", str(payload["profile"]))
        table.add_row(
            "Evidence",
            str(effective["evidence_classification"]),
        )
        table.add_row("Data source", str(effective["data_source_kind"]))
        table.add_row("Phase", str(payload["phase"]))
        table.add_row(
            "Seeds",
            ", ".join(map(str, payload["seeds"])),
        )
        table.add_row(
            "Budgets",
            ", ".join(map(str, payload["budgets"])),
        )
        table.add_row(
            "Tasks",
            ", ".join(
                f"{name}={count}"
                for name, count in payload["task_counts"].items()
            ),
        )
        table.add_row("Known total", str(payload["known_task_total"]))
        table.add_row("QA total", "unavailable (integrity gates only)")
        table.add_row(
            "Folds",
            f"OOF={effective['bce_oof_folds']}, inner={effective['inner_folds']}",
        )
        table.add_row("Candidate pool", str(effective["candidate_pool_size"]))
        table.add_row("Data", data_label)
        table.add_row("Output", str(payload["output_path"]))
        table.add_row("Execution", str(payload["execution"]))
        table.add_row(
            "Evidence notice",
            "\n".join(str(item) for item in evidence_notice),
        )
        if execution_notes:
            table.add_row(
                "Operational notes",
                "\n".join(str(item) for item in execution_notes),
            )
        state.console().print(table)
    elif show_plan:
        print(f"PLAN profile={payload['profile']}", file=state.stdout)
        print(
            "PLAN evidence_classification="
            f"{_format_status_value(effective['evidence_classification'])}",
            file=state.stdout,
        )
        print(
            f"PLAN data_source_kind={effective['data_source_kind']}",
            file=state.stdout,
        )
        print(f"PLAN phase={payload['phase']}", file=state.stdout)
        print(
            "PLAN seeds=" + ",".join(map(str, payload["seeds"])),
            file=state.stdout,
        )
        print(
            "PLAN budgets=" + ",".join(map(str, payload["budgets"])),
            file=state.stdout,
        )
        for name, count in payload["task_counts"].items():
            print(f"PLAN tasks.{name}={count}", file=state.stdout)
        print(
            f"PLAN known_task_total={payload['known_task_total']}",
            file=state.stdout,
        )
        print("PLAN qa_task_total=unavailable", file=state.stdout)
        print(
            f"PLAN folds.bce_oof={effective['bce_oof_folds']}",
            file=state.stdout,
        )
        print(
            f"PLAN folds.inner={effective['inner_folds']}",
            file=state.stdout,
        )
        print(
            f"PLAN candidate_pool_size={effective['candidate_pool_size']}",
            file=state.stdout,
        )
        print(f"PLAN data_status={plan.data_status}", file=state.stdout)
        print(
            "PLAN data_path="
            + (
                str(data["path"])
                if data["path"] is not None
                else "not_applicable"
            ),
            file=state.stdout,
        )
        print(
            "PLAN output_path="
            f"{display_path(plan.repository_root, plan.output_path)}",
            file=state.stdout,
        )
        print(
            f"PLAN execution={payload['execution']}",
            file=state.stdout,
        )
        for notice in evidence_notice:
            print(
                f"PLAN evidence_notice={json.dumps(str(notice), ensure_ascii=True)}",
                file=state.stdout,
            )
        for notice in execution_notes:
            print(
                f"PLAN execution_note={json.dumps(str(notice), ensure_ascii=True)}",
                file=state.stdout,
            )
    for error in plan.prerequisite_errors:
        render_error(state, command, error)


def execute_experiment(
    plan: ExperimentPlan,
    state: ShellState,
    *,
    command: str = "run",
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[int, dict[str, object]]:
    payload = plan.as_dict()
    effective = payload["effective_config"]
    task_counts = payload["task_counts"]
    evidence_notices = payload["evidence_notice"]
    if not isinstance(effective, Mapping):
        raise TypeError("Effective experiment configuration must be a mapping.")
    if not isinstance(task_counts, Mapping):
        raise TypeError("Experiment task counts must be a mapping.")
    if not isinstance(evidence_notices, list):
        raise TypeError("Experiment evidence notices must be a list.")
    if not callable(clock):
        raise TypeError("clock must be callable.")
    output = LongCommandProjection(
        state=state,
        command=command,
        command_kind="run",
        profile=str(payload["profile"]),
        evidence_classification=str(effective["evidence_classification"]),
        output_path=str(payload["output_path"]),
        evidence_notices=tuple(str(item) for item in evidence_notices),
        expected_counts={str(key): int(value) for key, value in task_counts.items()},
        seeds=tuple(int(value) for value in payload["seeds"]),
        budgets=tuple(int(value) for value in payload["budgets"]),
        inner_fold_total=int(effective["inner_folds"]),
        clock=clock,
    )
    config: ExperimentConfig = replace(
        plan.config,
        event_sink=output,
    )
    output.start()
    captured_stdout = _EventLogWriter(output)
    try:
        with redirect_stdout(captured_stdout):
            try:
                result = run_experiment(config)
            finally:
                captured_stdout.flush()
    except KeyboardInterrupt as exc:
        output.mark_failure(interrupted=True)
        details = output.snapshot()
        details.update(
            {
                "result": "interrupted",
                "error_code": "FD-INTERRUPTED",
                "error_message": "Experiment interrupted.",
                "error_recovery": [
                    "Partial experiment outputs are not COMPLETE.",
                    "Fit-level resume is unsupported.",
                    "Start a new run with a new --output path when ready.",
                ],
                "partial_output_possible": True,
                "fit_level_resume": "unsupported",
                "suggested_command": (
                    f"fraud-detection run --profile {plan.profile} "
                    "--output <NEW_PATH>"
                ),
                "_exception": exc,
            }
        )
        return 130, details
    except Exception as exc:
        output.mark_failure()
        details = output.snapshot()
        if isinstance(exc, ProductError):
            code = exc.code
            summary = exc.summary
            recovery = list(exc.recovery)
            exit_code = exc.exit_code
        else:
            code = "FD-EXPERIMENT-EXECUTION"
            phase = details.get("failed_phase") or "the active phase"
            summary = f"Experiment execution failed during {phase}."
            recovery = [
                "Inspect the reported phase and operation before retrying.",
                "Fit-level resume is unsupported; use a new output directory.",
            ]
            exit_code = 1
        details.update(
            {
                "result": "fail",
                "error_code": code,
                "error_message": summary,
                "error_recovery": recovery,
                "partial_output_possible": True,
                "fit_level_resume": "unsupported",
                "suggested_command": (
                    f"fraud-detection run --profile {plan.profile} "
                    "--output <NEW_PATH>"
                ),
                "_exception": exc,
            }
        )
        return exit_code, details
    else:
        if result.status != "COMPLETE":
            raise RuntimeError(
                f"Unexpected complete-run status: {result.status!r}."
            )
        output.mark_success(
            completed=(
                "preflight",
                "inner",
                "selection",
                "final",
                "aggregation",
                "qa",
            )
        )
        details = output.snapshot()
        details["event_completed_phases"] = details.pop("completed_phases")
        details.update(
            {
                "result": "pass",
                "output_root": result.output_root,
                "output_path": display_path(
                    plan.repository_root,
                    result.output_root,
                ),
                "requested_phase": result.requested_phase,
                "result_status": result.status,
                "qa_status": "PASS",
                "completed_phases": list(result.completed_phases),
                "manifest_paths": {
                    "run": display_path(
                        plan.repository_root,
                        result.output_root / "RUN_MANIFEST.json",
                    )
                },
                "suggested_command": (
                    "fraud-detection build "
                    + display_path(plan.repository_root, result.output_root)
                ),
            }
        )
        return 0, details
    finally:
        output.close()


def render_run_result(
    state: ShellState,
    plan: ExperimentPlan,
    returncode: int,
    details: dict[str, object],
    *,
    command: str = "run",
) -> None:
    result = str(details["result"])
    public_details = {
        key: value for key, value in details.items() if not key.startswith("_")
    }
    effective = plan.as_dict()["effective_config"]
    if not isinstance(effective, Mapping):
        raise TypeError("Effective experiment configuration must be a mapping.")
    if state.json_output:
        payload = {
            "schema": JSON_SCHEMA,
            "command": command,
            "status": "pass" if returncode == 0 else "error",
            "result": result,
            "exit_code": returncode,
            "profile": plan.profile,
            "evidence_classification": effective["evidence_classification"],
            "output_path": display_path(plan.repository_root, plan.output_path),
            **public_details,
        }
        _emit_json(state, payload)
    elif returncode:
        error = ProductError(
            str(details["error_code"]),
            str(details["error_message"]),
            tuple(str(item) for item in details["error_recovery"]),
            {
                "failed_phase": details.get("failed_phase"),
                "failed_operation": details.get("failed_operation"),
                "partial_output_possible": details["partial_output_possible"],
                "fit_level_resume": details["fit_level_resume"],
            },
            returncode,
        )
        render_error(state, command, error)
        print(
            f"NEXT {details['suggested_command']}",
            file=state.stderr,
        )
    elif state.rich_enabled:
        table = Table(title="Experiment result", show_header=False)
        table.add_column("Field", style="bold cyan")
        table.add_column("Value")
        table.add_row("Status", "[green]COMPLETE[/]")
        table.add_row("QA", "[green]PASS[/]")
        table.add_row("Profile", str(plan.profile))
        table.add_row("Evidence", str(effective["evidence_classification"]))
        table.add_row("Elapsed", f"{float(details['elapsed_seconds']):.1f}s")
        table.add_row("Output", str(details["output_path"]))
        manifest_paths = details["manifest_paths"]
        if not isinstance(manifest_paths, Mapping):
            raise TypeError("Experiment manifest paths must be a mapping.")
        for path in manifest_paths.values():
            table.add_row("Manifest", str(path))
        state.console().print(table)
        for notice in plan.as_dict()["evidence_notice"]:
            state.console().print(str(notice))
        state.console().print(f"Next: {details['suggested_command']}")
    else:
        print(
            f"RESULT command={command.replace(' ', '_')} result=pass "
            f"exit_code={returncode} "
            f"elapsed_seconds={float(details['elapsed_seconds']):.3f} "
            "run_status=COMPLETE qa_status=PASS",
            file=state.stdout,
        )
        print(
            f"RUN_RESULT profile={plan.profile} "
            "evidence_classification="
            f"{_format_status_value(effective['evidence_classification'])} "
            f"output_path={_format_status_value(details['output_path'])}",
            file=state.stdout,
        )
        for kind, path in details["manifest_paths"].items():
            print(
                f"MANIFEST kind={kind} path={_format_status_value(path)}",
                file=state.stdout,
            )
        for notice in plan.as_dict()["evidence_notice"]:
            print(f"EVIDENCE {notice}", file=state.stdout)
        print(f"NEXT {details['suggested_command']}", file=state.stdout)
    if returncode and state.debug:
        exception = details.get("_exception")
        if isinstance(exception, BaseException):
            traceback.print_exception(exception, file=state.stderr)
