"""Console state and stable Rich, plain-text, JSON, and error adapters."""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from fraud_detection.errors import ProductError
from fraud_detection.setup import (
    DiagnosticFinding,
    DiagnosticReport,
    SetupFailure,
    SetupResult,
)

JSON_SCHEMA = "fraud_detection.cli.v1"
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _format_status_value(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class CommandFinding:
    """One setup-command rendering value."""

    status: str
    summary: str
    code: str | None = None
    recovery: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class CommandReport:
    """Small mutable report used only by command adapters."""

    findings: list[CommandFinding] = field(default_factory=list)

    def add(
        self,
        status: str,
        summary: str,
        *,
        code: str | None = None,
        recovery: tuple[str, ...] = (),
        details: dict[str, object] | None = None,
    ) -> None:
        self.findings.append(
            CommandFinding(
                status=status,
                summary=summary,
                code=code,
                recovery=recovery,
                details=dict(details or {}),
            )
        )

    def passed(self, summary: str) -> None:
        self.add("PASS", summary)

    def info(self, summary: str) -> None:
        self.add("INFO", summary)

    @property
    def has_failures(self) -> bool:
        return any(finding.status == "FAIL" for finding in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(finding.status == "WARN" for finding in self.findings)


@dataclass
class ShellState:
    """Global output and interaction settings for one CLI invocation."""

    plain: bool = False
    json_output: bool = False
    verbose: bool = False
    debug: bool = False
    no_color: bool = False
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)
    stdin: TextIO = field(default_factory=lambda: sys.stdin)

    @property
    def rich_enabled(self) -> bool:
        return (
            not self.plain
            and not self.json_output
            and bool(getattr(self.stdout, "isatty", lambda: False)())
        )

    @property
    def interactive_input(self) -> bool:
        return (
            not self.json_output
            and bool(getattr(self.stdin, "isatty", lambda: False)())
        )

    def console(self, *, error: bool = False) -> Console:
        return Console(
            file=self.stderr if error else self.stdout,
            force_terminal=self.rich_enabled and not self.no_color,
            no_color=self.no_color,
            color_system=None if self.no_color else "auto",
            highlight=False,
        )


_RUN_PHASES = (
    ("preflight", "Preflight"),
    ("inner", "Inner validation"),
    ("selection", "Selection Freeze"),
    ("final", "Final execution"),
    ("aggregation", "Aggregation"),
    ("qa", "QA"),
)
_BUILD_PHASES = (
    ("source", "Source run validation"),
    ("data", "Presentation data"),
    ("figures", "Figures"),
    ("tables", "Tables"),
    ("preview", "Optional LaTeX preview"),
    ("final", "Final manifests/completion"),
)
_COUNTER_EVENTS = {
    "inner-bce-complete": ("inner_bce_fits", "BCE fits"),
    "inner-ranker-complete": ("inner_ranker_fits", "ranker fits"),
    "selection-freeze-config-complete": (
        "selection_freeze_configurations",
        "frozen configurations",
    ),
    "final-bce-complete": ("final_bce_fits", "BCE fits"),
    "final-ranker-complete": ("final_ranker_fits", "ranker fits"),
    "build-data-frame": ("prepared_data", "prepared data"),
    "render-figure": ("figures", "figures"),
    "render-table": ("tables", "tables"),
}
_SYMBOLS = {
    "completed": "\u2713",
    "active": "\u25b6",
    "pending": "\u25cb",
    "warning": "!",
    "failed": "\u2717",
    "skipped": "\u2013",
}
_RUN_COUNTER_LABELS = {
    "inner_bce_fits": "Inner BCE",
    "inner_ranker_fits": "Inner rankers",
    "selection_freeze_configurations": "Selection",
    "final_bce_fits": "Final BCE",
    "final_ranker_fits": "Final rankers",
}
_METHOD_LABELS = {
    "selected_candidate_amount_gain": "Amount-Gain",
    "selected_candidate_p_only": "p-only",
}
_LOG_PREFIX = r"(?:\[[^\[\]\r\n]+\] )?"
_INNER_BCE_LOG = re.compile(
    rf"^{_LOG_PREFIX}inner BCE seed=(?P<seed>[0-9]+) "
    r"fold=(?P<fold>[0-9]+)$"
)
_INNER_RANKER_LOG = re.compile(
    rf"^{_LOG_PREFIX}inner ranker seed=(?P<seed>[0-9]+) "
    r"fold=(?P<fold>[0-9]+) k=(?P<budget>[0-9]+) "
    r"gain=(?P<gain>[A-Za-z0-9_-]+)$"
)
_FINAL_MODELS_LOG = re.compile(
    rf"^{_LOG_PREFIX}final models seed=(?P<seed>[0-9]+) "
    r"k=(?P<budget>[0-9]+) gain=(?P<gain>[A-Za-z0-9_-]+) "
    r"trees=(?P<trees>[0-9]+)$"
)


def _clean_long_value(value: object) -> object:
    if isinstance(value, str):
        return ANSI_PATTERN.sub("", value)
    if isinstance(value, Mapping):
        return {
            str(key): _clean_long_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_clean_long_value(item) for item in value]
    return value


class _LongCommandView:
    """Dynamic Rich renderable; Live supplies the one-second UI heartbeat."""

    def __init__(self, projection: LongCommandProjection) -> None:
        self.projection = projection

    def __rich_console__(self, _console: Console, _options: object) -> object:
        yield self.projection.live_table()


@dataclass
class LongCommandProjection:
    """Bounded, in-memory projection of existing run or build events."""

    state: ShellState
    command: str
    command_kind: str
    profile: str
    evidence_classification: str
    output_path: str
    evidence_notices: tuple[str, ...]
    expected_counts: Mapping[str, int] = field(default_factory=dict)
    seeds: tuple[int, ...] = ()
    budgets: tuple[int, ...] = ()
    inner_fold_total: int | None = None
    clock: Callable[[], float] = time.perf_counter
    events: list[dict[str, object]] = field(default_factory=list, init=False)
    warnings: list[str] = field(default_factory=list, init=False)
    completed_phases: list[str] = field(default_factory=list, init=False)
    current_phase: str | None = field(default=None, init=False)
    current_operation: str | None = field(default=None, init=False)
    current_seed: int | None = field(default=None, init=False)
    current_inner_fold: int | None = field(default=None, init=False)
    current_budget: int | None = field(default=None, init=False)
    current_gain: str | None = field(default=None, init=False)
    current_path: str | None = field(default=None, init=False)
    current_tree_count: int | None = field(default=None, init=False)
    current_operation_started: float | None = field(default=None, init=False)
    last_event_at: float = field(init=False)
    run_completed: bool = field(default=False, init=False)
    latest_completed_operation: str | None = field(default=None, init=False)
    failed_phase: str | None = field(default=None, init=False)
    failed_operation: str | None = field(default=None, init=False)
    _phase_status: dict[str, str] = field(default_factory=dict, init=False)
    _phase_counts: dict[str, dict[str, int | None]] = field(
        default_factory=dict,
        init=False,
    )
    _event_counts: dict[str, int] = field(default_factory=dict, init=False)
    _last_observed_operation: str | None = field(default=None, init=False)
    _started: float = field(init=False)
    _completed_at: float | None = field(default=None, init=False)
    _live: Live | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.command_kind not in {"run", "build"}:
            raise ValueError("Long command kind must be 'run' or 'build'.")
        self._started = self.clock()
        self.last_event_at = self._started
        self._phase_status = {
            key: "pending" for key, _label in self.phase_definitions
        }
        if self.command_kind == "build":
            self._mark_completed("source")
        self._phase_counts = {
            str(name): {"completed": 0, "total": int(total)}
            for name, total in self.expected_counts.items()
        }
        if self.evidence_classification != "thesis-evidentiary":
            self.warnings.extend(self.evidence_notices)

    @property
    def phase_definitions(self) -> tuple[tuple[str, str], ...]:
        return _RUN_PHASES if self.command_kind == "run" else _BUILD_PHASES

    @property
    def elapsed_seconds(self) -> float:
        end = self._completed_at if self._completed_at is not None else self.clock()
        return max(0.0, end - self._started)

    def start(self) -> None:
        if self.state.json_output:
            return
        if self.state.rich_enabled:
            self._live = Live(
                _LongCommandView(self),
                console=self.state.console(),
                auto_refresh=True,
                refresh_per_second=1,
            )
            self._live.start(refresh=True)
            return
        print(
            f"START command={self.command.replace(' ', '_')} "
            f"profile={self.profile} "
            "evidence_classification="
            f"{_format_status_value(self.evidence_classification)} "
            f"output_path={_format_status_value(self.output_path)}",
            file=self.state.stdout,
        )
        for notice in self.evidence_notices:
            print(f"EVIDENCE {notice}", file=self.state.stdout)

    def __call__(self, kind: str, payload: Mapping[str, object]) -> None:
        clean = _clean_long_value(payload)
        if not isinstance(clean, dict):
            raise TypeError("Long-command event payload must be a mapping.")
        if kind not in {"status", "log"}:
            raise ValueError(f"Unsupported long-command event kind: {kind!r}.")
        self.events.append({"kind": kind, "payload": clean})
        self.last_event_at = self.clock()
        if kind == "log":
            message = str(clean.get("message", ""))
            recognized = self._consume_log(message)
            if (
                not recognized
                and self.state.verbose
                and not self.state.json_output
            ):
                if self.state.rich_enabled and self._live is not None:
                    self._live.console.print(message, markup=False)
                else:
                    print(message, file=self.state.stdout)
            return
        self._consume_status(clean)

    def _consume_log(self, message: str) -> bool:
        match = _INNER_BCE_LOG.fullmatch(message)
        if match is not None:
            self._set_active_operation(
                phase="inner",
                operation="BCE",
                seed=int(match.group("seed")),
                inner_fold=int(match.group("fold")),
            )
            return True
        match = _INNER_RANKER_LOG.fullmatch(message)
        if match is not None:
            self._set_active_operation(
                phase="inner",
                operation="Amount-Gain ranker",
                seed=int(match.group("seed")),
                inner_fold=int(match.group("fold")),
                budget=int(match.group("budget")),
                gain=match.group("gain"),
            )
            return True
        match = _FINAL_MODELS_LOG.fullmatch(message)
        if match is not None:
            self._set_active_operation(
                phase="final",
                operation="final rankers",
                seed=int(match.group("seed")),
                budget=int(match.group("budget")),
                gain=match.group("gain"),
                tree_count=int(match.group("trees")),
            )
            return True
        return False

    def _set_active_operation(
        self,
        *,
        phase: str,
        operation: str,
        seed: int | None = None,
        inner_fold: int | None = None,
        budget: int | None = None,
        gain: str | None = None,
        path: str | None = None,
        tree_count: int | None = None,
    ) -> None:
        self.current_phase = phase
        if self._phase_status[phase] == "pending":
            self._phase_status[phase] = "active"
        self.current_operation = operation
        self._last_observed_operation = operation
        self.current_seed = seed
        self.current_inner_fold = inner_fold
        self.current_budget = budget
        self.current_gain = gain
        self.current_path = path
        self.current_tree_count = tree_count
        self.current_operation_started = self.clock()

    def _consume_status(self, payload: Mapping[str, object]) -> None:
        event = str(payload.get("event", "unknown"))
        level = str(payload.get("level", "INFO")).upper()
        fields = payload.get("fields", {})
        if not isinstance(fields, Mapping):
            raise TypeError("Long-command status fields must be a mapping.")
        self._event_counts[event] = self._event_counts.get(event, 0) + 1
        phase = self._phase_from_event(event, fields)
        if phase is not None:
            if phase != self.current_phase:
                self._clear_operation_context()
            self.current_phase = phase
            if self._phase_status[phase] == "pending":
                self._phase_status[phase] = "active"
        counter = _COUNTER_EVENTS.get(event)
        if counter is not None:
            name, operation = counter
            self._last_observed_operation = operation
            completed = fields.get("completed")
            total = fields.get("total")
            if isinstance(completed, int) and not isinstance(completed, bool):
                known_total = (
                    total
                    if isinstance(total, int) and not isinstance(total, bool)
                    else self._phase_counts.get(name, {}).get("total")
                )
                self._phase_counts[name] = {
                    "completed": completed,
                    "total": known_total,
                }
                self.latest_completed_operation = self._completed_label(
                    event,
                    fields,
                    completed,
                    known_total,
                )
        elif event.endswith("-start") and event != "phase-start":
            self.current_operation = self._operation_label(event)
            self.current_operation_started = self.clock()
        if counter is None and self._is_complete_event(event, fields):
            if phase is not None:
                self._mark_completed(phase)
            if not (
                event in {"phase-complete", "artifact-manifest-complete"}
                and self.latest_completed_operation is not None
            ):
                self.latest_completed_operation = self._completion_event_label(
                    event,
                    fields,
                    phase,
                )
        human_level = (
            "INFO"
            if self._optional_preview_skipped(event, fields)
            else level
        )
        if human_level == "WARN":
            warning = self._status_text(event, fields)
            self.warnings.append(warning)
            if phase is not None:
                self._phase_status[phase] = "warning"
        elif human_level == "FAIL":
            self.failed_phase = phase or self.current_phase
            self.failed_operation = self.current_operation or event
            if self.failed_phase is not None:
                self._phase_status[self.failed_phase] = "failed"
        if self.state.json_output:
            return
        if self.state.rich_enabled:
            if human_level in {"WARN", "FAIL"}:
                self.state.console(error=True).print(
                    f"{human_level} {self._status_text(event, fields)}",
                    markup=False,
                )
            return
        self._render_plain_transition(
            human_level,
            event,
            fields,
            phase,
            counter,
        )

    def _clear_operation_context(self) -> None:
        self.current_operation = None
        self._last_observed_operation = None
        self.current_seed = None
        self.current_inner_fold = None
        self.current_budget = None
        self.current_gain = None
        self.current_path = None
        self.current_tree_count = None
        self.current_operation_started = None

    @staticmethod
    def _integer_field(
        fields: Mapping[str, object],
        name: str,
    ) -> int | None:
        value = fields.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _optional_preview_skipped(
        event: str,
        fields: Mapping[str, object],
    ) -> bool:
        return (
            event == "table-preview"
            and fields.get("status") == "SKIPPED_NO_ENGINE"
        )

    @staticmethod
    def _completed_label(
        event: str,
        fields: Mapping[str, object],
        completed: int,
        total: int | None,
    ) -> str:
        if event == "inner-bce-complete":
            parts = ["BCE complete"]
        elif event == "inner-ranker-complete":
            parts = ["Amount-Gain ranker complete"]
        elif event == "selection-freeze-config-complete":
            parts = ["Selection frozen"]
        elif event == "final-bce-complete":
            parts = ["Final BCE complete"]
        elif event == "final-ranker-complete":
            path = fields.get("path")
            method = _METHOD_LABELS.get(str(path), "final")
            parts = [f"{method} final ranker complete"]
        else:
            parts = [event.replace("-", " ")]
        seed = LongCommandProjection._integer_field(fields, "seed")
        inner_fold = LongCommandProjection._integer_field(fields, "inner_fold")
        budget = LongCommandProjection._integer_field(fields, "budget")
        gain = fields.get("gain")
        if seed is not None:
            parts.append(f"seed {seed}")
        if inner_fold is not None:
            parts.append(f"fold {inner_fold}")
        if budget is not None:
            parts.append(f"k={budget}")
        if isinstance(gain, str):
            parts.append(f"gain={gain}")
        parts.append(f"{completed}/{total}" if total is not None else str(completed))
        return " · ".join(parts)

    def _completion_event_label(
        self,
        event: str,
        fields: Mapping[str, object],
        phase: str | None,
    ) -> str:
        if self._optional_preview_skipped(event, fields):
            return "Optional LaTeX preview — skipped (engine not available)"
        if event == "selection-freeze-complete":
            return "Selection frozen"
        if phase is not None and event in {
            "phase-complete",
            "preflight-complete",
            "aggregation-complete",
        }:
            return f"{dict(self.phase_definitions)[phase]} complete"
        return self._operation_label(event)

    def _phase_from_event(
        self,
        event: str,
        fields: Mapping[str, object],
    ) -> str | None:
        if self.command_kind == "build":
            if event.startswith("build-data"):
                return "data"
            if event.startswith("render-figure"):
                return "figures"
            if event == "table-preview":
                return "preview"
            if event.startswith("render-table"):
                return "tables"
            return self.current_phase
        if event.startswith("preflight") or event == "experiment-start":
            return "preflight"
        if event.startswith("selection"):
            return "selection"
        if event.startswith("aggregation"):
            return "aggregation"
        if event.startswith("inner"):
            return "inner"
        if event.startswith("final"):
            return "final"
        if event.startswith("qa"):
            return "qa"
        phase = fields.get("phase") or fields.get("active_phase")
        if phase in {"inner", "final", "qa"}:
            return str(phase)
        return self.current_phase

    @staticmethod
    def _is_complete_event(
        event: str,
        fields: Mapping[str, object],
    ) -> bool:
        if event in _COUNTER_EVENTS:
            return False
        if event == "table-preview":
            return str(fields.get("status", "")).startswith(
                ("COMPILED", "SKIPPED")
            )
        return event.endswith("-complete") or event == "phase-complete"

    def _mark_completed(self, phase: str) -> None:
        self._phase_status[phase] = "completed"
        label = dict(self.phase_definitions)[phase]
        if label not in self.completed_phases:
            self.completed_phases.append(label)

    @staticmethod
    def _operation_label(event: str) -> str:
        return event.replace("-", " ")

    @staticmethod
    def _status_text(event: str, fields: Mapping[str, object]) -> str:
        status = fields.get("status")
        return event if status is None else f"{event}: {status}"

    def _render_plain_transition(
        self,
        level: str,
        event: str,
        fields: Mapping[str, object],
        phase: str | None,
        counter: tuple[str, str] | None,
    ) -> None:
        phase_label = (
            dict(self.phase_definitions).get(phase, "Unknown")
            if phase is not None
            else "Unknown"
        )
        if level in {"WARN", "FAIL"}:
            print(
                f"{level} command={self.command.replace(' ', '_')} "
                f"phase={_format_status_value(phase_label)} "
                f"event={event}",
                file=self.state.stderr,
            )
        elif counter is not None:
            name, operation = counter
            count = self._phase_counts[name]
            print(
                f"PROGRESS command={self.command.replace(' ', '_')} "
                f"phase={_format_status_value(phase_label)} "
                f"operation={_format_status_value(operation)} "
                f"completed={count['completed']} total={count['total']}",
                file=self.state.stdout,
            )
        elif event.endswith("-start") or event == "phase-start":
            print(
                f"PHASE command={self.command.replace(' ', '_')} "
                f"status=START name={_format_status_value(phase_label)}",
                file=self.state.stdout,
            )
        elif self._is_complete_event(event, fields):
            print(
                f"PHASE command={self.command.replace(' ', '_')} "
                f"status=PASS name={_format_status_value(phase_label)}",
                file=self.state.stdout,
            )
        elif self.state.verbose:
            print(
                f"EVENT command={self.command.replace(' ', '_')} "
                f"phase={_format_status_value(phase_label)} name={event}",
                file=self.state.stdout,
            )

    def mark_success(self, *, completed: tuple[str, ...] = ()) -> None:
        self._completed_at = self.clock()
        for phase in completed:
            self._mark_completed(phase)
        if self.command_kind == "build":
            self._mark_completed("final")
        for count in self._phase_counts.values():
            if count["total"] is not None:
                count["completed"] = count["total"]
        self.run_completed = True
        self.current_operation = "complete"
        self.current_operation_started = None
        self.latest_completed_operation = (
            "experiment complete"
            if self.command_kind == "run"
            else "presentation build complete"
        )

    def mark_failure(
        self,
        *,
        interrupted: bool = False,
        phase: str | None = None,
        operation: str | None = None,
    ) -> None:
        phase = phase or self.current_phase
        if phase is not None:
            self.current_phase = phase
            self.failed_phase = phase
            self.failed_operation = (
                operation
                or self.current_operation
                or self._last_observed_operation
            )
            self._phase_status[phase] = "warning" if interrupted else "failed"

    def close(self) -> None:
        live = self._live
        self._live = None
        if live is not None:
            live.stop()

    def snapshot(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "completed_phases": list(self.completed_phases),
            "phase_counts": {
                name: dict(value) for name, value in self._phase_counts.items()
            },
            "warnings": list(self.warnings),
            "events": list(self.events),
            "event_summary": dict(sorted(self._event_counts.items())),
            "current_phase": (
                dict(self.phase_definitions).get(self.current_phase)
                if self.current_phase is not None
                else None
            ),
            "current_operation": self.current_operation,
            "latest_completed_operation": self.latest_completed_operation,
            "failed_phase": (
                dict(self.phase_definitions).get(self.failed_phase)
                if self.failed_phase is not None
                else None
            ),
            "failed_operation": self.failed_operation,
        }

    def live_table(self) -> Table:
        table = Table(
            title=(
                "Experiment run"
                if self.command_kind == "run"
                else "Presentation build"
            ),
            show_header=False,
        )
        table.add_column("Field", style="bold cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("Profile", self.profile)
        table.add_row("Evidence", self.evidence_classification)
        table.add_row("Output", self.output_path)
        table.add_row("Elapsed", f"{self.elapsed_seconds:.1f}s")
        for key, label in self.phase_definitions:
            status = self._phase_status[key]
            counters = [
                f"{name} {value['completed']}/{value['total']}"
                for name, value in self._phase_counts.items()
                if self._counter_phase(name) == key
            ]
            value = f"{_SYMBOLS[status]} {label}"
            if counters:
                value += " (" + ", ".join(counters) + ")"
            table.add_row("Phase", value)
        if self.command_kind == "run":
            for name, label in _RUN_COUNTER_LABELS.items():
                count = self._phase_counts.get(name)
                if count is not None and count["total"] is not None:
                    progress = Progress(
                        BarColumn(bar_width=20),
                        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
                        auto_refresh=False,
                    )
                    progress.add_task(
                        label,
                        total=count["total"],
                        completed=count["completed"] or 0,
                    )
                    table.add_row(label, progress)
        if self.run_completed:
            table.add_row("Current", "complete")
        else:
            self._add_current_operation_rows(table)
        table.add_row("Latest", self.latest_completed_operation or "none")
        if self.warnings:
            table.add_row("Warnings", "\n".join(self.warnings[-3:]))
        return table

    def _add_current_operation_rows(self, table: Table) -> None:
        if self.current_phase is None and self.current_operation is None:
            table.add_row("Current", "waiting for event")
            return
        table.add_row("Current operation", "")
        if self.current_phase is not None:
            table.add_row(
                "Phase",
                dict(self.phase_definitions)[self.current_phase],
            )
        if self.current_operation is not None:
            table.add_row("Operation", self.current_operation)
        if self.current_seed is not None:
            table.add_row("Seed", self._indexed_value(self.current_seed, self.seeds))
        if self.current_inner_fold is not None:
            fold = str(self.current_inner_fold)
            if self.inner_fold_total is not None:
                fold += f"/{self.inner_fold_total}"
            table.add_row("Inner fold", fold)
        if self.current_budget is not None:
            table.add_row(
                "Budget",
                self._indexed_value(self.current_budget, self.budgets),
            )
        if self.current_gain is not None:
            table.add_row("Gain", self.current_gain)
        if self.current_path is not None:
            table.add_row(
                "Path",
                _METHOD_LABELS.get(self.current_path, self.current_path),
            )
        if self.current_tree_count is not None:
            table.add_row("Trees", str(self.current_tree_count))
        now = self.clock()
        if self.current_operation_started is not None:
            table.add_row(
                "Running",
                f"{max(0.0, now - self.current_operation_started):.1f}s",
            )
        table.add_row(
            "Last event",
            f"{max(0.0, now - self.last_event_at):.1f}s ago",
        )

    @staticmethod
    def _indexed_value(value: int, values: tuple[int, ...]) -> str:
        try:
            index = values.index(value) + 1
        except ValueError:
            return str(value)
        return f"{value} ({index}/{len(values)})"

    def _counter_phase(self, name: str) -> str:
        if name.startswith("inner_"):
            return "inner"
        if name.startswith("selection_"):
            return "selection"
        if name.startswith("final_"):
            return "final"
        if name == "prepared_data":
            return "data"
        return name


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _emit_json(state: ShellState, payload: dict[str, object]) -> None:
    state.stdout.write(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    state.stdout.flush()


def _result_status(report: CommandReport | DiagnosticReport) -> str:
    if report.has_failures:
        return "fail"
    if report.has_warnings:
        return "pass_with_warnings"
    return "pass"


def _finding_payload(
    finding: CommandFinding | DiagnosticFinding,
) -> dict[str, object]:
    return {
        "status": finding.status.lower(),
        "code": finding.code or None,
        "summary": finding.summary,
        "recovery": list(finding.recovery),
        "details": finding.details,
    }


def render_report(
    state: ShellState,
    command: str,
    report: CommandReport | DiagnosticReport,
    *,
    exit_code: int,
    data: dict[str, object] | None = None,
) -> None:
    status = _result_status(report)
    if state.json_output:
        _emit_json(
            state,
            {
                "schema": JSON_SCHEMA,
                "command": command,
                "status": status,
                "exit_code": exit_code,
                "findings": [_finding_payload(item) for item in report.findings],
                "data": data or {},
            },
        )
        return
    if state.rich_enabled:
        table = Table(title=command, show_header=True, header_style="bold cyan")
        table.add_column("Status", no_wrap=True)
        table.add_column("Diagnostic")
        if state.verbose:
            table.add_column("Code", no_wrap=True)
        styles = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "INFO": "cyan"}
        for finding in report.findings:
            row = [
                f"[{styles[finding.status]}]{finding.status}[/]",
                finding.summary,
            ]
            if state.verbose:
                row.append(finding.code or "-")
            table.add_row(*row)
        state.console().print(table)
        state.console().print(
            f"[bold]{command}[/] status: [bold]{status.upper()}[/] "
            f"(exit {exit_code})"
        )
        return
    for finding in report.findings:
        print(f"{finding.status} {finding.summary}", file=state.stdout)
        if finding.code:
            print(f"ERROR_CODE {finding.code}", file=state.stdout)
        for index, step in enumerate(finding.recovery, start=1):
            print(f"RECOVERY {index} {step}", file=state.stdout)
    print(
        f"STATUS command={command} result={status} exit_code={exit_code}",
        file=state.stdout,
    )


@dataclass(frozen=True, slots=True)
class SetupProgressRecord:
    """One CLI-local projection of a public setup progress message."""

    status: str
    subject: str
    detail: str


def project_setup_progress(message: str) -> SetupProgressRecord:
    """Project the current setup callback text without owning setup state."""

    phase = re.fullmatch(
        r"(?P<status>START|PASS) (?P<subject>[^\r\n—]+) "
        r"— (?P<detail>[^\r\n]+)",
        message,
    )
    if phase is not None:
        return SetupProgressRecord(
            phase.group("status"),
            phase.group("subject").strip(),
            phase.group("detail").strip(),
        )
    if message.startswith("Creating or reusing virtual environment:"):
        return SetupProgressRecord(
            "RUNNING", "Virtual environment", "creating or reusing"
        )
    if message.startswith("Installing pinned environment with:"):
        return SetupProgressRecord(
            "RUNNING", "Dependencies and package", "installing pinned set"
        )
    if message.startswith("Downloaded and verified "):
        return SetupProgressRecord(
            "PASS", "Dataset", "validated · acquired"
        )
    if message.startswith("Reused verified "):
        return SetupProgressRecord(
            "PASS", "Dataset", "validated · reused"
        )
    if message.startswith("PyCharm interpreter:"):
        return SetupProgressRecord("PASS", "Project interpreter", "ready")
    return SetupProgressRecord("RUNNING", "Setup", message)


def _human_status(
    state: ShellState,
    status: str,
    subject: str,
    detail: str,
    *,
    error: bool = False,
) -> None:
    value = f"{status} {subject} — {detail}"
    if state.rich_enabled:
        styles = {
            "PASS": "green",
            "START": "cyan",
            "WARNING": "yellow",
            "FAIL": "red",
            "RUNNING": "cyan",
            "SKIPPED": "cyan",
        }
        state.console(error=error).print(
            f"[{styles.get(status, 'cyan')}]{status}[/] "
            f"[bold]{subject}[/] — {detail}"
        )
        return
    print(value, file=state.stderr if error else state.stdout)


def render_setup_plan(
    state: ShellState,
    *,
    repository_root: Path,
    environment_path: Path,
    dataset_action: str,
) -> None:
    """Display the setup mutation plan before invoking the public API."""

    if state.json_output:
        return
    print("SETUP_PLAN mutation=true", file=state.stdout)
    print(f"REPOSITORY_ROOT {repository_root}", file=state.stdout)
    print(f"PYTHON_EXECUTABLE {sys.executable}", file=state.stdout)
    print("PLAN Python validation", file=state.stdout)
    print(f"PLAN Virtual environment target={environment_path}", file=state.stdout)
    print("PLAN Dependency installation pinned=true", file=state.stdout)
    print("PLAN Package installation local=true", file=state.stdout)
    print(f"PLAN Dataset action={dataset_action}", file=state.stdout)
    print("PLAN Data identity validation", file=state.stdout)


def render_setup_progress(
    state: ShellState,
    record: SetupProgressRecord,
) -> None:
    """Render one setup callback projection in human modes only."""

    if not state.json_output:
        _human_status(state, record.status, record.subject, record.detail)


def _setup_success_findings(
    result: SetupResult,
) -> list[dict[str, object]]:
    dataset_qualifier = (
        "acquired" if result.dataset_status == "downloaded" else "reused"
    )
    return [
        {"status": "PASS", "subject": "Python", "detail": "validated"},
        {
            "status": "PASS",
            "subject": "Virtual environment",
            "detail": "prepared",
        },
        {
            "status": "PASS",
            "subject": "Dependencies and package",
            "detail": "installed",
        },
        {
            "status": "PASS",
            "subject": "Dataset",
            "detail": f"validated · {dataset_qualifier}",
        },
        {
            "status": "PASS",
            "subject": "Data identity",
            "detail": "validated",
        },
    ]


def render_setup_success(
    state: ShellState,
    *,
    repository_root: Path,
    environment_path: Path,
    result: SetupResult,
) -> None:
    """Render one completed setup result without starting follow-on work."""

    findings = _setup_success_findings(result)
    dataset_status = (
        "acquired" if result.dataset_status == "downloaded" else "reused"
    )
    if state.json_output:
        _emit_json(
            state,
            {
                "schema": JSON_SCHEMA,
                "command": "setup",
                "result": "pass",
                "exit_code": 0,
                "repository_root": repository_root,
                "python_executable": result.interpreter,
                "environment_path": environment_path,
                "environment_status": "prepared",
                "package_status": "installed",
                "dataset_status": dataset_status,
                "data_identity_status": "validated",
                "findings": findings,
                "suggested_command": "fraud-detection check --full",
                "next_commands": [
                    "fraud-detection check --full",
                    "fraud-detection run --profile smoke-synthetic",
                ],
            },
        )
        return
    for finding in findings:
        _human_status(
            state,
            str(finding["status"]),
            str(finding["subject"]),
            str(finding["detail"]),
        )
    print("RESULT command=setup result=pass exit_code=0", file=state.stdout)
    print(f"PYTHON_EXECUTABLE {result.interpreter}", file=state.stdout)
    print(f"ENVIRONMENT_PATH {environment_path}", file=state.stdout)
    print("NEXT fraud-detection check --full", file=state.stdout)
    print(
        "THEN fraud-detection run --profile smoke-synthetic",
        file=state.stdout,
    )


def _setup_failure_stage(code: str) -> str:
    prefixes = (
        ("FD-PYTHON", "Python validation"),
        ("FD-VENV", "Virtual environment"),
        ("FD-INSTALL", "Dependencies and package"),
        ("FD-KAGGLE", "Dataset acquisition"),
        ("FD-DOWNLOAD", "Dataset acquisition"),
        ("FD-DATA", "Dataset validation"),
    )
    return next(
        (stage for prefix, stage in prefixes if code.startswith(prefix)),
        "Setup",
    )


def render_setup_failure(
    state: ShellState,
    *,
    repository_root: Path,
    environment_path: Path,
    error: SetupFailure,
) -> None:
    """Render an expected setup failure with partial-change guidance."""

    stage = _setup_failure_stage(error.code)
    no_changes = error.details.get("side_effects_started") is False
    environment_status = "not_started" if no_changes else "partial_possible"
    finding = {
        "status": "FAIL",
        "code": error.code,
        "subject": stage,
        "detail": error.summary,
        "recovery": list(error.recovery),
        "details": error.details,
    }
    if state.json_output:
        _emit_json(
            state,
            {
                "schema": JSON_SCHEMA,
                "command": "setup",
                "result": "fail",
                "exit_code": error.exit_code,
                "repository_root": repository_root,
                "python_executable": sys.executable,
                "environment_path": environment_path,
                "environment_status": environment_status,
                "package_status": "unknown",
                "dataset_status": "unchanged_or_unknown",
                "data_identity_status": "not_validated",
                "failed_stage": stage,
                "partial_changes_possible": not no_changes,
                "findings": [finding],
                "suggested_command": "fraud-detection setup",
            },
        )
        return
    _human_status(state, "FAIL", stage, error.summary, error=True)
    print(
        "MODIFICATION none" if no_changes else "MODIFICATION partial changes may exist",
        file=state.stderr,
    )
    for index, step in enumerate(error.recovery, start=1):
        print(f"RECOVERY {index} {step}", file=state.stderr)
    print("NEXT fraud-detection setup", file=state.stderr)


def render_check_start(
    state: ShellState,
    *,
    mode: str,
    require_data: bool,
    repository_root: Path | None,
) -> None:
    """Display the bounded read-only diagnostic scope before dispatch."""

    if state.json_output:
        return
    print(
        f"CHECK mode={mode} require_data={str(require_data).lower()} read_only=true",
        file=state.stdout,
    )
    print(f"PYTHON_EXECUTABLE {sys.executable}", file=state.stdout)
    print(
        f"REPOSITORY_ROOT {repository_root or 'not-detected'}",
        file=state.stdout,
    )


def _diagnostic_status(status: str) -> str:
    return {"PASS": "PASS", "WARN": "WARNING", "FAIL": "FAIL"}.get(
        status,
        "PASS",
    )


def _diagnostic_counts(
    report: DiagnosticReport,
) -> dict[str, int]:
    counts = {"PASS": 0, "WARNING": 0, "FAIL": 0, "SKIPPED": 0}
    for finding in report.findings:
        counts[_diagnostic_status(finding.status)] += 1
    return counts


def _diagnostic_dataset_status(
    report: DiagnosticReport,
    *,
    require_data: bool,
) -> str:
    data_findings = [
        finding
        for finding in report.findings
        if finding.code.startswith("FD-DATA")
        or "data/creditcard.csv" in finding.summary
        or "data identity" in finding.summary.lower()
    ]
    if any(finding.code == "FD-DATA-MISSING" for finding in data_findings):
        return "missing"
    if any(finding.status == "FAIL" for finding in data_findings):
        return "invalid"
    if any(finding.status == "PASS" for finding in data_findings):
        return "validated"
    return "not_required" if not require_data else "not_validated"


def _diagnostic_suggestion(
    report: DiagnosticReport,
    *,
    mode: str,
    require_data: bool,
) -> str:
    if report.has_failures:
        setup_needed = any(
            finding.code.startswith("FD-DATA")
            or finding.code.startswith("FD-PACKAGE")
            or "pip check" in finding.summary.lower()
            or "requirements" in finding.summary.lower()
            or "not installed" in finding.summary.lower()
            for finding in report.findings
            if finding.status == "FAIL"
        )
        return "fraud-detection setup" if setup_needed else "fraud-detection check"
    if mode == "full" or require_data:
        return "fraud-detection run --profile smoke-synthetic"
    return "fraud-detection run --profile smoke-synthetic --dry-run"


def _diagnostic_why(finding: DiagnosticFinding) -> str | None:
    if finding.status != "FAIL":
        return None
    summary = finding.summary.lower()
    if finding.code.startswith("FD-DATA"):
        return "Canonical real-data validation cannot proceed."
    if (
        "pip check" in summary
        or "requirements" in summary
        or "not installed" in summary
    ):
        return "The active Python environment cannot be trusted for repository work."
    if finding.code.startswith(("FD-ROOT", "FD-GIT")):
        return "Repository-scoped diagnostics cannot complete reliably."
    return "A required operational prerequisite is not satisfied."


def render_diagnostic_report(
    state: ShellState,
    *,
    command: str,
    mode: str,
    require_data: bool,
    report: DiagnosticReport,
) -> None:
    """Render quick/full diagnostics with stable streams and JSON fields."""

    counts = _diagnostic_counts(report)
    dataset_status = _diagnostic_dataset_status(
        report,
        require_data=require_data,
    )
    suggested = _diagnostic_suggestion(
        report,
        mode=mode,
        require_data=require_data,
    )
    findings = [
        {
            "status": _diagnostic_status(finding.status),
            "code": finding.code or None,
            "subject": finding.summary.split(".", maxsplit=1)[0],
            "detail": finding.summary,
            "recovery": list(finding.recovery),
            "details": dict(finding.details),
            "why_it_matters": _diagnostic_why(finding),
        }
        for finding in report.findings
    ]
    result = "fail" if report.has_failures else "pass"
    if state.json_output:
        _emit_json(
            state,
            {
                "schema": JSON_SCHEMA,
                "command": command,
                "result": result,
                "exit_code": report.exit_code,
                "mode": mode,
                "require_data": require_data,
                "repository_root": report.repository_root,
                "python_executable": sys.executable,
                "passed_count": counts["PASS"],
                "warning_count": counts["WARNING"],
                "failed_count": counts["FAIL"],
                "skipped_count": counts["SKIPPED"],
                "dataset_status": dataset_status,
                "findings": findings,
                "suggested_command": suggested,
            },
        )
        return
    for finding, payload in zip(report.findings, findings, strict=True):
        status = str(payload["status"])
        is_problem = status in {"WARNING", "FAIL"}
        _human_status(
            state,
            status,
            str(payload["subject"]),
            str(payload["detail"]),
            error=is_problem,
        )
        if is_problem:
            why = _diagnostic_why(finding)
            if why is not None:
                print(f"WHY {why}", file=state.stderr)
            for index, step in enumerate(finding.recovery, start=1):
                print(f"RECOVERY {index} {step}", file=state.stderr)
        if state.verbose and finding.code:
            stream = state.stderr if is_problem else state.stdout
            print(f"CODE {finding.code}", file=stream)
            for name, value in sorted(finding.details.items()):
                print(f"DETAIL {name}={value}", file=stream)
    print(
        f"RESULT command={command} result={result} exit_code={report.exit_code} "
        f"mode={mode} passed={counts['PASS']} warnings={counts['WARNING']} "
        f"failed={counts['FAIL']} skipped={counts['SKIPPED']} "
        f"dataset={dataset_status}",
        file=state.stdout,
    )
    if report.has_failures:
        print("MODIFICATION none; diagnostics are read-only", file=state.stderr)
    print(f"NEXT {suggested}", file=state.stdout)


def render_error(state: ShellState, command: str, error: ProductError) -> None:
    if error.code == "FD-USAGE":
        category = "usage_error"
    elif error.code == "FD-INTERRUPTED":
        category = "interruption"
    elif error.code == "FD-RUN-INCOMPLETE":
        category = "incomplete_experiment_input"
    elif "OUTPUT" in error.code:
        category = "output_path_conflict"
    elif error.code == "FD-UNEXPECTED":
        category = "unexpected_internal_failure"
    else:
        category = "product_validation_failure"
    payload = {
        "schema": JSON_SCHEMA,
        "command": command,
        "status": "error",
        "result": "interrupted" if error.exit_code == 130 else "fail",
        "exit_code": error.exit_code,
        "error_category": category,
        "error": {
            "code": error.code,
            "summary": error.summary,
            "recovery": list(error.recovery),
            "details": error.details,
        },
    }
    if state.json_output:
        _emit_json(state, payload)
        return
    if state.rich_enabled:
        recovery = "\n".join(
            f"{index}. {step}" for index, step in enumerate(error.recovery, start=1)
        )
        detail_lines = (
            "\n".join(
                f"{name}: {value}"
                for name, value in sorted(error.details.items())
            )
            if state.verbose and error.details
            else ""
        )
        body = error.summary
        if detail_lines:
            body += "\n\n" + detail_lines
        if recovery:
            body += "\n\nRecovery\n" + recovery
        state.console(error=True).print(
            Panel(body, title=f"[red]{error.code}[/]", border_style="red")
        )
        return
    print(f"ERROR {error.code} {error.summary}", file=state.stderr)
    for name, value in sorted(error.details.items()):
        print(f"DETAIL {name}={value}", file=state.stderr)
    for index, step in enumerate(error.recovery, start=1):
        print(f"RECOVERY {index} {step}", file=state.stderr)
