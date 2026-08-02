"""Read-only semantic inspection rendering for the public inspect command."""

from __future__ import annotations

import json

from rich.table import Table

from fraud_detection.artifacts import InspectionResult
from fraud_detection.cli.output import JSON_SCHEMA, ShellState, _emit_json


def _plain_value(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value) or "none"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _summary_rows(result: InspectionResult) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = [
        ("Type", result.path_type),
        ("Status", result.status),
        ("Path", result.inspected_path),
    ]
    optional = (
        ("Profile", result.profile),
        ("Role", result.presentation_role),
        ("Evidence", result.evidence_classification),
        ("Source", result.source_kind),
        ("Completed phases", result.completed_phases or None),
        ("Missing phases", result.missing_phases or None),
        ("Artifacts", result.artifact_count),
        ("Checksums", result.checksum_status),
        ("Prepared data", result.prepared_data_count),
        ("Logical figures", result.figure_count),
        ("Figure files", result.figure_file_count),
        ("Logical tables", result.table_count),
        ("Table files", result.table_file_count),
        ("Presentation compatible", result.presentation_compatible),
    )
    rows.extend((label, value) for label, value in optional if value is not None)
    return rows


def render_inspection(
    state: ShellState,
    result: InspectionResult,
    *,
    command: str,
) -> None:
    """Render exactly one stable semantic inspection result."""

    if state.json_output:
        _emit_json(
            state,
            {
                "schema": JSON_SCHEMA,
                "command": command,
                "result": "success",
                "exit_code": 0,
                **result.as_dict(),
            },
        )
        return
    rows = _summary_rows(result)
    if state.rich_enabled:
        state.console().print(
            f"[bold]Type:[/] {result.path_type}  "
            f"[bold]Status:[/] {result.status}"
        )
        table = Table(title="Read-only inspection", show_header=False)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value")
        for label, value in rows[2:]:
            table.add_row(label, _plain_value(value))
        for name, value in result.details.items():
            table.add_row(name.replace("_", " ").title(), _plain_value(value))
        state.console().print(table)
        for warning in result.warnings:
            state.console().print(f"[yellow]Warning:[/] {warning}")
        if result.key_paths:
            state.console().print("[bold]Important paths[/]")
            for name, value in result.key_paths.items():
                state.console().print(f"  {name}: {value}")
        if result.suggested_command:
            state.console().print(f"[bold]Next:[/] {result.suggested_command}")
        return
    for label, value in rows:
        print(
            f"{label.upper().replace(' ', '_')} {_plain_value(value)}",
            file=state.stdout,
        )
    for warning in result.warnings:
        print(f"WARNING {warning}", file=state.stdout)
    for name, value in result.key_paths.items():
        print(f"IMPORTANT_PATH {name}={value}", file=state.stdout)
    for name, value in result.details.items():
        print(f"DETAIL {name}={_plain_value(value)}", file=state.stdout)
    if result.suggested_command:
        print(f"NEXT {result.suggested_command}", file=state.stdout)
    print(
        f"RESULT command={command.replace(' ', '_')} result=success exit_code=0",
        file=state.stdout,
    )
