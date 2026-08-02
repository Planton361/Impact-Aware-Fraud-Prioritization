import subprocess
from pathlib import Path

import pytest

import fraud_detection.setup.diagnostics as diagnostics
from fraud_detection.setup import DiagnosticReport, run_check, run_doctor

pytestmark = pytest.mark.contract


def test_doctor_returns_structured_report_without_console_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = run_doctor(repository_root=tmp_path)

    captured = capsys.readouterr()
    assert isinstance(report, DiagnosticReport)
    assert report.command == "doctor"
    assert report.repository_root == tmp_path.resolve()
    assert report.findings
    assert report.status == "FAIL"
    assert report.exit_code == 1
    assert captured.out == ""
    assert captured.err == ""
    assert run_doctor is diagnostics.run_doctor

    identity_report = diagnostics._Collector()
    diagnostics._record_required_data_identity(
        identity_report,
        reference_results_valid=True,
        dataset_valid=True,
    )
    assert len(identity_report.findings) == 1
    identity = identity_report.findings[0]
    assert identity.code == "FD-DATA-IDENTITY"
    assert identity.status == "PASS"
    assert identity.details == {
        "raw_rows": 284_807,
        "raw_fraud": 492,
        "raw_legitimate": 284_315,
        "deduplicated_rows": 283_726,
        "deduplicated_fraud": 473,
        "deduplicated_legitimate": 283_253,
        "removed_duplicates": 1_081,
        "raw_identity": diagnostics.EXPECTED_RAW_SHA256,
        "deduplicated_identity": diagnostics.EXPECTED_DEDUPLICATED_SHA256,
    }


def test_check_returns_structured_report_without_console_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = run_check(repository_root=tmp_path)

    captured = capsys.readouterr()
    assert isinstance(report, DiagnosticReport)
    assert report.command == "check"
    assert report.repository_root == tmp_path.resolve()
    assert report.findings
    assert report.status == "FAIL"
    assert report.exit_code == 1
    assert captured.out == ""
    assert captured.err == ""
    assert run_check is diagnostics.run_check


def test_documented_help_checks_full_public_cli(
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_runner(
        arguments: tuple[str, ...],
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path
        observed.append(tuple(arguments))
        return subprocess.CompletedProcess(arguments, 0, "help", "")

    report = diagnostics._Collector()
    passed = diagnostics._check_documented_command_help(
        tmp_path,
        report,
        process_runner=fake_runner,
    )

    assert passed
    assert len(observed) == 6
    package_commands = [command for command in observed if command[2] == "-c"]
    assert len(package_commands) == 6
    for arguments in (
        "main(['--help'])",
        "main(['setup', '--help'])",
        "main(['check', '--help'])",
        "main(['run', '--help'])",
        "main(['build', '--help'])",
        "main(['inspect', '--help'])",
    ):
        assert any(arguments in command[3] for command in package_commands)
    deleted_stage_root = "scripts" + "/thesis"
    assert all(deleted_stage_root not in " ".join(command) for command in observed)
    summaries = [finding.summary for finding in report.findings]
    assert any(
        "fraud-detection --help completed" in summary
        for summary in summaries
    )
    assert any(
        "fraud-detection setup --help completed" in summary
        for summary in summaries
    )
    assert any(
        "fraud-detection check --help completed" in summary
        for summary in summaries
    )
    for command in ("run", "build", "inspect"):
        assert any(
            f"fraud-detection {command} --help completed" in summary
            for summary in summaries
        )


def test_documented_help_reports_public_cli_failure(
    tmp_path: Path,
) -> None:
    def fake_runner(
        arguments: tuple[str, ...],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        failed = "main(['check', '--help'])" in " ".join(arguments)
        return subprocess.CompletedProcess(
            arguments,
            9 if failed else 0,
            "",
            "controlled package help failure\n" if failed else "",
        )

    report = diagnostics._Collector()
    passed = diagnostics._check_documented_command_help(
        tmp_path,
        report,
        process_runner=fake_runner,
    )

    assert not passed
    failures = [
        finding for finding in report.findings if finding.status == "FAIL"
    ]
    assert len(failures) == 1
    assert (
        failures[0].summary
        == "fraud-detection check --help failed with exit code 9: "
        "controlled package help failure"
    )
