"""Read-only doctor and bounded developer-check APIs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable, Literal, Sequence

from fraud_detection.artifacts import (
    _EXPECTED_REFERENCE_FILENAMES,
    find_repository_root,
)
from fraud_detection.experiment.config import (
    CANDIDATE_POOL_SIZE,
    EXPECTED_DEDUPLICATED_SHA256,
    EXPECTED_RAW_SHA256,
    METHOD_FAMILIES,
    OUTER_SEEDS,
    TARGET_BUDGETS,
)

from .environment import (
    PYTHON_CONTRACT,
    parse_pinned_requirements,
)

DiagnosticStatus = Literal["PASS", "WARN", "FAIL", "INFO"]
_ProcessRunner = Callable[
    [Sequence[str], Path],
    subprocess.CompletedProcess[str],
]
_VersionProvider = Callable[[str], str]
_FULL_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
_EXPECTED_RAW_ROWS = 284_807
_EXPECTED_RAW_FRAUD = 492
_EXPECTED_RAW_LEGITIMATE = 284_315
_EXPECTED_DEDUPLICATED_ROWS = 283_726
_EXPECTED_CLASS_COUNTS = {"legitimate": 283_253, "fraud": 473}
_EXPECTED_REMOVED_DUPLICATES = 1_081
_EXPECTED_DATA_HEADER = (
    "Time",
    *(f"V{index}" for index in range(1, 29)),
    "Amount",
    "Class",
)
_CENTRAL_SCHEMA = (
    "target_budget",
    "path",
    "plr_mean",
    "plr_sd",
    "fraud_at_k_mean",
    "fraud_at_k_sd",
    "precision_at_k_mean",
    "precision_at_k_sd",
    "recall_at_k_mean",
    "recall_at_k_sd",
)
_SELECTION_SCHEMA = (
    "seed",
    "target_budget",
    "selected_gain",
    "selection_status",
    "final_n_estimators",
    "truncation",
    "eval_at",
)
_CENTRAL_BUDGETS = frozenset({20, 50, 100})


@dataclass(frozen=True, slots=True)
class DiagnosticFinding:
    """One structured diagnostic outcome."""

    code: str
    status: DiagnosticStatus
    summary: str
    recovery: tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Structured, rendering-independent result of doctor or check."""

    command: Literal["doctor", "check"]
    repository_root: Path | None
    findings: tuple[DiagnosticFinding, ...]
    elapsed_seconds: float

    @property
    def status(self) -> Literal["PASS", "WARN", "FAIL"]:
        if self.has_failures:
            return "FAIL"
        if self.has_warnings:
            return "WARN"
        return "PASS"

    @property
    def exit_code(self) -> int:
        return 1 if self.has_failures else 0

    @property
    def has_failures(self) -> bool:
        return any(finding.status == "FAIL" for finding in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(finding.status == "WARN" for finding in self.findings)


@dataclass
class _Collector:
    findings: list[DiagnosticFinding] = field(default_factory=list)

    def add(
        self,
        status: DiagnosticStatus,
        summary: str,
        *,
        code: str | None = None,
        recovery: Sequence[str] = (),
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.findings.append(
            DiagnosticFinding(
                code=code or "",
                status=status,
                summary=summary,
                recovery=tuple(recovery),
                details=dict(details or {}),
            )
        )

    def passed(self, summary: str) -> None:
        self.add("PASS", summary)

    def warn(self, summary: str) -> None:
        self.add("WARN", summary)

    def fail(
        self,
        summary: str,
        *,
        code: str | None = None,
        recovery: Sequence[str] = (),
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.add(
            "FAIL",
            summary,
            code=code,
            recovery=recovery,
            details=details,
        )

    def info(self, summary: str) -> None:
        self.add("INFO", summary)


@dataclass(frozen=True)
class _GitState:
    is_checkout: bool
    branch: str | None = None
    detached: bool = False
    head: str | None = None
    upstream: str | None = None
    upstream_sha: str | None = None
    ahead: int | None = None
    behind: int | None = None
    staged: tuple[str, ...] = ()
    unstaged: tuple[str, ...] = ()
    untracked: tuple[str, ...] = ()


@dataclass(frozen=True)
class _DoctorState:
    root: Path | None
    git: _GitState | None


def _process_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _run_process(
    arguments: Sequence[str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run one controlled, captured subprocess without invoking a shell."""

    return subprocess.run(
        list(arguments),
        cwd=cwd.resolve(),
        check=False,
        capture_output=True,
        text=True,
        env=_process_environment(),
    )


def _completed_output(completed: subprocess.CompletedProcess[str]) -> str:
    parts = [
        value.strip()
        for value in (completed.stdout, completed.stderr)
        if value and value.strip()
    ]
    return " | ".join(parts)


def _git_failure_details(
    completed: subprocess.CompletedProcess[str],
) -> str:
    output = _completed_output(completed)
    suffix = f": {output}" if output else ""
    return f"exit code {completed.returncode}{suffix}"


def _git(
    root: Path,
    arguments: Sequence[str],
    process_runner: _ProcessRunner,
) -> subprocess.CompletedProcess[str]:
    return process_runner(["git", *arguments], root)


def _classify_porcelain(
    output: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    for line in output.splitlines():
        if not line:
            continue
        code = line[:2]
        path = line[3:] if len(line) > 3 else line
        if code == "??":
            untracked.append(path)
            continue
        if code[0] != " ":
            staged.append(path)
        if code[1] != " ":
            unstaged.append(path)
    return tuple(staged), tuple(unstaged), tuple(untracked)


def _inspect_git(
    root: Path,
    report: _Collector,
    process_runner: _ProcessRunner = _run_process,
) -> _GitState:
    try:
        version = _git(root, ["--version"], process_runner)
    except FileNotFoundError:
        report.fail("Git is not available.")
        return _GitState(is_checkout=False)
    if version.returncode != 0:
        report.fail(f"Git availability check failed: {_completed_output(version)}")
        return _GitState(is_checkout=False)
    report.passed((version.stdout or "Git is available.").strip())

    checkout = _git(root, ["rev-parse", "--is-inside-work-tree"], process_runner)
    if checkout.returncode != 0 or checkout.stdout.strip() != "true":
        report.warn("Repository files are present, but this is not a Git checkout.")
        return _GitState(is_checkout=False)

    head_result = _git(root, ["rev-parse", "HEAD"], process_runner)
    if head_result.returncode != 0:
        report.fail(f"Git could not resolve HEAD: {_completed_output(head_result)}")
        return _GitState(is_checkout=True)
    head = head_result.stdout.strip()
    if _FULL_SHA_PATTERN.fullmatch(head) is None:
        report.fail(f"Git returned an invalid full HEAD SHA: {head!r}.")
    else:
        report.passed(f"Git HEAD is {head}.")

    branch_result = _git(
        root,
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        process_runner,
    )
    if branch_result.returncode == 0:
        branch = branch_result.stdout.strip() or None
        detached = False
        if branch is None:
            report.fail("Git returned an empty branch name.")
        else:
            report.passed(f"Git branch is {branch}.")
    elif branch_result.returncode == 1:
        branch = None
        detached = True
        report.info("Git is in a clearly identified detached-HEAD state.")
    else:
        report.fail(
            "Git could not determine the branch or detached-HEAD state "
            f"({_git_failure_details(branch_result)})."
        )
        branch = None
        detached = False

    status_result = _git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        process_runner,
    )
    if status_result.returncode != 0:
        report.fail(f"Git status failed: {_completed_output(status_result)}")
        staged: tuple[str, ...] = ()
        unstaged: tuple[str, ...] = ()
        untracked: tuple[str, ...] = ()
    else:
        staged, unstaged, untracked = _classify_porcelain(status_result.stdout)
        dirty_count = len(staged) + len(unstaged) + len(untracked)
        if dirty_count:
            report.warn(
                "Working tree has "
                f"{len(staged)} staged, {len(unstaged)} unstaged, and "
                f"{len(untracked)} untracked non-ignored path(s)."
            )
        else:
            report.passed(
                "Working tree has no staged, unstaged, or untracked changes."
            )

    upstream: str | None = None
    upstream_sha: str | None = None
    ahead: int | None = None
    behind: int | None = None
    if detached:
        report.info("Detached HEAD has no branch upstream requirement.")
    elif branch is not None:
        remote_key = f"branch.{branch}.remote"
        merge_key = f"branch.{branch}.merge"
        remote_result = _git(root, ["config", "--get", remote_key], process_runner)
        merge_result = _git(root, ["config", "--get", merge_key], process_runner)
        config_failed = False
        config_missing: dict[str, bool] = {}
        for key, result in (
            (remote_key, remote_result),
            (merge_key, merge_result),
        ):
            output = _completed_output(result)
            if result.returncode == 0:
                config_missing[key] = False
            elif result.returncode == 1 and not output:
                config_missing[key] = True
            else:
                report.fail(
                    f"Git could not read upstream setting {key} "
                    f"({_git_failure_details(result)})."
                )
                config_failed = True

        if not config_failed:
            remote_missing = config_missing[remote_key]
            merge_missing = config_missing[merge_key]
            if remote_missing and merge_missing:
                report.warn(
                    "No upstream is configured; no repository setting was changed."
                )
            elif remote_missing != merge_missing:
                report.fail(
                    f"Git upstream configuration for branch {branch} is incomplete: "
                    f"{remote_key} and {merge_key} must both be configured."
                )
            elif not remote_result.stdout.strip() or not merge_result.stdout.strip():
                report.fail(
                    f"Git upstream configuration for branch {branch} contains an "
                    "empty remote or merge value."
                )
            else:
                upstream_result = _git(
                    root,
                    [
                        "rev-parse",
                        "--abbrev-ref",
                        "--symbolic-full-name",
                        "@{upstream}",
                    ],
                    process_runner,
                )
                if upstream_result.returncode != 0:
                    report.fail(
                        "Git could not resolve the configured symbolic upstream "
                        f"({_git_failure_details(upstream_result)})."
                    )
                else:
                    upstream = upstream_result.stdout.strip() or None
                    if upstream is None:
                        report.fail(
                            "Git returned an empty name for the configured upstream."
                        )

        if upstream is not None:
            upstream_sha_result = _git(
                root,
                ["rev-parse", "@{upstream}"],
                process_runner,
            )
            if upstream_sha_result.returncode != 0:
                report.fail(
                    "Git could not resolve the configured upstream commit "
                    f"({_git_failure_details(upstream_sha_result)})."
                )
            else:
                candidate_sha = upstream_sha_result.stdout.strip()
                if _FULL_SHA_PATTERN.fullmatch(candidate_sha) is None:
                    report.fail(
                        f"Git returned an invalid upstream SHA: "
                        f"{candidate_sha!r}."
                    )
                else:
                    upstream_sha = candidate_sha

        if upstream is not None and upstream_sha is not None:
            counts = _git(
                root,
                ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
                process_runner,
            )
            if counts.returncode != 0:
                report.fail(
                    "Git could not determine ahead/behind "
                    f"({_git_failure_details(counts)})."
                )
            else:
                try:
                    ahead_text, behind_text = counts.stdout.split()
                    ahead, behind = int(ahead_text), int(behind_text)
                except (TypeError, ValueError):
                    report.fail(
                        f"Git returned invalid ahead/behind data: "
                        f"{counts.stdout!r}."
                    )
                else:
                    report.passed(
                        f"Upstream {upstream} is available at {upstream_sha}; "
                        f"ahead {ahead}, behind {behind}."
                    )

    return _GitState(
        is_checkout=True,
        branch=branch,
        detached=detached,
        head=head,
        upstream=upstream,
        upstream_sha=upstream_sha,
        ahead=ahead,
        behind=behind,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
    )


def _check_python(
    report: _Collector,
    version_info: Sequence[int] | None = None,
) -> bool:
    version = tuple(version_info or sys.version_info)
    display = ".".join(str(part) for part in version[:3])
    if len(version) >= 2 and version[0] == 3 and version[1] == 12:
        report.passed(f"Python {display} satisfies {PYTHON_CONTRACT}.")
        return True
    report.fail(f"Python {display} does not satisfy {PYTHON_CONTRACT}.")
    return False


def _check_environment(
    root: Path,
    report: _Collector,
    *,
    version_provider: _VersionProvider = metadata.version,
    process_runner: _ProcessRunner = _run_process,
) -> bool:
    passed = True
    requirement_paths = (
        root / "environment" / "final_experiment_requirements.txt",
        root / "environment" / "bootstrap_requirements.txt",
    )
    for path in requirement_paths:
        if not path.is_file():
            report.fail(f"Missing requirements file: {path.relative_to(root)}.")
            passed = False
            continue
        try:
            pins = parse_pinned_requirements(path)
        except (OSError, UnicodeError, ValueError) as exc:
            report.fail(str(exc))
            passed = False
            continue
        for distribution, expected in pins:
            try:
                actual = version_provider(distribution)
            except metadata.PackageNotFoundError:
                report.fail(
                    f"{distribution}=={expected} is required by "
                    f"{path.relative_to(root)}, but is not installed."
                )
                passed = False
            except Exception as exc:
                report.fail(f"Could not inspect {distribution}: {exc}")
                passed = False
            else:
                if actual == expected:
                    report.passed(
                        f"{distribution}=={actual} matches the pinned version."
                    )
                else:
                    report.fail(
                        f"{distribution}=={actual} does not match the pinned "
                        f"version {expected}."
                    )
                    passed = False

    pip_check = process_runner([sys.executable, "-m", "pip", "check"], root)
    if pip_check.returncode == 0:
        report.passed("python -m pip check completed successfully.")
    else:
        output = _completed_output(pip_check)
        report.fail(
            "python -m pip check failed" + (f": {output}" if output else ".")
        )
        passed = False
    return passed


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _int_field(row: dict[str, str], key: str, path: Path) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path.name} contains an invalid {key!r} value."
        ) from exc


def _check_reference_results(root: Path, report: _Collector) -> bool:
    reference_root = root / "reference_results"
    if not reference_root.is_dir():
        report.fail("Missing reference_results directory.")
        return False
    observed_files = {
        path.name for path in reference_root.iterdir() if path.is_file()
    }
    if observed_files != _EXPECTED_REFERENCE_FILENAMES:
        missing = sorted(_EXPECTED_REFERENCE_FILENAMES - observed_files)
        additional = sorted(observed_files - _EXPECTED_REFERENCE_FILENAMES)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if additional:
            details.append("additional " + ", ".join(additional))
        report.fail("Reference file set differs: " + "; ".join(details) + ".")
        passed = False
    else:
        report.passed(
            "reference_results contains exactly the three approved files."
        )
        passed = True

    central_path = reference_root / "central_topk_results.csv"
    if central_path.is_file():
        try:
            header, rows = _read_csv(central_path)
            pairs = {
                (_int_field(row, "target_budget", central_path), row["path"])
                for row in rows
            }
            budgets = {
                _int_field(row, "target_budget", central_path)
                for row in rows
            }
            paths = {row["path"] for row in rows}
            central_ok = (
                header == list(_CENTRAL_SCHEMA)
                and len(rows) == len(_CENTRAL_BUDGETS) * len(METHOD_FAMILIES)
                and budgets == _CENTRAL_BUDGETS
                and paths == set(METHOD_FAMILIES)
                and len(pairs) == len(rows)
            )
        except (OSError, UnicodeError, csv.Error, ValueError, KeyError) as exc:
            report.fail(f"central_topk_results.csv is invalid: {exc}")
            central_ok = False
        else:
            if central_ok:
                report.passed(
                    "central_topk_results.csv has 12 unique budget-path rows, "
                    "three budgets, four paths, and the expected schema."
                )
            else:
                report.fail(
                    "central_topk_results.csv does not match its frozen contract."
                )
        passed = passed and central_ok

    selection_path = reference_root / "selected_configuration_summary.csv"
    if selection_path.is_file():
        try:
            header, rows = _read_csv(selection_path)
            pairs = {
                (
                    _int_field(row, "seed", selection_path),
                    _int_field(row, "target_budget", selection_path),
                )
                for row in rows
            }
            seeds = {_int_field(row, "seed", selection_path) for row in rows}
            budgets = {
                _int_field(row, "target_budget", selection_path)
                for row in rows
            }
            expected_pairs = {
                (seed, budget)
                for seed in OUTER_SEEDS
                for budget in TARGET_BUDGETS
            }
            selection_ok = (
                header == list(_SELECTION_SCHEMA)
                and len(rows) == len(expected_pairs)
                and seeds == set(OUTER_SEEDS)
                and budgets == set(TARGET_BUDGETS)
                and pairs == expected_pairs
            )
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            report.fail(
                f"selected_configuration_summary.csv is invalid: {exc}"
            )
            selection_ok = False
        else:
            if selection_ok:
                report.passed(
                    "selected_configuration_summary.csv has the unique "
                    "complete 5 x 7 seed-budget grid."
                )
            else:
                report.fail(
                    "selected_configuration_summary.csv does not match its "
                    "frozen contract."
                )
        passed = passed and selection_ok

    identity_path = reference_root / "data_identity.json"
    if identity_path.is_file():
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            identity_ok = (
                identity["raw_row_count"] == _EXPECTED_RAW_ROWS
                and identity["raw_sha256"] == EXPECTED_RAW_SHA256
                and identity["deduplicated_row_count"]
                == _EXPECTED_DEDUPLICATED_ROWS
                and identity["deduplicated_dataframe_sha256"]
                == EXPECTED_DEDUPLICATED_SHA256
                and identity["class_counts"] == _EXPECTED_CLASS_COUNTS
                and identity["outer_seeds"] == list(OUTER_SEEDS)
                and identity["budget_grid"] == list(TARGET_BUDGETS)
                and identity["candidate_pool_size"] == CANDIDATE_POOL_SIZE
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as exc:
            report.fail(f"data_identity.json is invalid: {exc}")
            identity_ok = False
        else:
            if identity_ok:
                report.passed(
                    "data_identity.json matches all frozen identities."
                )
            else:
                report.fail(
                    "data_identity.json does not match the frozen identities."
                )
        passed = passed and identity_ok
    return passed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_data(
    root: Path,
    report: _Collector,
    *,
    require_data: bool,
) -> bool:
    data_path = root / "data" / "creditcard.csv"
    if not data_path.is_file():
        message = "data/creditcard.csv is absent."
        if require_data:
            report.fail(
                message + " --require-data makes it mandatory.",
                code="FD-DATA-MISSING",
                recovery=(
                    "Run fraud-detection setup.",
                    "Re-run the command after the canonical CSV is verified.",
                ),
                details={"path": str(data_path)},
            )
            return False
        report.warn(message + " The dataset is optional for this check.")
        return True

    passed = True
    try:
        observed_hash = _sha256_file(data_path)
    except OSError as exc:
        report.fail(f"Could not hash data/creditcard.csv: {exc}")
        return False
    if observed_hash == EXPECTED_RAW_SHA256:
        report.passed("data/creditcard.csv has the registered raw SHA-256.")
    else:
        report.fail(
            "data/creditcard.csv has the wrong SHA-256: "
            f"{observed_hash}; expected {EXPECTED_RAW_SHA256}.",
            code="FD-DATA-HASH",
            recovery=(
                "Move the mismatching CSV to a safe location outside data/.",
                "Run fraud-detection setup to obtain and verify the canonical CSV.",
            ),
            details={
                "path": str(data_path),
                "actual_sha256": observed_hash,
                "expected_sha256": EXPECTED_RAW_SHA256,
            },
        )
        passed = False

    try:
        with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            row_count = sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        report.fail(f"Could not read data/creditcard.csv: {exc}")
        return False
    if header == list(_EXPECTED_DATA_HEADER):
        report.passed("data/creditcard.csv has the expected 31-column schema.")
    else:
        report.fail(
            "data/creditcard.csv does not have the expected 31-column schema."
        )
        passed = False
    if row_count == _EXPECTED_RAW_ROWS:
        report.passed(f"data/creditcard.csv has {_EXPECTED_RAW_ROWS} data rows.")
    else:
        report.fail(
            f"data/creditcard.csv has {row_count} data rows; "
            f"expected {_EXPECTED_RAW_ROWS}."
        )
        passed = False
    return passed


def _record_required_data_identity(
    report: _Collector,
    *,
    reference_results_valid: bool,
    dataset_valid: bool,
) -> None:
    if not reference_results_valid or not dataset_valid:
        return
    report.add(
        "PASS",
        "Canonical data identity, raw counts, and stable deduplication contract "
        "are validated.",
        code="FD-DATA-IDENTITY",
        details={
            "raw_rows": _EXPECTED_RAW_ROWS,
            "raw_fraud": _EXPECTED_RAW_FRAUD,
            "raw_legitimate": _EXPECTED_RAW_LEGITIMATE,
            "deduplicated_rows": _EXPECTED_DEDUPLICATED_ROWS,
            "deduplicated_fraud": _EXPECTED_CLASS_COUNTS["fraud"],
            "deduplicated_legitimate": _EXPECTED_CLASS_COUNTS["legitimate"],
            "removed_duplicates": _EXPECTED_REMOVED_DUPLICATES,
            "raw_identity": EXPECTED_RAW_SHA256,
            "deduplicated_identity": EXPECTED_DEDUPLICATED_SHA256,
        },
    )


def _perform_doctor_checks(
    report: _Collector,
    *,
    require_data: bool,
    root: Path | None = None,
    process_runner: _ProcessRunner = _run_process,
    version_provider: _VersionProvider = metadata.version,
) -> _DoctorState:
    _check_python(report)
    detected_root = root or find_repository_root()
    if detected_root is None:
        report.fail(
            "Repository root was not found from installed module path "
            f"{Path(__file__).resolve()}."
        )
        return _DoctorState(root=None, git=None)
    detected_root = detected_root.resolve()
    report.passed(f"Repository root is {detected_root}.")
    package_init = Path(__file__).resolve().parents[1] / "__init__.py"
    expected_package_init = (
        detected_root / "src" / "fraud_detection" / "__init__.py"
    ).resolve()
    if package_init == expected_package_init and package_init.is_file():
        report.passed(
            "The active fraud_detection package imports from this repository."
        )
    else:
        report.fail(
            "The active fraud_detection package does not import from this "
            "repository.",
            code="FD-PACKAGE-IMPORT",
            recovery=(
                "Run fraud-detection setup from the repository checkout.",
            ),
            details={
                "active_package": str(package_init),
                "expected_package": str(expected_package_init),
            },
        )

    git_state = _inspect_git(detected_root, report, process_runner)
    _check_environment(
        detected_root,
        report,
        version_provider=version_provider,
        process_runner=process_runner,
    )
    reference_results_valid = _check_reference_results(detected_root, report)
    dataset_valid = _check_data(
        detected_root,
        report,
        require_data=require_data,
    )
    if require_data:
        _record_required_data_identity(
            report,
            reference_results_valid=reference_results_valid,
            dataset_valid=dataset_valid,
        )
    return _DoctorState(root=detected_root, git=git_state)


def _check_tracked_python_syntax(
    root: Path,
    report: _Collector,
    process_runner: _ProcessRunner = _run_process,
) -> bool:
    listed = _git(
        root,
        ["ls-files", "--", "src", "scripts", "tests"],
        process_runner,
    )
    if listed.returncode != 0:
        report.fail(
            f"Could not list tracked Python files: {_completed_output(listed)}"
        )
        return False
    paths = [
        line.strip()
        for line in listed.stdout.splitlines()
        if line.strip().endswith(".py")
    ]
    passed = True
    for relative in paths:
        path = root / relative
        try:
            source = path.read_text(encoding="utf-8-sig")
            compile(source, relative, "exec")
        except SyntaxError as exc:
            report.fail(
                f"Syntax check failed for {relative}:{exc.lineno}: {exc.msg}."
            )
            passed = False
        except (OSError, UnicodeError) as exc:
            report.fail(f"Syntax check could not read {relative}: {exc}")
            passed = False
    if passed:
        report.passed(
            f"In-memory syntax check passed for {len(paths)} tracked Python files."
        )
    return passed


def _check_imports(
    root: Path,
    report: _Collector,
    process_runner: _ProcessRunner = _run_process,
) -> bool:
    code = (
        "from pathlib import Path; import sys; "
        "import fraud_detection; import fraud_detection.cli; "
        "root=Path(sys.argv[1]).resolve(); "
        "package=Path(fraud_detection.__file__).resolve(); "
        "cli=Path(fraud_detection.cli.__file__).resolve(); "
        "assert package.is_relative_to(root), package; "
        "assert cli.is_relative_to(root), cli; "
        "print(package); print(cli)"
    )
    completed = process_runner(
        [sys.executable, "-B", "-c", code, str(root)],
        root,
    )
    if completed.returncode == 0:
        locations = " | ".join(completed.stdout.strip().splitlines())
        report.passed(f"Repository package imports succeeded: {locations}.")
        return True
    report.fail(f"Repository package import failed: {_completed_output(completed)}")
    return False


def _last_output_line(completed: subprocess.CompletedProcess[str]) -> str:
    lines = [
        line.strip()
        for value in (completed.stdout, completed.stderr)
        for line in (value or "").splitlines()
        if line.strip()
    ]
    return lines[-1] if lines else "no subprocess output"


def _check_documented_command_help(
    root: Path,
    report: _Collector,
    process_runner: _ProcessRunner = _run_process,
) -> bool:
    commands = (
        (
            "fraud-detection --help",
            (
                sys.executable,
                "-B",
                "-c",
                "from fraud_detection.cli import main; "
                "raise SystemExit(main(['--help']))",
            ),
        ),
        (
            "fraud-detection setup --help",
            (
                sys.executable,
                "-B",
                "-c",
                "from fraud_detection.cli import main; "
                "raise SystemExit(main(['setup', '--help']))",
            ),
        ),
        (
            "fraud-detection check --help",
            (
                sys.executable,
                "-B",
                "-c",
                "from fraud_detection.cli import main; "
                "raise SystemExit(main(['check', '--help']))",
            ),
        ),
        (
            "fraud-detection run --help",
            (
                sys.executable,
                "-B",
                "-c",
                "from fraud_detection.cli import main; "
                "raise SystemExit(main(['run', '--help']))",
            ),
        ),
        (
            "fraud-detection build --help",
            (
                sys.executable,
                "-B",
                "-c",
                "from fraud_detection.cli import main; "
                "raise SystemExit(main(['build', '--help']))",
            ),
        ),
        (
            "fraud-detection inspect --help",
            (
                sys.executable,
                "-B",
                "-c",
                "from fraud_detection.cli import main; "
                "raise SystemExit(main(['inspect', '--help']))",
            ),
        ),
    )
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        completed_results = tuple(
            executor.map(
                lambda item: process_runner(item[1], root),
                commands,
            )
        )
    passed = True
    for (label, _), completed in zip(
        commands,
        completed_results,
        strict=True,
    ):
        if completed.returncode == 0:
            report.passed(f"{label} completed with exit code 0.")
        else:
            report.fail(
                f"{label} failed with exit code "
                f"{completed.returncode}: {_last_output_line(completed)}"
            )
            passed = False
    return passed


def _structured_report(
    command: Literal["doctor", "check"],
    state: _DoctorState,
    report: _Collector,
    started: float,
) -> DiagnosticReport:
    return DiagnosticReport(
        command=command,
        repository_root=state.root,
        findings=tuple(report.findings),
        elapsed_seconds=time.perf_counter() - started,
    )


def run_doctor(
    *,
    repository_root: Path | None = None,
    require_data: bool = False,
) -> DiagnosticReport:
    """Run read-only diagnostics without rendering console output."""

    started = time.perf_counter()
    report = _Collector()
    state = _perform_doctor_checks(
        report,
        require_data=require_data,
        root=repository_root,
    )
    return _structured_report("doctor", state, report, started)


def run_check(
    *,
    repository_root: Path | None = None,
    require_data: bool = False,
) -> DiagnosticReport:
    """Run the bounded check without pytest, setup, build, or project work."""

    started = time.perf_counter()
    report = _Collector()
    state = _perform_doctor_checks(
        report,
        require_data=require_data,
        root=repository_root,
    )
    if state.root is None:
        report.fail(
            "Fast-check stages require a detected repository root.",
            code="FD-ROOT-NOT-FOUND",
            recovery=("Run the command from a repository checkout.",),
        )
    elif state.git is None or not state.git.is_checkout:
        report.fail(
            "Fast-check stages require a Git checkout.",
            code="FD-GIT-CHECKOUT",
            recovery=("Run the command from a repository checkout.",),
        )
    else:
        _check_tracked_python_syntax(state.root, report)
        _check_imports(state.root, report)
        _check_documented_command_help(state.root, report)
    report.info(f"Fast check runtime: {time.perf_counter() - started:.2f}s.")
    return _structured_report("check", state, report, started)
