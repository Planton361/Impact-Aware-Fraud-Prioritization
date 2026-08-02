"""Standard-library-only local setup for fresh clones and the product shell."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import venv
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Literal, Sequence

DATASET_HANDLE = "mlg-ulb/creditcardfraud"
DATASET_FILE = "creditcard.csv"
EXPECTED_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
PYTHON_CONTRACT = ">=3.12,<3.13"
_PIN_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)=="
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)"
)
_REPOSITORY_MARKERS = (
    "pyproject.toml",
    "README.md",
    "src/fraud_detection/__init__.py",
)


def find_repository_root(path: Path | None = None) -> Path | None:
    """Locate the checkout at or above a supplied path or the cwd."""

    detected = Path.cwd() if path is None else Path(path)
    detected = detected.resolve()
    start = detected if detected.is_dir() else detected.parent
    for candidate in (start, *start.parents):
        if all((candidate / name).is_file() for name in _REPOSITORY_MARKERS):
            return candidate
    return None


@dataclass
class SetupFailure(Exception):
    """Expected setup failure that is safe to render without a traceback."""

    code: str
    summary: str
    recovery: tuple[str, ...]
    details: dict[str, object] = field(default_factory=dict)
    exit_code: int = 1

    def __str__(self) -> str:
        return self.summary


@dataclass(frozen=True, slots=True)
class SetupResult:
    """Structured result of a completed local setup."""

    dataset_path: Path
    dataset_status: Literal["downloaded", "reused"]
    interpreter: Path
    expected_sha256: str


def parse_pinned_requirements(path: Path) -> list[tuple[str, str]]:
    """Parse exact pins and reject unsupported or duplicate requirements."""

    pins: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"{path.name}:{line_number} contains an unsupported requirement: "
                f"{raw_line!r}"
            )
        name = match.group("name")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in seen:
            raise ValueError(f"{path.name}:{line_number} duplicates {name}.")
        seen.add(normalized)
        pins.append((name, match.group("version")))
    return pins


def validate_python_version(version_info: Sequence[int] | None = None) -> None:
    version = tuple(version_info or sys.version_info)
    if version[:2] == (3, 12):
        return
    actual = ".".join(str(part) for part in version[:2])
    raise SetupFailure(
        "FD-PYTHON-VERSION",
        f"Python 3.12 is required; setup is running on Python {actual}.",
        (
            "Install Python 3.12.",
            "Run this bootstrap explicitly with the Python 3.12 interpreter.",
        ),
        {
            "actual": actual,
            "required": PYTHON_CONTRACT,
            "side_effects_started": False,
        },
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def venv_python(venv_dir: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return venv_dir / relative


def _redact_secrets(value: str) -> str:
    patterns = (
        r"(?i)(KAGGLE_API_TOKEN\s*[=:]\s*)\S+",
        r"(?i)(token\s*[=:]\s*)\S+",
    )
    redacted = value
    for pattern in patterns:
        redacted = re.sub(pattern, r"\1<redacted>", redacted)
    return redacted


def _install_command(
    command: Sequence[str],
    *,
    capture_output: bool,
) -> None:
    kwargs: dict[str, object] = {"check": True}
    if capture_output:
        kwargs.update({"capture_output": True, "text": True})
    try:
        subprocess.run(list(command), **kwargs)
    except subprocess.CalledProcessError as exc:
        details = _redact_secrets(
            (getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or "").strip()
        )
        raise SetupFailure(
            "FD-INSTALL-FAILED",
            "A pinned environment installation command failed.",
            (
                "Check package-index access and available disk space.",
                "Re-run setup; completed installation steps are safe to reuse.",
            ),
            {
                "command": list(command),
                "exit_code": exc.returncode,
                "output": details,
            },
        ) from exc
    except subprocess.SubprocessError as exc:
        raise SetupFailure(
            "FD-INSTALL-FAILED",
            "A pinned environment installation command failed.",
            (
                "Check package-index access and the local Python installation.",
                "Re-run setup after correcting the reported subprocess issue.",
            ),
            {"command": list(command), "reason": str(exc)},
        ) from exc
    except OSError as exc:
        raise SetupFailure(
            "FD-INSTALL-START",
            "A pinned environment installation command could not be started.",
            (
                "Check the virtual-environment interpreter and file permissions.",
                "Re-run setup after correcting the local environment.",
            ),
            {"command": list(command), "reason": str(exc)},
        ) from exc


def install_environment(
    root: Path,
    python_path: Path,
    *,
    capture_output: bool = False,
    progress: Callable[[str], None] | None = None,
) -> None:
    final_requirements = root / "environment" / "final_experiment_requirements.txt"
    bootstrap_requirements = root / "environment" / "bootstrap_requirements.txt"
    emit = progress or (lambda _message: None)
    phases = (
        (
            "START pip — installing pinned version 26.2",
            "PASS pip — installed",
            [str(python_path), "-m", "pip", "install", "pip==26.2"],
        ),
        (
            "START Runtime dependencies — installing pinned scientific environment",
            "PASS Runtime dependencies — installed",
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "-r",
                str(final_requirements),
            ],
        ),
        (
            "START KaggleHub — installing pinned acquisition dependency",
            "PASS KaggleHub — installed",
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "-r",
                str(bootstrap_requirements),
            ],
        ),
        (
            "START Repository package — installing editable local package",
            "PASS Repository package — installed",
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "-e",
                str(root),
                "--no-deps",
            ],
        ),
    )
    for start_message, pass_message, command in phases:
        emit(start_message)
        _install_command(command, capture_output=capture_output)
        emit(pass_message)


def download_creditcard_csv(download_dir: Path, python_path: Path) -> Path:
    code = (
        "import kagglehub; "
        f"print(kagglehub.dataset_download({DATASET_HANDLE!r}, "
        f"path={DATASET_FILE!r}, output_dir={str(download_dir)!r}))"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = _redact_secrets(
            (exc.stderr or exc.stdout or f"exit code {exc.returncode}").strip()
        )
        raise SetupFailure(
            "FD-KAGGLE-DOWNLOAD",
            "KaggleHub could not download the canonical dataset.",
            (
                "Keep credentials outside the repository and authenticate with "
                "KAGGLE_API_TOKEN, ~/.kaggle/kaggle.json, or kagglehub.login().",
                "Check network access, then rerun setup.",
            ),
            {"reason": details},
        ) from exc
    except subprocess.SubprocessError as exc:
        raise SetupFailure(
            "FD-KAGGLE-DOWNLOAD",
            "KaggleHub could not download the canonical dataset.",
            (
                "Check KaggleHub authentication and network access.",
                "Rerun setup without placing credentials in the repository.",
            ),
            {"reason": _redact_secrets(str(exc))},
        ) from exc
    except OSError as exc:
        raise SetupFailure(
            "FD-KAGGLE-START",
            "The KaggleHub download helper could not be started.",
            (
                "Check the virtual-environment interpreter.",
                "Re-run setup after correcting the environment.",
            ),
            {"reason": str(exc)},
        ) from exc
    reported = result.stdout.strip().splitlines()
    if not reported:
        raise SetupFailure(
            "FD-KAGGLE-RESULT",
            "KaggleHub returned no download path.",
            ("Check KaggleHub authentication and rerun setup.",),
        )
    candidate = Path(reported[-1].strip()).expanduser()
    if candidate.is_dir():
        candidate = candidate / DATASET_FILE
    if not candidate.is_file():
        raise SetupFailure(
            "FD-KAGGLE-RESULT",
            f"KaggleHub did not produce {DATASET_FILE}.",
            ("Check the dataset handle and KaggleHub output, then rerun setup.",),
            {"reported_path": str(candidate)},
        )
    return candidate


def _archive_failure(summary: str, archive: Path) -> SetupFailure:
    return SetupFailure(
        "FD-DATA-ARCHIVE",
        summary,
        (
            "No dataset file was installed.",
            "Check the Kaggle dataset payload, then rerun setup.",
        ),
        {"archive": str(archive)},
    )


def _resolve_downloaded_csv(candidate: Path, download_dir: Path) -> Path:
    try:
        with candidate.open("rb") as stream:
            signature = stream.read(4)
        is_zip = zipfile.is_zipfile(candidate)
    except OSError as exc:
        raise SetupFailure(
            "FD-DATA-READ",
            "The downloaded dataset payload could not be inspected.",
            (
                "No dataset file was installed.",
                "Check local permissions and rerun setup.",
            ),
            {"path": str(candidate), "reason": str(exc)},
        ) from exc
    if not is_zip:
        if signature.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            raise _archive_failure("The downloaded ZIP payload is invalid.", candidate)
        return candidate

    try:
        with zipfile.ZipFile(candidate) as archive:
            files = [member for member in archive.infolist() if not member.is_dir()]
            matches = [
                member
                for member in files
                if PurePosixPath(member.filename.replace("\\", "/")).name
                == DATASET_FILE
            ]
            if len(matches) != 1:
                raise _archive_failure(
                    "The downloaded ZIP must contain exactly one creditcard.csv file.",
                    candidate,
                )
            if len(files) != 1:
                raise _archive_failure(
                    "The downloaded ZIP contains unexpected additional files.",
                    candidate,
                )
            member = matches[0]
            normalized = member.filename.replace("\\", "/")
            member_path = PurePosixPath(normalized)
            mode = member.external_attr >> 16
            if (
                member.filename.startswith(("/", "\\"))
                or re.match(r"^[A-Za-z]:", member.filename) is not None
                or member_path.is_absolute()
                or ".." in member_path.parts
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
            ):
                raise _archive_failure(
                    "The downloaded ZIP contains an unsafe creditcard.csv member.",
                    candidate,
                )
            extraction_dir = Path(
                tempfile.mkdtemp(prefix=".creditcard-extract-", dir=download_dir)
            ).resolve()
            extracted = extraction_dir.joinpath(*member_path.parts).resolve()
            if not extracted.is_relative_to(extraction_dir):
                raise _archive_failure(
                    "The downloaded ZIP member escapes the controlled directory.",
                    candidate,
                )
            extracted.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, extracted.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            return extracted
    except SetupFailure:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SetupFailure(
            "FD-DATA-ARCHIVE",
            "The downloaded ZIP payload could not be extracted safely.",
            (
                "No dataset file was installed.",
                "Check the Kaggle dataset payload, then rerun setup.",
            ),
            {"archive": str(candidate), "reason": str(exc)},
        ) from exc


def _hash_or_failure(path: Path) -> str:
    try:
        return sha256_file(path)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupFailure(
            "FD-DATA-READ",
            "The dataset file could not be hashed.",
            (
                "Check file permissions and local storage.",
                "Retry setup without modifying the file in place.",
            ),
            {"path": str(path), "reason": str(exc)},
        ) from exc


def validate_existing_dataset(data_dir: Path) -> bool:
    """Fail before setup side effects when an existing CSV is non-canonical."""

    target = data_dir / DATASET_FILE
    if not target.exists():
        return False
    actual = _hash_or_failure(target)
    if actual != EXPECTED_SHA256:
        raise SetupFailure(
            "FD-DATA-HASH",
            "The existing dataset has the wrong SHA-256 and will not be overwritten.",
            (
                "Move the mismatching CSV to a safe location outside data/.",
                "Rerun setup to download and verify the canonical CSV.",
            ),
            {
                "path": str(target),
                "actual_sha256": actual,
                "expected_sha256": EXPECTED_SHA256,
                "side_effects_started": False,
            },
        )
    return True


def ensure_dataset(
    data_dir: Path,
    python_path: Path,
    *,
    downloader: Callable[[Path, Path], Path] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, bool]:
    emit = progress or (lambda _message: None)
    target = data_dir / DATASET_FILE
    if target.exists():
        emit("START Dataset — reusing existing canonical CSV")
        emit("START Dataset — verifying SHA-256")
        actual = _hash_or_failure(target)
        if actual == EXPECTED_SHA256:
            emit("PASS Dataset — SHA-256 verified")
            emit("PASS Dataset — canonical CSV reused")
            return target, False
        raise SetupFailure(
            "FD-DATA-HASH",
            "The existing dataset has the wrong SHA-256 and will not be overwritten.",
            (
                "Move the mismatching CSV to a safe location outside data/.",
                "Rerun setup to download and verify the canonical CSV.",
            ),
            {
                "path": str(target),
                "actual_sha256": actual,
                "expected_sha256": EXPECTED_SHA256,
            },
        )
    try:
        emit("START Dataset — acquiring canonical Kaggle payload")
        data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".creditcard-download-",
            dir=data_dir,
        ) as raw_dir:
            downloaded = (downloader or download_creditcard_csv)(
                Path(raw_dir),
                python_path,
            )
            downloaded = Path(downloaded)
            if downloaded.is_dir():
                downloaded = downloaded / DATASET_FILE
            if not downloaded.is_file():
                raise SetupFailure(
                    "FD-KAGGLE-RESULT",
                    f"KaggleHub did not produce {DATASET_FILE}.",
                    ("No dataset file was installed; check KaggleHub and rerun setup.",),
                    {"reported_path": str(downloaded)},
                )
            emit("START Dataset — resolving archive payload")
            downloaded = _resolve_downloaded_csv(downloaded, Path(raw_dir))
            emit("START Dataset — verifying SHA-256")
            actual = _hash_or_failure(downloaded)
            if actual != EXPECTED_SHA256:
                raise SetupFailure(
                    "FD-DOWNLOAD-HASH",
                    "The downloaded dataset has the wrong SHA-256 and was not installed.",
                    (
                        "Do not move the downloaded file into data/.",
                        "Check the official dataset source before rerunning setup.",
                    ),
                    {
                        "path": str(downloaded),
                        "actual_sha256": actual,
                        "expected_sha256": EXPECTED_SHA256,
                        "target": str(target),
                    },
                )
            emit("PASS Dataset — SHA-256 verified")
            os.replace(downloaded, target)
            emit("PASS Dataset — canonical CSV installed")
    except SetupFailure:
        raise
    except OSError as exc:
        raise SetupFailure(
            "FD-DATA-INSTALL",
            "The verified dataset could not be installed atomically.",
            (
                "Check data/ permissions and available disk space.",
                "Rerun setup; an existing file will never be overwritten.",
            ),
            {"path": str(target), "reason": str(exc)},
        ) from exc
    return target, True


def run_setup(
    *,
    repository_root: Path,
    capture_install_output: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SetupResult:
    """Execute the shared setup workflow after a side-effect-free version gate."""

    emit = progress or (lambda _message: None)
    root = repository_root.resolve()
    emit("START Repository and Python toolchain — validating")
    validate_python_version()
    emit("PASS Repository and Python toolchain — ready")
    data_dir = root / "data"
    venv_dir = root / ".venv"
    validate_existing_dataset(data_dir)
    emit("START Virtual environment — creating or reusing")
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
    except OSError as exc:
        raise SetupFailure(
            "FD-VENV-CREATE",
            "The local virtual environment could not be created or reused.",
            (
                "Check directory permissions and available disk space.",
                "Rerun setup after correcting the local issue.",
            ),
            {"path": str(venv_dir), "reason": str(exc)},
        ) from exc
    python_path = venv_python(venv_dir)
    emit("PASS Virtual environment — ready")
    install_environment(
        root,
        python_path,
        capture_output=capture_install_output,
        progress=emit,
    )
    dataset_path, downloaded = ensure_dataset(
        data_dir,
        python_path,
        progress=emit,
    )
    return SetupResult(
        dataset_path=dataset_path,
        dataset_status="downloaded" if downloaded else "reused",
        interpreter=python_path,
        expected_sha256=EXPECTED_SHA256,
    )


def _render_bootstrap_failure(error: SetupFailure) -> None:
    print(f"ERROR {error.code} {error.summary}", file=sys.stderr)
    for name, value in sorted(error.details.items()):
        print(f"DETAIL {name}={value}", file=sys.stderr)
    for index, step in enumerate(error.recovery, start=1):
        print(f"RECOVERY {index} {step}", file=sys.stderr)


def _installed_cli(interpreter: Path) -> Path:
    name = "fraud-detection.exe" if os.name == "nt" else "fraud-detection"
    return interpreter.with_name(name)


def _run_installed_full_check(root: Path, interpreter: Path) -> None:
    command = [str(_installed_cli(interpreter)), "check", "--full"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupFailure(
            "FD-POST-CHECK-START",
            "The installed full diagnostic could not be started.",
            (
                "Inspect the project environment and package installation.",
                "Rerun bootstrap after correcting the reported local issue.",
            ),
            {
                "command": command,
                "reason": str(exc),
                "side_effects_started": True,
            },
        ) from exc
    if completed.returncode != 0:
        output = (completed.stderr or completed.stdout or "").strip()
        raise SetupFailure(
            "FD-POST-CHECK-FAILED",
            "The installed full diagnostic failed after setup.",
            (
                "Review the diagnostic failure without deleting the environment.",
                "Correct the reported issue, then rerun bootstrap.",
            ),
            {
                "command": command,
                "exit_code": completed.returncode,
                "output": _redact_secrets(output),
                "side_effects_started": True,
            },
        )


def _bootstrap_next_command(root: Path, interpreter: Path) -> str:
    cli = _installed_cli(interpreter)
    try:
        relative = cli.relative_to(root)
    except ValueError:
        executable = str(cli)
    else:
        executable = str(relative) if os.name == "nt" else relative.as_posix()
        executable = f".\\{executable}" if os.name == "nt" else executable
    return f"{executable} run --profile smoke-synthetic"


def _discover_repository_root(path: Path) -> Path | None:
    return find_repository_root(path)


def bootstrap_main(root: Path | None = None) -> int:
    try:
        candidate = Path.cwd() if root is None else root
        repository_root = _discover_repository_root(candidate)
        if repository_root is None or (
            root is not None and repository_root != root.resolve()
        ):
            raise SetupFailure(
                "FD-ROOT-NOT-FOUND",
                "Repository root was not found.",
                ("Run bootstrap from the repository checkout.",),
                {"path": str(candidate.resolve()), "side_effects_started": False},
            )
        result = run_setup(
            repository_root=repository_root,
            capture_install_output=False,
            progress=print,
        )
        print("START Full diagnostic — validating installed repository")
        _run_installed_full_check(repository_root, result.interpreter)
        print("PASS Full diagnostic — completed")
    except SetupFailure as exc:
        _render_bootstrap_failure(exc)
        return exc.exit_code
    except Exception as exc:
        _render_bootstrap_failure(
            SetupFailure(
                "FD-SETUP-UNEXPECTED",
                f"{type(exc).__name__}: {exc}",
                (
                    "Check the reported local error.",
                    "Retry the bootstrap after correcting it.",
                ),
            )
        )
        return 1
    dataset_state = "acquired" if result.dataset_status == "downloaded" else "reused"
    print("Setup: PASS")
    print(f"Environment: {result.interpreter}")
    print("Package: installed")
    print("Validation: PASS (full)")
    print(f"Dataset: validated ({dataset_state})")
    print(f"Next: {_bootstrap_next_command(repository_root, result.interpreter)}")
    return 0
