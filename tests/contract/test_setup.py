import os
import shutil
import subprocess
from collections.abc import Callable
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from zipfile import ZipFile

import pytest

import fraud_detection.setup.environment as setup

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_UV_ASSET = "uv-x86_64-pc-windows-msvc.zip"
WINDOWS_UV_SHA256 = "8fcb0cb46e1229065e344758980924e569bef5882ef45f46fada8fb24e06b74a"


def _assert_windows_bootstrap_contract(tmp_path: Path) -> None:
    compiler_shell = shutil.which("powershell") or shutil.which("pwsh")
    powershell = shutil.which("pwsh") or compiler_shell
    if os.name != "nt" or powershell is None:
        return

    fixture_root = tmp_path / "windows-bootstrap-contract"
    fixture_root.mkdir()
    fixture_source = fixture_root / "uv-fixture.cs"
    fixture_binary = fixture_root / "uv-fixture.exe"
    fixture_source.write_text(
        "using System;\n"
        "using System.Diagnostics;\n"
        "using System.IO;\n"
        "public static class UvFixture {\n"
        "  public static int Main(string[] args) {\n"
        "    string record = Environment.GetEnvironmentVariable(\"MOCK_UV_RECORD\");\n"
        "    if (record != null) File.AppendAllText(record, string.Join(\" \", args) + Environment.NewLine);\n"
        "    if (args.Length == 1 && args[0] == \"--version\") {\n"
        "      Console.WriteLine(Environment.GetEnvironmentVariable(\"MOCK_UV_VERSION\"));\n"
        "      int code; return int.TryParse(Environment.GetEnvironmentVariable(\"MOCK_UV_EXIT\"), out code) ? code : 0;\n"
        "    }\n"
        "    if (args.Length >= 3 && args[0] == \"--no-config\" && args[1] == \"python\" && args[2] == \"find\") {\n"
        "      string active = Environment.GetEnvironmentVariable(\"VIRTUAL_ENV\");\n"
        "      bool system = Array.IndexOf(args, \"--system\") >= 0;\n"
        "      Console.WriteLine(!system && !String.IsNullOrEmpty(active) ? Path.Combine(active, \"Scripts\", \"python.exe\") : Path.Combine(Environment.GetEnvironmentVariable(\"UV_PYTHON_INSTALL_DIR\"), \"fixture\", \"python.exe\")); return 0;\n"
        "    }\n"
        "    string executable = Process.GetCurrentProcess().MainModule.FileName;\n"
        "    if (args.Length >= 2 && args[0] == \"-c\" && executable.IndexOf(\".venv\", StringComparison.OrdinalIgnoreCase) >= 0) {\n"
        "      if (args[1].Contains(\"paths=(\")) return 1;\n"
        "      return Environment.GetEnvironmentVariable(\"MOCK_VENV_VALID\") == \"0\" ? 1 : 0;\n"
        "    }\n"
        "    return 0;\n"
        "  }\n"
        "}\n",
        encoding="ascii",
    )
    compiler = fixture_root / "compile-fixture.ps1"
    compiler.write_text(
        "param([string]$Source, [string]$Output)\n"
        "Add-Type -TypeDefinition (Get-Content -LiteralPath $Source -Raw) "
        "-OutputAssembly $Output -OutputType ConsoleApplication\n",
        encoding="ascii",
    )
    compiled = subprocess.run(
        [
            compiler_shell,
            "-NoProfile",
            "-File",
            str(compiler),
            "-Source",
            str(fixture_source),
            "-Output",
            str(fixture_binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr

    def create_fixture(
        name: str,
        members: tuple[str, ...],
        version_output: str = "uv 0.12.1",
        version_exit: int = 0,
    ) -> tuple[Path, dict[str, str], Path, Path]:
        clone = fixture_root / name
        (clone / "scripts").mkdir(parents=True)
        (clone / "environment").mkdir()
        shutil.copy2(ROOT / "scripts" / "bootstrap.ps1", clone / "scripts")
        archive = fixture_root / f"{name}.zip"
        with ZipFile(archive, "w") as created:
            if not members:
                created.writestr("README.txt", "no executable\n")
            for member in members:
                created.write(fixture_binary, member)
        archive_hash = sha256(archive.read_bytes()).hexdigest()
        lock = (ROOT / "environment" / "bootstrap_toolchain.lock").read_text(
            encoding="utf-8"
        ).replace(WINDOWS_UV_SHA256, archive_hash, 1)
        (clone / "environment" / "bootstrap_toolchain.lock").write_text(
            lock,
            encoding="utf-8",
        )
        platform_dir = clone / ".tools" / "uv" / "0.12.1" / "windows-x86_64"
        platform_dir.mkdir(parents=True)
        shutil.copy2(archive, platform_dir / WINDOWS_UV_ASSET)
        managed_python = clone / ".tools" / "python" / "fixture" / "python.exe"
        managed_python.parent.mkdir(parents=True)
        shutil.copy2(fixture_binary, managed_python)
        record = fixture_root / f"{name}-record.txt"
        environment = os.environ.copy()
        environment.update(
            {
                "MOCK_UV_RECORD": str(record),
                "MOCK_UV_VERSION": version_output,
                "MOCK_UV_EXIT": str(version_exit),
                "MOCK_VENV_VALID": "1",
            }
        )
        return clone, environment, record, archive

    def run_fixture(
        clone: Path,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-File",
                str(clone / "scripts" / "bootstrap.ps1"),
            ],
            cwd=clone,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    root_clone, root_environment, root_record, _ = create_fixture(
        "root", ("uv.exe",)
    )
    root_result = run_fixture(root_clone, root_environment)
    assert root_result.returncode == 0, root_result.stderr
    venv_python = root_clone / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    shutil.copy2(fixture_binary, venv_python)
    venv_marker = root_clone / ".venv" / "keep"
    venv_marker.write_text("reuse\n", encoding="ascii")
    root_environment["VIRTUAL_ENV"] = str(root_clone / ".venv")
    repeat_result = run_fixture(root_clone, root_environment)
    assert repeat_result.returncode == 0, repeat_result.stderr
    assert venv_marker.read_text(encoding="ascii") == "reuse\n"
    final_dir = root_clone / ".tools" / "uv" / "0.12.1" / "windows-x86_64"
    assert {path.name for path in final_dir.iterdir()} == {WINDOWS_UV_ASSET, "uv.exe"}
    root_calls = root_record.read_text(encoding="utf-8").splitlines()
    assert root_calls.count("--version") == 4
    assert [line for line in root_calls if "python find" in line] == [
        "--no-config python find --managed-python --system cpython@3.12.13",
        "--no-config python find --managed-python --system cpython@3.12.13",
    ]
    root_environment["MOCK_VENV_VALID"] = "0"
    replacement_result = run_fixture(root_clone, root_environment)
    assert replacement_result.returncode == 0, replacement_result.stderr
    assert not (root_clone / ".venv").exists()

    metadata_clone, metadata_environment, _, _ = create_fixture(
        "metadata",
        ("nested/uv.exe",),
        "uv 0.12.1 (329541a50 2026-07-31 x86_64-pc-windows-msvc)",
    )
    metadata_result = run_fixture(metadata_clone, metadata_environment)
    assert metadata_result.returncode == 0, metadata_result.stderr

    for name, members in (
        ("missing", ()),
        ("duplicate", ("one/uv.exe", "two/uv.exe")),
    ):
        clone, environment, _, _ = create_fixture(name, members)
        result = run_fixture(clone, environment)
        assert result.returncode != 0
        assert "FD-UV-EXECUTABLE" in result.stderr

    rejected_outputs = (
        "uv 0.12.2 (metadata)",
        "uv 0.12.10 (metadata)",
        "uv 0.12.1 unexpected-text",
        "prefix uv 0.12.1",
        "uv 0.12.1\nuv 0.12.1",
        "uv 0.12.1 ((metadata))",
        "uv 0.12.1 (one) (two)",
        "",
    )
    for index, output in enumerate(rejected_outputs):
        clone, environment, _, _ = create_fixture(
            f"version-{index}", ("uv.exe",), output
        )
        result = run_fixture(clone, environment)
        assert result.returncode != 0
        assert "FD-UV-VERSION" in result.stderr

    exit_clone, exit_environment, _, _ = create_fixture(
        "version-exit", ("uv.exe",), "uv 0.12.1", 7
    )
    exit_result = run_fixture(exit_clone, exit_environment)
    assert exit_result.returncode != 0
    assert "FD-UV-VERSION" in exit_result.stderr

    checksum_clone, checksum_environment, checksum_record, checksum_archive = (
        create_fixture("checksum", ("uv.exe",))
    )
    checksum_lock = checksum_clone / "environment" / "bootstrap_toolchain.lock"
    checksum_lock.write_text(
        checksum_lock.read_text(encoding="utf-8").replace(
            sha256(checksum_archive.read_bytes()).hexdigest(),
            "0" * 64,
        ),
        encoding="utf-8",
    )
    harness = fixture_root / "checksum-harness.ps1"
    harness.write_text(
        "param([string]$Bootstrap)\n"
        "function Invoke-WebRequest { param([string]$Uri, [string]$OutFile) "
        "Copy-Item -LiteralPath $env:MOCK_UV_ARCHIVE -Destination $OutFile }\n"
        "function Expand-Archive { throw 'Extraction must not start.' }\n"
        "& $Bootstrap\n"
        "exit $LASTEXITCODE\n",
        encoding="ascii",
    )
    checksum_environment["MOCK_UV_ARCHIVE"] = str(checksum_archive)
    checksum = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-File",
            str(harness),
            "-Bootstrap",
            str(checksum_clone / "scripts" / "bootstrap.ps1"),
        ],
        cwd=checksum_clone,
        env=checksum_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checksum.returncode != 0
    assert "FD-CHECKSUM" in checksum.stderr
    assert not checksum_record.exists()


def test_existing_verified_dataset_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / setup.DATASET_FILE
    target.write_bytes(b"verified")
    monkeypatch.setattr(
        setup,
        "_hash_or_failure",
        lambda _path: setup.EXPECTED_SHA256,
    )

    def unexpected_download(_directory: Path, _python: Path) -> Path:
        raise AssertionError("download must not start")

    progress: list[str] = []
    path, downloaded = setup.ensure_dataset(
        data_dir,
        Path("python"),
        downloader=unexpected_download,
        progress=progress.append,
    )

    assert path == target
    assert downloaded is False
    assert target.read_bytes() == b"verified"
    assert progress == [
        "START Dataset — reusing existing canonical CSV",
        "START Dataset — verifying SHA-256",
        "PASS Dataset — SHA-256 verified",
        "PASS Dataset — canonical CSV reused",
    ]


def test_wrong_existing_dataset_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / setup.DATASET_FILE
    target.write_bytes(b"keep-me")
    monkeypatch.setattr(setup, "_hash_or_failure", lambda _path: "wrong")

    def unexpected_download(_directory: Path, _python: Path) -> Path:
        raise AssertionError("download must not start")

    with pytest.raises(setup.SetupFailure, match="will not be overwritten"):
        setup.ensure_dataset(
            data_dir,
            Path("python"),
            downloader=unexpected_download,
        )

    assert target.read_bytes() == b"keep-me"


def test_wrong_python_version_stops_before_any_setup_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_validator = setup.validate_python_version

    def reject_version() -> None:
        original_validator((3, 11, 9))

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("setup side effect must not start")

    monkeypatch.setattr(setup, "validate_python_version", reject_version)
    monkeypatch.setattr(setup.venv, "EnvBuilder", unexpected)
    monkeypatch.setattr(setup, "install_environment", unexpected)
    monkeypatch.setattr(setup, "ensure_dataset", unexpected)

    with pytest.raises(setup.SetupFailure) as captured:
        setup.run_setup(repository_root=tmp_path)

    assert captured.value.code == "FD-PYTHON-VERSION"
    assert captured.value.details["side_effects_started"] is False


def test_download_is_verified_before_atomic_installation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_replace = setup.os.replace

    def verified(path: Path) -> str:
        assert path.read_bytes() == b"canonical-csv"
        events.append("verified")
        return setup.EXPECTED_SHA256

    def atomic_replace(source: Path, destination: Path) -> None:
        events.append("installed")
        original_replace(source, destination)

    monkeypatch.setattr(setup, "_hash_or_failure", verified)
    monkeypatch.setattr(setup.os, "replace", atomic_replace)

    for name, member in (
        ("direct", None),
        ("zip-root", "creditcard.csv"),
        ("zip-nested", "payload/creditcard.csv"),
    ):
        progress: list[str] = []

        def downloader(
            directory: Path,
            _python: Path,
            *,
            archive_member: str | None = member,
        ) -> Path:
            downloaded = directory / setup.DATASET_FILE
            if archive_member is None:
                downloaded.write_bytes(b"canonical-csv")
            else:
                with ZipFile(downloaded, "w") as archive:
                    archive.writestr(archive_member, b"canonical-csv")
            return downloaded

        target, downloaded = setup.ensure_dataset(
            tmp_path / name / "data",
            Path("python"),
            downloader=downloader,
            progress=progress.append,
        )

        assert downloaded is True
        assert target.name == setup.DATASET_FILE
        assert target.read_bytes() == b"canonical-csv"
        assert not target.read_bytes().startswith(b"PK\x03\x04")
        assert progress == [
            "START Dataset — acquiring canonical Kaggle payload",
            "START Dataset — resolving archive payload",
            "START Dataset — verifying SHA-256",
            "PASS Dataset — SHA-256 verified",
            "PASS Dataset — canonical CSV installed",
        ]

    assert events == ["verified", "installed"] * 3


def test_download_mismatch_leaves_target_absent(
    tmp_path: Path,
) -> None:
    cases: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("wrong-direct", "direct", ()),
        ("wrong-zip", "zip", ("creditcard.csv",)),
        ("invalid-zip", "invalid", ()),
        ("missing-member", "zip", ("other.csv",)),
        ("multiple-members", "zip", ("creditcard.csv", "nested/creditcard.csv")),
        ("unexpected-file", "zip", ("creditcard.csv", "notes.txt")),
        ("traversal", "zip", ("../creditcard.csv",)),
        ("absolute", "zip", ("/creditcard.csv",)),
        ("drive", "zip", ("C:/creditcard.csv",)),
    )
    for name, payload_type, members in cases:
        data_dir = tmp_path / name / "data"

        def downloader(
            directory: Path,
            _python: Path,
            *,
            kind: str = payload_type,
            archive_members: tuple[str, ...] = members,
        ) -> Path:
            downloaded = directory / setup.DATASET_FILE
            if kind == "direct":
                downloaded.write_bytes(b"wrong-csv")
            elif kind == "invalid":
                downloaded.write_bytes(b"PK\x03\x04invalid-zip")
            else:
                with ZipFile(downloaded, "w") as archive:
                    for member in archive_members:
                        archive.writestr(member, b"wrong-csv")
            return downloaded

        expected_code = (
            "FD-DOWNLOAD-HASH"
            if name in {"wrong-direct", "wrong-zip"}
            else "FD-DATA-ARCHIVE"
        )
        with pytest.raises(setup.SetupFailure) as captured:
            setup.ensure_dataset(
                data_dir,
                Path("python"),
                downloader=downloader,
            )

        assert captured.value.code == expected_code
        assert not (data_dir / setup.DATASET_FILE).exists()


def test_installation_failure_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fail_install(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        raise subprocess.CalledProcessError(
            9,
            command,
            stderr="package index unavailable",
        )

    monkeypatch.setattr(setup.subprocess, "run", fail_install)

    failure_progress: list[str] = []
    with pytest.raises(setup.SetupFailure) as captured:
        setup.install_environment(
            tmp_path,
            Path("python"),
            capture_output=True,
            progress=failure_progress.append,
        )

    failure = captured.value
    assert failure.code == "FD-INSTALL-FAILED"
    assert failure.details["exit_code"] == 9
    assert failure.details["output"] == "package index unavailable"
    assert any("disk space" in step for step in failure.recovery)
    assert commands == [["python", "-m", "pip", "install", "pip==26.2"]]
    assert failure_progress == ["START pip — installing pinned version 26.2"]

    install_kwargs: list[dict[str, object]] = []
    def successful_install(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        install_kwargs.append(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(setup.subprocess, "run", successful_install)
    progress: list[str] = []
    setup.install_environment(
        tmp_path,
        Path("python"),
        capture_output=False,
        progress=progress.append,
    )

    assert commands == [
        ["python", "-m", "pip", "install", "pip==26.2"],
        ["python", "-m", "pip", "install", "pip==26.2"],
        [
            "python",
            "-m",
            "pip",
            "install",
            "-r",
            str(tmp_path / "environment" / "final_experiment_requirements.txt"),
        ],
        [
            "python",
            "-m",
            "pip",
            "install",
            "-r",
            str(tmp_path / "environment" / "bootstrap_requirements.txt"),
        ],
        [
            "python",
            "-m",
            "pip",
            "install",
            "-e",
            str(tmp_path),
            "--no-deps",
        ],
    ]
    assert progress == [
        "START pip — installing pinned version 26.2",
        "PASS pip — installed",
        "START Runtime dependencies — installing pinned scientific environment",
        "PASS Runtime dependencies — installed",
        "START KaggleHub — installing pinned acquisition dependency",
        "PASS KaggleHub — installed",
        "START Repository package — installing editable local package",
        "PASS Repository package — installed",
    ]
    assert all("capture_output" not in kwargs for kwargs in install_kwargs)
    assert all("text" not in kwargs for kwargs in install_kwargs)
    assert all("--upgrade" not in command for command in commands)
    assert all("build" not in command for command in commands)


def test_kaggle_failure_keeps_reason_and_recovery_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_download(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            1,
            ["python"],
            stderr="authentication denied token=secret-value",
        )

    monkeypatch.setattr(setup.subprocess, "run", fail_download)

    with pytest.raises(setup.SetupFailure) as captured:
        setup.download_creditcard_csv(tmp_path, Path("python"))

    failure = captured.value
    assert failure.code == "FD-KAGGLE-DOWNLOAD"
    assert "authentication denied" in str(failure.details["reason"])
    assert "secret-value" not in str(failure.details["reason"])
    assert "authenticate" in " ".join(failure.recovery).lower()


def test_mocked_setup_happy_path_has_safe_high_level_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _assert_windows_bootstrap_contract(tmp_path)

    events: list[str] = []
    capture_modes: list[bool] = []
    package_root = tmp_path / "src" / "fraud_detection"
    package_root.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("fixture\n", encoding="utf-8")
    (package_root / "__init__.py").write_text("\n", encoding="utf-8")

    class FakeBuilder:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def create(self, _path: Path) -> None:
            events.append("environment")

    def validate_data(_data_dir: Path) -> bool:
        events.append("validation")
        return False

    def install(
        _root: Path,
        _python: Path,
        *,
        capture_output: bool,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        capture_modes.append(capture_output)
        events.append("installation")
        assert progress is not None
        for message in (
            "START pip — installing pinned version 26.2",
            "PASS pip — installed",
            "START Runtime dependencies — installing pinned scientific environment",
            "PASS Runtime dependencies — installed",
            "START KaggleHub — installing pinned acquisition dependency",
            "PASS KaggleHub — installed",
            "START Repository package — installing editable local package",
            "PASS Repository package — installed",
        ):
            progress(message)

    def place_dataset(
        data_dir: Path,
        _python: Path,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[Path, bool]:
        events.append("dataset")
        assert progress is not None
        for message in (
            "START Dataset — acquiring canonical Kaggle payload",
            "START Dataset — resolving archive payload",
            "START Dataset — verifying SHA-256",
            "PASS Dataset — SHA-256 verified",
            "PASS Dataset — canonical CSV installed",
        ):
            progress(message)
        return data_dir / setup.DATASET_FILE, True

    monkeypatch.setattr(setup, "validate_existing_dataset", validate_data)
    monkeypatch.setattr(setup.venv, "EnvBuilder", FakeBuilder)
    monkeypatch.setattr(setup, "install_environment", install)
    monkeypatch.setattr(setup, "ensure_dataset", place_dataset)

    progress: list[str] = []
    result = setup.run_setup(
        repository_root=tmp_path,
        capture_install_output=True,
        progress=progress.append,
    )

    assert events == ["validation", "environment", "installation", "dataset"]
    assert result.dataset_status == "downloaded"
    assert result.dataset_path == tmp_path / "data" / setup.DATASET_FILE
    assert result.interpreter == setup.venv_python(tmp_path / ".venv")
    assert result.expected_sha256 == setup.EXPECTED_SHA256
    assert capture_modes == [True]
    assert progress == [
        "START Repository and Python toolchain — validating",
        "PASS Repository and Python toolchain — ready",
        "START Virtual environment — creating or reusing",
        "PASS Virtual environment — ready",
        "START pip — installing pinned version 26.2",
        "PASS pip — installed",
        "START Runtime dependencies — installing pinned scientific environment",
        "PASS Runtime dependencies — installed",
        "START KaggleHub — installing pinned acquisition dependency",
        "PASS KaggleHub — installed",
        "START Repository package — installing editable local package",
        "PASS Repository package — installed",
        "START Dataset — acquiring canonical Kaggle payload",
        "START Dataset — resolving archive payload",
        "START Dataset — verifying SHA-256",
        "PASS Dataset — SHA-256 verified",
        "PASS Dataset — canonical CSV installed",
    ]

    public_setup = import_module("fraud_detection.setup")
    diagnostic_module = import_module("fraud_detection.setup.diagnostics")
    assert public_setup.__all__ == [
        "SetupFailure",
        "SetupResult",
        "DiagnosticFinding",
        "DiagnosticReport",
        "run_setup",
        "run_doctor",
        "run_check",
    ]
    assert public_setup.SetupFailure is setup.SetupFailure
    assert public_setup.SetupResult is setup.SetupResult
    assert public_setup.DiagnosticFinding is diagnostic_module.DiagnosticFinding
    assert public_setup.DiagnosticReport is diagnostic_module.DiagnosticReport
    assert public_setup.run_setup is setup.run_setup
    assert public_setup.run_doctor is diagnostic_module.run_doctor
    assert public_setup.run_check is diagnostic_module.run_check
    for implementation_name in (
        "bootstrap_main",
        "PYTHON_CONTRACT",
        "parse_pinned_requirements",
        "find_repository_root",
    ):
        with pytest.raises(AttributeError):
            getattr(public_setup, implementation_name)

    installed_full_check = setup._run_installed_full_check
    validation_calls: list[tuple[Path, Path]] = []

    def validate_cli(root: Path, interpreter: Path) -> None:
        validation_calls.append((root, interpreter))

    monkeypatch.setattr(setup, "_run_installed_full_check", validate_cli)
    returncode = setup.bootstrap_main(tmp_path)
    captured = capsys.readouterr()

    assert returncode == 0
    assert captured.err == ""
    assert capture_modes[-1] is False
    assert validation_calls == [
        (tmp_path.resolve(), setup.venv_python(tmp_path / ".venv"))
    ]
    assert "Setup: PASS" in captured.out
    assert "Package: installed" in captured.out
    assert "Validation: PASS (full)" in captured.out
    assert "Dataset: validated (acquired)" in captured.out
    assert "START pip — installing pinned version 26.2" in captured.out
    assert "PASS Repository package — installed" in captured.out
    assert "START Full diagnostic — validating installed repository" in captured.out
    assert "PASS Full diagnostic — completed" in captured.out
    assert "run --profile smoke-synthetic" in captured.out
    assert "Next: fraud-detection check --full" not in captured.out
    assert "fraud-detection doctor" not in captured.out

    monkeypatch.chdir(tmp_path)
    root_none_returncode = setup.bootstrap_main()
    captured = capsys.readouterr()
    assert root_none_returncode == 0
    assert captured.err == ""
    assert validation_calls[-1] == (
        tmp_path.resolve(),
        setup.venv_python(tmp_path / ".venv"),
    )

    unmarked_root = tmp_path.parent / "unmarked-bootstrap-root"
    unmarked_root.mkdir()
    original_run_setup = setup.run_setup
    monkeypatch.setattr(
        setup,
        "run_setup",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("setup must not start without a checkout")
        ),
    )
    failed_root = setup.bootstrap_main(unmarked_root)
    captured = capsys.readouterr()
    assert failed_root == 1
    assert "FD-ROOT-NOT-FOUND" in captured.err
    assert unmarked_root.is_dir()
    monkeypatch.setattr(setup, "run_setup", original_run_setup)

    def fail_validation(_root: Path, _interpreter: Path) -> None:
        raise setup.SetupFailure(
            "FD-POST-CHECK-FAILED",
            "controlled installed diagnostic failure",
            ("Correct the environment and rerun bootstrap.",),
            exit_code=7,
        )

    monkeypatch.setattr(setup, "_run_installed_full_check", fail_validation)
    failed = setup.bootstrap_main(tmp_path)
    captured = capsys.readouterr()

    assert failed == 7
    assert "FD-POST-CHECK-FAILED" in captured.err
    assert "Setup: PASS" not in captured.out
    assert "run --profile smoke-synthetic" not in captured.out

    process_calls: list[tuple[list[str], Path]] = []

    def pass_process(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        process_calls.append((command, Path(str(kwargs["cwd"]))))
        return subprocess.CompletedProcess(command, 0, "PASS", "")

    monkeypatch.setattr(setup.subprocess, "run", pass_process)
    interpreter = setup.venv_python(tmp_path / ".venv")
    installed_full_check(tmp_path, interpreter)

    assert process_calls == [
        (
            [str(setup._installed_cli(interpreter)), "check", "--full"],
            tmp_path,
        )
    ]

    def fail_process(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, "", "diagnostic failed")

    monkeypatch.setattr(setup.subprocess, "run", fail_process)
    with pytest.raises(setup.SetupFailure) as post_check:
        installed_full_check(tmp_path, interpreter)
    assert post_check.value.code == "FD-POST-CHECK-FAILED"
    assert post_check.value.details["exit_code"] == 9
